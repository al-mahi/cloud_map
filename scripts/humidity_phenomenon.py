#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mlp
from mpl_toolkits.mplot3d.axes3d import Axes3D
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from std_msgs.msg import String, Float32
import serial
import datetime as dt


class SenseHumidity(object):
    def __init__(self, scale=20, dim=2):
        self._scale = scale
        self._dim = dim
        self._space = tuple([scale for _ in range(dim)])
        self._true_hum_A = 0.
        self._true_hum_B = 0.
        self._true_hum_C = 0.
        self._pose_A = Pose()
        self._pose_B = Pose()
        self._pose_C = Pose()
        self._serial = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=2)

    def callback_sensor_pose_A(self, pose): self._pose_A = pose
    def callback_sensor_pose_B(self, pose): self._pose_B = pose
    def callback_sensor_pose_C(self, pose): self._pose_C = pose

    def cast_humidity(self):
        rospy.init_node("Humidity", log_level=rospy.DEBUG)
        rate = rospy.Rate(2)
        rospy.Subscriber("/solo/A/pose_euclid", Pose, callback=self.callback_sensor_pose_A)
        rospy.Subscriber("/solo/B/pose_euclid", Pose, callback=self.callback_sensor_pose_B)
        rospy.Subscriber("/solo/C/pose_euclid", Pose, callback=self.callback_sensor_pose_C)
        pub_A = rospy.Publisher("A/humidity", data_class=Float32, queue_size=10)
        pub_B = rospy.Publisher("B/humidity", data_class=Float32, queue_size=10)
        pub_C = rospy.Publisher("C/humidity", data_class=Float32, queue_size=10)

        while not rospy.is_shutdown():
            xA, yA, zA = map(int, map(round, np.array(self._pose_A.position.__getstate__())[:]))
            xB, yB, zB = map(int, map(round, np.array(self._pose_B.position.__getstate__())[:]))
            xC, yC, zC = map(int, map(round, np.array(self._pose_C.position.__getstate__())[:]))
            try:
                msg = str(self._serial.readline())
                if msg.startswith("B_H:"):
                    self._true_hum_B = float(msg[:-1].split(":")[1])
            except Exception as e:
                rospy.logdebug("[A Temp Read Error]: {}".format(e.message))
            rospy.logdebug("Hum: B({},{},{}) {}".format(xB, yB, zB, self._true_hum_B))
            if self._dim == 3:
                pub_A.publish(self._true_hum_A)
                pub_B.publish(self._true_hum_B)
                pub_C.publish(self._true_hum_C)
            if self._dim == 2:
                pub_A.publish(self._true_hum_A)
                pub_B.publish(self._true_hum_B)
                pub_C.publish(self._true_hum_C)
            rospy.logdebug("B_HUM[{}]@{}:{}".format(dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime(
                "%H:%M:%S"), self._pose_B.position.__getstate__()[:], self._true_hum_B))
            rate.sleep()


if __name__ == "__main__":
    scale = 20
    dim = 2
    print("Reading temperature sensor")
    if rospy.has_param("/scale"): scale = int(rospy.get_param("/scale"))
    if rospy.has_param("/dim"): dim = int(rospy.get_param("/dim"))
    profile = SenseHumidity(scale, dim)
    profile.cast_humidity()
    # profile.plot_static()


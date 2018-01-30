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
# for flightgear
from simulator import Simulator
import os


class SenseTemperature(object):
    def __init__(self, scale=20, dim=2):
        self._scale = scale
        self._dim = dim
        self._space = tuple([scale for _ in range(dim)])
        self._true_tmp_A = 0.
        self._true_tmp_B = 0.
        self._true_tmp_C = 0.
        self._pose_A = Pose()
        self._pose_B = Pose()
        self._pose_C = Pose()
        self._serial = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=2)

    def callback_sensor_pose_A(self, pose): self._pose_A = pose
    def callback_sensor_pose_B(self, pose): self._pose_B = pose
    def callback_sensor_pose_C(self, pose): self._pose_C = pose

    def cast_temperature(self):
        rospy.init_node("Temperature", log_level=rospy.DEBUG)
        rate = rospy.Rate(2)
        vendor_of_A = rospy.get_param('/As_vendor')
        vendor_of_B = rospy.get_param('/Bs_vendor')
        vendor_of_C = rospy.get_param('/Cs_vendor')
        rospy.Subscriber("/"+vendor_of_A+"/A/pose_euclid", Pose, callback=self.callback_sensor_pose_A)
        rospy.Subscriber("/"+vendor_of_B+"/B/pose_euclid", Pose, callback=self.callback_sensor_pose_B)
        rospy.Subscriber("/"+vendor_of_C+"/C/pose_euclid", Pose, callback=self.callback_sensor_pose_C)
        pub_A = rospy.Publisher("A/temperature", data_class=Float32, queue_size=10)
        pub_B = rospy.Publisher("B/temperature", data_class=Float32, queue_size=10)
        pub_C = rospy.Publisher("C/temperature", data_class=Float32, queue_size=10)
        UDP_IP = os.environ.get("ROS_IP")
        simA = Simulator({'IP': UDP_IP, "port_send": 41001, "port_recv": 41101})
        simB = Simulator({'IP': UDP_IP, "port_send": 42001, "port_recv": 42102})


        while not rospy.is_shutdown():
            xA, yA, zA = map(int, map(round, np.array(self._pose_A.position.__getstate__())[:]))
            xB, yB, zB = map(int, map(round, np.array(self._pose_B.position.__getstate__())[:]))
            xC, yC, zC = map(int, map(round, np.array(self._pose_C.position.__getstate__())[:]))

            # need to change this file to a generalized version of sensor reading, robot vendor and reading method
            if vendor_of_A=='solo':
                try:
                    msg = str(self._serial.readline())
                    if msg.startswith("A_T:"):
                        self._true_tmp_A = 1.8*float(msg[:-1].split(":")[1])+32.
                except Exception as e:
                    rospy.logdebug("[A Temp Read Error]: {}".format(e.message))
            elif vendor_of_A=='flightgear':
                self._true_tmp_A = 1.8* float(simA.FGRecv()[38])+32.
            rospy.logdebug("Temp: A({},{},{}) {}".format(xA, yA, zA, self._true_tmp_A))
            if self._dim == 3:
                pub_A.publish(self._true_tmp_A)
                pub_B.publish(self._true_tmp_B)
                pub_C.publish(self._true_tmp_C)
            if self._dim == 2:
                pub_A.publish(self._true_tmp_A)
                pub_B.publish(self._true_tmp_B)
                pub_C.publish(self._true_tmp_C)
            # rospy.logdebug("A_TMP[{}]@{}:{}".format(dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime(
            #     "%H:%M:%S"), self._pose_A.position.__getstate__()[:], self._true_tmp_A))
            rate.sleep()


if __name__ == "__main__":
    scale = 20
    dim = 2
    print("Reading temperature sensor")
    if rospy.has_param("/scale"): scale = int(rospy.get_param("/scale"))
    if rospy.has_param("/dim"): dim = int(rospy.get_param("/dim"))
    profile = SenseTemperature(scale, dim)
    profile.cast_temperature()
    # profile.plot_static()


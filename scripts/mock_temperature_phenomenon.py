#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mlp
from mpl_toolkits.mplot3d.axes3d import Axes3D
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from std_msgs.msg import String, Float32


class MockTemperature(object):
    def __init__(self, scale=20):
        self._scale = scale
        self._true_tmp = np.zeros((self._scale, self._scale, self._scale))
        self._pose_A = Pose()
        self._pose_B = Pose()
        self._pose_C = Pose()
        self.ground_tmp = 90.  # F
        self.above_tmp = 70.  # F
        self.del_tmp = (self.ground_tmp - self.above_tmp)/self._scale

        for i in range(int(scale)):
            self._true_tmp[:, :, i] = self.ground_tmp - i * self.del_tmp
        self._true_tmp[:, :, 10:13] -= 4.
        self._true_tmp[:, :, 11] -= 8.

    @property
    def true_tmp(self):
        """
        :rtype: np.ndarray
        """
        return self._true_tmp

    @true_tmp.setter
    def true_tmp(self, value):
        self._true_tmp = value

    def callback_sensor_pose_A(self, pose): self._pose_A = pose
    def callback_sensor_pose_B(self, pose): self._pose_B = pose
    def callback_sensor_pose_C(self, pose): self._pose_C = pose

    def cast_temperature(self):
        rospy.init_node("Sensors")
        rate = rospy.Rate(1)
        rospy.Subscriber("/solo/A/pose_euclid", Pose, callback=self.callback_sensor_pose_A)
        rospy.Subscriber("/solo/B/pose_euclid", Pose, callback=self.callback_sensor_pose_B)
        rospy.Subscriber("/solo/C/pose_euclid", Pose, callback=self.callback_sensor_pose_C)
        pub_A = rospy.Publisher("A/mock_temperature", data_class=Float32, queue_size=10)
        pub_B = rospy.Publisher("B/mock_temperature", data_class=Float32, queue_size=10)
        pub_C = rospy.Publisher("C/mock_temperature", data_class=Float32, queue_size=10)

        while not rospy.is_shutdown():
            xA, yA, zA = map(int, np.array(self._pose_A.position.__getstate__())[:])
            xB, yB, zB = map(int, np.array(self._pose_B.position.__getstate__())[:])
            xC, yC, zC = map(int, np.array(self._pose_C.position.__getstate__())[:])
            print("psoe & temp", xA, yA, zA, self._true_tmp[xA, yA, zA])
            pub_A.publish(self._true_tmp[xA, yA, zA])
            pub_B.publish(self._true_tmp[xB, yB, zB])
            pub_C.publish(self._true_tmp[xC, yC, zC])
            rate.sleep()

    def plot_static(self):
        fig = plt.figure(figsize=plt.figaspect(1./3.))
        cax1 = fig.add_axes([.01, .05, .02, .9])
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        norm = mlp.colors.Normalize(vmin=np.min(self.true_tmp), vmax=np.max(self.true_tmp), clip=True)
        x, y, z = np.meshgrid(np.arange(0, self._scale, dtype='int'), np.arange(0, self._scale, dtype='int'),
                              np.arange(0, self._scale, dtype='int'))
        p = ax1.scatter(x, y, z, c=self.true_tmp, norm=norm, alpha=.7)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Alt")
        ax1.set_title("Static Temperature")
        ax1.set_xlim(0., self._scale)
        ax1.set_ylim(0., self._scale)
        ax1.set_zlim(0., self._scale)
        cb = plt.colorbar(p, cax=cax1)
        plt.show()

if __name__ == "__main__":
    scale = 20
    print("Mocking temperature sensor")
    if rospy.has_param("/scale"): scale = int(rospy.get_param("/scale"))
    profile = MockTemperature(scale)
    profile.cast_temperature()
    # profile.plot_static()


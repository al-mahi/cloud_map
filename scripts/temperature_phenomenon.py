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
    def __init__(self, scale=20, dim=2):
        self._scale = scale
        self._dim = dim
        self._space = tuple([scale for _ in range(dim)])
        self._true_tmp = np.zeros(self._space)
        self._pose_A = Pose()
        self._pose_B = Pose()
        self._pose_C = Pose()
        self.ground_tmp = 90.  # F
        self.above_tmp = 70.  # F
        self.del_tmp = (self.ground_tmp - self.above_tmp)/self._scale
        if self._dim == 3:
            for i in range(int(scale)):
                self._true_tmp[:, :, i] = self.ground_tmp - i * self.del_tmp
            self._true_tmp[:, :, int((self._scale/2)-1):int((self._scale/2)+1)] -= 4.
            self._true_tmp[:, :, int((self._scale/2))] -= 8.

        if self._dim == 2:
            for i in range(int(scale)):
                self._true_tmp[:, i] = self.ground_tmp - i * self.del_tmp
            self._true_tmp[:, 10:14] -= 2.
            self._true_tmp[:, 11:12] -= 4.

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
        rate = rospy.Rate(2)
        rospy.Subscriber("/UAV_FW/A/pose", Pose, callback=self.callback_sensor_pose_A)
        rospy.Subscriber("/UAV_FW/B/pose", Pose, callback=self.callback_sensor_pose_B)
        rospy.Subscriber("/UAV_FW/C/pose", Pose, callback=self.callback_sensor_pose_C)
        pub_A = rospy.Publisher("A/mock_temperature", data_class=Float32, queue_size=10)
        pub_B = rospy.Publisher("B/mock_temperature", data_class=Float32, queue_size=10)
        pub_C = rospy.Publisher("C/mock_temperature", data_class=Float32, queue_size=10)
        while not rospy.is_shutdown():
            xA, yA, zA = map(int, map(round, np.array(self._pose_A.position.__getstate__())[:]))
            xB, yB, zB = map(int, map(round, np.array(self._pose_B.position.__getstate__())[:]))
            xC, yC, zC = map(int, map(round, np.array(self._pose_C.position.__getstate__())[:]))
            if self._dim == 3:
                pub_A.publish(self._true_tmp[yA, xA, zA])
                pub_B.publish(self._true_tmp[yB, xB, zB])
                pub_C.publish(self._true_tmp[yC, xC, zC])
            if self._dim == 2:
                pub_A.publish(self._true_tmp[yA, xA])
                pub_B.publish(self._true_tmp[yB, xB])
                pub_C.publish(self._true_tmp[yC, xC])
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
    dim = 2
    print("Reading temperature sensor")
    if rospy.has_param("/scale"): scale = int(rospy.get_param("/scale"))
    if rospy.has_param("/dim"): dim = int(rospy.get_param("/dim"))
    profile = MockTemperature(scale, dim)
    profile.cast_temperature()
    # profile.plot_static()


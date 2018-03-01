#!/usr/bin/python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os
import rospy
from cloud_map.msg import euclidean_location, Belief, path_history
import datetime as dt

class Visualization(object):
    def __init__(self, name, d):
        self.name = name
        self.d = d
        self.pose = np.zeros(3)
        self.neighbor_names = []
        self._pose_received = {}
        self._intention_received = {}
        self._intention_sent = {}
        self._intention_self = np.zeros(shape=(d, d, d))
        self._cutoff_percentile = 10  # the threshold intention of a voxel for visibility
        self._path_history = {"xs": [], "ys": [], "zs": []}
        self._goal_euclid = np.zeros(d)

    @property
    def tag(self):
        return "Viz{}[{}]:".format(self.name, dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%H:%M:%S"))

    def update_all(self, num, unused, ax1self, ax2cax1):
        self.update_self_viz(num=num, unused_iterable=unused, ax=ax1self, cax=ax2cax1)

    def start_node(self):
        rospy.init_node(self.name, log_level=rospy.DEBUG)
        freq = 1
        rate = rospy.Rate(freq)
        rospy.logdebug("Launch Visual Node {} with neighbors {}".format(self.name, self.neighbor_names))
        rospy.Subscriber("/UAV/" + self.name + "/intention_self", data_class=Belief, callback=self.callback_intention_self)
        rospy.Subscriber("/UAV/" + self.name + "/pose", data_class=euclidean_location, callback=self.callback_pose)
        # rospy.Subscriber("/UAV/" + self.name + "/path_history", path_history, callback=self.callback_path_history)
        rospy.Subscriber("/UAV/{}/next_way_point_euclid".format(self.name), data_class=euclidean_location,
                         callback=self.callback_next_euclidean_way_point)
        for nm in self.neighbor_names:
            rospy.Subscriber("/UAV/" + nm + "/pose", euclidean_location, callback=self.callback_pose)

        # Attaching 3D axis to the figure
        fig_num = {"A": 0, "B": 1, "C": 2, "D": 3}
        fig = plt.figure(fig_num[self.name])
        ax1self = fig.add_subplot(1, 1, 1, projection='3d')
        x, y, z = np.meshgrid(np.arange(0, self.d, dtype='int'), np.arange(0, self.d, dtype='int'), np.arange(0, self.d, dtype='int'))

        unused = [ax1self.scatter(x, y, z)]  # dummy iterable required for animation

        ax2cax1 = fig.add_axes([.02, .2, .02, .6])

        ind = 0
        interval = 1000 / freq
        anims = [None]
        anims[ind] = animation.FuncAnimation(fig, self.update_all, 10000, fargs=(unused, ax1self, ax2cax1), interval=interval,
                                             blit=False, repeat=False)

        plt.suptitle("Robot {}'s Joint Belief".format(self.name))
        # plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        while not rospy.is_shutdown():
            rate.sleep()

    def update_self_viz(self, num, unused_iterable, ax, cax):
        """
        :type num: int
        :type dists: list
        :type ax: Axes3D
        :return:
        """
        if num > 30 and num%4==0:
            try:
                plt.savefig("{}/frames{}/{}_{:04d}_joint.png".format(os.path.dirname(__file__), self.name, self.name, num))
            except ValueError as ex:
                rospy.logerr("{}: Viz ".format(self.name, ex.message, ex.args))
        ax.clear()
        # Setting the axes properties
        try:
            ax.set_xlim3d([0.0, float(self.d)+1.])
            ax.set_ylim3d([0.0, float(self.d)+1.])
            ax.set_zlim3d([0.0, float(self.d)+1.])
        except ValueError:
            pass

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xticks(np.linspace(0., self.d, num=5.))
        ax.set_yticks(np.linspace(0., self.d, num=5.))
        ax.set_zticks(np.linspace(0., self.d, num=5.))

        ax.set_title("{}'s joint frame#{}".format(self.name, num))
        t = self._intention_self
        # ind = np.where((t > np.percentile(t, self._cutoff_percentile))
        #                | (t < np.percentile(t, 100.-self._cutoff_percentile))
        #                & (t != t.max()))
        # ind = np.where((t > np.percentile(t, self._cutoff_percentile)))
        ind = np.where(t > self._cutoff_percentile/100.)

        norm = mpl.colors.Normalize(vmin=np.min(self._intention_self), vmax=np.max(self._intention_self), clip=True)
        x, y, z = ind[0], ind[1], ind[2]
        # ax.text(self.pose[0], self.pose[1], self.pose[2], "{}({:.1f},{:.1f},{:.1f})".format(self.name, self.pose[0], self.pose[1], self.pose[2]), fontsize='small')
        ax.text(self.pose[0], self.pose[1], self.pose[2], "{}".format(self.name), fontsize='small')
        ax.text(self._goal_euclid[0], self._goal_euclid[1], self._goal_euclid[2], "G", fontsize='small')
        for nm in self.neighbor_names:
            if self._pose_received.has_key(nm):
                p3 = self._pose_received[nm]
                ax.text(p3[0], p3[1], p3[2], "{}({:.1f},{:.1f},{:.1f})".format(nm, p3[0], p3[1], p3[2]), fontsize='small')
        p = ax.scatter(x, y, z, c=t[ind], norm=norm, alpha=.6)
        if all(val is not None for val in self._path_history.values()):
            try:
                path_line = ax.plot(xs=self._path_history["xs"], ys=self._path_history["ys"],
                                    zs=self._path_history["zs"], c='r')
            except ValueError as ex:
                rospy.logdebug("{} {}".format(self.tag, ex.message))
        cb = plt.colorbar(p, cax=cax)
        cb.set_label("Joint dist. at A")
        return p

    def callback_pose(self, pose):
        """
        :type pose: euclidean_location
        """
        p = np.array(pose.__getstate__()[1:]).astype(float)
        self._pose_received[pose.header.frame_id] = p
        if pose.header.frame_id == self.name:
            self.pose = p
            self._path_history["xs"].append(pose.x)
            self._path_history["ys"].append(pose.y)
            self._path_history["zs"].append(pose.z)
            # self._path_history["xs"] = self._path_history["xs"][-200:]
            # self._path_history["ys"] = self._path_history["ys"][-200:]
            # self._path_history["zs"] = self._path_history["zs"][-200:]

    def callback_next_euclidean_way_point(self, goal_euclid):
        """
        :param goal_euclid: goal in euclidian coordinate
        :type goal_euclid: euclidean_location
        :return:
        """
        if goal_euclid is not None:
            self._goal_euclid = np.array(goal_euclid.__getstate__()[1:]).astype(float)

    def callback_intention_self(self, pdf_intention):
        """
        :type pdf_intention: Belief
        """
        tmp = np.asanyarray(pdf_intention.data).reshape(self.d, self.d, self.d)
        # a fter intorucing the walls. assign zeros to walls instead of high value so that they dont block the
        # visualization of the actual dist.
        # th = 8
        # tmp[0:th, :, :] = tmp[:, 0:th, :] = tmp[:, :, 0:th] = 0.
        # tmp[self.d-1-th:, :, :] = tmp[:, self.d-1-th:, :] = tmp[:, :, self.d-1-th:] = 0.
        self._intention_self = tmp

    # def callback_path_history(self, msg):
    #     """:type msg: path_history"""
    #     if msg.header.frame_id == self.name:
    #         self._path_history["xs"] = msg.xs
    #         self._path_history["ys"] = msg.xs
    #         self._path_history["zs"] = msg.xs


def visualizer(name):
    # delete previous output plots
    cmd = "rm /home/alien/catkin_ws/src/cloud_map/scripts/frames{}/*".format(name)
    os.system(cmd)
    # default scale 2 will be overwritten by rosparam space
    scale = int(rospy.get_param("/scale"))
    viz = Visualization(name=name, d=scale)
    if rospy.has_param("/Visual/" + name + "/neighbors"):
        neighbours = rospy.get_param("/Visual/" + name + "/neighbors").split("_")
        for n in neighbours:
            # todo validate the format of n
            if n != '':
                viz.neighbor_names.append(n)
    viz.start_node()

    print("--------------------------------------------------------------------------")


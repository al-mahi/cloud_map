#!/usr/bin/python

"""
============
3D animation
============

A simple example of an animated plot... In 3D!
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import os
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from cloud_map.msg import Belief
from mpl_toolkits.axes_grid1 import make_axes_locatable
from subprocess import call
import datetime as dt
# plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
from cloud_map.msg import Belief


class Visualization(object):
    def __init__(self, name, d):
        self.name = name
        self.scale = d
        self.pose = np.zeros(2)
        self.neighbor_names = []
        self._pose_received = {}
        self._intention_received = {}
        self._intention_sent = {}
        self._intention_self = Belief()

    def start_node(self):
        rospy.init_node(self.name, log_level=rospy.DEBUG)
        rate = rospy.Rate(1)
        # for nm in self.neighbor_names:
        #     rospy.logdebug("[Visual] {} waiting for {}.".format(self.name, nm))
        #     rospy.wait_for_message("/UAV/" + nm + "/pose", Pose, timeout=5.)
        rospy.logdebug("Launch Visual Node {} with neighbors {}".format(self.name, self.neighbor_names))
        rospy.Subscriber("/UAV/" + self.name + "/intention_self", Belief, callback=self.callback_intention_self)
        rospy.Subscriber("/UAV/" + self.name + "/pose", Pose, callback=self.callback_pose, callback_args=self.name)
        if len(self.neighbor_names) > 0:
            for nm in self.neighbor_names:
                rospy.Subscriber("/UAV/" + nm + "/pose", Pose, callback=self.callback_pose, callback_args=nm)
            rospy.Subscriber("/UAV/" + self.name + "/intention_sent", Belief, callback=self.callback_intention_sent)
            rospy.Subscriber("/UAV/" + self.name + "/intention_received", Belief, callback=self.callback_intention_received)
        # Attaching 3D axis to the figure
        fig_num = {"A":0, "B":1, "C":2}
        fig = plt.figure(fig_num[self.name])
        # ax = p3.Axes3D(fig.add_subplot(2,1,1))
        ax1self = fig.add_subplot(3, 2, 1)
        # ax2color = fig.add_subplot(3, 2, 2)
        ax3recv1 = fig.add_subplot(3, 2, 3)
        ax4recv2 = fig.add_subplot(3, 2, 5)
        ax5sent1 = fig.add_subplot(3, 2, 4)
        ax6sent2 = fig.add_subplot(3, 2, 6)
        x, y = np.meshgrid(np.arange(0, self.scale, dtype='int'), np.arange(0, self.scale, dtype='int'))

        unused = [ax1self.scatter(x, y)]  # dummy iterable required for animation

        ax2cax1 = fig.add_axes([.01, .65, .01, .25])
        ax2cax3 = fig.add_axes([.01, .35, .01, .25])
        ax2cax4 = fig.add_axes([.01, .05, .01, .25])
        ax2cax5 = fig.add_axes([.60, .65, .01, .25])
        ax2cax6 = fig.add_axes([.80, .65, .01, .25])

        # Creating the Animation object
        # Contains plots for the 5  that I am plotting joint, A->B, A<-B, A->C, A<-C


        # anim1 = animation.FuncAnimation(fig, self.update_self_viz, 1000, fargs=(unused, ax1self, ax2cax1), interval=50, blit=False)
        # anim1.save("{}_movie.gif".format(self.name), writer='imagemagick')
        # anims = []
        # anims.append(animation.FuncAnimation(
        #     fig, self.update_self_viz, 1000, fargs=(unused, ax1self, ax2cax1), interval=50, blit=False))
        ind = 0
        anims = [None, None, None, None, None]
        anims[ind] = animation.FuncAnimation(
            fig, self.update_self_viz, 1000, fargs=(unused, ax1self, ax2cax1), interval=50, blit=False)

        if len(self.neighbor_names) > 0:
            for to_uav, ax_sent, cax in zip(self.neighbor_names, [ax5sent1, ax6sent2], [ax2cax5, ax2cax6]):
                # anim2 = animation.FuncAnimation(fig, self.update_sent_viz, 1000, fargs=(unused, ax_sent, to_uav, cax),
                #                                interval=50, blit=False)
                # anim2.save("{}_sentto_{}.gif".format(self.name,to_uav), writer='imagemagick', fps=30)
                # anims.append(animation.FuncAnimation(fig, self.update_sent_viz, 1000, fargs=(unused, ax_sent, to_uav, cax),
                #                                      interval=50, blit=False))
                ind += 1
                anims[ind] = animation.FuncAnimation(
                    fig, self.update_sent_viz, 1000, fargs=(unused, ax_sent, to_uav, cax), interval=50, blit=False)

            for from_uav, ax_rec, cax in zip(self.neighbor_names, [ax3recv1, ax4recv2], [ax2cax3, ax2cax4]):
                # anim3 = animation.FuncAnimation(fig, self.update_received_viz, 1000,
                #                                fargs=(unused, ax_rec, from_uav, cax), interval=50, blit=False)
                # anim3.save("{}_recfrm_{}.gif".format(self.name, from_uav), writer='imagemagick', fps=30)
                # anims.append(animation.FuncAnimation(fig, self.update_received_viz, 1000,
                #                                      fargs=(unused, ax_rec, from_uav, cax), interval=50, blit=False))
                ind += 1
                anims[ind] = animation.FuncAnimation(
                    fig, self.update_received_viz, 1000, fargs=(unused, ax_rec, from_uav, cax), interval=50, blit=False)

        plt.suptitle("Robot {}'s Perspective-{}".format(self.name,dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%M:%S.%f")), fontsize=10)
        # plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        while not rospy.is_shutdown():
            pass

    def update_self_viz(self, num, unused_iterable, ax, cax):
        """
        :type num: int
        :type dists: list
        :type ax: Axes
        :return:
        """
        if num < 200:
            plt.savefig("{}/frames{}/{}_{:04d}_joint.png".format(os.path.dirname(__file__), self.name, self.name, num))
        ax.clear()
        # Setting the axes properties
        ax.set_xlim([0.0, float(self.scale)])
        ax.set_xlabel('X')

        ax.set_ylim([0.0, float(self.scale)])
        ax.set_ylabel('Y')

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title("{}-joint-#-{}-T1-{}-T0-{}".
                     format(self.name, num, dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%M:%S.%f"),
                            dt.datetime.fromtimestamp(self._intention_self.header.stamp.to_time()).strftime("%M:%S.%f")), fontsize=10)
        t = np.asanyarray(self._intention_self.data).reshape(self.scale, self.scale)

        ind = np.where(t > .0001)
        norm = mpl.colors.Normalize(vmin=np.min(t), vmax=np.max(t), clip=True)
        x, y = ind[0], ind[1]
        ax.text(self.pose[0], self.pose[1], "{}({},{})".format(self.name, self.pose[0], self.pose[1]), fontsize='small')
        for nm in self.neighbor_names:
            if self._pose_received.has_key(nm):
                p2 = self._pose_received[nm]
                # fix3d not self pose
                ax.text(p2[0], p2[1], "{}({},{})".format(nm, p2[0], p2[1]), fontsize='small')

        p = ax.scatter(x, y, c=t[ind], norm=norm)
        # x, y = np.meshgrid(np.arange(0, self.scale, dtype='int'), np.arange(0, self.scale, dtype='int'))
        # p = plt.contour(x, y, t)
        # plt.clabel(p, inline=True,fontsize=10)
        cb = plt.colorbar(p, cax=cax)
        cb.set_label("Joint dist. at A")
        return p

    def update_sent_viz(self, num, unused_iterable, ax, to_uav, cax):
        """
        :type num: int
        :type dists: list
        :type ax: Axes
        :return:
        """

        # frames dir should be created to run
        # if num < 200:
        #     plt.savefig("{}/frames{}/{}_to_{}_{:04d}.png".format(os.path.dirname(__file__), self.name, self.name, to_uav, num))
        ax.clear()
        # Setting the axes properties
        ax.set_xlim([0.0, float(self.scale)])
        # ax.set_xlabel('X')

        ax.set_ylim([0.0, float(self.scale)])
        # ax.set_ylabel('Y')

        ax.set_xticks([])
        ax.set_yticks([])

        ax.text(self.pose[0], self.pose[1], self.name)
        for nm in self.neighbor_names:
            if self._pose_received.has_key(nm):
                p2 = self._pose_received[nm]
                ax.text(p2[0], p2[1], nm)
        if self._intention_sent.has_key(to_uav):
            ax.set_title("{}>{}#{}-T1-{}-T0-{}".format(self.name, to_uav, num,dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%M:%S.%f"),
                                                       dt.datetime.fromtimestamp(self._intention_sent[to_uav].header.stamp.to_time()).strftime("%M:%S.%f")), fontsize=10)


            t = np.asanyarray(self._intention_sent[to_uav].data).reshape(self.scale, self.scale)
            norm = mpl.colors.Normalize(vmin=np.min(t), vmax=np.max(t), clip=True)
            ind = np.where(t > .0001)
            x, y = ind[0], ind[1]
            p = ax.scatter(x, y, c=t[ind], norm=norm)
            # x, y = np.meshgrid(np.arange(0, self.scale, dtype='int'), np.arange(0, self.scale, dtype='int'))
            # print(x.shape, y.shape, t.shape, self.scale)
            # p = plt.contour(x, y, t)
            # plt.clabel(p, inline=True,fontsize=10)
            # uncomment when save in fig
            cb = plt.colorbar(p, cax=cax)
            cb.set_label("{}-->{}".format(self.name, to_uav, num))
            cb.ax.set_xticklabels([np.min(t[ind]), np.max(t[ind])])
            return p
        else:
            # print("No update sent to {}".format(to_uav))
            return unused_iterable[-1]

    def update_received_viz(self, num, unused_iterable, ax, from_uav, cax):
        """
        :type num: int
        :type dists: list
        :type ax: Axes3D
        :return:
        """
        # if num < 200:
        #     plt.savefig("{}/frames{}/{}_from_{}_{:04d}.png.png".format(os.path.dirname(__file__), self.name, self.name, from_uav, num))
        ax.clear()
        # Setting the axes properties
        ax.set_xlim([0.0, float(self.scale)])
        ax.set_xlabel('X')

        ax.set_ylim([0.0, float(self.scale)])
        ax.set_ylabel('Y')

        ax.set_xticks([])
        ax.set_yticks([])

        # ax.set_title("{}<--{}#{}-{}".format(self.name, from_uav, num,del_t.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%M:%S.%.f")), fontsize=10)

        ax.text(self.pose[0], self.pose[1], self.name)
        for nm in self.neighbor_names:
            if self._pose_received.has_key(nm):
                p3 = self._pose_received[nm]
                ax.text(p3[0], p3[1], nm)

        if self._intention_received.has_key(from_uav):
            ax.set_title("{}<{}#{}-T1-{}-T0-{}".format(self.name, from_uav, num,dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%M:%S.%f"),
                                                       dt.datetime.fromtimestamp(self._intention_received[from_uav].header.stamp.to_time()).strftime("%M:%S.%f")), fontsize=10)
            t = np.asanyarray(self._intention_received[from_uav].data).reshape(self.scale, self.scale)
            norm = mpl.colors.Normalize(vmin=np.min(t), vmax=np.max(t), clip=True)
            ind = np.where(t > .0001)
            x, y = ind[0], ind[1]
            p = ax.scatter(x, y, c=t[ind], norm=norm)
            # x, y = np.meshgrid(np.arange(0, self.scale, dtype='int'), np.arange(0, self.scale, dtype='int'))
            # p = plt.contour(x, y, t)
            # plt.clabel(p, inline=True,fontsize=10)
            # uncomment when save in fig
            cb = plt.colorbar(p, cax=cax)
            cb.set_label("{}<--{}".format(self.name, from_uav, num))
            cb.ax.set_xticklabels([np.min(t[ind]), np.max(t[ind])])
            return p
        else:
            print("no update received from {}!".format(from_uav))
            return unused_iterable[-1]

    def callback_pose(self, pose, from_uav):
        """
        :type pdf_intention: Pose
        """
        if from_uav == self.name:
            self.pose = np.array(pose.position.__getstate__()[:2]).astype(int)
        else:
            self._pose_received[from_uav] = np.array(pose.position.__getstate__()).astype(int)

    def callback_intention_sent(self, pdf_intention):
        """
        :type pdf_intention: Belief
        """
        # tmp = np.asanyarray(pdf_intention.data).reshape(self.scale, self.scale)
        self._intention_sent[pdf_intention.header.frame_id[2]] = pdf_intention #tmp

    def callback_intention_received(self, pdf_intention):
        """
        :type pdf_intention: Belief
        """
        # tmp = np.asanyarray(pdf_intention.data).reshape(self.scale, self.scale)
        self._intention_received[pdf_intention.header.frame_id] = pdf_intention #tmp

    def callback_intention_self(self, pdf_intention):
        """
        :type pdf_intention: Belief
        """
        # tmp = np.asanyarray(pdf_intention.data).reshape(self.scale, self.scale)
        self._intention_self = pdf_intention #tmp


def visualizer(name):
    # delete previous output plots
    cmd = "rm /home/alien/catkin_ws/src/cloud_map/scripts/frames{}/*".format(name)
    os.system(cmd)
    cmd = "rm /home/alien/catkin_ws/src/cloud_map/scripts/gifs/*".format(name)
    os.system(cmd)
    # default scale 2 will be overwritten by rosparam space
    scale = int(rospy.get_param("/scale"))
    viz = Visualization(name=name, d=scale)
    if rospy.has_param("/Visual/" + name + "/neighbors"):
        neighbours = rospy.get_param("/Visual/" + name + "/neighbors").split("_")
        for n in neighbours:
            # todo validate the format of n
            if n!= '':
                viz.neighbor_names.append(n)
    viz.start_node()

    print("-----------------------making gifs-----------------------")
    os.system("sh /home/alien/catkin_ws/src/cloud_map/scripts/gifify.sh")


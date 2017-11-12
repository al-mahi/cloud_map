#!/usr/bin/python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib import cm
import os
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from cloud_map.msg import Belief
from mpl_toolkits.axes_grid1 import make_axes_locatable
from subprocess import call
from std_msgs.msg import String, Float32
from scipy.interpolate import griddata


class VisualizeHumidity(object):
    def __init__(self, name, dim, scale, q_size=60):
        self._name = name
        self._dim = dim
        self._scale = scale
        self._space = tuple([scale for _ in range(dim)])
        self._dim = len(self._space)
        self._cutoff_percentile = 90
        self._measured_humidity = np.nan * np.ones(self._space)
        self._inferred_humidity = np.zeros(self._space)
        self._phi_humidity_change = np.zeros(self._space)
        self._pose = Pose()

    def callback_sensor_pose_euclid(self, pose):
        self._pose = pose

    def update_measured_viz(self, num, unused_iterable, ax, cax):
        """
        :type num: int
        :type dists: list
        :type ax: Axes3D
        :return:
        """
        plt.savefig("{}/frames{}/{}_{:04d}_joint.png".format(os.path.dirname(__file__), self._name, self._name, num))
        ax.clear()
        # Setting the axes properties

        try:
            ax.set_xlim3d([0.0, float(self._scale)])
            ax.set_ylim3d([0.0, float(self._scale)])
            ax.set_zlim3d([0.0, float(self._scale)])
        except ValueError:
            pass
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        ax.set_zlabel('Z')
        ax.set_xticks([])
        ax.set_yticks([])
        F = self._measured_humidity
        ax.set_title("{}_Measured#{}".format(self._name, num))
        t = np.nan_to_num(F)
        ind = np.where(t > 0)
        norm = mpl.colors.BoundaryNorm(np.linspace(60, 90, 20), plt.cm.jet.N)
        x, y, z = ind[0], ind[1], ind[2]
        p = ax.scatter(x, y, z, c=t[ind], norm=norm, alpha=.6)
        cb = plt.colorbar(p, cax=cax, norm=norm)
        cb.set_label("{} in F".format(self._name))
        return p

    def callback_humidity(self, hum):
        """
        :type hum: Float32
        :return:
        """
        val = float(hum.data)
        x, y, z = map(int, map(round, np.array(self._pose.position.__getstate__())[:]))
        if self._dim == 3:
            self._measured_humidity[y, x, z] = val
        if self._dim == 2:
            self._measured_humidity[y, x] = val
        # calc humidity change
        mask = ~np.isnan(self._measured_humidity)
        values = self._measured_humidity[mask]
        points = mask.nonzero()

        method = 'nearest'

        if self._dim == 3:
            xx, yy, zz = np.meshgrid(np.arange(self._scale), np.arange(self._scale), np.arange(self._scale))
            self._inferred_humidity = griddata(points, values, (xx, yy, zz), method=method)

        if self._dim == 2:
            xx, yy = np.meshgrid(np.arange(self._scale), np.arange(self._scale))
            self._inferred_humidity = griddata(points, values, (xx, yy), method=method)
        grad = np.gradient(self._inferred_humidity)
        if self._dim == 3: self._phi_humidity_change = np.sqrt(grad[0]**2 + grad[1]**2 + grad[2]**2)
        if self._dim == 2: self._phi_humidity_change = np.sqrt(grad[0]**2 + grad[1]**2)
        # self._phi_humidity_change = np.nan_to_num(self._phi_humidity_change)
        self._phi_humidity_change /= np.sum(self._phi_humidity_change)
        # print("{}\nnan count ={}".format(self._name, np.count_nonzero(np.isnan(self._measured_humidity))))
        self._phi_humidity_change = np.nan_to_num(self._phi_humidity_change)
        self._phi_humidity_change = 1. - self._phi_humidity_change

    def update_gradient_viz(self, num, unused_iterable, ax):
        """
        :type num: int
        :type dists: list
        :type ax: Axes3D
        :return:
            """
        plt.savefig("{}/frames{}/{}_{:04d}_joint.png".format(os.path.dirname(__file__), self._name, self._name, num))
        ax.clear()
        # Setting the axes properties
        try:
            ax.set_xlim3d([0.0, float(self._scale)])
            ax.set_ylim3d([0.0, float(self._scale)])
            ax.set_zlim3d([0.0, float(self._scale)])
        except ValueError:
            pass

        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        ax.set_zlabel('Z')

        ax.set_xticks([])
        ax.set_yticks([])
        F = self._phi_humidity_change
        ax.set_title("{} gradient#{}".format(self._name, num))
        t = np.nan_to_num(F)
        ind = np.where((t > np.percentile(t, self._cutoff_percentile))
                       | (t < np.percentile(t, 100-self._cutoff_percentile))
                       & (t != t.max()))
        norm = mpl.colors.Normalize(vmin=np.min(F), vmax=np.max(F), clip=True)
        x, y, z = ind[0], ind[1], ind[2]
        p = ax.scatter(x, y, z, c=t[ind], norm=norm, alpha=.6)

        return p

    def update_inferred_viz(self, num, unused_iterable, ax, cax):
        """
        :type num: int
        :type dists: list
        :type ax: Axes3D
        :return:
            """
        plt.savefig("{}/frames{}/{}_{:04d}_joint.png".format(os.path.dirname(__file__), self._name, self._name, num))
        ax.clear()
        # Setting the axes properties
        try:
            ax.set_xlim3d([0.0, float(self._scale)])
            ax.set_ylim3d([0.0, float(self._scale)])
            ax.set_zlim3d([0.0, float(self._scale)])
        except ValueError:
            pass
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        ax.set_zlabel('Z')

        ax.set_xticks([])
        ax.set_yticks([])
        F = self._inferred_humidity

        npts = 60
        x = np.arange(0, self._scale)
        y = np.arange(0, self._scale)
        z = np.arange(0, self._scale)
        px, py, pz = np.random.choice(x, npts), np.random.choice(y, npts), np.random.choice(z, npts)
        ax.set_title("{} inferred#{}".format(self._name, num))
        t = np.nan_to_num(F[py, px, pz])
        norm = mpl.colors.Normalize(vmin=np.min(F), vmax=np.max(F), clip=True)
        p = ax.scatter(px, py, pz, c=t, norm=norm, alpha=.6)
        cb = plt.colorbar(p, cax=cax)
        cb.set_label("{} in F".format(self._name))
        return p

    def update_vertical_viz(self, num, unused_iterable, ax):
        """
        :type num: int
        :type dists: list
        :type ax: Axes3D
        :return:
        """
        plt.savefig("{}/frames{}/{}_{:04d}_joint.png".format(os.path.dirname(__file__), self._name, self._name, num))
        ax.clear()

        F = np.sum(self._inferred_humidity, axis=(0,1)) / self._scale**2
        ax.set_title("{} vertical profiel#{}".format(self._name, num))
        ax.yaxis.tick_right()
        ax.set_xlabel("{}".format(self._name))
        t = np.nan_to_num(F)
        p = ax.plot(t, np.arange(self._scale))
        ax.set_label("Vertical {}".format(self._name))
        return p

    def start(self):
        rospy.init_node(self._name, log_level=rospy.DEBUG)
        rate = rospy.Rate(2)
        rospy.Subscriber("/Sensors/B/humidity", Float32, callback=self.callback_humidity)
        rospy.Subscriber("/UAV/B/pose", Pose, callback=self.callback_sensor_pose_euclid)


        fig = plt.figure(2)
        # ax = p3.Axes3D(fig.add_subplot(2,1,1))
        ax1measured = fig.add_subplot(2, 2, 1, projection='3d')
        ax2inferred = fig.add_subplot(2, 2, 3, projection='3d')
        ax3gradient = fig.add_subplot(2, 2, 2, projection='3d')
        ax4vertical = fig.add_subplot(2, 2, 4)

        x, y, z = np.meshgrid(np.arange(0, self._scale, dtype='int'), np.arange(0, self._scale, dtype='int'), np.arange(0, self._scale, dtype='int'))

        unused = [ax1measured.scatter(x, y, z)]  # dummy iterable required for animation

        cax1 = fig.add_axes([.01, .65, .01, .25])
        cax2 = fig.add_axes([.01, .15, .01, .25])

        ind = 0
        interval = 300
        anims = [None, None, None, None, None]
        anims[0] = animation.FuncAnimation(fig, self.update_measured_viz, 1000, fargs=(unused, ax1measured, cax1),interval=interval, blit=False)
        anims[1] = animation.FuncAnimation(fig, self.update_inferred_viz, 1000, fargs=(unused, ax2inferred, cax2),interval=interval, blit=False)
        anims[2] = animation.FuncAnimation(fig, self.update_gradient_viz, 1000, fargs=(unused, ax3gradient),interval=interval, blit=False)
        anims[3] = animation.FuncAnimation(fig, self.update_vertical_viz, 1000, fargs=(unused, ax4vertical),interval=interval, blit=False)

        plt.show()
        while not rospy.is_shutdown():
            rate.sleep()


def visualize_humidity(name="Humidity"):
    cmd = "rm /home/alien/catkin_ws/src/cloud_map/scripts/frames{}/*".format(name)
    os.system(cmd)
    dim = int(rospy.get_param("/dim"))
    if dim == 2:
        rospy.logdebug("humidity visulaization is only available on 3D!")
        exit()
    scale = int(rospy.get_param("/scale"))
    humidity_viz = VisualizeHumidity(name=name, dim=dim, scale=scale)
    humidity_viz.start()


if __name__ == '__main__':
    print("visual humidity!!!!!")
    visualize_humidity()

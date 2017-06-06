#!/usr/bin/python

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import time
import mpl_toolkits.mplot3d.axes3d as p3


class uav(object):
    def __init__(self, name, d=8, sensor_tmp=False, sensor_hum=False, sensor_pressure=False):
        self.name = name
        self._d = d
        self.position = [2, 3, 6]
        self._intention_unexplored = np.ones(shape=(d, d, d)) / (d**3.)  # uniform prior

        dist = multivariate_normal(mean=self.position, cov=self._d * np.identity(3))
        indices = np.ndindex(self._d, self._d, self._d)
        self._intention_unexplored = np.array(
            map(lambda x: dist.pdf(np.array([x[0], x[1], x[2]])), indices), dtype=np.float32).reshape(self._d, self._d, self._d)
        # add walls as less interesting place to go
        walls = np.zeros(self._intention_unexplored.shape, dtype=np.float32)
        walls[0, :, :] = walls[:, 0, :] = walls[:, :, 0] = 1./self._d**3
        walls[self._d-1, :, :] = walls[:, self._d-1, :] = walls[:, :, self._d-1] = 1./self._d**3
        self._intention_unexplored += walls
        self._last_visited = np.zeros(shape=(d, d, d))

    def visualize(self, num):
        fig = plt.figure(self.name)
        ax = fig.add_subplot(211,projection='3d')
        # Setting the axes properties
        ax.set_xlim3d([0.0, float(self._d)])
        ax.set_xlabel('X')

        ax.set_ylim3d([0.0, float(self._d)])
        ax.set_ylabel('Y')

        ax.set_zlim3d([0.0, float(self._d)])
        ax.set_zlabel('Z')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_title("{}'s joint frame#{}".format(self.name, num))
        t = self._intention_unexplored
        ind = np.where(t > .0001)
        norm = mpl.colors.Normalize(vmin=np.min(t), vmax=np.max(t), clip=True)
        x, y, z = ind[0], ind[1], ind[2]
        ax.text(self.position[0], self.position[1], self.position[2], self.name)
        ax.scatter(x, y, z, c=t[ind], norm=norm, alpha=.8)
        # plt.show()
        plt.savefig("{}/gdtest/{}_{:04d}_grad.png".format(os.path.dirname(__file__), self.name, num))

    def fly(self):
        #  current position
        x, y, z = map(int, self.position[:])
        # can not work with small value in gradient. so workaround is normalize the vel along all axis
        # then if value along an axis is more than 1/3 then move one unit towards that axis
        # global gradient calculation is not feasible in real-time for two reasons
        # 1. it will be very slow
        # 2. numerical un-stability of extreme values
        # Fix to this problem is slicing out the local distribution and calculate gradient of that
        # confining gradient calculation withing range
        # todo find out a better solution
        local_range = int(self._d/2)

        # slice local dist. calc indices
        # index of local dist. s- starting
        # index of local dist. e- ending
        sx = max(0, (x + 1) - local_range)
        sy = max(0, (y + 1) - local_range)
        sz = max(0, (z + 1) - local_range)
        ex = min(self._d, x + local_range)
        ey = min(self._d, y + local_range)
        ez = min(self._d, z + local_range)
        local_dist = self._intention_unexplored[sx:ex, sy:ey, sz:ez]
        # translate position in the sliced frame of ref
        nx, ny, nz = x-sx-1, y-sy-1, z-sz-1
        # todo more debuggin required here because the plot is not convincing
        print("{} pos {}".format(self.name, self.position))
        # where to fly. find the optimum in intention dist of the inverse
        local_gradient = np.array(np.gradient(1.-local_dist), dtype=np.float32)
        gradient = local_gradient[:, nx, ny, nz]
        print("{} grad {}".format(self.name, gradient))
        # need to scale because the gradient is a dist. over the space thus a small number.
        # take an gamma value that ensures at least movement of one unit in coordinate. an alpha
        # taking gamma 1000
        gamma = 3250.
        scaled_gradient = gamma * gradient
        print("{} new_grad {}".format(self.name, scaled_gradient))
        #  new_gradient has to be int rather than float. so round it
        rounded_gradient = np.around(scaled_gradient)
        print("{} rounded_grad {}".format(self.name, rounded_gradient))
        new_pos = self.position + rounded_gradient
        print("new pos [{}] = {}".format(new_pos, self._intention_unexplored[int(new_pos[0]), int(new_pos[1]), int(new_pos[2])]))
        self.position = new_pos

if __name__ == '__main__':
    uav = uav('A')
    for i in range(100):
        print("iteration {}".format(i))
        uav.visualize(i)
        uav.fly()
        # time.sleep(1)
    print("min at dist[{}] = {}".format(np.where(uav._intention_unexplored == uav._intention_unexplored.min()), uav._intention_unexplored.min()))

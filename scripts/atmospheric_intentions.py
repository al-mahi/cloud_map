#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mlp
from mpl_toolkits.mplot3d.axes3d import Axes3D


class StaticAtmosphere(object):
    def __init__(self, d=300):
        self._d = float(d)
        self._true_tmp = np.zeros((d, d, d))
        self._true_pas = np.zeros((d, d, d))
        self._true_hum = np.zeros((d, d, d))

        self.ground_tmp = 60.  # F
        self.above_tmp = 80.  # F
        self.del_tmp = (self.above_tmp - self.ground_tmp)/self._d
        self.ground_pas = 101.  # KPa
        self.above_pas = 97.
        self.del_pas = (self.above_pas - self.ground_pas)/self._d
        self.ground_hum = 90.
        self.above_hum = 60.
        self.del_hum = (self.above_hum - self.ground_hum)/self._d

        for i in range(int(d)):
            self._true_tmp[:, :, i] = self.ground_tmp + i * self.del_tmp
            self._true_pas[:, :, i] = self.ground_pas + i * self.del_pas
            self._true_hum[:, :, i] = self.ground_hum + i * self.del_hum

    @property
    def true_tmp(self):
        """
        :rtype: np.ndarray
        """
        return self._true_tmp

    @true_tmp.setter
    def true_tmp(self, value):
        self._true_tmp = value

    @property
    def true_pas(self):
        """
        :rtype: np.ndarray
        """
        return self._true_pas

    @true_pas.setter
    def true_pas(self, value):
        self._true_pas = value

    @property
    def true_hum(self):
        """
        :rtype: np.ndarray
        """
        return self._true_hum

    @true_hum.setter
    def true_hum(self, value):
        self._true_hum = value


def plot_static():
    sample = 10
    # -----------------------------------tmepreatur plot----------------------------------------------------------------
    fig = plt.figure(figsize=plt.figaspect(1./3.))
    ax1 = fig.add_subplot(3, 1, 1, projection='3d')
    norm = mlp.colors.Normalize(vmin=np.min(atm.true_tmp), vmax=np.max(atm.true_tmp), clip=True)
    atm.true_tmp[:50, 50:100, 50:200] = atm.above_tmp
    for i in range(d):
        X = np.random.randint(low=0, high=d, size=sample)
        Y = np.random.randint(low=0, high=d, size=sample)
        Z = i*np.ones(sample, dtype='int')
        T = atm.true_tmp[X, Y, Z]
        ax1.scatter(X, Y, Z, c=T, norm=norm, marker='o')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Alt")
    ax1.set_title("Static Temperature")
    ax1.set_xlim(0., d)
    ax1.set_ylim(0., d)
    ax1.set_zlim(0., d)

    # ---------------------------------pressure plot--------------------------------------------------------------------
    ax2 = fig.add_subplot(3, 1, 2, projection='3d')
    norm = mlp.colors.Normalize(vmin=np.min(atm.true_pas), vmax=np.max(atm.true_pas), clip=True)
    atm.true_pas[50:200, 100:200, 50:100] = atm.above_pas
    for i in range(d):
        X = np.random.randint(low=0, high=d, size=sample)
        Y = np.random.randint(low=0, high=d, size=sample)
        Z = i*np.ones(sample, dtype='int')
        T = atm.true_pas[X, Y, Z]
        ax2.scatter(X, Y, Z, c=T, norm=norm, marker='o')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Alt")
    ax2.set_title("Static Pressure")
    ax2.set_xlim(0., d)
    ax2.set_ylim(0., d)
    ax2.set_zlim(0., d)

    # ---------------------------------humidity plot--------------------------------------------------------------------
    ax3 = fig.add_subplot(3, 1, 3, projection='3d')
    norm = mlp.colors.Normalize(vmin=np.min(atm.true_hum), vmax=np.max(atm.true_hum), clip=True)
    atm.true_hum[200:d, 50:100, 250:d] = atm.ground_hum
    for i in range(d):
        X = np.random.randint(low=0, high=d, size=sample)
        Y = np.random.randint(low=0, high=d, size=sample)
        Z = i*np.ones(sample, dtype='int')
        T = atm.true_hum[X, Y, Z]
        ax3.scatter(X, Y, Z, c=T, norm=norm, marker='o')
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Alt")
    ax3.set_title("Static Humidity")
    ax3.set_xlim(0., d)
    ax3.set_ylim(0., d)
    ax3.set_zlim(0., d)

    plt.suptitle("Static Atmosphere.")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    d = 300
    atm = StaticAtmosphere(d)
    plot_static()

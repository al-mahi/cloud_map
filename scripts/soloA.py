#!/usr/bin/python

import numpy as np
import rospy
from solo_2d import solo_2d

if __name__ == '__main__':
    name = "A"
    port = int(rospy.get_param("/solo/{}/port".format(name)))
    solo = solo_2d(name=name, port=port)
    solo.arm_and_takeoff(start_at_euclid=[5., 3., 8.])

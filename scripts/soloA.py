#!/usr/bin/python

import numpy as np
import rospy
from solo_3dr import solo_3dr

if __name__ == '__main__':
    name = "A"
    dim = int(rospy.get_param("/dim"))
    scale = int(rospy.get_param("/scale"))
    port = int(rospy.get_param("/solo/{}/port".format(name)))
    solo = solo_3dr(name=name, port=port, scale=scale, dim=dim)
    solo.arm_and_takeoff(start_at_euclid=[5., 3., 4.])

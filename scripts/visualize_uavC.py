#!/usr/bin/python
import rospy
import visualization_3d as viz3
import visualization_2d as viz2
import multiprocessing
name = "C"
dim = int(rospy.get_param("/dim"))
pool = multiprocessing.Pool()
if dim == 2:
    pool.map(viz2.visualizer(name=name), [])
elif dim == 3:
    pool.map(viz3.visualizer(name=name), [])

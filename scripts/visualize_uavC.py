#!/usr/bin/python
import rospy
import visualization_3d as viz3
import visualization_2d as viz2
import visualization_2d_contour as viz2cont
import visualization_2d_surface as viz2surf
import multiprocessing
name = "C"
dim = int(rospy.get_param("/dim"))
pool = multiprocessing.Pool()
if dim == 2:
    # pool.map(viz2.visualizer(name=name), [])
    # pool.map(viz2cont.visualizer(name=name), [])
    # pool.map(viz2surf.visualizer(name=name), [])
    viz2cont.visualizer(name=name)

elif dim == 3:
    # pool.map(viz3.visualizer(name=name), [])
    viz3.visualizer(name=name)

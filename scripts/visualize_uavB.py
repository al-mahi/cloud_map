#!/usr/bin/python
import rospy

# import visualization_3d as viz3
import visualization_3d_param as viz3
# import visualization_2d as viz2
import visualization_2d_contour as viz2cont
name = "B"
dim = int(rospy.get_param("/dim"))

if dim == 2:
    viz2cont.visualizer(name=name)
elif dim == 3:
    viz3.visualizer(name=name)

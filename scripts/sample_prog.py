#!/usr/bin/env python

"""
Sample program for flying one Solo drone
author: James Kostas
modified by: S M Al Mahi
"""
import rospy
import cPickle
from std_msgs.msg import String
import time

pub = rospy.Publisher('control_topic', String, queue_size=10)


def sendmsg(param): #sends a message to the control node
    mypickle = cPickle.dumps(param)
    pub.publish(mypickle)


def sample_prog():
    rospy.init_node('sample_prog', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    alt1 = 10
    points = [[36.1116, -97.1056, alt1], [36.1109, -97.1056, alt1], [36.1109, -97.1062, alt1],
              [36.1116, -97.1056, alt1], [36.1116, -97.1056, alt1 + 10]]
    veh_num = 0 # our vehicle number, corresponds to 14550

    print "takeoff"
    sendmsg(["takeoff", veh_num, 10])
    time.sleep(10)

    # go through the points:
    for n in range(len(points)):
        # sendmsg(["yaw", veh_num, 1])
        sendmsg(["goto", veh_num, points[n][0], points[n][1], points[n][2]])
        print "going to point %s" % n
        time.sleep(8)

    print "land"
    sendmsg(["land", veh_num])

    while not rospy.is_shutdown():


        rate.sleep()

if __name__ == '__main__':
    sample_prog()

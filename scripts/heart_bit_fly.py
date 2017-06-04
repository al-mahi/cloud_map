#!/usr/bin/python


import rospy
from std_msgs.msg import String

if __name__ == '__main__':
    rospy.init_node("heart_bit_fly")
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        pub = rospy.Publisher(name="fly", data_class=String, queue_size=10)
        pub.publish("fly")
        rate.sleep()


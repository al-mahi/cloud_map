#!/usr/bin/python

import numpy as np
import rospy
from fg_fixed_wing_interface import fg_fw_interface
import socket

if __name__ == '__main__':
    name = "D"
    dim = int(rospy.get_param("/dim"))
    scale = int(rospy.get_param("/scale"))
    port_send = int(rospy.get_param("/flightgear/port_send".format(name)))
    port_recv = int(rospy.get_param("/flightgear/port_recv".format(name)))
    instance_num = int(rospy.get_param("/{}s_instance".format(name)))

    fgfw_pid = fg_fw_interface(name=name, port_send=port_send + instance_num, port_recv=port_recv + instance_num)
    fgfw_pid.simple_goto()

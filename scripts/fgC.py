#!/usr/bin/python

import numpy as np
import rospy
from flightgear_quad_static import flightgear_quad_static
from flightgear_fixed_wing import flightgear_fixed_wing
import socket

if __name__ == '__main__':
    name = "C"
    dim = int(rospy.get_param("/dim"))
    scale = int(rospy.get_param("/scale"))
    port_send = int(rospy.get_param("/flightgear/port_send".format(name)))
    port_recv = int(rospy.get_param("/flightgear/port_recv".format(name)))
    instance_num = int(rospy.get_param("/{}s_instance".format(name)))
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 8080))
    server_ip = s.getsockname()[0]
    port = s.getsockname()[1]
    s.close()
    fg_uav = flightgear_quad_static(name=name,instance=instance_num,server_id=1, server_ip=server_ip,
                                    port_send=port_send + instance_num, port_recv=port_recv+instance_num, scale=scale,
                                 dim=dim)
    fg_uav.arm_and_takeoff(start_at_euclid=[30., 28., 32.])

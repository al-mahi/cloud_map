#!/usr/bin/python

import numpy as np
import rospy
import socket
from FgLibs import FGthread

if __name__ == '__main__':
    name = "B"
    port_send = int(rospy.get_param("/flightgear/port_send".format(name)))
    port_recv = int(rospy.get_param("/flightgear/port_recv".format(name)))
    instance_num = int(rospy.get_param("/{}s_instance".format(name)))
    vehicle = rospy.get_param('/{}s_type'.format(name))
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 8080))
    server_ip = s.getsockname()[0]
    port = s.getsockname()[1]
    s.close()
    FGthread(
        server_id = 1, instance=instance_num, controller_hostIP=server_ip, freq_in=100, freq_out=100,
        vehicle=vehicle, lat=36.1342542738, lon=-97.0762114789, alt=50,
        iheading=45, ivel=60, ithrottle=0.0001)  # 0.1 -> throttle

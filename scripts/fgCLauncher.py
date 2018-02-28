#!/usr/bin/python

import numpy as np
import rospy
import socket
from FgLibs import FGthread
from cloud_map.msg import geo_location

def euclid_to_geo(NS, EW, UD):
    """
    Converts euclid NED coordinate and converts it to gps latitude and longitude.
    displacement in meters (N and E are positive, S and W are negative), and outputs the new lat/long
    CAUTION: the numbers below are set for use near Stillwater will change at other lattitudes
    :param NS: set as y axis of euclidean coordinate lat
    :param EW: set as x axis of euclidean coordinate lon
    :param UD: set as z axis of eculidean coordinate alt
    :rtype: geo_location
    """
    origin_lat = 36.1333333
    origin_lon = -97.0771
    origin_alt = 5.  # meter
    meters_per_alt = 4.
    meters_per_disposition = 4.
    meters_per_lat = 110961.03  # meters per degree of latitude for use near Stillwater
    meters_per_lon = 90037.25  # meters per degree of longitude
    pose = geo_location()
    lon = origin_lon + meters_per_disposition * EW / meters_per_lon
    lat = origin_lat + meters_per_disposition * NS / meters_per_lat
    alt = origin_alt + meters_per_alt * UD
    pose.longitude = lon
    pose.latitude = lat
    pose.altitude = alt
    return pose


if __name__ == '__main__':
    name = "C"
    port_send = int(rospy.get_param("/flightgear/port_send".format(name)))
    port_recv = int(rospy.get_param("/flightgear/port_recv".format(name)))
    instance_num = int(rospy.get_param("/{}s_instance".format(name)))
    vehicle = rospy.get_param('/{}s_type'.format(name))
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 8080))
    server_ip = s.getsockname()[0]
    port = s.getsockname()[1]
    s.close()
    start_at = euclid_to_geo(NS=15., EW=25, UD=11)
    FGthread(
        server_id = 1, instance=instance_num, controller_hostIP=server_ip,
        control_input_config='ControlInputMageQuadTeleport', freq_in=1, freq_out=1, vehicle=vehicle,
        lat=start_at.latitude, lon=start_at.longitude, alt=start_at.altitude, iheading=45, ivel=60, ithrottle=0.0001)


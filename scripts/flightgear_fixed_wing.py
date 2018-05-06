#!/usr/bin/env python

from __future__ import print_function
from std_msgs.msg import Bool
import datetime as dt
import rospy
import socket
from cloud_map.msg import *
import numpy as np


class flightgear_fixed_wing(object):
    def __init__(self, name, instance, server_id, server_ip, port_send, port_recv, dim=3, scale=16):
        """
        :param port: port number of solo controller to connect
        :param altitude: in meters
        """
        self._name = name
        self._instance = instance
        self._port_send = port_send
        self._port_recv = port_recv
        self._server_id = server_id
        self._server_ip = server_ip
        self._goal_gps = geo_location()
        self._goal_euclid = euclidean_location()
        self._dim = int(rospy.get_param("/dim"))
        self._scale = int(rospy.get_param("/scale"))
        self._space = tuple([self._scale for _ in range(self._dim)])
        # cowboy cricket ground bowling end 36.133642, -97.076528
        self._origin_lat = 36.169097  #36.1333333
        self._origin_lon = -97.088101  #-97.0771
        self._sea_level_ft = int(rospy.get_param("/sea_level_ft"))
        self._origin_alt = 50.
        self._meters_per_alt = 8.
        self._meters_per_disposition = 10.
        self._meters_per_lat = 110961.03  # meters per degree of latitude for use near Stillwater
        self._meters_per_lon = 90037.25  # meters per degree of longitude
        self._tol_meter = .05  # drone to be considered reached a goal if it is withing tol_meter within the goal
        self._tol_lat = 1.e-6
        self._tol_lon = 1.e-6
        self._tol_alt = 0.5

        self._max_lon = self._origin_lon + (self._meters_per_disposition * self._scale) / self._meters_per_lon
        self._max_lat = self._origin_lat + (self._meters_per_disposition * self._scale) / self._meters_per_lat
        self._max_alt = self._origin_alt + (self._meters_per_alt * self._scale)
        print("mx lon lat alt = {}, {}, {}".format(self._max_lon, self._max_lat, self._max_alt))
        self._center_lon = (self._origin_lon + self._max_lon) / 2.
        self._center_lat = (self._origin_lat + self._max_lat) / 2.
        self._center_alt = (self._origin_alt + self._max_alt) / 2.

        self._pose_gps = geo_location()
        self._orientation = orientation_euler()
        self._temperature = temperature()
        self._humidity = humidity()
        self._co2 = CO_2()
        self._velocity_ms = twist_ms()

        self._pub_pose_gps = None
        self._pub_pose_euclid = None
        self._pub_distance_to_goal = None
        self._pub_next_goal_gps = None
        self._pub_temperature = None
        self._pub_humidity = None
        self._pub_orientation_euler = None
        self._pub_co2 = None
        self._pub_vel_ms = None
        self._pub_vel_euclid = None
        self._is_ready = False
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("google.com", 80))
            UDP_IP = s.getsockname()[0]
            s.close()
        except socket.error:
            rospy.logdebug("{}Network connection unavailable...".format(self.tag))
            exit(-1)
        self._udp_ip = UDP_IP

    @property
    def tag(self):
        return "fgfw{}[{}]:".format(self._name, dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%H:%M:%S"))

    def arm_and_takeoff(self, start_at_euclid=None):
        """
        Init ROS node.
        Arms vehicle and fly_grad to aTargetAltitude (in meters).
        """
        rospy.init_node(self._name, log_level=rospy.DEBUG)
        rate = rospy.Rate(10)
        rospy.Subscriber("/fg_interface/{}/sensor_data".format(self._name), data_class=sensor_data,
                         callback=self.callback_fg_sensor)
        # longitude EW = x axis and latitude NS = y axis
        # send solo to initial location
        print('center = ', self._center_lat, self._center_lon, self._center_alt)
        self._goal_euclid = euclidean_location(header=self._name, x=start_at_euclid[0], y=start_at_euclid[1], z=start_at_euclid[2])
        self._goal_gps = self.euclid_to_geo(NS=self._goal_euclid.y, EW=self._goal_euclid.x, UD=self._goal_euclid.z)

        rospy.logdebug("{}Sending to initial goal (x,y,z)=({}) (lon, lat, alt)=({},{},{}) tol=({},{},{})".format(
            self.tag, start_at_euclid, self._goal_gps.longitude, self._goal_gps.latitude,
            self._goal_gps.altitude, self._tol_lon, self._tol_lat, self._tol_alt)
        )
        self._pub_next_goal_gps = rospy.Publisher(self._name + '/next_way_point_gps', data_class=geo_location, queue_size=10)

        while True:
            pose = self.pose_in_euclid()
            if (0. < pose.x < self._scale) and (0 < pose.y < self._scale):
                break
            self._goal_gps = self._pose_gps
            self._pub_next_goal_gps.publish(self._pose_gps)
            rospy.logdebug("{} Waiting....\npose read....\n{}".format(self.tag, pose))
            rospy.sleep(5)
        wait = 60 + np.random.randint(low=5, high=15)
        while wait > 0:
            rospy.logdebug("{} Waiting....{}".format(self.tag, wait))
            wait -= 5
            rospy.sleep(5)

        pub_ready = rospy.Publisher('{}/ready'.format(self._name), data_class=Bool, queue_size=1)
        rospy.Subscriber("/UAV/{}/next_way_point_euclid".format(self._name), data_class=euclidean_location,
                         callback=self.callback_next_euclidean_way_point)
        self._pub_pose_gps = rospy.Publisher(self._name + '/pose_gps', data_class=geo_location, queue_size=10)
        self._pub_pose_euclid = rospy.Publisher(self._name + '/pose_euclid', data_class=euclidean_location, queue_size=10)
        self._pub_next_goal_gps = rospy.Publisher(self._name + '/next_way_point_gps', data_class=geo_location, queue_size=10)
        self._pub_co2 = rospy.Publisher(self._name + '/co2', data_class=CO_2, queue_size=10)
        self._pub_orientation_euler = rospy.Publisher(self._name + '/orientation_euler', data_class=orientation_euler, queue_size=10)
        self._pub_humidity = rospy.Publisher(self._name + '/humidity', data_class=humidity, queue_size=10)
        self._pub_temperature = rospy.Publisher(self._name + '/temperature', data_class=temperature, queue_size=10)
        self._pub_vel_ms = rospy.Publisher(name="{}/vel_ms".format(self._name), data_class=twist_ms, queue_size=10)
        self._pub_vel_euclid = rospy.Publisher(name="{}/vel_euclid".format(self._name), data_class=twist_euclid, queue_size=10)

        while not rospy.is_shutdown():
            pub_ready.publish(True)
            self._pub_pose_gps.publish(self._pose_gps)
            self._pub_pose_euclid.publish(self.pose_in_euclid())
            self._pub_next_goal_gps.publish(self._goal_gps)
            if self._co2.density > 0.1:
                self._pub_co2.publish(self._co2)
            self._pub_orientation_euler.publish(self._orientation)
            self._pub_humidity.publish(self._humidity)
            self._pub_temperature.publish(self._temperature)
            self._pub_vel_ms.publish(self._velocity_ms)
            self._pub_vel_euclid.publish(self.velocity_euclid)
            rate.sleep()

    def euclid_to_geo(self, NS, EW, UD):
        """
        Converts euclid NED coordinate and converts it to gps latitude and longitude.
        displacement in meters (N and E are positive, S and W are negative), and outputs the new lat/long
        CAUTION: the numbers below are set for use near Stillwater will change at other lattitudes
        :param NS: set as y axis of euclidean coordinate lat
        :param EW: set as x axis of euclidean coordinate lon
        :param UD: set as z axis of eculidean coordinate alt
        :rtype: geo_location
        """
        pose = geo_location()
        lon = self._origin_lon + self._meters_per_disposition * EW / self._meters_per_lon
        lat = self._origin_lat + self._meters_per_disposition * NS / self._meters_per_lat
        alt = self._origin_alt + self._meters_per_alt * UD + self._sea_level_ft
        pose.longitude = lon
        pose.latitude = lat
        pose.altitude = alt
        return pose

    def pose_in_euclid(self):
        """
        Converts euclid NED coordinate and converts it to gps latitude and longitude.
        displacement in meters (N and E are positive, S and W are negative), and outputs the new lat/long
        CAUTION: the numbers below are set for use near Stillwater will change at other lattitudes
        :param lon: set as y axis of euclidean coordinate lon
        :param lat: set as x axis of euclidean coordinate lat
        :return: Pose in euclid
        :rtype: euclidean_location
        """
        pose = euclidean_location()
        pose.header.frame_id = self._name
        pose.x = ((self._pose_gps.longitude - self._origin_lon)/(self._max_lon - self._origin_lon)) * float(self._scale)
        pose.y = ((self._pose_gps.latitude - self._origin_lat)/(self._max_lat - self._origin_lat)) * float(self._scale)
        pose.z = ((self._pose_gps.altitude - self._origin_alt)/(self._max_alt - self._origin_alt)) * float(self._scale)
        if (self._pose_gps.latitude < self._origin_lat or self._pose_gps.longitude < self._origin_lon or self._pose_gps.altitude >= self._max_alt or self._pose_gps.longitude >= self._max_lon or self._pose_gps.latitude >= self._max_lat) and self._is_ready:
            rospy.logdebug("{} Loiter because went out of boundary!!! psoe={} (lon,lat,alt)=({},{},{})".format(
                self.tag, pose.__getstate__(), self._pose_gps.longitude,
                self._pose_gps.latitude, self._pose_gps.altitude))
            rospy.signal_shutdown("{} Went out of boundary".format(self.tag))
            self._pub_ready(False)
        return pose

    @property
    def velocity_euclid(self):
        """:rtype twist_euclid"""
        vel = twist_euclid()
        vel.header.frame_id = self._name
        vel.x = (self._velocity_ms.x / (self._meters_per_lon * (self._max_lon - self._origin_lon))) * self._scale
        vel.y = (self._velocity_ms.y / (self._meters_per_lat * (self._max_lat - self._origin_lat))) * self._scale
        vel.z = (self._velocity_ms.z / (self._meters_per_alt * (self._max_alt - self._origin_alt))) * self._scale
        return vel

    def callback_next_euclidean_way_point(self, goal_euclid):
        """
        :param goal_euclid: goal in euclidian coordinate
        :type goal_euclid: euclidean_location
        :return:
        """
        if goal_euclid is not None:
            self._goal_euclid = goal_euclid
            # longitude EW = x axis and latitude NS = y axis, E is +x, N is +y
            self._goal_gps = self.euclid_to_geo(NS=goal_euclid.y, EW=goal_euclid.x, UD=goal_euclid.z)
            rospy.logdebug("{}New Goal (x,y,z)=({},{},{}) (lat,long,alt)=({},{},{})".format(
                self.tag, self._goal_euclid.x, self._goal_euclid.y, self._goal_euclid.z,
                self._goal_gps.latitude, self._goal_gps.longitude, self._goal_gps.altitude)
            )
        else:
            rospy.logdebug("{} No goal waypoint received yet.".format(self.tag))

    def callback_fg_sensor(self, sensor):
        """
        :type sensor: sensor_data
        """
        self._pose_gps.header.frame_id = self._name
        self._pose_gps.latitude = float(sensor.Pos_n)
        self._pose_gps.longitude = float(sensor.Pos_e)
        self._pose_gps.altitude = float(sensor.Pos_d) - self._sea_level_ft

        self._orientation.header.frame_id = self._name
        self._orientation.roll = float(sensor.roll_deg)
        self._orientation.pitch = float(sensor.pitch_deg)
        self._orientation.yaw = float(sensor.yaw_deg)

        self._temperature.header.frame_id = self._name
        self._temperature.temperature_f = 1.8 * float(sensor.Temperature_degc) + 32.0

        self._humidity.header.frame_id = self._name
        self._humidity.rel_humidity = float(sensor.Relative_humidity)

        self._co2.header.frame_id = self._name
        self._co2.density = float(sensor.CO2Density)

        self._velocity_ms.header.frame_id = self._name
        self._velocity_ms.x = float(sensor.V_e_ms)
        self._velocity_ms.y = float(sensor.V_n_ms)
        self._velocity_ms.z = float(sensor.V_d_ms)


#!/usr/bin/env python

from __future__ import print_function
from std_msgs.msg import Bool
import datetime as dt
import rospy
# for flightgear
from simulator import Simulator
import socket
from cloud_map.msg import *


class flightgear_quad_static(object):
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
        self._origin_lat = 36.1333333
        self._origin_lon = -97.0771
        self._origin_alt = 5.  # meter
        self._meters_per_alt = 4.
        self._meters_per_disposition = 4.
        self._meters_per_lat = 110961.03  # meters per degree of latitude for use near Stillwater
        self._meters_per_lon = 90037.25  # meters per degree of longitude
        self._tol_meter = .05  # drone to be considered reached a goal if it is withing tol_meter within the goal
        self._tol_lat = 1.e-6
        self._tol_lon = 1.e-6
        self._tol_alt = 0.5

        self._max_lon = self._origin_lon + (self._meters_per_disposition * self._scale) / self._meters_per_lon
        self._max_lat = self._origin_lat + (self._meters_per_disposition * self._scale) / self._meters_per_lat
        self._max_alt = self._origin_alt + (self._meters_per_disposition * self._scale)

        self._pose_gps = geo_location()

        self._pub_pose_gps = None
        self._pub_pose_euclid = None
        self._pub_distance_to_goal = None
        self._is_ready = False
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("google.com", 80))
            UDP_IP = s.getsockname()[0]
            s.close()
        except socket.error:
            rospy.logdebug("{}Network connection unavailable...".format(self.tag))
            exit(-1)
        self._sim = Simulator({'IP': UDP_IP, "port_send": self._port_send, "port_recv": self._port_recv})

    @property
    def tag(self):
        return "fg_{}[{}]:".format(self._name, dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%H:%M:%S"))

    def arm_and_takeoff(self, start_at_euclid=None):
        """
        Init ROS node.
        Arms vehicle and fly_grad to aTargetAltitude (in meters).
        """
        rospy.init_node(self._name, log_level=rospy.DEBUG)
        rate = rospy.Rate(1)

        rospy.logdebug("{} init node max (lon, lat, alt)=({},{},{})".format(
            self.tag, self._max_lon, self._max_lat, self._max_alt))

        # longitude EW = x axis and latitude NS = y axis
        # send solo to initial location

        # FGthread(
        #     server_id = self._server_id, instance=self._instance, controller_hostIP=self._server_ip, freq_in=100, freq_out=100,
        #     vehicle='quad', lat=self._origin_lat, lon=self._origin_lon, alt=self._origin_alt,
        #     iheading=45, ivel=60, ithrottle=0)  # 0.1 -> throttle

        pub_ready = rospy.Publisher('{}/ready'.format(self._name), data_class=Bool, queue_size=1)

        self._goal_euclid = euclidean_location()
        self._goal_gps = self.euclid_to_geo(NS=self._goal_euclid.y, EW=self._goal_euclid.x,
                                        UD=self._goal_euclid.z)

        # rospy.logdebug("{}Sending to initial goal (x,y,z)=({}) (lon, lat, alt)=({},{},{}) tol=({},{},{})".format(
        #     self.tag, start_at_euclid, self._goal_gps.position.y, self._goal_gps.position.x,
        #     self._goal_gps.position.z, self._tol_lon, self._tol_lat, self._tol_alt)
        # )

        rospy.Subscriber("/fg_interface/{}/sensor_data".format(self._name), data_class=sensor_data,
                         callback=self.callback_fg_sensor)
        while True:
            pose = self.pose_in_euclid()
            if (0. < pose.x < self._scale) and (0 < pose.y < self._scale):
                break
            rospy.logdebug("{} Waiting....\npose read....\n{}".format(self.tag, pose))
            rospy.sleep(5)
        rospy.sleep(5)
        rospy.logdebug("{} Ready\npose read:\n{}".format(self.tag, pose))
        # rospy.Subscriber("/UAV/{}/next_way_point_euclid".format(self._name), data_class=Pose,
        #                  callback=self.callback_next_euclidean_way_point)
        self._pub_pose_gps = rospy.Publisher(self._name + '/pose_gps', data_class=geo_location, queue_size=10)
        self._pub_pose_euclid = rospy.Publisher(self._name + '/pose_euclid', data_class=euclidean_location, queue_size=10)

        # pub_fly = rospy.Publisher("{}/fly_grad".format(self._name), data_class=String, queue_size=10)
        # pub_fly.publish("fly_grad")

        self._pose_gps.header.frame_id = self._name

        while not rospy.is_shutdown():
            pub_ready.publish(True)
            self._pub_pose_gps.publish(self._pose_gps)
            self._pub_pose_euclid.publish(self.pose_in_euclid())

            # if self._goal_gps is not None:
            #     simple_goto(lat=self._goal_gps.position.y, lon=self._goal_gps.position.x, alt=self._goal_gps.position.z,
            #                 callasign=self._name, sim=self._sim)
            #
            #     reached_lon = np.isclose(self._goal_gps.position.x, self.gps_loc.position.x,
            #                              atol=self._tol_lon)
            #     reached_lat = np.isclose(self._goal_gps.position.y, self.gps_loc.position.y,
            #                              atol=self._tol_lat)
            #     reached_alt = np.isclose(self._goal_gps.position.z, self.gps_loc.position.z,
            #                              atol=self._tol_alt)
            #
            #     dif_lon = self.gps_loc.position.x - self._goal_gps.position.x
            #     dif_lat = self.gps_loc.position.y - self._goal_gps.position.y
            #     dif_alt = self.gps_loc.position.z - self._goal_gps.position.z
            #
            #     if reached_lat and reached_lon and reached_alt:
            #         pos_eu = self.pose_in_euclid()
            #         rospy.logdebug("{}[{}]Reached goal @(lon,lat,alt)=({},{},{}) goal({},{},{}) dif=({},{},{}) @(x,y,z)=({},{},{}) "
            #                        "goal_eu=({},{},{})".format(self._tag, self.time_tag,
            #             self.gps_loc.position.x,
            #             self.gps_loc.position.y,
            #             self.gps_loc.position.z,
            #             self._goal_gps.position.x, self._goal_gps.position.y, self._goal_gps.position.z, dif_lon,
            #             dif_lat, dif_alt, pos_eu.position.x, pos_eu.position.y, pos_eu.position.z,
            #             self._goal_euclid.position.x, self._goal_euclid.position.y, self._goal_euclid.position.z))
            #         self._goal_gps = None
            #         self._goal_euclid = None
            #     else:
            #         pos_eu = self.pose_in_euclid()
            #         rospy.logdebug("{}[{}]@(lon,lat,alt)=({},{},{}) goal({},{},{}) dif=({},{},{}) @(x,y,z)=({},{},{}) goal_eu=({},{},{})".format(
            #             self._tag, del_t.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%H:%M:%S"),
            #             self.gps_loc.position.x,
            #             self.gps_loc.position.y,
            #             self.gps_loc.position.z,
            #             self._goal_gps.position.x, self._goal_gps.position.y, self._goal_gps.position.z,
            #             dif_lon, dif_lat, dif_alt, pos_eu.position.x, pos_eu.position.y,
            #             pos_eu.position.z,self._goal_euclid.position.x, self._goal_euclid.position.y,
            #             self._goal_euclid.position.z))
            #         pub_fly.publish("wait")
            #
            # else:
            #     rospy.logdebug("{}[{}]Waiting for new goal".format(self._tag, del_t.datetime.fromtimestamp(
            #         rospy.Time.now().to_time()).strftime("%H:%M:%S")))
            #     pub_fly.publish("fly_grad")

            rate.sleep()

    def euclid_to_geo(self, NS, EW, UD):
        """
        Converts euclid NED coordinate and converts it to gps latitude and longitude.
        displacement in meters (N and E are positive, S and W are negative), and outputs the new lat/long
        CAUTION: the numbers below are set for use near Stillwater will change at other lattitudes
        :param NS: set as y axis of euclidean coordinate lat
        :param EW: set as x axis of euclidean coordinate lon
        :param UD: set as z axis of eculidean coordinate alt
        :rtype: Pose
        """
        pose = geo_location()
        lon = self._origin_lon + self._meters_per_disposition * EW / self._meters_per_lon
        lat = self._origin_lat + self._meters_per_disposition * NS / self._meters_per_lat
        alt = self._origin_alt + self._meters_per_alt * UD
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
        :rtype: Pose
        """
        pose = euclidean_location()
        pose.header.frame_id = self._name
        lon = self._pose_gps.longitude
        lat = self._pose_gps.latitude
        alt = self._pose_gps.altitude
        pose.x = ((lon - self._origin_lon)/(self._max_lon - self._origin_lon)) * float(self._scale)
        pose.y = ((lat - self._origin_lat)/(self._max_lat - self._origin_lat)) * float(self._scale)
        pose.z = ((alt - self._origin_alt)/(self._max_alt - self._origin_alt)) * float(self._scale)
        if (lat < self._origin_lat or lon < self._origin_lon or alt >= self._max_alt or lon >= self._max_lon or lat >= self._max_lat) and self._is_ready:
            rospy.logdebug("{} Out of boundary!!! pose={} (lon,lat,alt)=({},{},{})".format(
                self.tag, pose.__getstate__(),self._pose_gps.longitude,
                self._pose_gps.latitude, self._pose_gps.altitude))
        return pose

    def callback_next_euclidean_way_point(self, goal_euclid):
        """
        :param goal_euclid: goal in euclidian coordinate
        :type goal_euclid: Pose
        :return:
        """
        if goal_euclid is not None:
            self._goal_euclid = goal_euclid
            # longitude EW = x axis and latitude NS = y axis, E is +x, N is +y
            self._goal_gps = self.euclid_to_geo(NS=goal_euclid.position.y, EW=goal_euclid.position.x,
                                                UD=goal_euclid.position.z)

            rospy.logdebug("{}New Goal (x,y,z)=({},{},{}) (lat,long,alt)=({},{},{})".format(self.tag,
                self._goal_euclid.position.x, self._goal_euclid.position.y, self._goal_euclid.position.z,
                self._goal_gps.position.y, self._goal_gps.position.x, self._goal_gps.position.z)
            )
        else:
            rospy.logdebug("{} No goal waypoint received yet.".format(self.tag))

    def callback_fg_sensor(self, sensor):
        """
        :type sensor: sensor_data
        """
        self._pose_gps.latitude = float(sensor.Pos_n)
        self._pose_gps.longitude = float(sensor.Pos_e)
        self._pose_gps.altitude = float(sensor.Pos_d)

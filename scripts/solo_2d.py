#!/usr/bin/python
"""
author: James Kostas
modified by: S M Al Mahi
"""
from __future__ import print_function
import numpy as np
import dronekit
import rospy
import time
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from tf.transformations import quaternion_from_euler
from std_msgs.msg import String, Float32, Bool


class solo_2d(object):
    def __init__(self, name, port):
        """
        :param port: port number of solo controller to connect
        :param altitude: in meters
        """
        self._name = name
        self._port = port
        self._tag = "[solo_{}]".format(port)
        self._goal_gps = Pose(Point(0., 0., 0.), Quaternion(*quaternion_from_euler(0., 0., 0.)))
        self._goal_euclid = Pose(Point(0., 0., 0.), Quaternion(*quaternion_from_euler(0., 0., 0.)))
        self._dim = 2
        # cowboy cricket ground bowling end 36.133642, -97.076528
        self._origin_lat = 36.133642
        self._origin_lon = -97.076528
        self._alt = 8
        self._meters_per_disposition = 2.
        self._meters_per_lat = 110961.03  # meters per degree of latitude for use near Stillwater
        self._meters_per_lon = 90037.25  # meters per degree of longitude
        try:
            self._vehicle = dronekit.connect("udpin:0.0.0.0:{}".format(port))
        except Exception as e:
            print("{}Could not connect!!! {}".format(self._tag, port, e.message))
            exit(code=-1)
        self._pub_pose_gps = None
        self._pub_pose_euclid = None
        self._pub_distance_to_goal = None

    def arm_and_takeoff(self, aTargetAltitude=10, starting_coordinate=None):
        """
        Init ROS node.
        Arms vehicle and fly to aTargetAltitude (in meters).
        """
        rospy.init_node(self._name, log_level=rospy.DEBUG)
        rate = rospy.Rate(1)
        rospy.logdebug("{}Basic pre-arm checks".format(self._tag))
        # Don't try to arm until autopilot is ready
        while not self._vehicle.is_armable:
            rospy.logdebug("{}Waiting for vehicle to initialise...".format(self._tag))
            time.sleep(1)

        rospy.logdebug("{}Arming motors".format(self._tag))
        # Copter should arm in GUIDED mode
        self._vehicle.mode = dronekit.VehicleMode("GUIDED")
        self._vehicle.armed = True

        # Confirm vehicle armed before attempting to take off
        while not self._vehicle.armed:
            rospy.logdebug("{}Waiting for arming...".format(self._tag))
            time.sleep(1)

        self._alt = aTargetAltitude
        rospy.logdebug("Taking off to{}".format(self._alt))
        self._vehicle.simple_takeoff(self._alt)
        time.sleep(10)

        rospy.Subscriber("/UAV/{}/next_way_point_euclid".format(self._name), data_class=Pose,
                         callback=self.callback_go_to_next_euclidean_way_point)
        rospy.Subscriber("/UAV/{}/land".format(self._name), data_class=String, callback=self.callback_land)
        self._pub_pose_gps = rospy.Publisher(self._name + '/pose_gps', data_class=Pose, queue_size=10)
        self._pub_pose_euclid = rospy.Publisher(self._name + '/pose_euclid', data_class=Pose, queue_size=10)
        self._pub_distance_to_goal = rospy.Publisher(self._name + '/distance_from_goal', data_class=Float32, queue_size=10)
        while not rospy.is_shutdown():
            pose_gps = Pose()
            pose_gps.position.x = self._vehicle.location.global_relative_frame.lon
            pose_gps.position.y = self._vehicle.location.global_relative_frame.lat
            self._pub_pose_gps.publish(pose_gps)
            self._pub_pose_euclid.publish(self.pose_in_euclid())
            self._pub_distance_to_goal.publish(self.get_distance_to_goal())
            rate.sleep()

    def euclid_to_geo(self, NS, EW):
        """
        Converts euclid NED coordinate and converts it to gps latitude and longitude.
        displacement in meters (N and E are positive, S and W are negative), and outputs the new lat/long
        CAUTION: the numbers below are set for use near Stillwater will change at other lattitudes
        :param NS: set as y axis of euclidean coordinate
        :param EW: set as x axis of euclidean coordinate
        :return:
        """
        return self._origin_lat + self._meters_per_disposition * NS / self._meters_per_lat,\
               self._origin_lon + self._meters_per_disposition * EW / self._meters_per_lon

    def pose_in_euclid(self):
        """
        Converts euclid NED coordinate and converts it to gps latitude and longitude.
        displacement in meters (N and E are positive, S and W are negative), and outputs the new lat/long
        CAUTION: the numbers below are set for use near Stillwater will change at other lattitudes
        :param lon: set as y axis of euclidean coordinate
        :param lat: set as x axis of euclidean coordinate
        :return: Pose in euclid
        :rtype: Pose
        """
        pose = Pose()
        pose.position.x = (self._vehicle.location.global_relative_frame.lon - self._origin_lon) * self._meters_per_lon / self._meters_per_disposition
        pose.position.y = (self._vehicle.location.global_relative_frame.lat - self._origin_lat) * self._meters_per_lat / self._meters_per_disposition
        rospy.logdebug("{}Pose (x,y)=({},{}) actual (lon,lat)=({},{})".format(self._tag, pose.position.x, pose.position.y,
                                                                         self._vehicle.location.global_relative_frame.lon, self._vehicle.location.global_relative_frame.lat))
        return pose

    def callback_go_to_next_euclidean_way_point(self, goal_euclid):
        """
        :param goal_euclid: goal in euclidian coordinate
        :type goal_euclid: Pose
        :return:
        """
        self._goal_euclid = goal_euclid
        lat, lon = self.euclid_to_geo(NS=goal_euclid.position.y, EW=goal_euclid.position.x)
        # longitude EW = x axis and latitude NS = y axis
        self._goal_gps.position.x = lon
        self._goal_gps.position.y = lat
        rospy.logdebug("{}Going to waypoint x={} y={} lat={} long={}".format(self._tag,goal_euclid.position.x, goal_euclid.position.y, lat, lon))
        self._vehicle.simple_goto(dronekit.LocationGlobalRelative(self._goal_gps.position.y, self._goal_gps.position.x,
                                                                  self._alt))

    def callback_land(self, msg):
        rospy.logdebug("{}Returning to Launch".format(self._tag))
        self._vehicle.mode = dronekit.VehicleMode("RTL")

        # Close vehicle object before exiting script
        rospy.logdebug("{}Close vehicle object".format(self._tag))
        self._vehicle.close()

    def get_distance_to_goal(self):
        pose = self.pose_in_euclid()
        d = np.sqrt((self._goal_euclid.position.x - pose.position.x)**2. +
                       (self._goal_euclid.position.y - pose.position.y)**2.)
        if d < 0.5:
            rospy.logdebug("{}reached goal (long,lat)=({},{}) (x,y)=({},{})".format(
                self._tag, self._vehicle.location.global_relative_frame.lon,
                self._vehicle.location.global_relative_frame.lat, pose.position.x, pose.position.y))
        return d







#!/usr/bin/python
"""
author: James Kostas
modified by: S M Al Mahi
"""
from __future__ import print_function
import numpy as np
import dronekit
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from tf.transformations import quaternion_from_euler
from std_msgs.msg import String, Float32, Bool
import time

class solo_2d(object):
    def __init__(self, name, port, scale, dim):
        """
        :param port: port number of solo controller to connect
        :param altitude: in meters
        """
        self._name = name
        self._port = port
        self._tag = "[solo_{}]".format(port)
        self._goal_gps = Pose(Point(0., 0., 0.), Quaternion(*quaternion_from_euler(0., 0., 0.)))
        self._goal_euclid = Pose(Point(0., 0., 0.), Quaternion(*quaternion_from_euler(0., 0., 0.)))
        self._dim = int(rospy.get_param("/dim"))
        self._scale = int(rospy.get_param("/scale"))
        self._space = tuple([scale for _ in range(dim)])
        # cowboy cricket ground bowling end 36.133642, -97.076528
        self._origin_lat = 36.133450
        self._origin_lon = -97.076666

        self._log_file = open("log_poses_{}_{}.txt".format(self._tag, time.time()), mode='a+')
        self._origin_alt = 10.  # meter
        if name == 'A':
            self._origin_alt = 10.5  # meter
        self._goal_alt = 20
        self._meters_per_alt = 5.
        self._meters_per_disposition = 5.
        self._meters_per_lat = 110961.03  # meters per degree of latitude for use near Stillwater
        self._meters_per_lon = 90037.25  # meters per degree of longitude

        self.gps_grid = np.empty(shape=self._space, dtype=(float, 3))

        max_lon = self._origin_lon + (self._meters_per_disposition * self._scale) / self._meters_per_lon
        max_lat = self._origin_lat + (self._meters_per_disposition * self._scale) / self._meters_per_lat
        max_alt = self._origin_alt + (self._meters_per_disposition * self._scale) / self._meters_per_alt

        try:
            self._vehicle = dronekit.connect("udpin:0.0.0.0:{}".format(port))
        except Exception as e:
            print("{}Could not connect!!! {}".format(self._tag, port, e.message))
            exit(code=-1)
        self._pub_pose_gps = None
        self._pub_pose_euclid = None
        self._pub_distance_to_goal = None

    def arm_and_takeoff(self, start_at_euclid=None):
        """
        Init ROS node.
        Arms vehicle and fly to aTargetAltitude (in meters).
        """
        rospy.init_node(self._name, log_level=rospy.DEBUG)
        rospy.logdebug("{} init node".format(self._tag))
        rate = rospy.Rate(1)

        print("{}Basic pre-arm checks".format(self._tag))
        # Don't try to arm until autopilot is ready
        while not self._vehicle.is_armable:
            print("{}Waiting for vehicle to initialise...".format(self._tag))
            rate.sleep(1)

        rospy.logdebug("{}Arming motors".format(self._tag))
        # Copter should arm in GUIDED mode
        self._vehicle.mode = dronekit.VehicleMode("GUIDED")
        self._vehicle.armed = True

        # Confirm vehicle armed before attempting to take off
        while not self._vehicle.armed:
            print("{}Waiting for arming...".format(self._tag))
            rate.sleep(1)

        print("{}Taking off to {}m".format(self._tag, self._origin_alt))
        self._vehicle.simple_takeoff(self._origin_alt)

        # give solo time to take off
        rate.sleep(10)
        self._goal_euclid = Pose(Point(start_at_euclid[0], start_at_euclid[1], start_at_euclid[2]), Quaternion(
            *quaternion_from_euler(0., 0., 0.)))
        self._goal_gps = self.euclid_to_geo(NS=self._goal_euclid.position.y, EW=self._goal_euclid.position.x,
                                            UD=self._goal_euclid.position.z)
        # longitude EW = x axis and latitude NS = y axis
        # send solo to initial location
        self._vehicle.simple_goto(dronekit.LocationGlobalRelative(
            lat=self._goal_gps.position.y, lon=self._goal_gps.position.x, alt=self._goal_gps.position.z))
        print("{}Sending to initial goal (x,y,z)=({}) (lon, lat, alt)=({},{},{})".format(
            self._tag, start_at_euclid, self._goal_gps.position.y, self._goal_gps.position.x, self._goal_gps.position.z)
        )
        rate.sleep(3)

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
            pose_gps.position.z = self._vehicle.location.global_relative_frame.alt
            self._pub_pose_gps.publish(pose_gps)
            self._pub_pose_euclid.publish(self.pose_in_euclid())
            self._pub_distance_to_goal.publish(self.get_distance_to_goal())
            rate.sleep()

    def euclid_to_geo(self, NS, EW, UD):
        """
        Converts euclid NED coordinate and converts it to gps latitude and longitude.
        displacement in meters (N and E are positive, S and W are negative), and outputs the new lat/long
        CAUTION: the numbers below are set for use near Stillwater will change at other lattitudes
        :param NS: set as y axis of euclidean coordinate
        :param EW: set as x axis of euclidean coordinate
        :rtype: Pose
        """
        pose = Pose()
        lon = self._origin_lat + self._meters_per_disposition * NS / self._meters_per_lat
        lat = self._origin_lon + self._meters_per_disposition * EW / self._meters_per_lon
        alt = self._origin_alt + self._meters_per_alt * UD
        pose.position.x = lon
        pose.position.y = lat
        pose.position.z = alt
        return pose

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
        pose.position.z = (self._vehicle.location.global_relative_frame.alt - self._origin_alt) / self._meters_per_alt
        rospy.logdebug("{}Pose (x,y,z)=({},{},{}) actual (lon,lat,alt)=({},{},{})"
                       .format(self._tag, pose.position.x, pose.position.y, pose.position.z,
                       self._vehicle.location.global_relative_frame.lon,
                       self._vehicle.location.global_relative_frame.lat,
                       self._vehicle.location.global_relative_frame.alt))
        # self._log_file.write("{}_{}_ {},{},{} {},{},{}\n".format(self._tag, rospy.Time.now(), self._vehicle.location.global_relative_frame.lon,
        #                                               self._vehicle.location.global_relative_frame.lat,
        #                                               self._vehicle.location.global_relative_frame.alt, pose.position.x, pose.position.y, pose.position.z))

        if pose.position.x < -1. or pose.position.y < -1. or pose.position.x >= self._scale or pose.position.y >= self._scale:
            rospy.logdebug("{} landing because went out of boundary!!! ".format(self._tag))
            self._vehicle.mode = dronekit.VehicleMode("LAND")
        return pose

    def callback_go_to_next_euclidean_way_point(self, goal_euclid):
        """
        :param goal_euclid: goal in euclidian coordinate
        :type goal_euclid: Pose
        :return:
        """
        if goal_euclid is None:
            rospy.logdebug("{} No goal waypoint received yet.".format(self._tag))
            return

        self._goal_euclid = goal_euclid
        # longitude EW = x axis and latitude NS = y axis, E is +x, N is +y
        self._goal_gps = self.euclid_to_geo(NS=goal_euclid.position.y, EW=goal_euclid.position.x, UD=goal_euclid.position.z)

        rospy.logdebug("{}Going to waypoint (x,y,z)=({},{},{}) (lat,long,alt)=({},{},{})".format(
            self._tag, goal_euclid.position.x, goal_euclid.position.y, goal_euclid.position.z,
            self._goal_gps.position.y, self._goal_gps.position.x, self._goal_gps.position.z))

        self._vehicle.simple_goto(dronekit.LocationGlobalRelative(self._goal_gps.position.y, self._goal_gps.position.x,
                                                                  self._goal_gps.position.z))

    def callback_land(self, msg):
        rospy.logdebug("{}Returning to Launch".format(self._tag))
        self._vehicle.mode = dronekit.VehicleMode("RTL")

        # Close vehicle object before exiting script
        rospy.logdebug("{}Close vehicle object".format(self._tag))
        self._vehicle.close()

        self._log_file.close()

    def get_distance_to_goal(self):
        pose = self.pose_in_euclid()
        if self._dim == 3:
            d = np.sqrt((self._goal_euclid.position.x - pose.position.x)**2. + (self._goal_euclid.position.y - pose.position.y)**2. + (self._goal_euclid.position.z - pose.position.z)**2.)
        if self._dim == 2:
            d = np.sqrt((self._goal_euclid.position.x - pose.position.x)**2. + (self._goal_euclid.position.y - pose.position.y)**2.)
        rospy.logdebug("{}goal_gps={} dz_gps={} scaled_alt={} actual_alt={} d={}".format(self._tag, self._goal_gps.position.z,
                        (self._goal_gps.position.z - self._vehicle.location.global_relative_frame.alt)**2.,
                        pose.position.z,
                        self._vehicle.location.global_relative_frame.alt, d))
        tol = 0.5
        if self._dim == 2: tol = 0.5
        if self._dim == 3: tol = 0.8

        if d < tol:
            rospy.logdebug("{}reached goal (long,lat,alt)=({},{},{}) (x,y)=({},{},{})".format(
                self._tag, self._vehicle.location.global_relative_frame.lon,
                self._vehicle.location.global_relative_frame.lat, self._vehicle.location.global_relative_frame.alt, pose.position.x, pose.position.y, pose.position.z))
        return d







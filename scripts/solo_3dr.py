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
import datetime as dt


class solo_3dr(object):
    def __init__(self, name, port, scale, dim):
        """
        :param port: port number of solo controller to connect
        :param altitude: in meters
        """
        self._name = name
        self._port = port
        self._tag = "[solo_{}]".format(port)
        self._goal_gps = Pose(Point(0., 0., 0.), Quaternion(*quaternion_from_euler(0., 0., 0.)))
        self._goal_euclid = Pose()
        self._dim = int(rospy.get_param("/dim"))
        self._scale = int(rospy.get_param("/scale"))
        self._space = tuple([scale for _ in range(dim)])
        # cowboy cricket ground bowling end 36.133642, -97.076528
        self._origin_lat = 36.169097   #36.1333333
        self._origin_lon = -97.088101  #-97.0771
        self._origin_alt = 4.  # meter
        if name == 'A':
            self._origin_alt = 4.1  # meter
        self._meters_per_alt = 4.
        self._meters_per_disposition = 4.
        self._meters_per_lat = 110961.03  # meters per degree of latitude for use near Stillwater
        self._meters_per_lon = 90037.25  # meters per degree of longitude
        self._tol_meter = .05  # drone to be considered reached a goal if it is withing tol_meter within the goal
        self._tol_lat = 1.e-7
        self._tol_lon = 1.e-7
        self._tol_alt = 0.5

        self._max_lon = self._origin_lon + (self._meters_per_disposition * self._scale) / self._meters_per_lon
        self._max_lat = self._origin_lat + (self._meters_per_disposition * self._scale) / self._meters_per_lat
        self._max_alt = self._origin_alt + (self._meters_per_disposition * self._scale)

        try:
            self._vehicle = dronekit.connect("udpin:0.0.0.0:{}".format(port))
        except Exception as e:
            print("{}Could not connect!!! {}".format(self._tag, port, e.message))
            exit(code=-1)
        self._pub_pose_gps = None
        self._pub_pose_euclid = None
        self._pub_distance_to_goal = None
        self._is_ready = False

    def arm_and_takeoff(self, start_at_euclid=None):
        """
        Init ROS node.
        Arms vehicle and fly_grad to aTargetAltitude (in meters).
        """
        rospy.init_node(self._name, log_level=rospy.DEBUG)
        rospy.logdebug("{}[{}] init node max (lon, lat, alt)=({},{},{})".format(self._tag,
                                                 dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime(
                                                     "%H:%M:%S"), self._max_lon, self._max_lat, self._max_alt))
        rate = rospy.Rate(1)

        rospy.logdebug("{}Basic pre-arm checks".format(self._tag))
        rospy.logdebug("{} Mode {}".format(self._name, self._vehicle.mode))
        # Don't try to arm until autopilot is ready
        while not self._vehicle.is_armable:
            rospy.logdebug("{}[{}]Waiting for vehicle to initialise...".format(self._tag, dt.datetime.fromtimestamp(
                rospy.Time.now().to_time()).strftime("%H:%M:%S")))
            rate.sleep()

        rospy.logdebug("{}[{}]Arming motors".format(self._tag,
                                                    dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime(
                                                        "%H:%M:%S")))
        # Copter should arm in GUIDED mode
        self._vehicle.mode = dronekit.VehicleMode("GUIDED")
        self._vehicle.armed = True

        # Confirm vehicle armed before attempting to take off
        while not self._vehicle.armed:
            rospy.logdebug("{}Waiting for arming...".format(self._tag))
            rate.sleep()

        rospy.logdebug("{}[{}]Taking off to {}m".format(self._tag,
                                                        dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime(
                                                            "%H:%M:%S"), self._origin_alt))

        pub_ready = rospy.Publisher('{}/ready'.format(self._name), data_class=Bool, queue_size=1)
        self._vehicle.simple_takeoff(self._origin_alt)
        while True:
            rospy.logdebug("{}[{}] Altitude: {}".format(self._tag,
                                                        dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime(
                                                            "%H:%M:%S"),
                                                        self._vehicle.location.global_relative_frame.alt))
            # Break and return from function just below target altitude.
            if self._vehicle.location.global_relative_frame.alt >= self._origin_alt * .9:
                rospy.logdebug("{}[{}]Reached target altitude".format(self._tag, dt.datetime.fromtimestamp(
                    rospy.Time.now().to_time()).strftime("%H:%M:%S")))
                break
            pub_ready.publish(Bool(False))
            rospy.sleep(8)

        self._goal_euclid = Pose(Point(start_at_euclid[0], start_at_euclid[1], start_at_euclid[2]), Quaternion(
            *quaternion_from_euler(0., 0., 0.)))
        self._goal_gps = self.euclid_to_geo(NS=self._goal_euclid.position.y, EW=self._goal_euclid.position.x,
                                            UD=self._goal_euclid.position.z)
        # longitude EW = x axis and latitude NS = y axis
        # send solo to initial location
        self._vehicle.simple_goto(
            dronekit.LocationGlobalRelative(
                lat=self._goal_gps.position.y, lon=self._goal_gps.position.x, alt=self._goal_gps.position.z),
            groundspeed=10.
        )
        rospy.logdebug("{}[{}]Sending to initial goal (x,y,z)=({}) (lon, lat, alt)=({},{},{}) tol=({},{},{})".format(
            self._tag, dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%H:%M:%S"),
            start_at_euclid, self._goal_gps.position.y, self._goal_gps.position.x, self._goal_gps.position.z,
            self._tol_lon, self._tol_lat, self._tol_alt)
        )

        while True:
            reached_lon = np.isclose(self._goal_gps.position.x, self._vehicle.location.global_relative_frame.lon,
                                     atol=self._tol_lon)
            reached_lat = np.isclose(self._goal_gps.position.y, self._vehicle.location.global_relative_frame.lat,
                                     atol=self._tol_lat)
            reached_alt = np.isclose(self._goal_gps.position.z, self._vehicle.location.global_relative_frame.alt,
                                     atol=self._tol_alt)
            dif_lon = self._vehicle.location.global_relative_frame.lon - self._goal_gps.position.x
            dif_lat = self._vehicle.location.global_relative_frame.lat - self._goal_gps.position.y
            dif_alt = self._vehicle.location.global_relative_frame.alt - self._goal_gps.position.z

            if reached_lat and reached_lon and reached_alt:
                rospy.logdebug("{}[{}]Reached initial goal".format(self._tag, dt.datetime.fromtimestamp(
                    rospy.Time.now().to_time()).strftime("%H:%M:%S")))
                self._goal_gps = None
                self._goal_euclid = None
                self._is_ready = True
                break
            pose = self.pose_in_euclid()
            rospy.logdebug("{}[{}]@(lon,lat,alt)=({},{},{}) goal=({},{},{}) dif=({},{},{}) @(x,y,z)=({},{},{})".format(
                self._tag, dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%H:%M:%S"),
                self._vehicle.location.global_relative_frame.lon,
                self._vehicle.location.global_relative_frame.lat,
                self._vehicle.location.global_relative_frame.alt,
                self._goal_gps.position.x, self._goal_gps.position.y, self._goal_gps.position.z, dif_lon, dif_lat,
                dif_alt, pose.position.x, pose.position.y, pose.position.z, )
            )
            pub_ready.publish(Bool(False))
            rospy.sleep(3)

        pub_ready.publish(True)
        rospy.Subscriber("/UAV/{}/next_way_point_euclid".format(self._name), data_class=Pose,
                         callback=self.callback_next_euclidean_way_point)
        rospy.Subscriber("/UAV/{}/land".format(self._name), data_class=String, callback=self.callback_land)
        rospy.Subscriber("/UAV/{}/loiter".format(self._name), data_class=String, callback=self.callback_loiter)

        self._pub_pose_gps = rospy.Publisher(self._name + '/pose_gps', data_class=Pose, queue_size=10)
        self._pub_pose_euclid = rospy.Publisher(self._name + '/pose_euclid', data_class=Pose, queue_size=10)
        pub_fly = rospy.Publisher("{}/fly_grad".format(self._name), data_class=String, queue_size=10)

        pub_fly.publish("fly_grad")

        while not rospy.is_shutdown():
            pose_gps = Pose()
            pose_gps.position.x = self._vehicle.location.global_relative_frame.lon
            pose_gps.position.y = self._vehicle.location.global_relative_frame.lat
            pose_gps.position.z = self._vehicle.location.global_relative_frame.alt
            self._pub_pose_gps.publish(pose_gps)
            self._pub_pose_euclid.publish(self.pose_in_euclid())

            if self._goal_gps is not None:
                self._vehicle.simple_goto(
                    dronekit.LocationGlobalRelative(
                        lat=self._goal_gps.position.y, lon=self._goal_gps.position.x, alt=self._goal_gps.position.z
                    ),
                    groundspeed=4.
                )

                reached_lon = np.isclose(self._goal_gps.position.x, self._vehicle.location.global_relative_frame.lon,
                                         atol=self._tol_lon)
                reached_lat = np.isclose(self._goal_gps.position.y, self._vehicle.location.global_relative_frame.lat,
                                         atol=self._tol_lat)
                reached_alt = np.isclose(self._goal_gps.position.z, self._vehicle.location.global_relative_frame.alt,
                                         atol=self._tol_alt)

                dif_lon = self._vehicle.location.global_relative_frame.lon - self._goal_gps.position.x
                dif_lat = self._vehicle.location.global_relative_frame.lat - self._goal_gps.position.y
                dif_alt = self._vehicle.location.global_relative_frame.alt - self._goal_gps.position.z

                if reached_lat and reached_lon and reached_alt:
                    pos_eu = self.pose_in_euclid()
                    rospy.logdebug("{}[{}]Reached goal @(lon,lat,alt)=({},{},{}) goal({},{},{}) dif=({},{},{}) @(x,y,z)=({},{},{}) "
                                   "goal_eu=({},{},{})".format(
                        self._tag, dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%H:%M:%S"),
                        self._vehicle.location.global_relative_frame.lon,
                        self._vehicle.location.global_relative_frame.lat,
                        self._vehicle.location.global_relative_frame.alt,
                        self._goal_gps.position.x, self._goal_gps.position.y, self._goal_gps.position.z, dif_lon,
                        dif_lat, dif_alt, pos_eu.position.x, pos_eu.position.y, pos_eu.position.z,
                        self._goal_euclid.position.x, self._goal_euclid.position.y, self._goal_euclid.position.z))
                    self._goal_gps = None
                    self._goal_euclid = None
                else:
                    pos_eu = self.pose_in_euclid()
                    rospy.logdebug("{}[{}]@(lon,lat,alt)=({},{},{}) goal({},{},{}) dif=({},{},{}) @(x,y,z)=({},{},{}) goal_eu=({},{},{})".format(
                        self._tag, dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%H:%M:%S"),
                        self._vehicle.location.global_relative_frame.lon,
                        self._vehicle.location.global_relative_frame.lat,
                        self._vehicle.location.global_relative_frame.alt,
                        self._goal_gps.position.x, self._goal_gps.position.y, self._goal_gps.position.z,
                        dif_lon, dif_lat, dif_alt, pos_eu.position.x, pos_eu.position.y,
                        pos_eu.position.z,self._goal_euclid.position.x, self._goal_euclid.position.y,
                        self._goal_euclid.position.z))
                    pub_fly.publish("wait")

            else:
                rospy.logdebug("{}[{}]Waiting for new goal".format(self._tag, dt.datetime.fromtimestamp(
                    rospy.Time.now().to_time()).strftime("%H:%M:%S")))
                pub_fly.publish("fly_grad")

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
        pose = Pose()
        lon = self._origin_lon + self._meters_per_disposition * EW / self._meters_per_lon
        lat = self._origin_lat + self._meters_per_disposition * NS / self._meters_per_lat
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
        :param lon: set as y axis of euclidean coordinate lon
        :param lat: set as x axis of euclidean coordinate lat
        :return: Pose in euclid
        :rtype: Pose
        """
        pose = Pose()
        lon = self._vehicle.location.global_relative_frame.lon
        lat = self._vehicle.location.global_relative_frame.lat
        alt = self._vehicle.location.global_relative_frame.alt
        pose.position.x = ((lon - self._origin_lon)/(self._max_lon - self._origin_lon)) * float(self._scale)
        pose.position.y = ((lat - self._origin_lat)/(self._max_lat - self._origin_lat)) * float(self._scale)
        pose.position.z = ((alt - self._origin_alt)/(self._max_alt - self._origin_alt)) * float(self._scale)
        if (lat < self._origin_lat or lon < self._origin_lon or alt >= self._max_alt or lon >= self._max_lon or lat >= self._max_lat) and self._is_ready:
            rospy.logdebug("{} Loiter because went out of boundary!!! psoe={} (lon,lat,alt)=({},{},{})".format(
                self._tag, pose.position.__getstate__(),self._vehicle.location.global_relative_frame.lon,
                self._vehicle.location.global_relative_frame.lat, self._vehicle.location.global_relative_frame.alt))
            self._vehicle.mode = dronekit.VehicleMode("LOITER")
            rospy.signal_shutdown("{} Went out of boundary".format(self._tag))
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

            rospy.logdebug("{}[{}]New Goal (x,y,z)=({},{},{}) (lat,long,alt)=({},{},{})".format(
                self._tag, dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%H:%M:%S"),
                self._goal_euclid.position.x, self._goal_euclid.position.y, self._goal_euclid.position.z,
                self._goal_gps.position.y, self._goal_gps.position.x, self._goal_gps.position.z)
            )
        else:
            rospy.logdebug("{} No goal waypoint received yet.".format(self._tag))

    def callback_land(self, msg):
        rospy.logdebug("{}Returning To Launch".format(self._tag))
        self._vehicle.mode = dronekit.VehicleMode("RTL")

        # Close vehicle object before exiting script
        rospy.logdebug("{}Close vehicle object".format(self._tag))
        self._vehicle.close()

    def callback_loiter(self, msg):
        """
        :type msg: String
        :return:
        """
        rospy.logdebug("{}Mode {}".format(self._tag, msg.data))
        self._vehicle.mode = dronekit.VehicleMode("LOITER")

        if msg.data=="GUIDED":
            self._vehicle.mode = dronekit.VehicleMode("GUIDED")
            rospy.logdebug("{}Change Mode {}".format(self._tag, msg.data))

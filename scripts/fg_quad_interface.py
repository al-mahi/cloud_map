#!/usr/bin/env python

from simulator import Simulator
import rospy
import socket
from math import *
from cloud_map.msg import sensor_data, geo_location
from std_msgs.msg import Bool
# from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
import numpy as np
import datetime as dt


class PID:
    def __init__(self, k_p, k_i, k_d, del_t):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self.del_t = del_t


class fg_quad_interface(object):
    def __init__(self, name, port_send, port_recv):
        self._name = name
        self._port_send = port_send
        self._port_recv = port_recv
        self._goal = geo_location()
        self._sensor = sensor_data()
        self._velocity = 1.
        self._is_ready = False
        self._reached_goal = False
        self._rospy_rate = 1
        self._sea_level_ft = int(rospy.get_param("/sea_level_ft"))

    def callback_goal_gps(self, goal):
        """:type goal: geo_location"""
        self._goal = goal
        rospy.logdebug("{}:[{},{},{}]->{}".format(self.tag, self._sensor.Pos_e, self._sensor.Pos_n, self._sensor.Pos_d,
                                                  goal.__getstate__()[1:]))
        # if self._reached_goal and goal:
        #     # print "reached goal new goal ", goal.__getstate__()[1:]
        #     self._goal = goal
        #     self._reached_goal = False

    def callback_is_robot_ready(self, msg):
        """:type msg: Bool"""
        self._is_ready = bool(msg.data)

    def callback_fg_sensor(self, sensor):
        """:type sensor: sensor_data"""
        self._sensor = sensor

    @property
    def tag(self):
        return "{}[{}]".format(self._name, dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%H:%M:%S"))

    def simple_goto(self):
        """
        Continuously Listens to goal_gps topic and set controller to guide aircraft towards the goal
        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("google.com", 80))
            UDP_IP = s.getsockname()[0]
            s.close()
        except socket.error:
            rospy.logdebug("{}:Network connection unavailable...".format(self.tag))
            exit(-1)
        sock_params = {'IP': UDP_IP, "port_send": self._port_send, "port_recv": self._port_recv}
        sim = Simulator(sock_params)
        sensor = sensor_data()

        rospy.init_node(self._name)
        rate = rospy.Rate(self._rospy_rate)
        rospy.Subscriber("/flightgear/{}/next_way_point_gps".format(self._name), data_class=geo_location,
                         callback=self.callback_goal_gps)
        vendor = rospy.get_param('/{}s_vendor'.format(self._name))
        rospy.Subscriber("/{}/{}/ready".format(vendor, self._name), Bool, callback=self.callback_is_robot_ready)

        pub_sensor = rospy.Publisher('{}/sensor_data'.format(self._name), data_class=sensor_data, queue_size=1)

        distance = 0.0

        i = 0

        while not rospy.is_shutdown():
            i = (i + 1) % 30
            fg_data = sim.FGRecv()
            sensor.Pos_n = fg_data[0]
            sensor.Pos_e = fg_data[1]
            sensor.Pos_d = (fg_data[2])  # alt from sea level in feet.
            sensor.V_n_ms = fg_data[3]
            sensor.V_e_ms = fg_data[4]
            sensor.V_d_ms = fg_data[5]
            sensor.u = fg_data[6]
            sensor.v = fg_data[7]
            sensor.w = fg_data[8]
            sensor.roll_deg = fg_data[9]
            sensor.pitch_deg = fg_data[10]
            sensor.yaw_deg = fg_data[11]
            sensor.p_body = fg_data[12]
            sensor.q_body = fg_data[13]
            sensor.r_body = fg_data[14]
            sensor.V_airspeed = fg_data[15]

            sensor.hour = fg_data[29]
            sensor.min = fg_data[30]
            sensor.sec = fg_data[31]
            sensor.Odometer = fg_data[32]
            sensor.CO2Density = fg_data[33]
            sensor.CO2alt_ft = fg_data[34]
            sensor.CO2lat_deg = fg_data[35]
            sensor.CO2lon_deg = fg_data[36]
            sensor.CO2heading_deg = fg_data[37]
            sensor.Temperature_degc = fg_data[38]
            sensor.Relative_humidity = fg_data[39]
            sensor.Pressure_inhg = fg_data[40]
            sensor.Dewpoint_degc = fg_data[41]
            sensor.wind_speed_kt = fg_data[42]
            sensor.wind_heading_deg = fg_data[43]
            poslat = float(sensor.Pos_n)
            poslon = float(sensor.Pos_e)
            posalt = float(sensor.Pos_d)
            lat1 = float(sensor.Pos_n)
            lon1 = float(sensor.Pos_e)
            alt1 = float(sensor.Pos_d)  # feet
            heading = sensor.yaw_deg

            lat2 = self._goal.latitude
            lon2 = self._goal.longitude
            alt2 = self._goal.altitude

            dlat = lat2 - lat1
            dlon = lon2 - lon1
            dalt = alt2 - alt1

            distlat = dlat * 110961.03
            distlon = dlon * 90037.25
            distalt = dalt
            distance = sqrt(distlat ** 2 + distlon ** 2 + distalt ** 2)

            reslat = lat2
            reslon = lon2
            resalt = alt2

            if not self._is_ready or distance > 40:
                print "{} quad not ready because is d={}>40?{} is_ready?{}".format(self._name, distance, distance > 40, self._is_ready)
                reslat = lat1
                reslon = lon1
                resalt = alt1

            commands = {
                "throttle": float(0.00),
                "elevator": float(0.0),
                "aileron": float(0.00),
                "rudder": float(0.00),
                "poslat": float(reslat),  # float(reslat),
                "poslon": float(reslon),  # float(reslon),
                "posalt": float(resalt)  # float(resalt)
            }
            # if self._is_ready:
            # if i%self._rospy_rate == 0:
            # print("{}{:02d} num={} lat=cur={} nxt={} g={} lon {} {} {} alt {} {} {} d={}".format(self._name, i, dist_linspace_num, poslat, reslat, lat2, poslon, reslon, lon2, posalt, resalt, alt2, distance))
            if distance < 40.:
                sim.FGSend(commands, for_model="quad")
            pub_sensor.publish(sensor)
            rate.sleep()

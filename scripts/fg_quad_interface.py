#!/usr/bin/env python

from __future__ import print_function
from simulator import Simulator
import rospy
import socket
from math import *
from cloud_map.msg import sensor_data, geo_location, euclidean_location
# from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
import numpy as np


class fg_quad_interface(object):
    def __init__(self, name, port_send, port_recv):
        self._name = name
        self._port_send = port_send
        self._port_recv = port_recv
        self._goal = geo_location()
        self._sensor = sensor_data()

    def callback_goal_gps(self, goal):
        self._goal = goal

    def callback_fg_sensor(self, sensor):
        self._sensor = sensor

    def simple_goto(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("google.com", 80))
            UDP_IP = s.getsockname()[0]
            s.close()
        except socket.error:
            rospy.logdebug("{}[{}]:Network connection unavailable...".format(self._name,self.time_tag))
            exit(-1)

        sock_params = {'IP': UDP_IP, "port_send": self._port_send, "port_recv": self._port_recv}
        sim = Simulator(sock_params)
        sensor = sensor_data()

        rospy.init_node(self._name)
        rospy.Subscriber("/flightgear/{}/next_way_point_gps".format(self._name), data_class=geo_location,
                         callback=self.callback_goal_gps)

        pub_sensor = rospy.Publisher('{}/sensor_data'.format(self._name), data_class=sensor_data, queue_size=1)

        while not rospy.is_shutdown():
            fg_data = sim.FGRecv()
            sensor.Pos_n = fg_data[0]
            sensor.Pos_e = fg_data[1]
            sensor.Pos_d = fg_data[2]
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

            lat1 = self._sensor.Pos_n
            lon1 = self._sensor.Pos_e
            alt1 = self._sensor.Pos_d
            heading = self._sensor.yaw_deg

            lat2 = self._goal.latitude
            lon2 = self._goal.longitude
            alt2 = self._goal.altitude

            ### Uclidian # GPS -> Distance based
            x1 = alt1 * cos(lat1) * sin(lon1)
            y1 = alt1 * sin(lat1)
            z1 = alt1 * cos(lat1) * cos(lon1)

            x2 = alt2 * cos(lat2) * sin(lon2)
            y2 = alt2 * sin(lat2)
            z2 = alt2 * cos(lat2) * cos(lon2)

            dist = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

            dlat = lat2 - lat1
            dlon = lon2 - lon1
            dalt = alt2 - alt1

            Aaltitude = alt1
            Oppsite  = alt2

            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            # c = 2 * atan2(sqrt(a), sqrt(1-a))
            c = 2. * asin(sqrt(a))
            Base = 6371 * c + np.random.rand()*0.1

            Bearing =atan2(cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(lon2-lon1), sin(lon2-lon1)*cos(lat2))

            Bearing = degrees(Bearing)
            Bearing = (Bearing + 360) % 360
            Bearing = (90 - Bearing + 360) % 360

            Base2 = Base * 1000.
            distance = (Base * 2 + Oppsite * 2) / 2
            Caltitude = Oppsite - Aaltitude

            a = Oppsite/Base
            b = atan(a)
            c = degrees(b)

            distance = distance / 1000.
            throttle_low = 0.005
            if dalt > 0:
                elevator = -0.008
                throttle = throttle_low + 0.1*fabs(elevator)
            if dalt < 0:
                elevator = 0.008
                throttle = throttle_low + 0.1*fabs(elevator)
            if dalt == 0:
                elevator = 0
            # PID Controller (2 Jung)
            # heading = 45
            if (Bearing - heading + 360) % 360 > 180:
                aileron = -0.8 * (float(((Bearing - heading + 360) % 360)-180.)/180.)
                throttle = throttle_low + 0.001*fabs(aileron)
            if (Bearing - heading + 360) % 360 < 180:
                aileron = 0.8 * (float(((Bearing - heading + 360) % 360)-180.)/180.)
                throttle = throttle_low + 0.001*fabs(aileron)

            # commands = {"throttle":throttle, "elevator": elevator, "aileron": aileron, "rudder": 0.0}
            # sim.FGSend(commands)

            pub_sensor.publish(sensor)

            rospy.sleep(1)


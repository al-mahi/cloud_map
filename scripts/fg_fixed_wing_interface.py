#!/usr/bin/env python

from simulator import Simulator
import rospy
import socket
from math import *
from cloud_map.msg import sensor_data, geo_location
# from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
import numpy as np
import datetime as dt


class PID:
    def __init__(self, k_p, k_i, k_d, del_t):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self.del_t = del_t

        self.e_p = 0
        self.e_i = 0
        self.e_d = 0

        self.e_last = 0

    def pid_out(self, y_c, y, lim):
        self.e_p = y_c - y
        self.e_i = self.e_i + (self.del_t / 2) * (self.e_p + self.e_last)
        self.e_d = (self.e_p - self.e_last) * (2 / self.del_t) - self.e_d  # derivative error...check this again

        self.e_last = self.e_p

        # calculate unsaturated output

        u_unsat = self.k_p * self.e_p + self.k_i * self.e_i + self.k_d * self.e_d

        # saturate
        u_sat = saturate(u_unsat, lim)

        # antiwindup
        if self.k_i != 0:
            self.e_i = self.e_i + (self.del_t / self.k_i) * (u_sat - u_unsat)
        return u_sat


def saturate(u_in, limit):
    if u_in < -limit:
        u = -limit
    elif u_in > limit:
        u = limit
    else:
        u = u_in
    return u


class fg_fw_interface(object):
    def __init__(self, name, port_send, port_recv):
        self._name = name
        self._port_send = port_send
        self._port_recv = port_recv
        self._goal = geo_location()
        self._sensor = sensor_data()

        self._previous_error_alt = 0.
        self._previous_error_deg = 0.

        # kp = {}
        # ki = {}
        # kd = {}
        # kp["phi"] = 0.9
        # ki["phi"] = 0.00
        # kd["phi"] = 0.00
        # kp["theta"]= 4.9
        # ki["theta"]= 0.0
        # kd["theta"]= 0.00
        # kp["psi"]= 1.
        # ki["psi"]= 0.0
        # kd["psi"] = 0.0
        # kp["vel"] = 0.11
        # ki["vel"]= 0.00
        # kd["vel"] = 0.00
        # kp["alt"] = 0.005
        # ki["alt"] = 0.0000001
        # kd["alt"] = 0.00
        # kp["x"] = 4
        # ki["x"] = 0.00
        # kd["x"] = 0.00
        # del_t = 0.01
        # R = 20

        # # init the inner loop
        # self.PHI = PID(kp=kp["phi"], ki=ki["phi"], k_d=kd["phi"], del_t=del_t)  # roll pid init
        # self.THETA = PID(kp=kp["theta"], ki=ki["theta"], k_d=kd["theta"], del_t=del_t)  # pitch pid init
        # self.PSI = PID(kp=kp["psi"], ki=ki["psi"], k_d=kd["psi"], del_t=del_t)  # yaw pid init
        #
        # # init the outer loop
        # self.V = PID(kp=kp["vel"], ki=ki["vel"], k_d=kd["vel"], del_t=del_t)  # vel init
        # self.H = PID(kp=kp["alt"], ki=ki["alt"], k_d=kd["alt"], del_t=del_t)  # alt init
        # self.X = PID(kp=kp["x"], ki=ki["x"], k_d=kd["x"], del_t=del_t)  # roll pid init  # course init


    @property
    def tag(self):
        return "{}[{}]".format(self._name, dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%H:%M:%S"))

    def callback_goal_gps(self, goal):
        """:type goal: geo_location"""
        self._goal = goal

    def callback_fg_sensor(self, sensor):
        """:type sensor: sensor_data"""
        self._sensor = sensor

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
            rospy.logdebug("{}:Network connection unavailable...".format(self._name,self.tag))
            exit(-1)

        sock_params = {'IP': UDP_IP, "port_send": self._port_send, "port_recv": self._port_recv}
        sim = Simulator(sock_params)
        sensor = sensor_data()

        rospy.init_node(self._name)
        rate = rospy.Rate(50)
        rospy.Subscriber("/flightgear/{}/next_way_point_gps".format(self._name), data_class=geo_location,
                         callback=self.callback_goal_gps)
        pub_sensor = rospy.Publisher('{}/sensor_data'.format(self._name), data_class=sensor_data, queue_size=1)

        i = 0
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

            self._sensor = sensor

            lat1 = float(self._sensor.Pos_n)
            lon1 = float(self._sensor.Pos_e)
            alt1 = float(self._sensor.Pos_d)
            heading = self._sensor.yaw_deg

            if np.isclose(self._goal.latitude, 36.1340362495, atol=1e-8) and np.isclose(self._goal.longitude, 97.0762336919, atol=1e-8):
                rospy.logerr("{} pid: goal {}".format(self._name, self._goal))

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

            max_delta_lat, max_delta_lon = 0.00144194768199/10, 0.00177704227973/10

            max_aileron = 0.6
            max_elevator = 0.2
            max_throttle = 0.002

            kp_alerion = 1./360.
            # kp_alerion = 5.0/180.
            kp_throttle = 0.0
            kp_elevator = 0.1

            kd_alerion = 0.01
            kd_throttle = 0.0
            kd_elevator = 0.0

            throttle = max_throttle

            der_alt = dalt - self._previous_error_alt
            self._previous_error_alt = dalt
            output = kp_elevator * (np.abs(dalt)/50.) + kd_elevator * der_alt
            if dalt > 0: elevator = max(-max_elevator, -output)
            if dalt < 0: elevator = min( max_elevator,  output)

            err_theta = ((Bearing - heading + 360) % 360)
            if err_theta < 180:
                der_theta = err_theta - self._previous_error_deg
                turn = kp_alerion * err_theta + kd_alerion * der_theta
                aileron = min(turn, max_aileron)
            if err_theta > 180:
                der_theta = (360. - err_theta) - self._previous_error_deg
                turn = kp_alerion * (360 - err_theta) + kd_alerion * der_theta
                aileron = max(-turn, -max_aileron)
            print "i={:05} yw={:3.2f} g={:3.2f} err={:3.2f} prv={:3.2f} der_theta={:1.2f} kp*err={:1.2f} kd*der={:1.2f} turn={:1.2f} ail={:1.2f}".format(
                i, heading, Bearing, err_theta, self._previous_error_deg,  der_theta, kp_alerion * err_theta, kd_alerion * der_theta, turn, aileron)

            i += 1
            if i > 1000:
                throttle = 0.00001  # fix aileron first
                elevator = -0.0000001
                # aileron = 0.05

            commands = {"throttle":throttle, "elevator": elevator, "aileron": aileron, "rudder": 0.0}
            sim.FGSend(commands)
            pub_sensor.publish(sensor)
            rate.sleep()

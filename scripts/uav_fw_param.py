#!/usr/bin/python
import numpy as np
from scipy.stats import multivariate_normal
import rospy
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Bool
from cloud_map.msg import *
import datetime as dt


class Param(object):
    def __init__(self):
        self.pose_euclid = euclidean_location()
        self.co2 = CO_2()
        self.orientation_euler = orientation_euler()
        self.humidity = humidity()
        self.temperature = temperature()
        self.explored = []


def scale_dist(F, low=0., high=1., as_prob=False):
    """
    :type F: np.ndarray
    :rtype: np.ndarray
    """
    if as_prob:
        res = F - F.min()
        res /= np.sum(res)
    else:
        cut = (F.min() - low)
        res = F - cut
        res /= res.max()
        res *= high
    return np.nan_to_num(res)


class dummy_uav(object):
    def __init__(self, name, dim, scale, wsize=3):
        self._name = name
        self._scale = scale
        self._dim = dim
        self._space = tuple([scale for _ in range(dim)])
        self._pose = euclidean_location()
        self._vel_euclid = twist_euclid()
        self._orientation = orientation_euler()
        self._decay_explored = .3
        self._neighbors_names = []  # type: [str]
        self._pub_goal_euclid = rospy.Publisher("/UAV/{}/next_way_point_euclid".format(self._name),
                                                data_class=euclidean_location, queue_size=10)
        self._solo_is_ready = False
        self._param_q = {}  # type: dict[str, Param]
        self._joint_belief = np.zeros(shape=self._space)
        self._marked_as_out = False
        # self._path_history = path_history()
        # self._path_history.header.frame_id = self.name
        self._wsize = wsize  # radius of local space
        self.wspace = tuple([2 * self._wsize + 1 for _ in range(self._dim)])
        self._goal = euclidean_location()
        self._explored = np.zeros(self._space)
        self._goal.header.frame_id = self._name
        self._straight_goal = np.array(
            [15, 15, 25]
        )

    def windices(self, pos):
        """
        :type pos: np.ndarray
        :return: indices window which is local space around current position scaled by window size wsize
        :rtype: np.ndarray
        """
        dxyz = np.array(map(lambda ind: ind, np.ndindex(self.wspace))) - self._wsize
        window = dxyz + pos
        window = np.array(filter(lambda ind: 0 <= ind[0] < self._scale and
                                         0 <= ind[1] < self._scale and
                                         0 <= ind[2] < self._scale, window), dtype='int')
        return window

    def phi_avoid_collision(self, indices):
        """:rtype np.ndarray"""
        # res = np.zeros(shape=self._space)
        # fgx, fgy, fgz = self._pose.x, self._pose.y, self._pose.z
        # for nbnm in self.neighbour_names:
        #     nb = self._param_q[nbnm]
        #     vel = self._vel_euclid
        #     f1x, f1y, f1z = nb.pose_euclid.x, nb.pose_euclid.y, nb.pose_euclid.z
        #     f2x = (fgx + (vel.x / 10.) * (f1x - fgx))
        #     f2y = (fgy + (vel.y / 10.) * (f1y - fgy))
        #     f2z = (fgz + (vel.z / 02.) * (f1z - fgz))
        #
        #     def distance(arr):
        #         d1 = np.sqrt((arr[0] - f1x) ** 2. + (arr[1] - f1y) ** 2. + (arr[2] - f1z) ** 2.)
        #         d2 = np.sqrt((arr[0] - f2x) ** 2. + (arr[1] - f2y) ** 2. + (arr[2] - f2z) ** 2.)
        #         return d1 + d2
        #
        #     indices = np.ndindex(self._space)
        #     F = np.array(map(distance, indices), dtype=np.float32).reshape(self._space)
        #     F = 1. / F
        #     res += F
        # return res
        return np.zeros(self._space)

    def phi_explore(self, indices):
        """:rtype np.ndarray"""
        return self._explored

    def phi_boundary(self, indices):
        """:rtype np.ndarray"""
        F = np.zeros(self._space)
        mx = -np.inf
        for ind in indices:
            th = 10
            r = (self._scale-1)/2
            center = np.array([r, r, r])
            d = max(np.abs(ind - center))
            if d + th > r:
                F[tuple(ind)] = d
                mx = max(mx, d)

        return F / mx

    def phi_tunnel(self, indices):
        """:rtype np.ndarray"""
        F = np.zeros(self._space)
        return F

    def phi_valley(self, indices):
        """:rtype np.ndarray"""
        F = np.zeros(self._space)
        mx = -np.inf
        for ind in indices:
            d = ind[1] - 12.5
            F[tuple(ind)] = d
            mx = max(mx, d)
        return F / mx

    @property
    def name(self):
        return self._name

    @property
    def position(self):
        """:rtype: np.ndarray"""
        return np.array(self._pose.__getstate__()[1:])

    @property
    def neighbour_names(self):
        """
        :rtype: list[str]
        """
        return self._neighbors_names

    @property
    def tag(self):
        return "{}[{}]".format(self.name, dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%H:%M:%S"))

    def callback_pose(self, msg):
        """
        :type msg: euclidean_location
        """
        self._param_q[msg.header.frame_id].pose_euclid = msg
        self._param_q[msg.header.frame_id].explored.append(msg.__getstate__()[1:])
        if msg.header.frame_id == self.name:
            self._pose = msg
            # if self._solo_is_ready:
            #     self._path_history.xs.append(float(msg.x))
            #     self._path_history.xs.append(float(msg.y))
            #     self._path_history.xs.append(float(msg.z))
        # self._explored[int(msg.x), int(msg.y), int(msg.z)] = 1.

    def callback_co2(self, msg):
        """:type co2: CO_2"""
        self._param_q[msg.header.frame_id].co2 = msg

    def callback_orientation_euler(self, msg):
        """:type orientation_euler: orientation_euler"""
        self._param_q[msg.header.frame_id].orientation_euler = msg
        if msg.header.frame_id == self.name:
            self._orientation = msg

    def callback_humidity(self, msg):
        """:type msg: humidity"""
        self._param_q[msg.header.frame_id].humidity = msg

    def callback_temperature(self, msg):
        """:type msg: temperature"""
        self._param_q[msg.header.frame_id].temperature = msg

    def callback_vel_euclid(self, msg):
        """:type msg: twist_euclid"""
        if msg.header.frame_id == self._name:
            self._vel_euclid = msg

    def callback_human_goal_euclid(self, msg):
        """:type msg: euclidean_location"""
        print "human "
        msg.header.frame_id = self.name
        self._goal = msg

    def callback_is_robot_ready(self, ready):
        """
        :type ready: Bool
        """
        self._solo_is_ready = ready.data

    def fly_grad(self):
        self._marked_as_out = False
        x, y, z = self.position[:]
        # numpy gradient returns array ordered by axis number. x's axis number is 1 while y's 0 and z's 2
        dFdxy = np.array(np.gradient(self._joint_belief), dtype=np.float)
        #  rounded coordinate for accessing the gradient
        #  approximate gradient at fractional point by linear interpolation
        xr, yr, zr = map(int, map(round, [x, y, z]))
        fraction = np.array([x, y, z]) - np.array([xr, yr, zr])
        fraction_norm = fraction / np.sum(np.abs(fraction))
        fraction_norm[np.abs(fraction_norm) < .1] = 0.
        xn, yn, zn = map(int, np.array([xr, yr, zr]) + np.sign(fraction_norm))
        if xr != xn:
            rx = (x - xr) / (xn - xr)
            dFdx = dFdxy[0, yr, xr, zr] + rx * (dFdxy[0, yn, xn, zr] - dFdxy[0, yr, xr, zr])
        else:
            dFdx = dFdxy[0, yr, xr, zr]
        if yr != yn:
            ry = (y - yr) / (yn - yr)
            dFdy = dFdxy[1, yr, xr, zr] + ry * (dFdxy[1, yn, xn, zn] - dFdxy[1, yr, xr, zr])
        else:
            dFdy = dFdxy[1, yr, xr, zr]
        if zr != zn:
            rz = (z - zr) / (zn - zr)
            dFdz = dFdxy[2, yr, xr, zr] + rz * (dFdxy[2, yn, xn, zn] - dFdxy[2, yr, xr, zr])
        else:
            dFdz = dFdxy[2, yr, xr, zr]
        np.set_printoptions(precision=3)
        grad = np.array([dFdx, dFdy, dFdz])
        k = np.linalg.norm(grad)
        if np.isclose(k, 0., atol=1.e-6):
            # print("optimum!!!")
            # grad = np.random.uniform(-1, 1, self._dim)
            # k = np.sum(np.linalg.norm(grad))
            return
        dx = (grad / k)
        k = k * (self._scale/3.)
        old_pos = self.position

        new_pos = old_pos - k * dx
        for i in range(len(new_pos)):
            if round(new_pos[i]) < 0. or round(new_pos[i]) >= self._scale:
                rospy.logdebug("{} Pushing out! grad={}".format(self.tag, grad))
        self._goal.x = new_pos[0]
        self._goal.y = new_pos[1]
        self._goal.z = new_pos[2]
        np.set_printoptions(precision=3)
        rospy.logdebug("{}:{}->{} k={} grad={} dx={}\ndfdxr[{},{},{}]={} dfdxn[{},{},{}]={} dfdx[{},{},{}]={}\n"
                       "dfdyr[{},{},{}]={} dfdyn[{},{},{}]={} dfdy[{},{},{}]={}\n"
                       "dfdzr[{},{},{}]={} dfdzn[{},{},{}]={} dfdz[{},{},{}]={}".format(
            self.tag, old_pos[:], np.array(self._goal.__getstate__()[1:]), k, grad, dx,
            xr, yr, zr, dFdxy[1, yr, xr, zr], xn, yn, zn, dFdxy[1, yn, xn, zr], x, y, z, dFdx,
            xr, yr, zr, dFdxy[0, yr, xr, zr], xn, yn, zn, dFdxy[0, yn, xn, zr], x, y, z, dFdy,
            xr, yr, zr, dFdxy[2, yr, xr, zr], xn, yn, zn, dFdxy[2, yn, xn, zr], x, y, z, dFdz)
        )

    def fly_local_grad(self):
        self._marked_as_out = False
        t = old_pos = self.position
        yaw = self._orientation.yaw
        pitch = self._orientation.pitch
        #  intention weights, scale 0~1. Phi function must return value in scale 0~1
        weight = {
            "boundary":     1.00,  # better not to loose the robot by letting it out of a boundary
            "collision":    0.00,  # damage due to collision may be repairable
            "tunnel":       0.00,
            "valley":       0.00,
            "explored":     0.00,
        }
        F = np.zeros(self._space)

        dxyz = np.array(map(lambda ind: ind, np.ndindex(self.wspace))) - self._wsize
        wdxyz = dxyz + old_pos
        windices = np.array(
            filter(lambda ind: 0 <= ind[0] < self._scale and 0 <= ind[1] < self._scale and 0 <= ind[2] < self._scale,
                   wdxyz), dtype='int')

        for k in weight.keys():
            w = weight[k]
            if k=="collision": intention = self.phi_avoid_collision(windices)
            elif k=="boundary": intention = self.phi_boundary(windices)
            elif k=="tunnel": intention = self.phi_tunnel(windices)
            elif k=="valley": intention = self.phi_valley(windices)
            elif k=="explored": intention = self.phi_valley(windices)
            F += (w * intention)

        F /= F.max()

        Rz = lambda theta: np.array([
            [np.cos(theta), -np.sin(theta), 0.],
            [np.sin(theta),  np.cos(theta), 0.],
            [0,                         0.,  1.]
        ])

        Ry = lambda theta: np.array([
            [ np.cos(theta), 0., -np.sin(theta)],
            [            0., 1.,             0.],
            [-np.sin(theta), 0.,  np.cos(theta)]
        ])

        L = self._wsize
        Oa = np.zeros(3)
        Ob = np.array([L, 0, 0])
        v = np.array([self._vel_euclid.x, self._vel_euclid.y, self._vel_euclid.z])
        v = v / np.max(np.abs(v))
        v = v * L + t
        Pb = v
        Pa = old_pos

        # theta_z = np.deg2rad(yaw % 360)
        # theta_y = np.rad2deg(pitch % 360)
        # Ra = np.dot(Rz(theta_z), Oa.T)
        # Ra = np.dot(Ry(theta_y), Ra.T)
        # Pa = Ra + t
        # Rb = np.dot(Rz(theta_z), Ob.T)
        # Rb = np.dot(Ry(theta_y), Rb.T)
        # Pb_theta = Rb + t

        tip = np.round(Pb)

        def lm(ind):
            return [ind, F[tuple(ind)], np.sum(np.abs(np.array(ind[:2]) - tip[:2])) +
                    .5 * np.sum(np.abs(np.array(ind[2]) - tip[2]))]

        argdist = map(lm, windices)
        argmin = sorted(argdist, key=lambda x: (x[1], x[2]))
        self._goal.x = argmin[0][0][0]
        self._goal.y = argmin[0][0][1]
        self._goal.z = argmin[0][0][2]
        np.set_printoptions(precision=3)
        rospy.logdebug("{}:{}->{} wsize={}".format(self.tag, old_pos[:], np.array(self._goal.__getstate__()[1:]), self._wsize))
        self._pub_goal_euclid.publish(self._goal)
        self._joint_belief = F

    def fly_straight_goal(self):
        old_pos = self.position
        d = np.sqrt(np.sum((old_pos-self._straight_goal)**2))
        if d < 3:
            yaw = self._orientation.yaw % 360
            diff = 5.
            if (yaw < 45) or (yaw > 360-45):
                self._straight_goal = old_pos + np.array([diff, 0, 0])
            elif 45 <= yaw < 90 + 45:
                self._straight_goal = old_pos + np.array([0, diff, 0])
            elif 90 + 45 <= yaw < 180 + 45:
                self._straight_goal = old_pos - np.array([diff, 0, 0])
            elif 180 + 45 <= yaw < 270 + 45:
                self._straight_goal = old_pos - np.array([0, diff, 0])

        self._goal.x = self._straight_goal[0]
        self._goal.y = self._straight_goal[1]
        self._goal.z = self._straight_goal[2]
        np.set_printoptions(precision=3)
        rospy.logdebug("{}:{}->{} wsize={} d={}".format(self.tag, old_pos[:], np.array(self._goal.__getstate__()[1:]),
                                                        self._wsize, d))
        self._pub_goal_euclid.publish(self._goal)

    def take_off(self):
        """
        uav rosnode launcher and intention publisher
        """
        rospy.init_node(self._name, log_level=rospy.DEBUG)
        rate = rospy.Rate(10)

        vendor = rospy.get_param('/{}s_vendor'.format(self._name))
        rospy.Subscriber("/{}/{}/ready".format(vendor, self._name), Bool, callback=self.callback_is_robot_ready)
        # rospy.Subscriber("/" + vendor + "/{}/fly_grad".format(self._name), String, callback=self.callback_fly)

        rospy.logdebug("{} UAV autonomous waiting for {} to be ready".format(self.tag, vendor))

        while not self._solo_is_ready:
            rospy.sleep(1)

        rospy.logdebug("{} {} is ready".format(self.tag, vendor))

        rospy.logdebug("{} Starting Autonomous mode......!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(self.tag))

        rospy.Subscriber(name="/{}/{}/pose_euclid".format(vendor, self._name), data_class=euclidean_location,
                         callback=self.callback_pose)
        rospy.Subscriber(name='/{}/{}/co2'.format(vendor, self._name), callback=self.callback_co2, data_class=CO_2)
        rospy.Subscriber(name='/{}/{}/orientation_euler'.format(vendor, self._name),
                         callback=self.callback_orientation_euler, data_class=orientation_euler)
        rospy.Subscriber(name='/{}/{}/humidity'.format(vendor, self._name), callback=self.callback_humidity,
                         data_class=humidity)
        rospy.Subscriber(name='/{}/{}/temperature'.format(vendor, self._name), callback=self.callback_temperature,
                         data_class=temperature)
        rospy.Subscriber(name='/{}/{}/vel_euclid'.format(vendor, self._name), callback=self.callback_vel_euclid,
                         data_class=twist_euclid)
        rospy.Subscriber(name='/{}/{}/human_goal_euclid'.format(vendor, self._name), callback=self.callback_human_goal_euclid,
                         data_class=euclidean_location)

        for from_uav in self.neighbour_names:
            self._param_q[from_uav] = Param()
        self._param_q[self.name] = Param()

        for from_uav in self.neighbour_names:
            if from_uav:
                rospy.Subscriber(name="/{}/{}/pose_euclid".format(vendor, from_uav), data_class=euclidean_location,
                                 callback=self.callback_pose)
                rospy.Subscriber(name='/{}/{}/co2'.format(vendor, from_uav), callback=self.callback_co2, data_class=CO_2)
                rospy.Subscriber(name='/{}/{}/orientation_euler'.format(vendor, from_uav),
                                 callback=self.callback_orientation_euler, data_class=orientation_euler)
                rospy.Subscriber(name='/{}/{}/humidity'.format(vendor, from_uav), callback=self.callback_humidity,
                                 data_class=humidity)
                rospy.Subscriber(name='/{}/{}/temperature'.format(vendor, from_uav), callback=self.callback_temperature,
                                 data_class=temperature)
                rospy.Subscriber(name='/{}/{}/vel_euclid'.format(vendor, from_uav), callback=self.callback_vel_euclid,
                                 data_class=twist_euclid)

        self._pose = euclidean_location()
        q_size = 10
        # fix3d
        pub_pose = rospy.Publisher(self.name + '/pose', data_class=euclidean_location, queue_size=q_size)
        pub_self = rospy.Publisher(self.name + '/intention_self', numpy_msg(Belief), queue_size=q_size)
        pub_path_history = rospy.Publisher(self.name + '/path_history', numpy_msg(path_history), queue_size=q_size)

        while not rospy.is_shutdown():
            if any(self.position < 0.) or any(self.position >= self._scale - 1.):
                self._marked_as_out = True
                rospy.logerr("{}:{} Out center={}".format(self.tag, self.position, self._goal))
                self._pub_goal_euclid.publish(self._goal)
            else:
                # self.fly_local_grad()
                self._pub_goal_euclid.publish(self._goal)
                # self.fly_straight_goal()
            # ---------------------publishing own belief for visualization----------------------------------------------
            pub_pose.publish(self._pose)
            msg_viz = Belief()
            msg_viz.header.frame_id = self.name
            msg_viz.data = self._joint_belief.ravel()
            msg_viz.header.stamp = rospy.Time.now()
            pub_self.publish(msg_viz)
            # pub_path_history.publish(self._path_history)
            rate.sleep()


def launch_uav(name, start_at):
    # default scale 2 will be overwritten by rosparam space
    dim = int(rospy.get_param("/dim"))
    scale = int(rospy.get_param("/scale"))
    uav = dummy_uav(name=name, dim=dim, scale=scale)
    # uav.defined_intention_keys = rospy.get_param("/PHI/" + name + "/intent").split('_')

    neighbors = []
    if rospy.has_param("/UAV/" + name + "/neighbors"):
        neighbors = rospy.get_param("/UAV/" + name + "/neighbors").split("_")
        rospy.logdebug("Launch UAV {} with neighbors {}".format(uav.name, neighbors))
        for n in neighbors:
            # todo validate the format of n
            uav.neighbour_names.append(n)

    print("UAV {}d launcing {} with neighboues {}".format(dim, name, neighbors))
    uav.take_off()

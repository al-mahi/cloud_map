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
    def __init__(self, name, dim, scale, wsize=1):
        self._name = name
        self._scale = scale
        self._dim = dim
        self._space = tuple([scale for _ in range(dim)])
        self._pose = euclidean_location()
        self._vel_euclid = twist_euclid()
        self._orientation = orientation_euler()
        self._decay_explored = .3
        self._neighbors_names = []  # type: [str]
        self._indirect_neighbors_names = set()  # type: set[str]
        self._pub_goal_euclid = rospy.Publisher("/UAV/{}/next_way_point_euclid".format(self._name),
                                                data_class=euclidean_location, queue_size=10)
        self._solo_is_ready = False
        self._param_q = {}  # type: dict[str, Param]
        self._param_q_indirect = {}  # type: dict[str, Param]
        self._joint_belief = np.zeros(shape=self._space)
        self._marked_as_out = False
        # self._path_history = path_history()
        # self._path_history.header.frame_id = self.name
        self._wsize = wsize  # radius of local space
        self.wspace = tuple([2 * self._wsize + 1 for _ in range(self._dim)])
        self._goal = euclidean_location()
        self._explored = np.zeros(self._space)
        self._co2_density = np.zeros(self._space)
        # self._high_co2s = {(43, 22, 7): 0.6032779812812805, (40, 23, 6): 0.5902690291404724}  # type: dict[tuple, float]
        self._high_co2s = {}  # type: dict[tuple, float]
        self._goal.header.frame_id = self._name
        self._rospy_rate = 1
        self._logger_i = 0

    def windices(self, pos):
        """
        :type pos: np.ndarray
        :return: indices window which is local space around current position scaled by window size wsize
        :rtype: np.ndarray
        """
        # converting to numpy array is crucial; ndindex can only be used once whereas numpy array can used several times
        dxyz = np.array(map(lambda ind: np.array(ind), np.ndindex(self.wspace))) - self._wsize
        window = dxyz + np.round(pos)
        window = np.array(filter(lambda ind: 0 <= ind[0] < self._scale and
                                             0 <= ind[1] < self._scale and
                                             0 <= ind[2] < self._scale, window), dtype='int')
        return window

    def phi_avoid_collision(self, ind):
        """:rtype np.ndarray"""
        # all phi should be normalized wrt the global possible maximum for the defined space. max of this distance
        # if mistakenly normalized by local max then the affect is even if the actual cell value is low in that
        # space it will be treated wrt to other local cell and the max int local will be weighted
        mx_d = 10.  # internal param
        sum = 0.
        for nbnm in self._neighbors_names:
            nb = self._param_q[nbnm]
            f1x, f1y, f1z = nb.pose_euclid.x, nb.pose_euclid.y, nb.pose_euclid.z
            d = np.sqrt(np.sum((ind[0] - f1x) ** 2. + (ind[1] - f1y) ** 2. + (ind[2] - f1z) ** 2.))
            if d > mx_d: d = mx_d
            sum += d
        res = 1. - sum/(len(self._neighbors_names) * mx_d)
        if res < 0.:
            rospy.logerr("{}:phi avoid collision scaling error! res={} sum={}".format(self.tag, res, sum))
            return -0.0
        return res

    def phi_avoid_collision_indirect(self, ind):
        """:rtype np.ndarray"""
        if len(self._indirect_neighbors_names)==0:
            return 0.
        try:
            mx_d = 10.  # internal param
            sum = 0.
            for nbnm in self._indirect_neighbors_names:
                nb = self._param_q_indirect[nbnm]
                f1x, f1y, f1z = nb.pose_euclid.x, nb.pose_euclid.y, nb.pose_euclid.z
                d = np.sqrt(np.sum((ind[0] - f1x) ** 2. + (ind[1] - f1y) ** 2. + (ind[2] - f1z) ** 2.))
                if d > mx_d: d = mx_d
                sum += d
            res = 1. - sum/(len(self._indirect_neighbors_names) * mx_d)
            if res < 0.:
                rospy.logerr("{}:phi avoid collision scaling error! res={} sum={}".format(self.tag, res, sum))
                return -0.0
            return res
        except KeyError as ex:
            rospy.logerr("{}: phi_avoid_collision_indirect {} indrect neighbors {}".format(self.tag, ex.message, self._indirect_neighbors_names))
            return 1.

    def phi_co2(self, ind):
        """:rtype np.ndarray"""
        res = 1.
        fmax = -np.inf
        if len(self._high_co2s.keys()) == 0:
            return 1.
        for co2ind in self._high_co2s.keys():
            if self._explored[ind] > 0: return 1.
            f = np.sqrt(np.sum((np.array(co2ind)-np.array(ind)-self._wsize)**4.))
            fmax = max(f, fmax)
            f = np.sqrt(np.sum((np.array(co2ind)-np.array(ind)+self._wsize)**4.))
            fmax = max(f, fmax)

        for co2ind in self._high_co2s.keys():
            rho = self._high_co2s[co2ind]
            d = np.sqrt(np.sum((np.array(co2ind)-np.array(ind))**4))
            f = (1.-rho) * d
        return f/fmax

    def phi_explore(self, ind):
        """:rtype np.ndarray"""
        return min(4., self._explored[ind])/4. if self._explored[ind]>0. else 0.

    def phi_boundary(self, ind):
        """:rtype np.ndarray"""
        th = 4
        rmax = self._scale/2
        center = rmax * np.ones(3)
        r = max(np.abs(ind - center))
        if r + th > rmax: return r/rmax
        return 0.

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
    def tag(self):
        return "{}[{}]".format(self.name, dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%H:%M:%S"))

    def fly_grad(self):
        self._marked_as_out = False
        x, y, z = self.position[:]
        # numpy gradient returns array ordered by axis number. x's axis number is 1 while y's 0 and z's 2
        dFdxy = np.array(np.gradient(self._joint_belief), dtype=np.float)
        #  rounded coordinate for accessing the gradient
        # approximate gradient at fractional point by linear interpolation
        xr, yr, zr = map(int, map(round, [x, y, z]))
        fraction = np.array([x, y, z]) - np.array([xr, yr, zr])
        fraction_norm = fraction / np.sum(np.abs(fraction))
        fraction_norm[np.abs(fraction_norm) < .1] = 0
        xn, yn, zn = map(int, np.array([xr, yr, zr]) + np.sign(fraction_norm))
        if xr != xn:
            rx = (x - xr) / (xn - xr)
            dFdx = dFdxy[1, yr, xr, zr] + rx * (dFdxy[1, yn, xn, zr] - dFdxy[1, yr, xr, zr])
        else:
            dFdx = dFdxy[1, yr, xr, zr]
        if yr != yn:
            ry = (y - yr) / (yn - yr)
            dFdy = dFdxy[0, yr, xr, zr] + ry * (dFdxy[0, yn, xn, zn] - dFdxy[0, yr, xr, zr])
        else:
            dFdy = dFdxy[0, yr, xr, zr]
        if zr != zn:
            rz = (z - zr) / (zn - zr)
            dFdz = dFdxy[2, yr, xr, zr] + rz * (dFdxy[2, yn, xn, zn] - dFdxy[2, yr, xr, zr])
        else:
            dFdz = dFdxy[2, yr, xr, zr]
        np.set_printoptions(precision=3)
        grad = np.array([dFdx, dFdy, dFdz])
        k = np.sum(np.abs(grad))
        if np.isclose(k, 0., atol=1.e-6):
            print("optimum!!!")
            grad = np.random.uniform(-1, 1, self._dim)
            k = np.sum(np.abs(grad))
        dx = (grad / k)  # + np.random.uniform(low=-1, high=1, size=dim)/100000.

        old_pos = self.position
        goal = euclidean_location()
        dx[np.random.random(self._dim) < 0.1] = 0.
        new_pos = old_pos - 2. * dx
        for i in range(len(new_pos)):
            if round(new_pos[i]) < 0. or round(new_pos[i]) >= self._scale:
                rospy.logdebug("{} Pushing out! grad={}".format(self.tag, grad))
        goal.x = new_pos[0]
        goal.y = new_pos[1]
        goal.z = new_pos[2]
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
            "boundary":             1.00,  # better not to loose the robot by letting it out of a boundary
            "collision":            0.50,  # damage due to collision may be repairable
            "collision_indirect":   0.80,
            "explored":             1.00,
        }

        if self.name=="A":
            weight["co2"] = 1.00
        F = np.zeros(self._space)
        windices = self.windices(old_pos)
        mn = np.inf
        mx = -np.inf
        argmin = None
        for ind in windices:
            wsum = 0
            ind = tuple(ind)
            for k in weight.keys():
                # careful because win
                p = 1.
                w = weight[k]
                if np.isclose(w, 0.): continue
                if   k == "collision":          p = self.phi_avoid_collision(ind)
                elif k == "collision_indirect": p = self.phi_avoid_collision_indirect(ind)
                elif k == "boundary":           p = self.phi_boundary(ind)
                elif k == "explored":           p = self.phi_explore(ind)
                elif k == "co2":                p = self.phi_co2(ind)
                prod = w * p
                wsum += prod
                # rospy.logdebug("{}:k={:20} F{}  {:1.3}={:1.3}*{:1.3} wsum {:1.3} mn={:1.3} mx={:1.3} {} {}".format(
                #         self.tag, ind, k, prod, w, p, wsum, mn, mx, wsum<mn, wsum>mx))
            F[ind] = wsum
            mx = max(wsum, mx)
            if mn > wsum:
                mn = wsum
                argmin = ind

        self._goal.x = argmin[0]
        self._goal.y = argmin[1]
        self._goal.z = argmin[2]
        self._pub_goal_euclid.publish(self._goal)
        self._joint_belief = F

    def callback_pose(self, msg):
        """
        :type msg: euclidean_location
        """
        self._param_q[msg.header.frame_id].pose_euclid = msg
        self._param_q[msg.header.frame_id].explored.append(msg.__getstate__()[1:])
        if msg.header.frame_id == self.name:
            self._pose = msg
        if 0 < msg.x < self._scale and 0 < msg.y < self._scale and 0 < msg.z < self._scale:
            self._explored[int(np.floor(msg.x)), int(np.floor(msg.y)), int(np.floor(msg.z))] += 1.
            self._explored[int(np.round(msg.x)), int(np.round(msg.y)), int(np.round(msg.z))] += 1.
            # if self._logger_i % self._rospy_rate == 0:
            #     print  "{} explored[{} {} {}]={} explored[{} {} {}]={}".format(self.name,
            #     int(np.floor(msg.x)), int(np.floor(msg.y)), int(np.floor(msg.z)),
            #     self._explored[int(np.floor(msg.x)), int(np.floor(msg.y)), int(np.floor(msg.z))],
            #     int(np.round(msg.x)), int(np.round(msg.y)), int(np.round(msg.z)),
            #     self._explored[int(np.round(msg.x)), int(np.round(msg.y)), int(np.round(msg.z))])

    def callback_belief_params(self, msg):
        """:type msg: belief_param"""
        if msg.header.frame_id == self.name:
            for x, y, z, co2, nn in zip(msg.pos_xs, msg.pos_ys, msg.pos_zs, msg.co2s, msg.neighbors):
                # xi, yi, zi = map(int, np.floor([x, y, z]))
                # xr, yr, zr = map(int, np.round([x, y, z]))
                # self._explored[xi, yi, zi] += 1.
                # self._explored[xr, yr, xr] += 1.
                if not self._param_q_indirect.has_key(nn):
                    self._param_q_indirect[nn] = Param()
                self._param_q_indirect[nn].pose_euclid.x = x
                self._param_q_indirect[nn].pose_euclid.y = y
                self._param_q_indirect[nn].pose_euclid.z = z
                self._indirect_neighbors_names.add(nn)

    def callback_co2(self, msg):
        """:type msg: CO_2"""
        self._param_q[msg.header.frame_id].co2 = msg
        if msg.density>.1:
            if msg.header.frame_id==self.name:
                xi, yi, zi = map(int, np.floor(self.position))
                xr, yr, zr = map(int, np.round(self.position))
                self._co2_density[xi, yi, zi] = msg.density
                self._co2_density[xr, yr, zr] = msg.density
            else:
                print "msg co2 recvd!!!!!!!!!!!!"
                pos = self._param_q[msg.header.frame_id].pose_euclid
                xi, yi, zi = map(int, np.floor([pos.x, pos.y, pos.z]))
                xr, yr, zr = map(int, np.round([pos.x, pos.y, pos.z]))
                self._high_co2s[(xi, yi, zi)] = float(msg.density)
                self._high_co2s[(xr, yr, zr)] = float(msg.density)

    def callback_orientation_euler(self, msg):
        """:type msg: orientation_euler"""
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

    def take_off(self):
        """
        uav rosnode launcher and intention publisher
        """
        rospy.init_node(self._name, log_level=rospy.DEBUG)
        rate = rospy.Rate(self._rospy_rate)
        self._pose = euclidean_location()

        vendor = rospy.get_param('/{}s_vendor'.format(self._name))
        rospy.Subscriber("/{}/{}/ready".format(vendor, self._name), Bool, callback=self.callback_is_robot_ready)
        # rospy.Subscriber("/" + vendor + "/{}/fly_grad".format(self._name), String, callback=self.callback_fly)
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
        rospy.Subscriber(name='/{}/{}/human_goal_euclid'.format(vendor, self._name),
                         callback=self.callback_human_goal_euclid,
                         data_class=euclidean_location)

        for from_uav in self._neighbors_names:
            self._param_q[from_uav] = Param()
        self._param_q[self.name] = Param()

        for from_uav in self._neighbors_names:
            if from_uav:
                rospy.Subscriber(name="/{}/{}/pose_euclid".format(vendor, from_uav), data_class=euclidean_location,
                                 callback=self.callback_pose)
                rospy.Subscriber(name='/{}/{}/co2'.format(vendor, from_uav), callback=self.callback_co2,
                                 data_class=CO_2)
                rospy.Subscriber(name='/{}/{}/orientation_euler'.format(vendor, from_uav),
                                 callback=self.callback_orientation_euler, data_class=orientation_euler)
                rospy.Subscriber(name='/{}/{}/humidity'.format(vendor, from_uav), callback=self.callback_humidity,
                                 data_class=humidity)
                rospy.Subscriber(name='/{}/{}/temperature'.format(vendor, from_uav), callback=self.callback_temperature,
                                 data_class=temperature)
                rospy.Subscriber(name='/{}/{}/vel_euclid'.format(vendor, from_uav), callback=self.callback_vel_euclid,
                                 data_class=twist_euclid)
                rospy.Subscriber(name='/UAV/{}/belief_params'.format(from_uav),callback=self.callback_belief_params,
                                 data_class=belief_param)

        rospy.logdebug("{} UAV autonomous waiting for {} to be ready".format(self.tag, vendor))

        while not self._solo_is_ready:
            rospy.sleep(1)

        self._goal = self._pose

        rospy.logdebug("{} {} is ready".format(self.tag, vendor))
        rospy.logdebug("{} Starting Autonomous mode......!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(self.tag))

        q_size = 10
        # fix3d
        pub_pose = rospy.Publisher(self.name + '/pose', data_class=euclidean_location, queue_size=q_size)
        pub_self = rospy.Publisher(self.name + '/intention_self', numpy_msg(Belief), queue_size=q_size)

        while not rospy.is_shutdown():
            self._logger_i = (self._logger_i + 1) % self._rospy_rate  # rate
            if any(self.position < 0.) or any(self.position >= self._scale - 1.):
                self._marked_as_out = True
                rospy.logerr("{}:{} Out center={}".format(self.tag, self.position, self._goal))
                self._pub_goal_euclid.publish(self._goal)
            else:
                if self.name != "H":
                    self.fly_local_grad()
                # self._pub_goal_euclid.publish(self._goal)

            # ---------------------publishing own belief for visualization----------------------------------------------
            pub_pose.publish(self._pose)
            msg_viz = Belief()
            msg_viz.header.frame_id = self.name
            msg_viz.data = self._joint_belief.ravel()
            msg_viz.header.stamp = rospy.Time.now()
            pub_self.publish(msg_viz)
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
            uav._neighbors_names.append(n)

    print("UAV {}d launching {} with neighbors {}".format(dim, name, neighbors))
    uav.take_off()

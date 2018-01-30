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
    def __init__(self, name, dim, scale):
        self._name = name
        self._scale = scale
        self._dim = dim
        self._space = tuple([scale for _ in range(dim)])
        self._pose = euclidean_location()
        self._vel_euclid = twist_euclid()
        self._decay_explored = .3
        self._neighbors_names = []  # type: [str]
        self._pub_goal_euclid = rospy.Publisher("/UAV/{}/next_way_point_euclid".format(self._name),
                                                data_class=euclidean_location, queue_size=10)
        self._solo_is_ready = False
        self._param_q = {}  # type: dict[str, Param]
        self._joint_belief = np.zeros(shape=self._space)

    @property
    def phi_avoid_collision(self):
        """:rtype np.ndarray"""
        res = np.zeros(shape=self._space)
        fgx, fgy, fgz = self._pose.x, self._pose.y, self._pose.z
        for nbnm in self.neighbour_names:
            nb = self._param_q[nbnm]
            vel = self._vel_euclid
            f1x, f1y, f1z = nb.pose_euclid.x, nb.pose_euclid.y, nb.pose_euclid.z

            f2x = (fgx + (vel.x / 10.) * (f1x - fgx))
            f2y = (fgy + (vel.y / 10.) * (f1y - fgy))
            f2z = (fgz + (vel.z / 02.) * (f1z - fgz))

            def distance(arr):
                d1 = np.sqrt((arr[0] - f1x) ** 2. + (arr[1] - f1y) ** 2. + (arr[2] - f1z) ** 2.)
                d2 = np.sqrt((arr[0] - f2x) ** 2. + (arr[1] - f2y) ** 2. + (arr[2] - f2z) ** 2.)
                return d1 + d2

            indices = np.ndindex(self._space)
            F = np.array(map(distance, indices), dtype=np.float32).reshape(self._space)
            F = 1. / F
            res += F
        return res

    @property
    def phi_explore(self):
        """:rtype np.ndarray"""
        return

    @property
    def phi_boundary(self):
        """:rtype np.ndarray"""
        def distance(ind):
            center = np.array([(self._scale-1)/2., (self._scale-1)/2., (self._scale-1)/2.])
            d = np.max(np.abs(ind - center))**8
            return d
        indices = np.ndindex(self._space)
        F = np.array(map(distance, indices), dtype=np.float32).reshape(self._space)
        F /= np.max(F)
        return F

    @property
    def name(self):
        return self._name

    @property
    def position(self):
        """
        :rtype: np.ndarray
        """
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

    def sum_product_algo2(self):
        """
        Algo 2 is taking sum instead of product.
        :param to_uav:
        """
        #  intention weight is the pick of the intention in a scale 0~1
        weight = {
            "boundary":     1.00,  # better not to loose the robot by letting it out of a boundary
            "collision":    0.95,  # damage due to collision may be repairable
        }
        F = weight["collision"] * self.phi_avoid_collision #+ weight["boundary"]  * self.phi_boundary
        self._joint_belief = F / np.max(F)

    def callback_pose(self, msg):
        """
        :type msg: euclidean_location
        """
        self._param_q[msg.header.frame_id].pose_euclid = msg
        self._param_q[msg.header.frame_id].explored.append(msg.__getstate__()[1:])
        if msg.header.frame_id == self.name:
            self._pose = msg

    def callback_co2(self, msg):
        """:type co2: CO_2"""
        self._param_q[msg.header.frame_id].co2 = msg

    def callback_orientation_euler(self, msg):
        """:type orientation_euler: orientation_euler"""
        self._param_q[msg.header.frame_id].orientation_euler = msg

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



    def callback_is_robot_ready(self, ready):
        """
        :type ready: Bool
        """
        self._solo_is_ready = ready.data

    def fly(self):
        x, y, z = self.position[:]
        dFdxy = np.array(np.gradient(self._joint_belief), dtype=np.float)
        #  rounded coordinate for accessing the gradient
        #  approximate gradient at fractional point by linear interpolation
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
        goal.header.frame_id = self.name
        dx[np.random.random(self._dim) < 0.1] = 0.
        new_pos = old_pos - 2. * dx
        for i in range(len(new_pos)):
            if round(new_pos[i]) < 0. or round(new_pos[i]) >= self._scale:
                rospy.logdebug("{} Pushing out! grad={}".format(self.tag, grad))
        goal.x = new_pos[0]
        goal.y = new_pos[1]
        if self._dim == 3:
            goal.z = new_pos[2]

        np.set_printoptions(precision=3)
        if self._dim == 2:
            rospy.logdebug(
                "{}:{}->{} grad={} dx={}\ndfdxr[{},{}]={} dfdxn[{},{}]={} dfdx[{},{}]={}\ndfdyr[{},{}]={} "
                "dfdyn[{},{}]={} dfdy[{},{}]={}".format(self.tag, old_pos[:], goal.position.__getstate__()[:], grad, dx,
                                                        xr, yr, dFdxy[1, yr, xr], xn, yn, dFdxy[1, yn, xn], x, y, dFdx,
                                                        xr, yr, dFdxy[0, yr, xr], xn, yn, dFdxy[0, yn, xn], x, y, dFdy)
            )
        if self._dim == 3:
            rospy.logdebug("{}:{}->{} grad={} dx={}\ndfdxr[{},{},{}]={} dfdxn[{},{},{}]={} dfdx[{},{},{}]={}\n"
                           "dfdyr[{},{},{}]={} dfdyn[{},{},{}]={} dfdy[{},{},{}]={}\n"
                           "dfdzr[{},{},{}]={} dfdzn[{},{},{}]={} dfdz[{},{},{}]={}".format(
                self.tag, old_pos[:], goal.__getstate__()[1:], grad, dx,
                xr, yr, zr, dFdxy[1, yr, xr, zr], xn, yn, zn, dFdxy[1, yn, xn, zr], x, y, z, dFdx,
                xr, yr, zr, dFdxy[0, yr, xr, zr], xn, yn, zn, dFdxy[0, yn, xn, zr], x, y, z, dFdy,
                xr, yr, zr, dFdxy[2, yr, xr, zr], xn, yn, zn, dFdxy[2, yn, xn, zr], x, y, z, dFdz)
            )
        self._pub_goal_euclid.publish(goal)

    def take_off(self, start_at):
        """
        uav rosnode launcher and intention publisher
        """
        rospy.init_node(self._name, log_level=rospy.DEBUG)
        rate = rospy.Rate(5)

        vendor = rospy.get_param('/{}s_vendor'.format(self._name))
        rospy.Subscriber("/{}/{}/ready".format(vendor, self._name), Bool, callback=self.callback_is_robot_ready)
        # rospy.Subscriber("/" + vendor + "/{}/fly".format(self._name), String, callback=self.callback_fly)

        rospy.logdebug("{} UAV autonomous waiting for {} to be ready".format(self.tag, vendor))

        while not self._solo_is_ready:
            rospy.sleep(1)

        rospy.logdebug("{} {} is ready".format(self.tag, vendor))

        # choice = raw_input("Start autonomous node? yes/no:\n>> ").lower()
        # while not choice == "yes":
        #     choice = raw_input("Start autonomous node? yes/no:\n>> ").lower()
        #     rospy.sleep(10)

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

        for from_uav in self.neighbour_names:
            self._param_q[from_uav] = Param()
        self._param_q[self.name] = Param()

        for from_uav in self.neighbour_names:
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

        while not rospy.is_shutdown():
            self.sum_product_algo2()
            self.fly()
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
            uav.neighbour_names.append(n)

    print("UAV {}d launcing {} with neighboues {}".format(dim, name, neighbors))
    uav.take_off(start_at=start_at)

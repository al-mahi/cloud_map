#!/usr/bin/python
import numpy as np
from scipy.stats import multivariate_normal
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from tf.transformations import quaternion_from_euler
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String, Float32, Bool
from cloud_map.msg import Belief
import datetime as dt


class dummy_uav(object):
    def __init__(self, name, dim, scale, sensor_tmp=False, sensor_hum=False, sensor_pressure=False):
        self._name = name
        self._scale = scale
        self._dim = dim
        self._space = tuple([scale for _ in range(dim)])
        self.has_tmp_sensor = sensor_tmp
        self.has_hum_sensor = sensor_hum
        self.has_pressure_sensor = sensor_pressure
        self._pose = Pose(Point(0., 0., 0.), Quaternion(*quaternion_from_euler(0., 0., 0.)))
        self._vel = Twist(Vector3(1., 0., 0.), Vector3(0., 0., 0.))
        self._stall_vel = 1.0
        self._defined_intention_keys = ["boundary", "unexplored", "tempchange", "avoidcollision", "humaninteresting",
                                        "humanannoying", "humiditychange"]
        # self._neighbors_intention_keys = ["boundary", "unexplored", "tempchange"]
        self._decay_explored = .3
        self._decay_belief = .3
        self._belief = np.ones(shape=self._space) / (scale**self._dim)
        self._intention_unexplored = np.ones(shape=self._space, dtype=np.float32) / (scale**self._dim)  # uniform prior
        self._intention_fusion = np.ones(shape=self._space, dtype=np.float32) / (scale**self._dim)  # uniform prior
        self._neighbors_names = []  # type: [str]
        self._msg_received = {}  # type: dict[str, Belief]
        self._msg_send = {}  # type: dict[str, np.ndarray]
        self._phi = {}  # type: dict[str, np.ndarray]
        for key in self._defined_intention_keys:
            self._phi[key] = np.zeros(self._space)
        self._pub_goal_euclid = rospy.Publisher("/UAV/{}/next_way_point_euclid".format(self._name), data_class=Pose, queue_size=10)
        self._solo_is_ready = False
        self._solo_wants_to_fly = False

    @property
    def name(self):
        return self._name

    @property
    def position(self):
        """
        :rtype: np.ndarray
        """
        return np.array(self._pose.position.__getstate__()[:self._dim])

    @property
    def neighbour_names(self):
        """
        :rtype: list[str]
        """
        return self._neighbors_names

    @property
    def msg_received(self):
        """
        :rtype: dict[str, Belief]
        """
        return self._msg_received

    @property
    def belief_position(self):
        """
        unexplored intention has been defined as a multivariate gaussian distribution having mean at the current pose
        of the uav. the distribution is over the 3d space
        :rtype: np.ndarray
        """
        dist = multivariate_normal(mean=self.position, cov=1. * self._scale * np.identity(self._dim))
        indices = np.ndindex(self._space)
        self._belief = np.array(map(lambda x: dist.pdf(np.array(x[:])), indices), dtype=np.float32).reshape(self._space)
        return self._belief

    @property
    def defined_intention_keys(self):
        """
        :rtype : list[str]
        """
        return self._defined_intention_keys

    @defined_intention_keys.setter
    def defined_intention_keys(self, value):
        self._defined_intention_keys = value
        for key in value:
            self._phi[key] = np.zeros(self._space)

    @property
    def tag(self):
        return "{}[{}]".format(self._name, dt.datetime.fromtimestamp(rospy.Time.now().to_time()). strftime("%H:%M:%S"))

    def sum_product_algo2(self):
        """
        Algo 2 is taking sum instead of product.
        :param to_uav:
        """
        #  intention weight is the pick of the intention in a scale 0~1
        weight_high = {
            "neighbors": 1.0,
            "unexplored": 0.3,
            "avoidcollision": 1.0,
            "boundary": 1.0,
            "tempchange": 0.6,
            "humiditychange": 0.6,
            "humaninteresting": 0.5,
            "humanannoying": 0.9,
        }
        weight_low = {
            "neighbors": 0.5,
            "unexplored": 0.1,
            "avoidcollision": 0.0,
            "boundary": 0.5,
            "tempchange": 0.1,
            "humiditychange": 0.1,
            "humaninteresting": 0.0,
            "humanannoying":0.5,
        }

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

        neighbors_belief = np.zeros(shape=self._space)
        for v in self.neighbour_names:
            if not self._msg_received[v].data is None:
                nb = np.array(np.asanyarray(self._msg_received[v].data, dtype=np.float32).reshape(self._space))
                if nb.max() > 0.:
                    neighbors_belief += scale_dist(nb, low=weight_low["neighbors"], high=weight_high["neighbors"])

        # build own belief
        # start with boundary
        if self._dim == 2:
            boundary = np.zeros(shape=(self._scale, self._scale))
            X, Y = np.meshgrid(np.arange(0, self._scale, 1.), np.arange(0, self._scale, 1.))
            # boundary = (X-self._scale/2)**4 + (Y-self._scale/2)**4
            boundary[:, 0] = boundary[0, :] = boundary[:, self._scale-1] = boundary[self._scale-1, :] = 1.
            boundary = scale_dist(boundary, low=weight_low["boundary"], high=weight_high["boundary"])

        if self._dim == 3:
            boundary = np.zeros(shape=(self._scale, self._scale, self._scale))
            X, Y, Z = np.meshgrid(np.arange(0, self._scale, 1.), np.arange(0, self._scale, 1.), np.arange(0, self._scale, 1.))
            boundary[0, :, :] = boundary[:, 0, :] = boundary[:, :, 0] = 1.
            boundary[self._scale-1, :, :] = boundary[:, self._scale-1, :] = boundary[:, :, self._scale-1] = 1.

        self._phi["boundary"] = boundary

        own_belief = np.zeros(shape=self._space)
        for key in self.defined_intention_keys:
            if self._phi.has_key(key):
                own_belief += scale_dist(self._phi[key], low=weight_low[key], high=weight_high[key])

        for key in ["tempchange", "humiditychange", "humanannoying", "humaninteresting", "boundary"]:
            own_belief -= scale_dist(self._phi[key], low=weight_low[key], high=weight_high[key])

        for v in self.neighbour_names:
            if not self._msg_received[v].data is None:
                nb = np.array(np.asanyarray(self._msg_received[v].data, dtype=np.float32).reshape(self._space))
                nb = scale_dist(nb, low=weight_low["neighbors"], high=weight_high["neighbors"])
                other_neighbor = neighbors_belief - nb
                tmp = other_neighbor + own_belief
                self._msg_send[v] = scale_dist(tmp, as_prob=True)

        for key in ["tempchange", "humiditychange", "humanannoying", "humaninteresting", "boundary"]:
            own_belief += scale_dist(self._phi[key], low=weight_low[key], high=weight_high[key])

        key = "avoidcollision"
        own_belief -= scale_dist(self._phi[key], low=weight_low[key], high=weight_high[key])

        new_intention = own_belief + neighbors_belief
        res = np.array(new_intention.reshape(self._space), dtype=np.float32)
        self._intention_fusion = scale_dist(res, as_prob=True)

    def fly(self):
        if self._dim == 2:
            x, y = self.position[:]
            dFdxy = np.array(np.gradient(self._intention_fusion), dtype=np.float)
            xr, yr = map(int, map(round, [x, y]))
            # approximate gradient at fractional point by linear interpolation
            fraction = np.array([x, y]) - np.array([xr, yr])
            fraction_norm = fraction / np.sum(np.abs(fraction))
            # fraction_norm[np.abs(fraction_norm) < np.random.random()/self._dim] = 0
            xn, yn = map(int, np.array([xr, yr]) + np.sign(fraction_norm))
            if xr != xn:
                rx = (x - xr) / (xn - xr)
                dFdx = dFdxy[1, yr, xr] + rx * (dFdxy[1, yn, xn] - dFdxy[1, yr, xr])
            else:
                dFdx = dFdxy[1, yr, xr]
            if yr != yn:
                ry = (y - yr) / (yn - yr)
                dFdy = dFdxy[0, yr, xr] + ry * (dFdxy[0, yn, xn] - dFdxy[0, yr, xr])
            else:
                dFdy = dFdxy[0, yr, xr]
            grad = np.array([dFdx, dFdy])
            k = np.sum(np.abs(grad))

            if np.isclose(k, 0., atol=1.e-6):
                rospy.logdebug("[{}]{} at optimum!".format(dt.datetime.fromtimestamp(rospy.Time.now().to_time()).
                                                           strftime("%H:%M:%S"), self.name))
                grad = np.random.uniform(-1, 1, self._dim)
                k = np.sum(np.abs(grad))
            dx = (grad / k)  # + np.random.uniform(low=-1, high=1, size=dim)/100000.

        if self._dim == 3:
            x, y, z = self.position[:]
            dFdxy = np.array(np.gradient(self._intention_fusion), dtype=np.float)
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
            dx = (grad / k) #+ np.random.uniform(low=-1, high=1, size=dim)/100000.

        old_pos = self.position
        goal = Pose()
        dx[np.random.random(self._dim) < 0.1] = 0.
        new_pos = old_pos - 2. * dx
        for i in range(len(new_pos)):
            if round(new_pos[i]) < 0. or round(new_pos[i]) >= self._scale:
                rospy.logdebug("[{}]{} Pushing out! grad={}".format(dt.datetime.fromtimestamp(
                    rospy.Time.now().to_time()).strftime("%M:%S.%f"), self.name, grad))
        goal.position.x = new_pos[0]
        goal.position.y = new_pos[1]
        if self._dim == 3:
            goal.position.z = new_pos[2]

        np.set_printoptions(precision=3)
        if self._dim == 2:
            rospy.logdebug(
                "[{}]{}:{}->{} grad={} dx={}\ndfdxr[{},{}]={} dfdxn[{},{}]={} dfdx[{},{}]={}\ndfdyr[{},{}]={} "
                "dfdyn[{},{}]={} dfdy[{},{}]={}".format(dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime(
                    "%H:%M:%S"), self.name, old_pos[:], goal.position.__getstate__()[:], grad,dx,
                            xr, yr, dFdxy[1, yr, xr], xn, yn,dFdxy[1, yn, xn], x, y, dFdx,
                            xr, yr, dFdxy[0, yr, xr],xn, yn, dFdxy[0, yn, xn], x, y, dFdy)
            )
        if self._dim == 3:
            rospy.logdebug("[{}]{}:{}->{} grad={} dx={}\ndfdxr[{},{},{}]={} dfdxn[{},{},{}]={} dfdx[{},{},{}]={}\n"
                  "dfdyr[{},{},{}]={} dfdyn[{},{},{}]={} dfdy[{},{},{}]={}\n"
                  "dfdzr[{},{},{}]={} dfdzn[{},{},{}]={} dfdz[{},{},{}]={}".format(
                dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime(
                "%H:%M:%S"), self.name, old_pos[:], goal.position.__getstate__()[:], grad,dx,
                xr, yr, zr, dFdxy[1, yr, xr, zr], xn, yn, zn, dFdxy[1, yn, xn, zr], x, y, z, dFdx,
                xr, yr, zr, dFdxy[0, yr, xr, zr], xn, yn, zn, dFdxy[0, yn, xn, zr], x, y, z, dFdy,
                xr, yr, zr, dFdxy[2, yr, xr, zr], xn, yn, zn, dFdxy[2, yn, xn, zr], x, y, z, dFdz)
            )
        self._pub_goal_euclid.publish(goal)

    def callback(self, pdf_intention):
        """
        ros topic listener
        receive message in factor graph
        :param pdf_intention: pdf of intention over the space
        :type pdf_intention: Belief
        """
        if pdf_intention.header.frame_id[2] == self.name:
            intention_data = np.nan_to_num(np.asanyarray(pdf_intention.data, dtype=np.float32))
            msg_formatted = Belief()
            msg_formatted.header = pdf_intention.header
            msg_formatted.data = intention_data.ravel()
            self._msg_received[pdf_intention.header.frame_id[0]] = msg_formatted

    def callback_sensor_pose(self, pose):
        """
        :type pose: Pose
        """
        self._pose = pose

    def callback_fly(self, msg_fly):
        """
        :type msg_fly: String
        """
        # listen to fly_grad message.
        if msg_fly.data == "fly_grad":
            self._solo_wants_to_fly = True

    def callback_phi_unexplored(self, pdf_unexplored):
        # :type pdf_unexplored: Belief
        self._phi["unexplored"] = np.asanyarray(pdf_unexplored.data, dtype=np.float32).reshape(self._space)

    def callback_phi_temp_change(self, pdf_tempchange):
        # :type pdf_tempchange: Belief
        self._phi["tempchange"] = np.asanyarray(pdf_tempchange.data, dtype=np.float32).reshape(self._space)

    def callback_phi_hum_change(self, pdf_humiditychange):
        # :type pdf_humiditychange: Belief
        self._phi["humiditychange"] = np.asanyarray(pdf_humiditychange.data, dtype=np.float32).reshape(self._space)

    def callback_phi_avoid_collision(self, pdf_avoid_collision):
        # :type pdf_avoid_collision: Belief
        self._phi["avoidcollision"] = np.asanyarray(pdf_avoid_collision.data, dtype=np.float32).reshape(self._space)

    def callback_phi_human_interesting(self, pdf_human_interesting):
        # :type pdf_human_interesting: Belief
        self._phi["humaninteresting"] = np.asanyarray(pdf_human_interesting.data, dtype=np.float32).reshape(self._space)

    def callback_phi_human_annoying(self, pdf_human_annoying):
        # :type pdf_avoid_collision: Belief
        self._phi["humanannoying"] = np.asanyarray(pdf_human_annoying.data, dtype=np.float32).reshape(self._space)

    def callback_is_robot_ready(self, ready):
        """
        :type ready: Bool
        """
        self._solo_is_ready = ready.data

    def take_off(self, start_at):
        """
        uav rosnode launcher and intention publisher
        """
        rospy.init_node(self._name, log_level=rospy.DEBUG)

        vendor = rospy.get_param('/{}s_vendor'.format(self._name))
        rospy.Subscriber("/" + vendor + "/{}/ready".format(self._name), Bool, callback=self.callback_is_robot_ready)
        rospy.Subscriber("/" + vendor + "/{}/fly_grad".format(self._name), String, callback=self.callback_fly)

        rospy.logdebug("[{}]{} UAV autonomous waiting for {} to be ready".format(dt.datetime.fromtimestamp(
            rospy.Time.now().to_time()).strftime("%H:%M:%S"), self._name, vendor))

        while not self._solo_is_ready:
            rospy.sleep(1)

        rospy.logdebug("[{}]{} {} is ready".format(dt.datetime.fromtimestamp(
                rospy.Time.now().to_time()).strftime("%H:%M:%S"), self._name, vendor ))

        # choice = raw_input("Start autonomous node? yes/no:\n>> ").lower()
        # while not choice == "yes":
        #     choice = raw_input("Start autonomous node? yes/no:\n>> ").lower()
        #     rospy.sleep(10)

        rospy.logdebug("[{}]{} Starting Autonomouse mode......!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(dt.datetime.fromtimestamp(
            rospy.Time.now().to_time()).strftime("%H:%M:%S"), self.name))

        rate = rospy.Rate(1)
        rospy.Subscriber("/{}/{}/pose_euclid".format(vendor, self._name), Pose, callback=self.callback_sensor_pose)

        rospy.Subscriber("/PHI/{}/unexplored".format(self.name), numpy_msg(Belief), callback=self.callback_phi_unexplored)
        rospy.Subscriber("/PHI/{}/avoid_collision".format(self.name), numpy_msg(Belief), callback=self.callback_phi_avoid_collision)
        rospy.Subscriber("/PHI/{}/temp_change".format(self.name), numpy_msg(Belief), callback=self.callback_phi_temp_change)
        rospy.Subscriber("/PHI/{}/humidity_change".format(self.name), numpy_msg(Belief), callback=self.callback_phi_hum_change)

        # if self.name == "B":
        #     rospy.Subscriber("/PHI/{}/human_interesting".format(self.name), numpy_msg(Belief), callback=self.callback_phi_human_interesting)
        #     rospy.Subscriber("/PHI/{}/human_annoying".format(self.name), numpy_msg(Belief), callback=self.callback_phi_human_annoying)

        for from_uav in self.neighbour_names:
            # initialize all other uavs intention uniformly
            init_belief = Belief()
            init_belief.header.frame_id = "{}>{}".format(from_uav, self.name)
            init_belief.header.stamp = rospy.Time.now()
            init_belief.data = np.ones(self._scale ** self._dim, dtype=np.float32).ravel()
            self._msg_received[from_uav] = init_belief

            topic = "/UAV/" + from_uav + "/intention_sent"
            rospy.Subscriber(topic, Belief, callback=self.callback)

        self._pose = Pose(Point(start_at[0], start_at[1], start_at[2]), Quaternion(*quaternion_from_euler(0., 0., np.pi)))
        q_size = 10
        # fix3d
        pub_pose = rospy.Publisher(self.name + '/pose', Pose, queue_size=q_size)
        pub_inbox = rospy.Publisher(self.name + '/intention_received', numpy_msg(Belief), queue_size=q_size)
        pub_self = rospy.Publisher(self.name + '/intention_self', numpy_msg(Belief), queue_size=q_size)
        pub_outbox = rospy.Publisher(self.name + '/intention_sent', numpy_msg(Belief), queue_size=q_size)

        while not rospy.is_shutdown():
            self.sum_product_algo2()

            if self._solo_wants_to_fly:
                self.fly()
                self._solo_wants_to_fly = False

            for to_uav in self.neighbour_names:
                msg = Belief()
                msg.header.frame_id = "{}>{}".format(self.name, to_uav)
                msg.data = self._msg_send[to_uav].ravel()
                msg.header.stamp = rospy.Time.now()
                pub_outbox.publish(msg)

            # ---------------------publishing own belief for visualization----------------------------------------------
            pub_pose.publish(self._pose)
            msg_viz = Belief()
            msg_viz.header.frame_id = self.name
            msg_viz.data = self._intention_fusion.ravel()
            msg_viz.header.stamp = rospy.Time.now()
            pub_self.publish(msg_viz)

            for from_uav in self.neighbour_names:
                msg_viz = self._msg_received[from_uav]
                msg_viz.header.frame_id = from_uav
                pub_inbox.publish(msg_viz)
            # ----------------------------------------------------------------------------------------------------------
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

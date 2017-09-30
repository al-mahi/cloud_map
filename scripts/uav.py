#!/usr/bin/python
import numpy as np
from scipy.stats import multivariate_normal
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from tf.transformations import quaternion_from_euler
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String, Float32
from cloud_map.msg import Belief
import datetime as dt
import time


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
        self._defined_intention_keys = ["boundary", "unexplored", "tempchange", "avoidcollision", "humaninteresting", "humanannoying"]
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

    @property
    def name(self):
        return self._name

    @property
    def pose(self):
        """
        :rtype: Pose
        """
        return self._pose

    @pose.setter
    def pose(self, value):
        self._pose = value

    @property
    def position(self):
        """
        :rtype: np.ndarray
        """
        return np.array(self._pose.position.__getstate__()[:self._dim]) + np.random.uniform(low=-1., high=1.) * 0.00001

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

    def sum_product_algo2(self):
        """
        Algo 2 is taking sum instead of product.
        :param to_uav:
        """
        # fix3d
        # this can be any formula for building own belief from sensor data
        neighbors_belief = np.zeros(shape=self._space)
        for v in self.neighbour_names:
            if not self._msg_received[v].data is None:
                neighbors_belief += np.asanyarray(self._msg_received[v].data, dtype=np.float32).reshape(self._space)

        intention_weight = {
            "unexplored": .2, "tempchange": .7, "avoidcollision": .8, "boundary": 1., "humaninteresting": .8, "humanannoying": 0.9
        }

        if self._dim == 2:
            boundary = np.zeros(shape=(self._scale, self._scale))
            X, Y = np.meshgrid(np.arange(0, self._scale, 1.), np.arange(0, self._scale, 1.))
            # boundary = (X-self._scale/2)**4 + (Y-self._scale/2)**4
            boundary[:, 0] = boundary[0, :] = boundary[:, self._scale-1] = boundary[self._scale-1, :] = 1.
            boundary[0, 0] = boundary[0, self._scale - 1] = boundary[self._scale-1, 0] = boundary[self._scale-1, self._scale-1] = 2.
            boundary /= np.sum(boundary)

        if self._dim == 3:
            # X, Y, Z = np.meshgrid(np.arange(0, self._scale, 1.), np.arange(0, self._scale, 1.), np.arange(0, self._scale, 1.))
            # boundary = (X-self._scale/2)**2. + (Y-self._scale/2)**2. + (Z-self._scale/2)**2.
            boundary = np.zeros(shape=self._space)
            boundary[:, :, 0] = boundary[:, 0, :] = boundary[0, :, :] = boundary[:, :, self._scale-1] = boundary[:, self._scale-1, :] = boundary[self._scale-1, :, :] = 1.
            boundary[0, 0, 0] = boundary[self._scale-1, self._scale-1, self._scale-1] = boundary[0, 0, self._scale-1] = 2.
            boundary[0, self._scale-1, self._scale-1] = boundary[0, self._scale-1, 0] = boundary[self._scale-1, 0, 0] = 2.
            boundary[self._scale-1, 0, self._scale-1] = boundary[self._scale-1, self._scale-1, 0] = 2.
            boundary /= np.sum(boundary)

        self._phi["boundary"] = boundary

        own_belief = np.zeros(shape=self._space)
        for key in self.defined_intention_keys:
            if self._phi.has_key(key):
                own_belief += (intention_weight[key] * self._phi[key])

        normalizer_k = np.sum(neighbors_belief)

        normalizer_k1 = np.sum(own_belief)

        own_belief -= (intention_weight["tempchange"] * self._phi["tempchange"])
        own_belief -= (intention_weight["humanannoying"] * self._phi["humanannoying"])
        own_belief -= (intention_weight["humaninteresting"] * self._phi["humaninteresting"])

        for v in self.neighbour_names:
            if not self._msg_received[v].data is None:
                belief_to_send = neighbors_belief - np.asanyarray(self._msg_received[v].data, dtype=np.float32).reshape(self._space) / normalizer_k

                tmp = belief_to_send + np.nan_to_num(own_belief / normalizer_k1)
                if np.sum(tmp) > 0.0000001:
                    self._msg_send[v] = tmp / np.sum(tmp)
                else:
                    self._msg_send[v] = tmp

        own_belief += (intention_weight["tempchange"] * self._phi["tempchange"])
        own_belief += (intention_weight["humanannoying"] * self._phi["humanannoying"])
        own_belief += (intention_weight["humaninteresting"] * self._phi["humaninteresting"])

        own_belief -= (0.7 * intention_weight["avoidcollision"] * self._phi["avoidcollision"])
        new_intention = own_belief + neighbors_belief
        res = new_intention / np.sum(new_intention)
        self._intention_fusion = np.array(res.reshape(self._space), dtype=np.float32)

    def fly(self):
        #  current position
        if self._dim == 2:
            x, y = map(int, map(round, self.position[:]))
        if self._dim == 3:
            x, y, z = map(int, map(round, self.position[:]))

        if self._dim == 2:
            gradient_at_cur_pos = np.array(np.gradient(1.-self._intention_fusion), dtype=np.float32)[:, x, y]
        if self._dim == 3:
            gradient_at_cur_pos = np.array(np.gradient(1.-self._intention_fusion), dtype=np.float32)[:, x, y, z]

        old_pos = self.position
        p = np.random.rand() / self._dim
        goal = Pose()

        for i in range(self._dim):
            if np.isclose(gradient_at_cur_pos[i], 0., atol=1.e-6) or np.isnan(gradient_at_cur_pos[i]):
                gradient_at_cur_pos[i] = 0.

        k = np.sum(np.abs(gradient_at_cur_pos))
        if not np.isclose(k, 0., atol=1.e-12):
            grad_norm = gradient_at_cur_pos/k
        else:
            grad_norm = np.random.uniform(-1., 1., self._dim)
            rospy.logdebug("[{}]{} at optimum!".format(dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%M:%S.%f"), self.name))

        while True:
            prob = np.abs(grad_norm)
            dx = np.sign(grad_norm) * (prob > p)
            new_pos = old_pos + dx
            valid_move = True
            for i in range(self._dim):
                if round(new_pos[i]) < 0 or round(new_pos[i]) >= self._scale:
                    valid_move = False
                    rospy.logdebug("[{}]{} Pushing out! grad={}".format(dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%M:%S.%f"), self.name, gradient_at_cur_pos))
                    if grad_norm[i] > 0.:
                        grad_norm[i] = 0.
                    if grad_norm[i] < 0.:
                        grad_norm[i] = 0
            if np.sum(np.abs(grad_norm)) > 1.e-8: grad_norm /= np.sum(np.abs(grad_norm))
            if valid_move: break

        self.pose.position.x = new_pos[0]
        self.pose.position.y = new_pos[1]
        if self._dim == 3:
            self.pose.position.z = new_pos[2]
        np.set_printoptions(precision=4)
        rospy.logdebug("[{}]{}:{}->{} grad={} grad_norm={} p={}".format(dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%M:%S.%f"), self.name, map(int, map(round, old_pos[:])), map(int, map(round, self.position[:])), gradient_at_cur_pos, grad_norm, p))
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
        self.pose = pose

    def callback_goal_reached(self, distance):
        """
        :type distance: Float32
        """
        rospy.logdebug("{} distance {}".format(self.name, distance))
        if float(distance.data) < 0.5:
            self.fly()

    def callback_phi_unexplored(self, pdf_unexplored):
        # :type pdf_unexplored: Belief
        self._phi["unexplored"] = np.asanyarray(pdf_unexplored.data, dtype=np.float32).reshape(self._space)

    def callback_phi_temp_change(self, pdf_tempchange):
        # :type pdf_tempchange: Belief
        self._phi["tempchange"] = np.asanyarray(pdf_tempchange.data, dtype=np.float32).reshape(self._space)

    def callback_phi_avoid_collision(self, pdf_avoid_collision):
        # :type pdf_avoid_collision: Belief
        self._phi["avoidcollision"] = np.asanyarray(pdf_avoid_collision.data, dtype=np.float32).reshape(self._space)

    def callback_phi_human_interesting(self, pdf_human_interesting):
        # :type pdf_human_interesting: Belief
        self._phi["humaninteresting"] = np.asanyarray(pdf_human_interesting.data, dtype=np.float32).reshape(self._space)

    def callback_phi_human_annoying(self, pdf_human_annoying):
        # :type pdf_avoid_collision: Belief
        self._phi["humanannoying"] = np.asanyarray(pdf_human_annoying.data, dtype=np.float32).reshape(self._space)

    def take_off(self, start_at):
        """
        uav rosnode launcher and intention publisher
        """

        choice = raw_input("Start autonomous node? Press any letter").lower()
        print("choice = ", choice)
        print("Starting autonomouse mode......")
        rospy.init_node(self._name, log_level=rospy.DEBUG)
        rate = rospy.Rate(2)
        rospy.Subscriber("/solo/{}/pose_euclid".format(self._name), Pose, callback=self.callback_sensor_pose)
        rospy.Subscriber("/solo/{}/distance_from_goal".format(self._name), Float32, callback=self.callback_goal_reached)

        rospy.Subscriber("/PHI/{}/unexplored".format(self.name), numpy_msg(Belief), callback=self.callback_phi_unexplored)
        rospy.Subscriber("/PHI/{}/avoid_collision".format(self.name), numpy_msg(Belief), callback=self.callback_phi_avoid_collision)
        rospy.Subscriber("/PHI/{}/temp_change".format(self.name), numpy_msg(Belief), callback=self.callback_phi_temp_change)

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
            # if self.has_tmp_sensor:
            #     rospy.Publisher(self.name + '/temp', numpy_msg(Belief), queue_size=q_size).publish(#not written yet)
            self.sum_product_algo2()
            for to_uav in self.neighbour_names:
                msg = Belief()
                msg.header.frame_id = "{}>{}".format(self.name, to_uav)
                msg.data = self._msg_send[to_uav].ravel()
                msg.header.stamp = rospy.Time.now()
                pub_outbox.publish(msg)

            # ---------------------publishing own belief for visualization----------------------------------------------
            pub_pose.publish(self.pose)
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
    uav.defined_intention_keys = rospy.get_param("/PHI/" + name + "/intent").split('_')

    neighbors = []
    if rospy.has_param("/UAV/" + name + "/neighbors"):
        neighbors = rospy.get_param("/UAV/" + name + "/neighbors").split("_")
        rospy.logdebug("Launch UAV {} with neighbors {}".format(uav.name, neighbors))
        for n in neighbors:
            # todo validate the format of n
            uav.neighbour_names.append(n)

    print("UAV {}d launcing {} with neighboues {}".format(dim, name, neighbors))
    uav.take_off(start_at=start_at)

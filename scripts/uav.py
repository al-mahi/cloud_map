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
        self._defined_intention_keys = ["unexplored", "tempchange"]
        self._decay_explored = .3
        self._decay_belief = .3
        self._belief = np.ones(shape=self._space) / (scale**self._dim)
        self._intention_unexplored = np.ones(shape=self._space, dtype=np.float32) / (scale**self._dim)  # uniform prior
        self._intention_fusion = np.ones(shape=self._space, dtype=np.float32) / (scale**self._dim)  # uniform prior
        self._neighbors_names = []  # type: [str]
        self._msg_received = {}  # type: dict[str, Belief]
        self._msg_send = {} # type: dict[str, np.ndarray]
        self._phi = {}  # type: dict[str, np.ndarray]
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
        dist = multivariate_normal(mean=self.position, cov=1.*self._scale * np.identity(self._dim))
        indices = np.ndindex(self._space)
        self._belief = np.array(
            map(lambda x: dist.pdf(np.array(x[:])), indices), dtype=np.float32).reshape(self._space)
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

    def sum_product_algo2(self):
        """
        Algo 2 is taking sum instead of product.
        :param to_uav:
        """
        # fix3d
        # this can be any formula for building own belief from sensor data
        new_intention = self.belief_position
        for v in self.neighbour_names:
            if not self._msg_received[v].data is None:
                new_intention += np.asanyarray(self._msg_received[v].data, dtype=np.float32).reshape(self._space)

        for key in self._defined_intention_keys:
            if self._phi.has_key(key):
                new_intention += self._phi[key]

        for v in self.neighbour_names:
            if not self._msg_received[v].data is None:
                tmp = new_intention - np.asanyarray(self._msg_received[v].data, dtype=np.float32).reshape(self._space)
                self._msg_send[v] = tmp / np.sum(tmp)

        new_intention -= self.belief_position
        res = new_intention / np.sum(new_intention)

        if self._dim == 2:
            X, Y = np.meshgrid(np.arange(0, self._scale, 1.), np.arange(0, self._scale, 1.))
            wall = (X-self._scale/2)**4 + (Y-self._scale/2)**4
            wall /= np.sum(wall)
            wall *= 0.05

        if self._dim == 3:
            # X, Y, Z = np.meshgrid(np.arange(0, self._scale, 1.), np.arange(0, self._scale, 1.), np.arange(0, self._scale, 1.))
            # wall = (X-self._scale/2)**4. + (Y-self._scale/2)**4. + (Z-self._scale/2)**4.
            wall = np.zeros(shape=(self._scale, self._scale, self._scale))
            wall[:, :, 0] = wall[:, 0, :] = wall[0, :, :] = wall[:, :, self._scale-1] = wall[:, self._scale-1, :] = wall[self._scale-1, :, :] = 1.
            wall /= np.sum(wall)
            wall *= 0.3

        self._intention_fusion = np.array(res.reshape(self._space), dtype=np.float32) + wall

    def fly(self):
        #  current position
        if self._dim == 2:
            x, y = map(int, self.position[:])
        if self._dim == 3:
            x, y, z = map(int, self.position[:])

        if self._dim == 2:
            gradient_at_cur_pos = np.array(np.gradient(1.-self._intention_fusion), dtype=np.float32)[:, x, y]
        if self._dim == 3:
            gradient_at_cur_pos = np.array(np.gradient(1.-self._intention_fusion), dtype=np.float32)[:, x, y, z]

        k = np.sqrt(np.sum(gradient_at_cur_pos**2.))
        scaled_grad = 1.5 * gradient_at_cur_pos/k

        old_pos = self.position
        goal = Pose()

        if not np.isclose(k, 0., atol=1.e-12):
            if 0. <= self.pose.position.x + scaled_grad[0] < self._scale - 1.:
                goal.position.x = self.pose.position.x + scaled_grad[0]
            if 0. <= self.pose.position.y + scaled_grad[1] < self._scale - 1.:
                goal.position.y = self.pose.position.y + scaled_grad[1]
            if self._dim == 3:
                if 0. <= self.pose.position.z + scaled_grad[2] < self._scale - 1.:
                    goal.position.z = self.pose.position.z + scaled_grad[2]

        # rospy.logdebug("{}:occupancy\n{}".format(self.name, self._intention_fusion))
        rospy.logdebug("[{}]{}:{}->{} grad {}|{} 0?{}".format(dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%M:%S.%f"), self.name, old_pos, np.array([goal.position.x, goal.position.y, goal.position.z]), gradient_at_cur_pos, scaled_grad, np.isclose(k, 0.)))
        self._pub_goal_euclid.publish(goal)

    def callback(self, pdf_intention):
        """
        ros topic listener
        receive message in factor graph
        :param pdf_intention:
        :param from_uav: From which uav the message was sent from
        :type pdf_intention: Belief
        """
        if pdf_intention.header.frame_id[2] == self.name:
            self._msg_received[pdf_intention.header.frame_id[0]] = pdf_intention

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

    def callback_phi_temp_change(self, pdf_unexplored):
        # :type pdf_unexplored: Belief
        self._phi["tempchange"] = np.asanyarray(pdf_unexplored.data, dtype=np.float32).reshape(self._space)

    def take_off(self):
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
        rospy.Subscriber("/PHI/{}/temp_change".format(self.name), numpy_msg(Belief), callback=self.callback_phi_temp_change)
        for from_uav in self.neighbour_names:
            topic = "/UAV/" + from_uav + "/intention_sent"
            rospy.Subscriber(topic, Belief, callback=self.callback)
            # initialize all other uavs intention uniformly
            init_belief = Belief()
            init_belief.header.frame_id = from_uav
            init_belief.header.stamp = rospy.Time.now()
            init_belief.data = np.ones(self._scale ** self._dim, dtype=np.float32).ravel()
            self._msg_received[from_uav] = init_belief

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


def launch_uav(name):
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

    print("UAV_2D launcing {} with neighbors {}".format(name, neighbors))
    uav.take_off()

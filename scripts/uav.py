#!/usr/bin/python
import numpy as np
from scipy.stats import multivariate_normal
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from tf.transformations import quaternion_from_euler
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String
from dummy_cloud_map.msg import Belief
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
        self._defined_intention_keys = ["unexplored"]
        self._decay_explored = .3
        self._decay_belief = .3
        self._belief = np.ones(shape=self._space) / (scale**self._dim)
        self._intention_unexplored = np.ones(shape=self._space, dtype=np.float32) / (scale**self._dim)  # uniform prior
        self._intention_fusion = np.ones(shape=self._space, dtype=np.float32) / (scale**self._dim)  # uniform prior
        self._neighbors_names = []  # type: [str]
        self._msg_received = {}  # type: dict[str, Belief]
        self._msg_send = {} # type: dict[str, np.ndarray]
        self._phi = {}  # type: dict[str, np.ndarray]

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
        dist = multivariate_normal(mean=self.position, cov=.5 * np.identity(self._dim))
        indices = np.ndindex(self._space)
        self._belief = np.array(
            map(lambda x: dist.pdf(np.array(x[:])), indices), dtype=np.float32).reshape(self._space)
        return self._belief

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

        wall = np.zeros(self._space)
        wall[:, 0] = wall[0, :] = wall[:, self._scale-1] = wall[self._scale-1, :] = .005

        self._intention_fusion = np.array(res.reshape(self._space), dtype=np.float32) + wall

    def fly(self):
        #  current position
        if self._dim == 2:
            x, y = map(int, self.position[:])
        if self._dim == 3:
            x, y, z = map(int, self.position[:])

        # global gradient calculation is not feasible in real-time for two reasons
        # 1. it will be very slow
        # 2. numerical un-stability of extreme values
        # confining gradient calculation withing local ranges
        # local range has critical effect
        # local_range = int(self._scale/3)
        # # slice local dist. calc indices; prefix s- starting, e- ending
        # # min and max of local dist should be calculated excluding the wall
        # sx = max(0, (x + 1) - local_range)
        # sy = max(0, (y + 1) - local_range)
        # ex = min(self._scale - 1, x + local_range)
        # ey = min(self._scale - 1, y + local_range)
        # local_dist = self._intention_fusion[sx:ex, sy:ey]
        # # translate cur position in the sliced frame of ref
        # nx, ny = x-sx-1, y-sy-1
        # # where to fly? gradient descent thus calculate gradient on the inverse function here complement prob.
        # local_gradient = np.array(np.gradient(1.-local_dist), dtype=np.float32)
        # gradient_at_cur_pos = local_gradient[:, nx, ny]

        # rospy.logdebug("fusion`\n{}\ngradient\n{}".format(1.-self._intention_fusion, np.gradient(1.-self._intention_fusion)))
        if self._dim == 2:
            gradient_at_cur_pos = np.array(np.gradient(1.-self._intention_fusion), dtype=np.float32)[:, x, y]
        if self._dim == 3:
            gradient_at_cur_pos = np.array(np.gradient(1.-self._intention_fusion), dtype=np.float32)[:, x, y, z]

        k = np.linalg.norm(gradient_at_cur_pos.ravel(), ord=np.inf)
        scaled_grad = gradient_at_cur_pos/k

        old_pos = self.position
        if not np.isclose(k, 0.):
            threshold = 0.3
            if scaled_grad[0] > (threshold) and self.pose.position.x + 1 < self._scale: self.pose.position.x += 1.
            if scaled_grad[1] > (threshold) and self.pose.position.y + 1 < self._scale: self.pose.position.y += 1.
            if scaled_grad[0] < (-1.*threshold) and self.pose.position.x > 0: self.pose.position.x -= 1.
            if scaled_grad[1] < (-1.*threshold) and self.pose.position.y > 0: self.pose.position.y -= 1.
            if self._dim == 3:
                if scaled_grad[2] > (threshold) and self.pose.position.z + 1 < self._scale: self.pose.position.z += 1.
                if scaled_grad[2] < (-1.*threshold) and self.pose.position.z > 0: self.pose.position.z -= 1.
        else:
            dx = np.random.randint(low=-1, high=2, size=self._dim)
            if 0 <= self.pose.position.x + dx[0] < self._scale: self.pose.position.x += dx[0]
            if 0 <= self.pose.position.y + dx[1] < self._scale: self.pose.position.y += dx[1]
            if self._dim == 3:
                if 0 <= self.pose.position.z + dx[2] < self._scale: self.pose.position.z += dx[2]
        rospy.logdebug("[{}]{}:{}->{} grad {}|{} k>>0{}".format(dt.datetime.fromtimestamp(rospy.Time.now().to_time()).strftime("%M:%S.%f"), self.name, old_pos, self.position, gradient_at_cur_pos, scaled_grad, np.isclose(k, 0.)))

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

    def callback_fly(self, msg_fly):
        """
        :type msg_fly: String
        """
        if msg_fly.data == "fly":
            self.fly()

    def callback_phi_unexplored(self, pdf_unexplored):
        # :type pdf_unexplored: Belief
        self._phi["unexplored"] = np.asanyarray(pdf_unexplored.data, dtype=np.float32).reshape(self._space)

    def take_off(self):
        """
        uav rosnode launcher and intention publisher
        """
        rospy.init_node(self._name, log_level=rospy.DEBUG)
        rate = rospy.Rate(4)
        rospy.Subscriber("/UAV_FW/fly", String, callback=self.callback_fly)
        rospy.Subscriber("/PHI/{}/unexplored".format(self.name), numpy_msg(Belief), callback=self.callback_phi_unexplored)
        for from_uav in self.neighbour_names:
            topic = "/UAV_FW/" + from_uav + "/intention_sent"
            rospy.Subscriber(topic, Belief, callback=self.callback)
            # initialize all other uavs intention uniformly
            init_belief = Belief()
            init_belief.header.frame_id = from_uav
            init_belief.header.stamp = rospy.Time.now()
            init_belief.data = np.ones(self._scale ** self._dim, dtype=np.float32).ravel()
            self._msg_received[from_uav] = init_belief
        self._pose = Pose(Point(1. * np.random.randint(0, self._scale), 1. * np.random.randint(0, self._scale), 0.),
                          Quaternion(*quaternion_from_euler(0., 0., np.pi)))

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

    if rospy.has_param("/UAV_FW/" + name + "/neighbors"):
        neighbours = rospy.get_param("/UAV_FW/" + name + "/neighbors").split("_")
        rospy.logdebug("Launch UAV {} with neighbors {}".format(uav.name, neighbours))
        for n in neighbours:
            # todo validate the format of n
            uav.neighbour_names.append(n)

    print("UAV_2D launcing {} with neighboues {}".format(name, neighbours))
    uav.take_off()

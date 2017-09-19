#!/usr/bin/python
import numpy as np
from scipy.stats import multivariate_normal
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from tf.transformations import quaternion_from_euler
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String, Float32
from cloud_map.msg import Belief

"""
This the module where we define different intention as probability distribution over the space.
"""
class Unexplored(object):
    def __init__(self, name, dim, scale, q_size=1500):
        self._name = name
        self._dim = dim
        self._scale = scale
        self._space = tuple([scale for _ in range(dim)])
        self._dim = len(self._space)
        self._q_size = q_size
        self._q_pose = []  # type: list[np.ndarray]
        self._phi_unexplored = np.zeros(self._space)
        self._contour_scale = .8 # np.array([.5, .5, 1.])
        self._phi_unexplored_norm = np.zeros(self._space)

    def callback_pose(self, pose):
        """
        :type pose: Pose 
        :return: 
        """
        # dist = multivariate_normal(mean=np.array(pose.position.__getstate__()[:self._dim], dtype='float'),
        #                            cov=(self._scale * self._contour_scale) * np.identity(self._dim))
        # indices = np.ndindex(self._space)
        # last = np.array(map(lambda x: dist.pdf(np.array(x[:])), indices), dtype=np.float32).reshape(self._space)
        # self._q_pose.append(last)
        # self._q_pose = self._q_pose[-self._q_size:]
        # self._phi_unexplored = np.zeros(self._space)
        # decay = np.linspace(start=0.8, stop=1., num=len(self._q_pose))
        # for i in range(len(self._q_pose)):
        #     self._phi_unexplored += self._q_pose[i] * decay[i]
        x, y, z = map(int, map(round, pose.position.__getstate__()[:]))
        if self._dim == 2: self._phi_unexplored[x, y] += 1.0
        if self._dim == 3: self._phi_unexplored[x, y, z] += 1.0
        self._phi_unexplored_norm = self._phi_unexplored / np.sum(self._phi_unexplored)

    def start(self):
        rospy.init_node(self._name, log_level=rospy.DEBUG)
        rate = rospy.Rate(4)
        rospy.Subscriber("/UAV/" + self._name + "/pose", Pose, callback=self.callback_pose)
        pub_unexplored = rospy.Publisher(self._name + '/unexplored', numpy_msg(Belief), queue_size=10.)
        while not rospy.is_shutdown():
            msg = Belief()
            msg.header.frame_id = self._name
            msg.header.stamp = rospy.Time.now()
            msg.data = self._phi_unexplored_norm.ravel()
            pub_unexplored.publish(msg)
            rate.sleep()


def phi_unexplored_node(name):
    dim = int(rospy.get_param("/dim"))
    scale = int(rospy.get_param("/scale"))
    if rospy.has_param("/PHI/" + name + "/intent"):
        intents = rospy.get_param("/PHI/" + name + "/intent") # type: str
        if "unexplored" in intents.split('_'):
            phi_unexplored = Unexplored(name=name, dim=dim, scale=scale)
            phi_unexplored.start()


class TemperatureChange(object):
    def __init__(self, name, dim, scale, q_size=60):
        self._name = name
        self._dim = dim
        self._scale = scale
        self._space = tuple([scale for _ in range(dim)])
        self._dim = len(self._space)
        self._measured_temp = np.nan * np.ones(self._space)
        self._phi_temp_change = np.zeros(self._space)
        self._pose = Pose()
        self._tag = "[Temp {}]".format(self._name)

        self._true_tmp = np.zeros(self._space)
        self.ground_tmp = 90.  # F
        self.above_tmp = 70.  # F
        self.del_tmp = (self.ground_tmp - self.above_tmp)/self._scale
        if self._dim == 3:
            for i in range(int(scale)):
                self._true_tmp[:, :, i] = self.ground_tmp - i * self.del_tmp
            self._true_tmp[:, :, int((self._scale/2)-1):int((self._scale/2)+1)] -= 4.
            self._true_tmp[:, :, int((self._scale/2))] -= 8.

        if self._dim == 2:
            for i in range(int(scale)):
                self._true_tmp[:, i] = self.ground_tmp - i * self.del_tmp
            self._true_tmp[:, 10:14] -= 2.
            self._true_tmp[:, 11:12] -= 4.
        self._measured_temp = self._true_tmp

    def callback_sensor_pose_euclid(self, pose):
        self._pose = pose

    def callback_temp(self, temp):
        """
        :type temp: Float32
        :return:
        """
        # temp = float(temp.data)
        # x, y, z = map(int, np.array(self._pose.position.__getstate__())[:])
        # if self._dim == 3:
        #     self._measured_temp[x, y, z] = temp
        # if self._dim == 2:
        #     self._measured_temp[x, y] = temp
        # calc temp change
        grad = np.gradient(self._measured_temp)
        if self._dim == 3: self._phi_temp_change = np.sqrt(grad[0]**2 + grad[1]**2 + grad[2]**2)
        if self._dim == 2: self._phi_temp_change = np.sqrt(grad[0]**2 + grad[1]**2)
        # self._phi_temp_change = np.nan_to_num(self._phi_temp_change)
        self._phi_temp_change /= np.sum(self._phi_temp_change)
        self._phi_temp_change = 1. - self._phi_temp_change

    def start(self):
        rospy.init_node(self._name, log_level=rospy.DEBUG)
        rate = rospy.Rate(2)
        rospy.Subscriber("/Sensors/" + self._name + "/mock_temperature", Float32, callback=self.callback_temp)
        rospy.Subscriber("/UAV/{}/pose".format(self._name), Pose, callback=self.callback_sensor_pose_euclid)
        pub_temp_change = rospy.Publisher("/PHI/{}/temp_change".format(self._name), numpy_msg(Belief), queue_size=10.)

        while not rospy.is_shutdown():
            msg = Belief()
            msg.header.frame_id = self._name
            msg.header.stamp = rospy.Time.now()
            msg.data = self._phi_temp_change.ravel()
            pub_temp_change.publish(msg)
            rate.sleep()


def phi_mock_temperature_node(name):
    dim = int(rospy.get_param("/dim"))
    scale = int(rospy.get_param("/scale"))
    if rospy.has_param("/PHI/" + name + "/intent"):
        intents = rospy.get_param("/PHI/" + name + "/intent") # type: str
        if "tempchange" in intents.split('_'):
            phi_temp = TemperatureChange(name=name, dim=dim, scale=scale)
            phi_temp.start()

class AvoidCollision(object):
    def __init__(self, name, dim, scale, q_size=60):
        self._name = name
        self._dim = dim
        self._scale = scale
        self._space = tuple([scale for _ in range(dim)])
        self._dim = len(self._space)
        self._phi_avoid_collision = np.zeros(self._space)
        self._pose = Pose()
        self._tag = "[Avoid {}]".format(self._name)

    def callback_sensor_pose_euclid(self, pose):
        self._pose = pose
        dist = multivariate_normal(mean=np.array(pose.position.__getstate__()[:self._dim], dtype='float'),
                                   cov= 0.2 * self._scale * np.identity(self._dim))
        indices = np.ndindex(self._space)
        self._phi_avoid_collision = np.array(map(lambda x: dist.pdf(np.array(x[:])), indices), dtype=np.float32).reshape(self._space)

    def start(self):
        rospy.init_node(self._name, log_level=rospy.DEBUG)
        rate = rospy.Rate(2)
        rospy.Subscriber("/UAV/{}/pose".format(self._name), Pose, callback=self.callback_sensor_pose_euclid)
        pub_avoid_collision = rospy.Publisher("/PHI/{}/avoid_collision".format(self._name), numpy_msg(Belief), queue_size=10.)

        while not rospy.is_shutdown():
            msg = Belief()
            msg.header.frame_id = self._name
            msg.header.stamp = rospy.Time.now()
            msg.data = self._phi_avoid_collision.ravel()
            pub_avoid_collision.publish(msg)
            rate.sleep()


def phi_avoid_collision_node(name):
    dim = int(rospy.get_param("/dim"))
    scale = int(rospy.get_param("/scale"))
    if rospy.has_param("/PHI/" + name + "/intent"):
        intents = rospy.get_param("/PHI/" + name + "/intent") # type: str
        if "avoidcollision" in intents.split('_'):
            phi_avoid_collision = AvoidCollision(name=name, dim=dim, scale=scale)
            phi_avoid_collision.start()


class HumanIntention(object):
    def __init__(self, name, dim, scale, q_size=60):
        self._name = name
        self._dim = dim
        self._scale = scale
        self._space = tuple([scale for _ in range(dim)])
        self._dim = len(self._space)
        self._pose = Pose()
        self.phi_human_annoying = np.zeros(self._space)
        self.phi_human_interesting = np.zeros(self._space)
        self._list_interesting = []  # type: [Pose]
        self._list_annoying = []  # type: [Pose]
        self._tag = "[HumanIntent {}]".format(self._name)

    def calc_interesting(self):
        # recalculate from other interesting goals
        tmp = np.zeros(self._space)
        for pose in self._list_interesting:
            indices = np.ndindex(self._space)
            dist = multivariate_normal(mean=np.array(pose.position.__getstate__()[:self._dim], dtype='float'),
                                       cov= 0.25 * self._scale * np.identity(self._dim))
            tmp += np.array(map(lambda x: dist.pdf(np.array(x[:])), indices), dtype=np.float32).reshape(self._space)
        if np.sum(tmp) > 0.:
            tmp /= np.sum(tmp)
            tmp = 1. - tmp
            # tmp /= np.sum(tmp)
            self.phi_human_interesting = tmp

    def callback_human_interesting(self, pose_interest):
        self._list_interesting.append(pose_interest)
        self.calc_interesting()

    def callback_human_annoying(self, pose_interest):
        self._list_annoying.append(pose_interest)
        tmp = np.zeros(self._space)
        for pose in self._list_annoying:
            indices = np.ndindex(self._space)
            dist = multivariate_normal(mean=np.array(pose.position.__getstate__()[:self._dim], dtype='float'),
                                       cov= 0.1 * self._scale * np.identity(self._dim))
            tmp += np.array(map(lambda x: dist.pdf(np.array(x[:])), indices), dtype=np.float32).reshape(self._space)
        if np.sum(tmp) > 0.:
            tmp /= np.sum(tmp)
            self.phi_human_annoying = tmp



    def callback_sensor_pose_euclid(self, pose):
        self._pose = pose
        for pose_interesting in self._list_interesting:
            goal = np.array(map(int, map(round, pose_interesting.position.__getstate__()[:self._dim])))
            current_pos = np.array(map(int, map(round, pose.position.__getstate__()[:self._dim])))
            dx = goal - current_pos

            print(current_pos, goal, np.sum(dx**2.))
            if np.isclose(np.sum(dx**2.), 0.):
                self._list_interesting.remove(pose_interesting)
                self.phi_human_interesting = np.zeros(self._space)
                # recalculate from other interesting goals
                self.calc_interesting()

    def start(self):
        rospy.init_node(self._name, log_level=rospy.DEBUG)
        rate = rospy.Rate(2)

        rospy.Subscriber("/UAV/{}/pose".format(self._name), Pose, callback=self.callback_sensor_pose_euclid)
        rospy.Subscriber("/Human/{}/interesting".format(self._name), Pose, callback=self.callback_human_interesting)
        rospy.Subscriber("/Human/{}/annoying".format(self._name), Pose, callback=self.callback_human_annoying)

        pub_human_annoying = rospy.Publisher("/PHI/{}/human_annoying".format(self._name), numpy_msg(Belief), queue_size=10.)
        pub_human_interesting = rospy.Publisher("/PHI/{}/human_interesting".format(self._name), numpy_msg(Belief), queue_size=10.)


        while not rospy.is_shutdown():
            msg1 = Belief()
            msg1.header.frame_id = self._name
            msg1.header.stamp = rospy.Time.now()
            msg1.data = self.phi_human_annoying.ravel()
            pub_human_annoying.publish(msg1)

            msg2 = Belief()
            msg2.header.frame_id = self._name
            msg2.header.stamp = rospy.Time.now()
            msg2.data = self.phi_human_interesting.ravel()
            pub_human_interesting.publish(msg2)

            rate.sleep()


def phi_human_intention_node(name):
    dim = int(rospy.get_param("/dim"))
    scale = int(rospy.get_param("/scale"))
    if rospy.has_param("/PHI/" + name + "/intent"):
        intents = rospy.get_param("/PHI/" + name + "/intent") # type: str
        print("human intention node ", intents)
        if "humaninteresting" in intents.split('_'):
            phi_human_intent = HumanIntention(name=name, dim=dim, scale=scale)
            phi_human_intent.start()


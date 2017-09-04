#!/usr/bin/python
import numpy as np
from scipy.stats import multivariate_normal
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from tf.transformations import quaternion_from_euler
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String, Float32
from cloud_map.msg import Belief


class Unexplored(object):
    def __init__(self, name, dim, scale, q_size=120):
        self._name = name
        self._dim = dim
        self._scale = scale
        self._space = tuple([scale for _ in range(dim)])
        self._dim = len(self._space)
        self._q_size = q_size
        self._q_pose = []  # type: list[np.ndarray]
        self._phi_unexplored = np.zeros(self._space)
        self._contour_scale = .3

    def callback_pose(self, pose):
        """
        :type pose: Pose 
        :return: 
        """
        dist = multivariate_normal(mean=np.array(pose.position.__getstate__()[:self._dim], dtype='int'),
                                   cov=(self._scale * self._contour_scale) * np.identity(self._dim))
        indices = np.ndindex(self._space)
        last = np.array(
            map(lambda x: dist.pdf(np.array(x[:])), indices), dtype=np.float32).reshape(self._space)

        self._q_pose.append(last)
        self._q_pose = self._q_pose[-self._q_size:]
        self._phi_unexplored = np.zeros(self._space)
        decay = np.linspace(start=0.6, stop=1., num=len(self._q_pose))
        for i in range(len(self._q_pose)):
            self._phi_unexplored += self._q_pose[i] * decay[i]
        self._phi_unexplored /= np.sum(self._phi_unexplored)

    def start(self):
        rospy.init_node(self._name, log_level=rospy.DEBUG)
        rate = rospy.Rate(4)
        rospy.Subscriber("/UAV/" + self._name + "/pose", Pose, callback=self.callback_pose)
        pub_unexplored = rospy.Publisher(self._name + '/unexplored', numpy_msg(Belief), queue_size=10.)
        while not rospy.is_shutdown():
            msg = Belief()
            msg.header.frame_id = self._name
            msg.header.stamp = rospy.Time.now()
            msg.data = self._phi_unexplored.ravel()
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


class MockTemperature(object):
    def __init__(self, name, dim, scale, q_size=60):
        self._name = name
        self._dim = dim
        self._scale = scale
        self._space = tuple([scale for _ in range(dim)])
        self._dim = len(self._space)
        self._measured_temp = 50. * np.zeros(self._space)
        self._phi_temp_change = np.zeros(self._space)
        self._pose = Pose()
        self._tag = "[Temp {}]".format(self._name)

    def callback_sensor_pose_euclid(self, pose): self._pose = pose

    def callback_mock_temp(self, temp):
        """
        :type temp: Float32
        :return:
        """
        x, y, z = map(int, np.array(self._pose.position.__getstate__())[:])
        self._measured_temp[x, y, z] = temp
        # calc temp change
        self._phi_temp_change = np.abs(self._measured_temp - temp)
        self._phi_temp_change /= np.sum(self._phi_temp_change)
        self._phi_temp_change = 1. - self._phi_temp_change
        rospy.logdebug("{} is {}".format(self._tag, temp))

    def start(self):
        rospy.init_node(self._name, log_level=rospy.DEBUG)
        rate = rospy.Rate(4)
        rospy.Subscriber("/Sensors/" + self._name + "/mock_temperature", Float32, callback=self.callback_mock_temp)
        rospy.Subscriber("/solo/{}/pose_euclid".format(self._name), Pose, callback=self.callback_sensor_pose_eulid)

        pub_temp_change = rospy.Publisher(self._name + '/temp_change', numpy_msg(Belief), queue_size=10.)
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
        intents = rospy.get_param("/PHI/" + name + "/intent")  # type: str
        if "mocktemperature" in intents.split('_'):
            phi_mock_temp = MockTemperature(name=name, dim=dim, scale=scale)
            phi_mock_temp.start()



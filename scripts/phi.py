#!/usr/bin/python
import numpy as np
from scipy.stats import multivariate_normal
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from tf.transformations import quaternion_from_euler
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String
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

    def callback_pose(self, pose):
        """
        :type pose: Pose 
        :return: 
        """
        dist = multivariate_normal(mean=np.array(pose.position.__getstate__()[:self._dim], dtype='int'),
                                   cov=self._scale * np.identity(self._dim))
        indices = np.ndindex(self._space)
        last = np.array(
            map(lambda x: dist.pdf(np.array(x[:])), indices), dtype=np.float32).reshape(self._space)

        self._q_pose.append(last)
        self._q_pose = self._q_pose[-self._q_size:]
        self._phi_unexplored = np.zeros(self._space)
        for p in self._q_pose:
                self._phi_unexplored += p
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



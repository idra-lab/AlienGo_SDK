import rospy
import numpy as np
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped

class PubSub():
    def __init__(self):
        self.imu_acc = np.zeros(3)
        self.imu_quat = np.zeros(4)
        self.imu_gyro = np.zeros(3)
        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.pose = np.zeros(7)
        self.twist = np.zeros(3)

    def callback_imu(self, data):
        self.imu_acc[0] = data
        self.imu_acc[1] = data
        self.imu_acc[2] = data

        self.imu_gyro[0] = data
        self.imu_gyro[1] = data
        self.imu_gyro[2] = data

        self.imu_quat[0] = data
        self.imu_quat[1] = data
        self.imu_quat[2] = data
        self.imu_quat[3] = data

    def callback_joint(self, data):
        for i in range(12):
            self.joint_pos[i] = data
            self.joint_vel[i] = data

    def callback_pose(self, data):
        for i in range(7):
            self.pose[i] = data

    def callback_twist(self, data):
        for i in range(3):
            self.twist[i] = data

    def init_subscribers(self, config_topics):
        rospy.Subscriber(config_topics['imu'], Imu, self.callback_imu)
        rospy.Subscriber(config_topics['joint_state'], JointState, self.callback_joint)
        rospy.Subscriber(config_topics['pose'], PoseWithCovarianceStamped, self.callback_pose)
        rospy.Subscriber(config_topics['twist'], TwistWithCovarianceStamped, self.callback_twist)

    def publish(self, qDes):
        try:
            pub = rospy.Publisher('/joint_state', JointState, queue_size=10)
            rospy.init_node('vel', anonymous=True)
            joint_pub = JointState()
            joint_pub.position = qDes
            pub.publish(joint_pub)
        except rospy.ROSInterruptException:
            pass
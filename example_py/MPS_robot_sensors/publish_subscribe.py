import rospy
import numpy as np
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry

class PubSub():
    def __init__(self):
        self.imu_acc = np.zeros(3)
        self.imu_quat = np.zeros(4)
        self.imu_gyro = np.zeros(3)
        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.pose = np.zeros(7)
        self.twist = np.zeros(6)


    def callback_imu(self, msg):
        # Timestamp
        self.imu_time = msg.header.stamp.to_sec()

        # Orientation quaternion [x, y, z, w]
        self.imu_quat  = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ], dtype=np.float64)

        # Angular velocity [x, y, z]
        self.imu_gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ], dtype=np.float64)

        # Linear acceleration [x, y, z]
        self.imu_acc = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ], dtype=np.float64)

    def odom_callback(self, msg):
        # Timestamp
        self.odom_time = msg.header.stamp.to_sec()

        # Position [x, y, z]
        self.pose = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
        ], dtype=np.float64)

        # Linear velocity [x, y, z]
        self.twist = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ], dtype=np.float64)

    def callback_joint(self, data):
        # Locosim interface provides:
        # LF hip, LF thigh, LF calf
        # LH hip, LH thigh, LH calf
        # RF hip, RF thigh, RF calf
        # RH hip, RH thigh, RH calf

        # Nominal neural network needs:
        # LF hip, RF hip, LH hip, RH hip
        # LF thigh, RF thigh, LH thigh, RH thigh
        # LF calf, RF calf, LH calf, RH calf

        # Backup neural network needs:
        # LF hip, LF thigh, LF calf
        # RF hip, RF thigh, RF calf
        # LH hip, LH thigh, LH calf
        # RH hip, RH thigh, RH calf

        indices = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        for i in range(12):
            self.joint_pos[i] = data.position[indices[i]]
            self.joint_vel[i] = data.velocity[indices[i]]


    def init_subscribers(self, config_topics):
        self.imu_sub = rospy.Subscriber(config_topics['imu'], Imu, self.callback_imu)
        self.joint_state_sub = rospy.Subscriber(config_topics['joint_states'], JointState, self.callback_joint)
        self.odom_sub = rospy.Subscriber(config_topics['odometry'], Odometry, self.odom_callback)
        #self.imu_sub = rospy.Subscriber(config_topics['twist'], TwistWithCovarianceStamped, self.callback_twist)
        #self.imu_sub = rospy.Subscriber(config_topics['pose'], PoseWithCovarianceStamped, self.callback_pose)


    def publish(self, qDes):
        try:
            pub = rospy.Publisher('/command', JointState, queue_size=10)
            rospy.init_node('vel', anonymous=True)
            joint_pub = JointState()
            joint_pub.position = qDes
            pub.publish(joint_pub)
        except rospy.ROSInterruptException:
            pass
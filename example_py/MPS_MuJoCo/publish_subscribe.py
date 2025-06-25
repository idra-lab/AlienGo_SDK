import rospy
import numpy as np
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from unitree_legged_msgs.msg import JointStateWithGains

class PubSub():
    def __init__(self):
        
        self.imu_acc = np.zeros(3)
        self.imu_quat = np.zeros(4)
        self.imu_gyro = np.zeros(3)
        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.pose = np.zeros(7)
        self.twist = np.zeros(6)
        self.joint_state_des = JointStateWithGains()

        self.joint_state_des.cmd.name = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                     "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",                                     
                                     "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", 
                                     "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]
        self.value_function_res = Float64()

    def callback_imu(self, msg):
        # Timestamp
        self.imu_time = msg.header.stamp.to_sec()

        # Orientation quaternion [x, y, z, w]
        self.imu_quat  = np.array([
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            
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
      # Data is received Ordered as in HyQ/ANYmal convention LF RF LH RH
      # we convert into Unitree convention RF LF RH LH
        unitree_ids = [3,4,5,0,1,2,9,10,11,6,7,8]
    
        for i in range(12):
            self.joint_pos[unitree_ids[i]] = data.position[i]
            self.joint_vel[unitree_ids[i]] = data.velocity[i]

    def init_subscribers(self, config_topics):
        self.imu_sub = rospy.Subscriber(config_topics['imu'], Imu, self.callback_imu)
        self.joint_state_sub = rospy.Subscriber(config_topics['joint_states'], JointState, self.callback_joint)
        self.odom_sub = rospy.Subscriber(config_topics['odometry'], Odometry, self.odom_callback)
        
    def init_publisher(self, config_topics):
        self.cmd_pub = rospy.Publisher(config_topics['command'], JointStateWithGains, queue_size=10)
        self.value_function = rospy.Publisher(config_topics['value_function'], Float64, queue_size=10)

    def publish(self, qpos, qvel, eff, Kp, Kd, V_safe):
        try:
            self.joint_state_des.cmd.position = qpos  # store Kp on the fake joint pos
            self.joint_state_des.cmd.velocity = qvel # store Kd on the fake joint vel                        
            self.joint_state_des.cmd.effort = eff
            self.joint_state_des.Kp = 12 * [Kp] # same gains for all joints
            self.joint_state_des.Kd = 12 * [Kd] # same gains for all joints
            self.cmd_pub.publish(self.joint_state_des)
            
            self.value_function_res = V_safe
            self.value_function.publish(self.value_function_res)
        except rospy.ROSInterruptException:
            pass
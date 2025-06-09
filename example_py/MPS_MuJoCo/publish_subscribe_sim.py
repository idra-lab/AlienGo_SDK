import rospy
import numpy as np
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry

class PubSub():
    def __init__(self):
        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.joint_eff = np.zeros(12)

    def callback_command(self, data):
        # Ordered as in MuJoCo
        for i in range(12):
            self.joint_pos[i] = data.position[i]
            self.joint_vel[i] = data.velocity[i]
            self.joint_eff[i] = data.effort[i]


    def init_subscribers(self):
        self.joint_state_sub = rospy.Subscriber('/command', JointState, self.callback_command)

    def init_publishers(self):
        self.imu_data_pub = rospy.Publisher('/aliengo/trunk_imu', Imu, queue_size=10)
        self.joint_state_pub = rospy.Publisher('/aliengo/joint_states', JointState, queue_size=10)
        self.odom_data_pub = rospy.Publisher('/aliengo/ground_truth', Odometry, queue_size=10)
        rospy.init_node('vel', anonymous=True)
        self.joint_pub = JointState()
        self.imu_pub = Imu()
        self.odom_pub = Odometry()
        
    def publisher(self, qpos, qvel, imu_gyro, imu_quat, imu_acc, pose, twist):
        try:         
            self.joint_pub.position = qpos
            self.joint_pub.velocity = qvel
            #print('imu_gyro[0]',imu_gyro[0])
            #print('imu_gyro',imu_gyro)
            self.imu_pub.angular_velocity.x = imu_gyro[0]
            self.imu_pub.angular_velocity.y = imu_gyro[1]
            self.imu_pub.angular_velocity.z = imu_gyro[2]

            self.imu_pub.orientation.w = imu_quat[0]
            self.imu_pub.orientation.x = imu_quat[1]
            self.imu_pub.orientation.y = imu_quat[2]
            self.imu_pub.orientation.z = imu_quat[3]

            self.imu_pub.linear_acceleration.x = imu_acc[0]
            self.imu_pub.linear_acceleration.y = imu_acc[1]
            self.imu_pub.linear_acceleration.z = imu_acc[2]
            
            self.odom_pub.pose.pose.position.x = pose[0]
            self.odom_pub.pose.pose.position.y = pose[1]
            self.odom_pub.pose.pose.position.z = pose[2]
            self.odom_pub.pose.pose.orientation.w = pose[3]
            self.odom_pub.pose.pose.orientation.x = pose[4]
            self.odom_pub.pose.pose.orientation.y = pose[5]
            self.odom_pub.pose.pose.orientation.z = pose[6]

            self.odom_pub.twist.twist.linear.x = twist[0]
            self.odom_pub.twist.twist.linear.y = twist[1]
            self.odom_pub.twist.twist.linear.z = twist[2]
            self.odom_pub.twist.twist.angular.x = twist[3]
            self.odom_pub.twist.twist.angular.y = twist[4]
            self.odom_pub.twist.twist.angular.z = twist[5]

            self.joint_state_pub.publish(self.joint_pub)
            self.imu_data_pub.publish(self.imu_pub)
            self.odom_data_pub.publish(self.odom_pub)
            
        except rospy.ROSInterruptException:
            pass
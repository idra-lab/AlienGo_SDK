import rospy
import numpy as np
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
import threading

class PubSub():
    def __init__(self):
        
        self.imu_acc = np.zeros(3)
        self.imu_quat = np.zeros(4)
        self.imu_gyro = np.zeros(3)
        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.pose = np.zeros(7)
        self.twist = np.zeros(6)
        self.joint_pub = JointState()
        self.condition = threading.Condition()
        self.imu_received = False
        self.joint_state_received = False
        self.odom_received = False


    def callback_imu(self, msg):
        print('Received IMU')
        with self.condition:
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
          # Let everybody know that the IMU message has been received
          self.imu_received = True
          self.condition.notify_all()


    def odom_callback(self, msg):
        print('Received ODOM')
        with self.condition:
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
        # Let everybody know that the Odometry message has been received
        
          self.odom_received = True
          self.condition.notify_all()

    def callback_joint(self, data):
      print('Received joint')
        # Ordered as in MuJoCo
      with self.condition:
        for i in range(12):
            self.joint_pos[i] = data.position[i]
            self.joint_vel[i] = data.velocity[i]
        # Let everybody know that the Odometry message has been received
        self.joint_state_received = True
        self.condition.notify_all()

    def init_subscribers(self, config_topics):
        self.imu_sub = rospy.Subscriber(config_topics['imu'], Imu, self.callback_imu)
        self.joint_state_sub = rospy.Subscriber(config_topics['joint_states'], JointState, self.callback_joint)
        self.odom_sub = rospy.Subscriber(config_topics['odometry'], Odometry, self.odom_callback)
        #self.imu_sub = rospy.Subscriber(config_topics['twist'], TwistWithCovarianceStamped, self.callback_twist)
        #self.imu_sub = rospy.Subscriber(config_topics['pose'], PoseWithCovarianceStamped, self.callback_pose)

    def wait_for_all_messages(self):
        with self.condition:
            # Wait until all three messages are received
            while not (self.imu_received and self.odom_received and self.joint_state_received):
                self.condition.wait()

            # Copy messages to return safely
            msgs = (self.imu_acc, self.imu_quat, self.imu_gyro, self.joint_pos, self.joint_vel, self.pose, self.twist)

            # Reset flags for next round
            self.imu_received = False
            self.odom_received = False
            self.joint_state_received = False

            return msgs
        
    def init_publisher(self):
        self.cmd_pub = rospy.Publisher('/command', JointState, queue_size=10)

    def publish(self, qpos, qvel, eff):
        try:
            #rospy.init_node('communicate_aliengo', anonymous=True)
            self.joint_pub.position = qpos
            self.joint_pub.velocity = qvel
            self.joint_pub.effort = eff
            self.cmd_pub.publish(self.joint_pub)
            print('Published control')
        except rospy.ROSInterruptException:
            pass
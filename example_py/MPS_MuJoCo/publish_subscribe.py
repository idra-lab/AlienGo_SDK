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
        self.joint_state_des = JointState()

        self.joint_state_des.name = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                     "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",                                     
                                     "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", 
                                     "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                                     "gains"] # Fake 13th joint to store gains

        self.condition = threading.Condition()
        self.imu_received = False
        self.joint_state_received = False
        self.odom_received = False


    def callback_imu(self, msg):
      #  print('received IMU')
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
      #  print('received odom')
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
      # print('received joint')
      # Data is received Ordered as in HyQ/ANYmal convention LF RF LH RH
      # we convert into Unitree convention RF LF RH LH
      unitree_ids = [3,4,5,0,1,2,9,10,11,6,7,8]
      with self.condition:
        for i in range(12):
            self.joint_pos[unitree_ids[i]] = data.position[i]
            self.joint_vel[unitree_ids[i]] = data.velocity[i]
        # Let everybody know that the Odometry message has been received
        
        self.joint_state_received = True
        self.condition.notify_all()

    def init_subscribers(self, config_topics):
        self.imu_sub = rospy.Subscriber(config_topics['imu'], Imu, self.callback_imu)
        self.joint_state_sub = rospy.Subscriber(config_topics['joint_states'], JointState, self.callback_joint)
        self.odom_sub = rospy.Subscriber(config_topics['odometry'], Odometry, self.odom_callback)
      #  print('subscribed')
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
        
    def init_publisher(self, config_topics):
        self.cmd_pub = rospy.Publisher(config_topics['command'], JointState, queue_size=10)

    def publish(self, qpos, qvel, eff, Kp, Kd):
        try:
            #rospy.init_node('communicate_aliengo', anonymous=True)
            self.joint_state_des.position = np.concatenate((qpos, [Kp])) # store Kp on the fake joint pos
            self.joint_state_des.velocity = np.concatenate((qvel, [Kd])) # store Kd on the fake joint vel
            # concatenate a zero to maintain size to 13
            self.joint_state_des.effort = np.concatenate((eff, [0])) 
         #   print('pos',qpos)
          #  print('vel',qvel)
          #  print('eff',eff)
            self.cmd_pub.publish(self.joint_state_des)
        except rospy.ROSInterruptException:
            pass
#!/usr/bin/env python3
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
import rospy as rp

pos_l_x,pos_l_y,pos_l_z,pos_a_x,pos_a_y,pos_a_z = 2,0,0,0,0,-1.8 

def pose_callback(msg):
    print('callback')
    rp.loginfo("("+ str(msg.linear.x) + "," + str(msg.linear.y) + "," + str(msg.angular.z)+ ")")
    global pos_l_x,pos_l_y,pos_l_z,pos_a_x,pos_a_y,pos_a_z 
    pos_l_x = msg.linear.x
    pos_l_y = msg.linear.y
    pos_l_z = msg.linear.z
    pos_a_x = msg.angular.x
    pos_a_y = msg.angular.y
    pos_a_z = msg.angular.z

if __name__ == '__main__':
    rp.init_node("turtle_inverse")
    sub = rp.Subscriber("/turtle1/cmd_vel", Twist, callback= pose_callback)

    rate = rp.Rate(1)
    rp.loginfo("Node has been started")

    while not rp.is_shutdown():
        cmd = Twist()

        cmd.linear.x = -1*pos_l_x
        cmd.linear.y = -1*pos_l_y
        cmd.linear.z = -1*pos_l_z
        cmd.angular.x = -1*pos_a_x
        cmd.angular.y = -1*pos_a_y
        cmd.angular.z = -1*pos_a_z
    
        pub = rp.Publisher("/turtle1/cmd_vel", Twist, queue_size=10)
        try:
            pub.publish(cmd)
        except rp.ServiceException as e:
            pass
        pos_l_x,pos_l_y,pos_l_z,pos_a_x,pos_a_y,pos_a_z = 2,0,0,0,0,-1.8 
    rate.sleep()
    rp.spin()
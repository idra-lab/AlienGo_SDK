#!/usr/bin/python

import sys
import time
import math
import numpy as np
import torch
import mps_code
import copy
import rospy
import publish_subscribe
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped

# Neural network and configuration imports
from config_loader import load_config, load_actor_network, load_value
from utils import scale_axis, quat_rotate_inverse, swap_legs
import pygame

import threading

simulator = True


# Remove if controller is not used 
# Initialize pygame and the joystick module
if not simulator:
    pygame.init()
    pygame.joystick.init()

    # Remove if controller is not used 
    # Check if there is at least one joystick (gamepad) connected
    if pygame.joystick.get_count() == 0:
        print("No joystick connected")
    else:
        joystick = pygame.joystick.Joystick(0)  # Get the first joystick
        joystick.init()
        print(f"Detected joystick: {joystick.get_name()}")

### Configuration and neural network setup
# Nominal policy
config_path = "config.yaml"
config = load_config(config_path)
actor_network = load_actor_network(config['nominal'],config['nominal']['paths']['checkpoint_path'])
scaling_factors = config['nominal']['scaling']
default_joint_angles = config['nominal']['robot']['default_joint_angles']

max_pos = config['nominal']['robot']['max_pos']
min_pos = config['nominal']['robot']['min_pos']
torque_values_n = config['nominal']['robot']['torque_values']
scaling_qdes = scaling_factors['factor']


imu_acc = np.zeros(3)
imu_quat = np.zeros(4)
imu_gyro = np.zeros(3)
joint_pos = np.zeros(12)
joint_vel = np.zeros(12)
pose = np.zeros(7)
twist = np.zeros(6)

# Value function
config_path_value = '../nn/VF_safe_MPC_small.pkl'
config_value = load_value(config_path_value)


def get_commands(): 
    """
    Retrieves joystick commands from a connected joystick using pygame.

    This function checks if there is exactly one joystick connected. If so, it processes
    joystick events, retrieves axis values, scales them, and applies a threshold to filter
    out small values. The resulting commands are returned as a numpy array.

    Raises:
        SystemExit: If no joystick or more than one joystick is connected.
    """

    if pygame.joystick.get_count() == 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get axis values (e.g., left stick, right stick, triggers)
        axes = [joystick.get_axis(i) for i in [0, 1, 2, 5]]
        #axes = [joystick.get_axis(i) for i in [0, 1, 4, 5]]
        #print("axes: ",axes)
        summed_axes = + (1+axes[2]) - (1+axes[3])  # Sum axis 2 and axis 5 # Invert signs between the two controllers
        axes = np.array([axes[0], axes[1], summed_axes])

        scaled_axes = [scale_axis(i, axes[i]) for i in range(len(axes))]
        scaled_axes[0], scaled_axes[1] =scaled_axes[1], scaled_axes[0]
        commands = np.array(scaled_axes)

        # Apply the threshold to commands
        threshold = 0.05  # Define the threshold value
        commands = np.array([x if abs(x) >= threshold else 0 for x in scaled_axes])

        return commands
    else:
        exit()
    
def get_safety_button(): 
    """
    Function to check if the safety button on the controller is pressed. 
    Y button for current joystick.
    """

    if pygame.joystick.get_count() == 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        safety_button = joystick.get_button(3)
        if safety_button:
            return True
       
    return False

# Shared variables
lock = threading.Lock()
latest_actions = np.zeros(12)  # Store the latest actions safely across threads
previous_actions = np.zeros(12)  # Store the previous actions
current_actions = np.zeros(12)
inference_ready = threading.Event()  # Event to signal new inference results
stop_threads = False  # Flag to stop threads gracefully

def compute_observation(scaling_factors, prev_actions1, nominal):
    """
    Compute the observation vector from the robot's state.
    Legs are swapped to match the order of the neural network input.
    SDK order = [FR, FL, RR, RL]
    nn order = [FL, FR, RL, RR]
    """
    # Remove if the controller is not used
    #commands = get_commands() # The stopping condition here is not evaluated
    
    # Add if the controller is not used
    if nominal:
        commands = np.array([0.,0.,0.]) # The stopping condition here is not evaluated
    else:
        commands = np.array([-0.45, -0.02, 0.])
    commands = np.array([0.,0.,0.])

    body_quat = np.array([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]])
    body_vel = np.array([imu_gyro[0], imu_gyro[1], imu_gyro[2]])
    joint_angles1 = [joint_pos[i] for i in range(12)]
    joint_angles = swap_legs(joint_angles1)
    joint_velocities1 = [joint_vel[i] for i in range(12)]
    joint_velocities = swap_legs(joint_velocities1)

    # Gravity vector in body frame
    gravity_body = quat_rotate_inverse(
        torch.tensor(body_quat, dtype=torch.float32).unsqueeze(0),
        torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    ).squeeze().numpy()

    prev_actions1 = np.copy(previous_actions)
    prev_actions = swap_legs(prev_actions1)

    # Scale observations
    scaled_body_vel = body_vel * scaling_factors['body_ang_vel']
    scaled_commands = commands[:2] * scaling_factors['commands']
    scaled_commands = np.append(scaled_commands, commands[2] * scaling_factors['body_ang_vel'])
    scaled_gravity_body = gravity_body * scaling_factors['gravity_body']
    scaled_joint_angles = np.array(joint_angles) * scaling_factors['joint_angles']
    scaled_joint_velocities = np.array(joint_velocities) * scaling_factors['joint_velocities']
    scaled_actions = prev_actions * scaling_factors['actions']

    # Concatenate into a single observation vector
    return np.concatenate((scaled_body_vel, scaled_commands, scaled_gravity_body, scaled_joint_angles, scaled_joint_velocities, scaled_actions))

def compute_actions(scaling_factors, previous_actions, nominal, actor_network):

    """
    Inference on the NN to retrive actions from observations.
    Legs are swapped to match the order of the neural network input.
    SDK order = [FR, FL, RR, RL]
    nn order = [FL, FR, RL, RR]
    """
    #global latest_actions, previous_actions, stop_threads
    #while not stop_threads:
        #start_time = time.time()
        
     #   inference_ready.wait()  # Wait for signal from the main thread
      #  inference_ready.clear()

    obs = compute_observation(scaling_factors, previous_actions, nominal)
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    obs_normalized = actor_network.norm_obs(obs_tensor)

    with torch.no_grad():
        new_actions1 = actor_network(obs_normalized).numpy()
    
    # Swap the actions to the correct order for SDK
    #new_actions = swap_legs(new_actions1)

    #with lock:
    #    previous_actions[:] = latest_actions  # Store current actions as previous
    #    latest_actions[:] = new_actions  # Update latest actions

    """ print(f"Inference completed in: {time.time() - start_time:.5f} seconds") """
    return new_actions1

def jointLinearInterpolation(initPos, targetPos, rate):
    """
    Performs a linear interpolation between initial and target joint positions.
    """
    rate = np.fmin(np.fmax(rate, 0.0), 1.0)
    p = initPos*(1-rate) + targetPos*rate
    return p

def check_safety_stops(imu_quat):
    """
    Check if the inclination of the robot base exceeds the threshold (pi/8) and checks the safety button as well.
    """
    #imu = state.imu
    body_quat = imu_quat#imu.quaternion  # Quaternion from qpos
    # Calculate inclination using arcsin formula
    inclination = 2 * np.arcsin(np.sqrt(body_quat[1]**2 + body_quat[2]**2))

    stop_button = get_safety_button()  # Check if the safety button is pressed

    if pygame.joystick.get_count() != 1:
        return True

    if stop_button:#inclination > np.pi/8:# or stop_button:
        return True
    else:
        return False

if __name__ == '__main__':
    rospy.init_node('communicate_aliengo')
    pubSub = publish_subscribe.PubSub()
    pubSub.init_publisher(config['controller']['topics'])
    pubSub.init_subscribers(config['controller']['topics'])
    

    # Initialize as recoverable
    is_rec = True

    d = {'FR_0':0, 'FR_1': 1, 'FR_2': 2,
         'FL_0':3, 'FL_1': 4, 'FL_2': 5, 
         'RR_0':6, 'RR_1': 7, 'RR_2': 8, 
         'RL_0':9, 'RL_1':10, 'RL_2':11 }

    legs = ['FR', 'FL', 'RR', 'RL']
    joints = ['_0', '_1', '_2']

    sin_mid_q = 4*[0.0, 0.7, -1.5] # Creates a 12-element list with the default joint angles for the standup
    dt = 0.002

    qInit = [0, 0, 0,
             0, 0, 0,
             0, 0, 0,
             0, 0, 0]
    
    qDes = [0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0]
    
    rate_count = 0
    Kp_n = config['nominal']['robot']['Kp_n']
    Kd_n = config['nominal']['robot']['Kd_n']
    # PD tuning parameters
    Kp = [Kp_n, Kp_n, Kp_n]
    Kd = [Kd_n, Kd_n, Kd_n]

    actions = torch.zeros(12, dtype=torch.float32)

    # Decimation factor to reduce the policy update frequency - Number of control action updates @ sim DT per policy DT
    # Decimation changed to 5 to have a 100 Hz main loop, like in the simulations
    decimation = 5
    mps = mps_code.MPS(decimation, max_pos, min_pos, torque_values_n, Kp_n, Kd_n, scaling_qdes, default_joint_angles, scaling_factors, config_value)
    torque_values = np.zeros(12)
    #state = sdk.LowState()

    motiontime = 0
    nominal = True

    disable_torques = False  # Flag to disable torques if inclination exceeds threshold or safety button is pressed
    # Start the inference thread
    #threading.Thread(target=compute_actions, args=(scaling_factors,), daemon=True).start()
    n_wait = 0
    while pubSub.cmd_pub.get_num_connections() < 1:
        n_wait += 1
       # print(n_wait)
        #pass
    #print('after wait')
    #time.sleep(2)
    #pubSub.publish(np.zeros(12), np.zeros(12), np.zeros(12), Kp_n, Kd_n)
    firstTime = True

    while not rospy.is_shutdown():
       # print('while')
        """
        Keeping the dt = 0.002, we need a decimation = 10 to keep the policy update frequency to 50Hz
        The main loop for sending commands is running at 500Hz
        """
        step_start = time.time()
        motiontime += 1
        
        #imu_acc, imu_quat, imu_gyro, joint_pos, joint_vel, pose, twist = pubSub.wait_for_all_messages()
       # print('after wait_for_all_messages')
        data_new = [pubSub.imu_acc,pubSub.imu_quat,pubSub.imu_gyro,pubSub.joint_pos,pubSub.joint_vel,pubSub.pose,pubSub.twist]
        imu_acc = data_new[0]
        imu_quat = data_new[1]
        imu_gyro = data_new[2]
        joint_pos = data_new[3]
        joint_vel = data_new[4]
        pose = data_new[5]
        twist = data_new[6]

        # Check base inclination and modify Kp, Kd if needed - to disable control torques
        if not simulator and check_safety_stops(pubSub.imu_quat):  # Using qpos to check inclination
            print("Safety condition triggered, disabling control gains")
            # Set Kp, Kd to 0 (disable control) for safety
          #  Kp = [0, 0, 0]  # Set Kp to 0 for all joints
          #  Kd = [0, 0, 0]  # Set Kd to 0 for all joints
            Kp = 0
            Kd = 0
           # pubSub.publish(qDes, np.zeros(12), torque_values*4, 0, 0)
            exit()

        # First, record initial position
        #'''
        if( motiontime >= 0 and motiontime < 1*(1/dt)):
            Kp = 0
            Kd = 0
            # Extract qInit values using dictionary keys
            #qInit = [state_robot.motorState[d[key]].q for key in d]
            qInit = [pubSub.joint_pos[i] for i in range(12) ]

        # second, move to the origin point of a sine movement with Kp Kd
        elif( motiontime >= 1*(1/dt) and motiontime < 7*(1/dt)):
            #exit()
            Kp = Kp_n
            Kd = Kd_n
            torque_values = torque_values_n
            rate_count += 1
            rate = rate_count / (5*(1/dt))

            # Here I don't switch the legs because the default joint angles are simmetric
            #qDes = [jointLinearInterpolation(qInit[i], default_joint_angles[i], rate) for i in range(12)]
            qDes = [jointLinearInterpolation(qInit[i], sin_mid_q[i], rate) for i in range(12)]
            qDes = np.clip(qDes, min_pos, max_pos)#'''
           # print(qDes)

        
        elif( motiontime >= 7*(1/dt)):# and is_rec):
            if firstTime:
                print("START PRONTO!")
                time.sleep(10.)
                firstTime = False
            #    print('scaling_factors',scaling_factors)
            #    print('previous_actions',previous_actions)
                print('after wait')                     
            if motiontime % decimation == 0:

                '''
                actor_network_copy = copy.deepcopy(actor_network)
                new_actions1 = compute_actions(scaling_factors, previous_actions, nominal, actor_network_copy)
                
                # Compute torque using nominal policy       
                # Check which policy should be used
                is_rec = mps.is_rec_single(qDes, pubSub.pose, pubSub.twist, pubSub.joint_pos, pubSub.joint_vel, actor_network_copy, np.copy(current_actions), swap_legs(new_actions1))
                is_rec = True
                if not is_rec:
                    nominal = False#'''

                
                #'''
                new_actions1 = compute_actions(scaling_factors, previous_actions, nominal, actor_network)
                previous_actions = current_actions
            # Trigger inference every `decimation` steps
            
                #inference_ready.set()

                # Get the latest available actions
                #with lock:  
                current_actions = swap_legs(new_actions1)
                #print(scaling_qdes * current_actions + np.array(default_joint_angles))
                
            qDes = scaling_qdes * current_actions + np.array(default_joint_angles)

            # Clip the joint angles to the joint limits
            qDes = np.clip(qDes, min_pos, max_pos)#'''
            #'''

            # MPS check
          
        
        #time.sleep(0.02)
        #print('before publish')
        if(motiontime >= 1*(1/dt)):
            pubSub.publish(qDes, np.zeros(12), torque_values*4, Kp_n, Kd_n)
            #print(motiontime)
        #print('after publish')
        # Temporize the loop to maintain the desired frequency
        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
    rospy.spin()
        # elapsed_time = time.time() - step_start  # Time taken for the loop iteration
        # print(f"Loop took: {elapsed_time:.6f} seconds ({1/elapsed_time:.2f} Hz)")
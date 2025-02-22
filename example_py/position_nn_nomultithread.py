#!/usr/bin/python

import sys
import time
import math
import numpy as np
import torch

sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# Neural network and configuration imports
from config_loader.config_loader import load_config, load_actor_network
from utils import scale_axis, quat_rotate_inverse, swap_legs, clip_torques_in_groups
import pygame

# Initialize pygame and the joystick module
pygame.init()
pygame.joystick.init()

# Check if there is at least one joystick (gamepad) connected
if pygame.joystick.get_count() == 0:
    print("No joystick connected")
else:
    joystick = pygame.joystick.Joystick(0)  # Get the first joystick
    joystick.init()
    print(f"Detected joystick: {joystick.get_name()}")

# Config and neural network setup
config_path = "config.yaml"
config = load_config(config_path)
actor_network = load_actor_network(config)
scaling_factors = config['scaling']
default_joint_angles = config['robot']['default_joint_angles']

# Low-level command parameters
TARGET_PORT = 8007
LOCAL_PORT = 8082
TARGET_IP = "192.168.123.10"

LOW_CMD_LENGTH = 610
LOW_STATE_LENGTH = 771

def get_commands(commands_original = np.array([0., 0., 0.])): 
    """
    Compute the robot commands from the joystick inputs.
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
    else:
        exit()

    return commands

def get_safety_button(): 
    """
    Compute the robot commands from the joystick inputs.
    """

    if pygame.joystick.get_count() == 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        safety_button = joystick.get_button(3)
        if safety_button:
            return True
       
    return False


def compute_observation(state, scaling_factors, prev_actions1):
    """
    Compute the observation vector from the robot's state.
    """
    commands = get_commands()

    imu = state.imu
    body_quat = np.array([imu.quaternion[1], imu.quaternion[2], imu.quaternion[3], imu.quaternion[0]])
    body_vel = np.array([imu.gyroscope[0], imu.gyroscope[1], imu.gyroscope[2]])
    joint_angles1 = [state.motorState[i].q for i in range(12)]
    joint_angles = swap_legs(joint_angles1)
    joint_velocities1 = [state.motorState[i].dq for i in range(12)]
    joint_velocities = swap_legs(joint_velocities1)

    # Gravity vector in body frame
    gravity_body = quat_rotate_inverse(
        torch.tensor(body_quat, dtype=torch.float32).unsqueeze(0),
        torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    ).squeeze().numpy()

    prev_actions = swap_legs(prev_actions1)

    #print(prev_actions)

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

def jointLinearInterpolation(initPos, targetPos, rate):
    """
    Performs a linear interpolation between initial and target positions.
    """
    rate = np.fmin(np.fmax(rate, 0.0), 1.0)
    p = initPos*(1-rate) + targetPos*rate
    return p

def check_safety_stops(state):
    """
    Check if the inclination (pitch or roll) of the robot base exceeds the threshold (pi/4).
    """
    imu = state.imu
    body_quat = imu.quaternion  # Quaternion from qpos
    # Calculate inclination using arcsin formula
    inclination = 2 * np.arcsin(np.sqrt(body_quat[1]**2 + body_quat[2]**2))

    stop_button = get_safety_button()  # Check if the safety button is pressed

    if pygame.joystick.get_count() != 1:
        return True

    if inclination > np.pi/8 or stop_button:
        return True
    else:
        return False


if __name__ == '__main__':

    d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
         'FL_0':3, 'FL_1':4, 'FL_2':5, 
         'RR_0':6, 'RR_1':7, 'RR_2':8, 
         'RL_0':9, 'RL_1':10, 'RL_2':11 }

    legs = ['FR', 'FL', 'RR', 'RL']
    joints = ['_0', '_1', '_2']
    torque_values = [-1.6, 0.0, 0.0]

    PosStopF  = math.pow(10,9)
    VelStopF  = 16000.0
    HIGHLEVEL = 0x00
    LOWLEVEL  = 0xff
    sin_mid_q = 4*[0.0, 0.7, -1.5]
    dt = 0.002

    qInit = [0, 0, 0,
             0, 0, 0,
             0, 0, 0,
             0, 0, 0]
    
    qDes = [0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0]
    
    current_actions = np.array([0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0])
    
    rate_count = 0

    # PD tuning parameters
    """ Kp = 0.*[100, 100, 100]
    Kd = 0.*[3, 3, 3] """

    Kp = [100, 100, 100]
    Kd = [3, 3, 3]

    actions = torch.zeros(12, dtype=torch.float32)

    # Decimation factor to reduce the policy update frequency - Number of control action updates @ sim DT per policy DT
    decimation = 1

    # Initialize the UDP connection
    udp = sdk.UDP(LOCAL_PORT, TARGET_IP, TARGET_PORT, LOW_CMD_LENGTH, LOW_STATE_LENGTH, -1)
    safe = sdk.Safety(sdk.LeggedType.Aliengo)
    # Initialize the command and state objects
    cmd = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)
    cmd.levelFlag = LOWLEVEL

    motiontime = 0

    disable_torques = False  # Flag to disable torques if inclination exceeds threshold

    while True:
        """
        Keeping the dt = 0.002, we need a decimation = 10 to keep the policy update frequency to 50Hz
        The main loop for sending commands is running at 500Hz
        """
        #time.sleep(dt)
        step_start = time.time()
        motiontime += 1
    
        udp.Recv()
        udp.GetRecv(state)

        # Check base inclination and modify Kp, Kd if needed - to disable control torques
        if check_safety_stops(state):  # Using qpos to check inclination
            print("Inclination exceeded pi/6, disabling control gains")
            # Set Kp, Kd to 0 (disable control) for safety
            Kp = [0, 0, 0]  # Set Kp to 0 for all joints
            Kd = [0, 0, 0]  # Set Kd to 0 for all joints
            exit()

        # First, record initial position
        if( motiontime >= 0 and motiontime < 1*(1/dt)):
            # Extract qInit values using dictionary keys
            qInit = [state.motorState[d[key]].q for key in d]

        # second, move to the origin point of a sine movement with Kp Kd
        elif( motiontime >= 1*(1/dt) and motiontime < 7*(1/dt)):
            rate_count += 1
            rate = rate_count / (5*(1/dt))

            # Here I don't switch the legs because the default joint angles are simmetric
            #qDes = [jointLinearInterpolation(qInit[i], default_joint_angles[i], rate) for i in range(12)]
            qDes = [jointLinearInterpolation(qInit[i], sin_mid_q[i], rate) for i in range(12)]
        
        elif( motiontime >= 7*(1/dt)):

            obs = compute_observation(state, scaling_factors, current_actions)
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            obs_normalized = actor_network.norm_obs(obs_tensor)

            with torch.no_grad():
                new_actions1 = actor_network(obs_normalized).numpy()
            
            # Swap the actions to the correct order for SDK
            current_actions = swap_legs(new_actions1)

            #print(current_actions)
            qDes = 0.5 * current_actions + np.array(default_joint_angles)

        # Clip the joint angles to the joint limits
        for i in range(4):
            qDes[i*3] = np.clip(qDes[i*3], -1.22, 1.22) # Hip joint
            qDes[i*3+1] = np.clip(qDes[i*3+1], 0.0, 1.8) # Thigh joint
            qDes[i*3+2] = np.clip(qDes[i*3+2], -2.78, -0.65) # Calf joint

        if motiontime >= 1*(1/dt):
            for leg_idx, leg in enumerate(legs):
                for joint_idx, joint in enumerate(joints):
                    key = f"{leg}{joint}"
                    cmd.motorCmd[d[key]].q = qDes[leg_idx * 3 + joint_idx]
                    cmd.motorCmd[d[key]].dq = 0
                    cmd.motorCmd[d[key]].Kp = Kp[joint_idx]
                    cmd.motorCmd[d[key]].Kd = Kd[joint_idx]
                    cmd.motorCmd[d[key]].tau = torque_values[joint_idx]

        """ temp = dt - (time.time() - step_start)
        if temp < 0:
            print(f"\033[31m{temp:.5f}\033[0m")
        else:
            print(f"\033[32m{temp:.5f}\033[0m") """
        
        # print(qDes)
           
        # Safety checks
        safe.PowerProtect(cmd, state, 5)
        safe.PositionLimit(cmd)

        #print(qDes)
        if motiontime > 5*(1/dt):
            safe.PositionProtect(cmd, state, 0.087)

        """ if (motiontime >= 5*(1/dt)):
            print([state.motorState[i].tauEst for i in range(12)]) """

        udp.SetSend(cmd)
        udp.Send()

        # Temporize the loop to maintain the desired frequency
        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        
        """ elapsed_time = time.time() - step_start  # Time taken for the loop iteration
        print(f"Loop took: {elapsed_time:.6f} seconds ({1/elapsed_time:.2f} Hz)") """
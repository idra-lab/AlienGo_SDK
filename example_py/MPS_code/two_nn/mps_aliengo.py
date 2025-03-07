#!/usr/bin/python

import sys
import time
import math
import numpy as np
import torch
import mps
import mujoco

sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# Neural network and configuration imports
from config_loader import load_config, load_nominal_network, load_backup_network
from utils import scale_axis, quat_rotate_inverse, swap_legs, clip_torques_in_groups, orderState, orderBackup, orderPositionBackup
import pygame

import threading

# Initialize pygame and the joystick module
pygame.init()
pygame.joystick.init()
pygame.display.set_mode((100,100))

# Check if there is at least one joystick (gamepad) connected
#if pygame.joystick.get_count() == 0:
#    print("No joystick connected")
#else:
#    joystick = pygame.joystick.Joystick(0)  # Get the first joystick
#    joystick.init()
#    print(f"Detected joystick: {joystick.get_name()}")

# Config and neural networks setup
# Nominal policy
config_path_nominal = "config.yaml"
config_nominal = load_config(config_path_nominal)
nominal_network = load_nominal_network(config_nominal)
scaling_factors_nominal = config_nominal['scaling']
default_joint_angles_nominal = config_nominal['robot']['default_joint_angles']

# Backup policy
config_path_backup = "config_mps.yaml"
config_backup = load_config(config_path_backup)
backup_network = load_backup_network(config_backup)
scaling_factors_backup = config_backup['scaling']
default_joint_angles_backup = config_backup['scaling']['default_joint_angles']
lim_tau = config_backup['actions']['bounds']['high']
lim_vel = config_backup['actions']['bounds']['vel']
Kp_sim = config_backup['robot']['kp_custom']
Kd_sim = config_backup['robot']['kd_custom']
min_height = 0.1
N_mps = 200
X_inv = 10e-2
prev_z_position = 0
prev_body_vel = np.zeros(3)

# Maximum number of times the main loop will be repeated
max_iter = 500
iter_mpc = 0

# MuJoCo simulation
desc_dir = '../models/aliengo_models'
xml = desc_dir + '/xml/aliengo.xml'
spec = mujoco.MjSpec()
spec.from_file(xml)
model_muj = spec.compile()
data_muj = mujoco.MjData(model_muj)

# Force application
step_rand = 0
force_body = model_muj.body('trunk').id
joint_limits = model_muj.jnt_range
limits_min = joint_limits[:, 0]
limits_max = joint_limits[:, 1]
jmax_compare = [limits_max[1], limits_max[3], limits_max[4], limits_max[6], limits_max[7], limits_max[9], limits_max[10], limits_max[12]]
jmin_compare = [limits_min[1], limits_min[3], limits_min[4], limits_min[6], limits_min[7], limits_min[9], limits_min[10], limits_min[12]]


# Disable backup at the beginning
is_rec = True

decimation = 0

# Low-level command parameters
TARGET_PORT = 8007
LOCAL_PORT = 8082
TARGET_IP = "192.168.123.10"

LOW_CMD_LENGTH = 610
LOW_STATE_LENGTH = 771
    
def get_safety_button(): 
    """Function to check if the safety button on the controller is pressed. Y buttoon for corrent joystick"""

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        if event.type == pygame.KEYDOWN:
            print("Keydown detected")
            if event.key == pygame.K_0:
                print("0 Pressed, QUITTING !!!!")
                return True

    return False

def mps(): 
    """Decide if the backup policy has to be used"""

    if pygame.joystick.get_count() == 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        backup_button = joystick.get_button(2)
        if backup_button:
            return True
       
    return False

# Shared variables
lock = threading.Lock()
latest_actions = np.zeros(12)  # Store the latest actions safely across threads
previous_actions = np.zeros(12)  # Store the previous actions
inference_ready = threading.Event()  # Event to signal new inference results
stop_threads = False  # Flag to stop threads gracefully

def compute_observation_nominal(state, scaling_factors):
    """
    Compute the observation vector from the robot's state.
    Legs are swapped to match the order of the neural network input.
    SDK order = [FR, FL, RR, RL]
    nn order = [FL, FR, RL, RR]
    """
    commands = np.array([0.5, 0., 0.]) # Velocity commands

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

    prev_actions1 = np.copy(previous_actions)
    prev_actions = (swap_legs(prev_actions1) - default_joint_angles_nominal) / 0.5

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

def compute_actions(state, high_state, scaling_factors):
    """
    Inference ont he nn to retrive actions from observations.
    Legs are swapped to match the order of the neural network input.
    SDK order = [FR, FL, RR, RL]
    nn order = [FL, FR, RL, RR]
    """
    global latest_actions, previous_actions, stop_threads
    while not stop_threads:
        start_time = time.time()
        
        inference_ready.wait()  # Wait for signal from the main thread
        inference_ready.clear()

        # Choose policy 
        if is_rec:
            obs = compute_observation_nominal(state, scaling_factors)
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            obs_normalized = nominal_network.norm_obs(obs_tensor)

            with torch.no_grad():
                new_actions1 = nominal_network(obs_normalized).numpy()
            
            # Swap the actions to the correct order for SDK
            new_actions = 0.5 * swap_legs(new_actions1) + np.array(default_joint_angles_nominal)            
            imu = state.imu
            body_quat = np.array([imu.quaternion[0], imu.quaternion[1], imu.quaternion[2], imu.quaternion[3]])
            body_vel = np.array([imu.gyroscope[0], imu.gyroscope[1], imu.gyroscope[2]])
            body_acc = np.array([imu.accelerometer[0], imu.accelerometer[1], imu.accelerometer[2]])
            z_vel = (high_state.bodyHeight - prev_z_position) / dt
            ang_acc = (body_vel - prev_body_vel) / dt
            velocities = np.concatenate((high_state.velocity, z_vel, body_vel, [state.motorState[d[key]].dq for key in d]))
            data_muj.qpos = np.concatenate((high_state.position, high_state.bodyHeight, body_quat, [state.motorState[d[key]].q for key in d]))
            data_muj.qvel = velocities
            data_muj.qacc = np.concatenate((body_acc, ang_acc, [state.motorState[d[key]].ddq for key in d]))
            data_muj.qacc_warmstart = data_muj.qacc.copy()
            mujoco.mj_forward(model_muj, data_muj)

            is_rec = mps.isRecSingle(min_height, qDes, decimation, lim_tau, model_muj, data_muj, N_mps, backup_network, Kp_sim, Kd_sim, X_inv, i, step_rand, force_body, jmax_compare, jmin_compare, lim_vel)

            # Use the chosen policy
            if np.all(np.abs(velocities) <= X_inv) and not is_rec:
                is_rec = True

        if not is_rec:
            obs = mps.compute_observation_backup(state)
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            obs_normalized = backup_network.norm_obs(obs_tensor)

            with torch.no_grad():
                pos_backup = backup_network(obs_normalized).numpy()
                #new_actions1 = order_state(pos_backup + default_joint_angles)
            
            # Swap the actions to the correct order for SDK
            new_actions = orderBackup((0.8 * new_actions) + np.array(default_joint_angles_backup))
            

        with lock:
            previous_actions[:] = latest_actions  # Store current actions as previous
            latest_actions[:] = new_actions  # Update latest actions
    
        """ print(f"Inference completed in: {time.time() - start_time:.5f} seconds") """

def jointLinearInterpolation(initPos, targetPos, rate):
    """
    Performs a linear interpolation between initial and target joint positions.
    """
    rate = np.fmin(np.fmax(rate, 0.0), 1.0)
    p = initPos*(1-rate) + targetPos*rate
    return p

def check_safety_stops(state):
    """
    Check if the inclination of the robot base exceeds the threshold (pi/8) and checks the safety button as well.
    """
    imu = state.imu
    body_quat = imu.quaternion  # Quaternion from qpos
    # Calculate inclination using arcsin formula
    inclination = 2 * np.arcsin(np.sqrt(body_quat[1]**2 + body_quat[2]**2))

    stop_button = get_safety_button()  # Check if the safety button is pressed

    if stop_button:#inclination > np.pi/4 or
        print('inclination', inclination*180/np.pi)
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
    sin_mid_q = 4*[0.0, 0.7, -1.5] # Creates a 12-elements list with the default joint angles for the standup
    dt = 0.002

    qInit = [0, 0, 0,
             0, 0, 0,
             0, 0, 0,
             0, 0, 0]
    
    qDes = [0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0]
    
    dqDes = [0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0]
    
    rate_count = 0

    # PD tuning parameters
    Kp_nominal = [100, 100, 100]
    Kd_nominal = [3, 3, 3]

    Kp_backup = [25, 25, 25]
    Kd_backup = [0.5, 0.5, 0.5]    

    actions = torch.zeros(12, dtype=torch.float32)

    # Decimation factor to reduce the policy update frequency - Number of control action updates @ sim DT per policy DT
    decimation = 5

    # Initialize the UDP connection
    udp = sdk.UDP(LOCAL_PORT, TARGET_IP, TARGET_PORT, LOW_CMD_LENGTH, LOW_STATE_LENGTH, -1)
    safe = sdk.Safety(sdk.LeggedType.Aliengo)
    # Initialize the command and state objects
    cmd = sdk.LowCmd()
    state = sdk.LowState()
    high_state = sdk.HighState()
    udp.InitCmdData(cmd)
    cmd.levelFlag = LOWLEVEL

    motiontime = 0

    disable_torques = False  # Flag to disable torques if inclination exceeds threshold or safety button is pressed

    # Start the inference thread
    threading.Thread(target=compute_actions, args=(state, high_state, scaling_factors_nominal), daemon=True).start()

    while iter_mpc < max_iter:
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
            print("Safety condition triggered, disabling control gains")
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

            # Trigger inference every `decimation` steps
            if motiontime % decimation == 0:
                inference_ready.set()

            # Get the latest available actions
            with lock:  
                qDes = np.copy(latest_actions)

            #print(current_actions)
            if is_rec:
                Kp = Kp_nominal
                Kd = Kd_nominal
                u = torque_values
                
            else:
                Kp = Kp_backup
                Kd = Kd_backup
                u = np.zeros(3)
            
            

        # Clip the joint angles to the joint limits
        for i in range(4):
            qDes[i*3] = np.clip(qDes[i*3], -1.22, 1.22) # Hip joint
            qDes[i*3+1] = np.clip(qDes[i*3+1], 0.0, 1.8) # Thigh joint
            qDes[i*3+2] = np.clip(qDes[i*3+2], -2.78, -0.65) # Calf joint

        prev_z_position = np.copy(high_state.bodyHeight)
        imu = state.imu
        prev_body_vel = np.array([imu.gyroscope[0], imu.gyroscope[1], imu.gyroscope[2]])
        if motiontime >= 1*(1/dt):
            for leg_idx, leg in enumerate(legs):
                for joint_idx, joint in enumerate(joints):
                    key = f"{leg}{joint}"
                    cmd.motorCmd[d[key]].q = qDes[leg_idx * 3 + joint_idx]
                    cmd.motorCmd[d[key]].dq = 0
                    cmd.motorCmd[d[key]].Kp = Kp[joint_idx]
                    cmd.motorCmd[d[key]].Kd = Kd[joint_idx]
                    cmd.motorCmd[d[key]].tau = torque_values[joint_idx] ##########Check, comm.h says: desired output torque

        """ temp = dt - (time.time() - step_start)
        if temp < 0:
            print(f"\033[31m{temp:.5f}\033[0m")
        else:
            print(f"\033[32m{temp:.5f}\033[0m") """
           
        """ Safety checks"""
        safe.PowerProtect(cmd, state, 7)
        safe.PositionLimit(cmd)

        if motiontime > 5*(1/dt):
            safe.PositionProtect(cmd, state, 0.087)

        udp.SetSend(cmd)
        udp.Send()

        # Temporize the loop to maintain the desired frequency
        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        
        # elapsed_time = time.time() - step_start  # Time taken for the loop iteration
        # print(f"Loop took: {elapsed_time:.6f} seconds ({1/elapsed_time:.2f} Hz)")
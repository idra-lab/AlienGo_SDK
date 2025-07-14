#!/usr/bin/python

import sys
import time
import math
import numpy as np
import torch
import mps as mps_code

sys.path.append('../../lib/python/amd64')
import robot_interface as sdk

# Neural network and configuration imports
from config_loader import load_config, load_actor_network, labels_state_dict, Backup, orderPosition, computeBackup, orderBackup
from utils import scale_axis, quat_rotate_inverse, swap_legs, clip_torques_in_groups
import pygame

import threading

import time
import csv

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

# Config backup network
# Data to preprocess state before sending it to the network
device = torch.device('cpu')
#if torch.cuda.is_available():
#        device = torch.device('cuda')

running_mean = torch.tensor([ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.21205050e+00,
                             -8.06230644e-03, -1.42328272e+00,  1.00789203e-01, -1.14620532e-01,
                              6.92671321e-02, -2.20188718e-01,  2.29067680e-01,  2.09180188e-01,
                              2.06069022e-01,  1.45354481e-01, -4.06120459e-01, -4.48609024e-01,
                             -3.88464357e-01, -3.97695643e-01, -6.33672667e-03, -1.95075319e-03,
                             -3.96507459e-03, -4.42201867e-02,  1.14105612e-01,  1.03970752e-01,
                              5.92749748e-02,  4.45349231e-02, -1.63696425e-01, -1.79439128e-01,
                             -1.44850614e-01, -1.38541725e-01,  3.80852309e-02, -1.01241193e-01,
                             -1.90169735e-02, -1.39184525e-01,  1.36771685e-01,  1.38808689e-01,
                              1.64408062e-01,  1.45937922e-01,  4.67878538e-02, -5.74250520e-04,
                             -6.52443376e-02, -1.06306087e-02], device=device, dtype=torch.float64)

running_variance = torch.tensor([1.45321798e-08, 1.45321798e-08, 1.45321798e-08, 1.02176822e+01,
                                 1.08386133e+01, 2.91442128e+01, 6.12955153e-02, 7.91401819e-02,
                                 6.40535954e-02, 3.81343809e-02, 8.86714353e-02, 6.60074706e-02,
                                 5.39578640e-02, 3.24022101e-02, 5.45107210e-02, 5.66611032e-02,
                                 6.45901655e-02, 6.92300715e-02, 6.07200216e+00, 5.75513999e+00,
                                 8.04183510e+00, 8.15766779e+00, 4.10619267e+00, 4.09391329e+00,
                                 5.59485546e+00, 5.44709528e+00, 6.09915664e+00, 6.84910827e+00,
                                 9.67759842e+00, 8.53798207e+00, 7.66597364e-02, 7.27123376e-02,
                                 8.60715622e-02, 6.56937354e-02, 8.89956898e-02, 8.91989478e-02,
                                 8.37767829e-02, 6.97930652e-02, 1.43025920e-01, 1.22080731e-01,
                                 1.49352419e-01, 1.33925065e-01], device=device, dtype=torch.float64)

epsilon = 1e-8

clip_threshold = 5.0

joint_def = torch.tensor([ 0.1000, -0.1000,  0.1000,
                        -0.1000,  0.8000,  0.8000,
                        1.0000,  1.0000, -1.5000,
                        -1.5000, -1.5000, -1.5000], device=device, dtype=torch.float64)

# Load neural network for backup policy
#PATH = '/home/jessica/SMPS/MuJoCo_Aligator_Full/backup_policy/test/FULL_STATE__NN_v3.pt'
PATH = '../nn/backup_stop.pt'
dict_policy = torch.load(PATH, map_location=torch.device(device))['policy']

new_keys = ["layers.0.weight", "layers.0.bias", "layers.2.weight", "layers.2.bias",
                "layers.4.weight", "layers.4.bias", "layers.6.weight", "layers.6.bias"]
old_keys = ["net.0.weight", "net.0.bias",      "net.2.weight",      "net.2.bias",
                "net.4.weight", "net.4.bias", "mean_layer.weight", "mean_layer.bias"] 

new_policy_dict = labels_state_dict(dict_policy, old_keys, new_keys)
backup_nn = Backup(running_mean, running_variance, epsilon, clip_threshold, joint_def, device)
backup_nn.load_state_dict(new_policy_dict)
last_action_backup = np.zeros(12)

# Low-level command parameters
TARGET_PORT = 8007
LOCAL_PORT = 8082
TARGET_IP = "192.168.123.10"

LOW_CMD_LENGTH = 610
LOW_STATE_LENGTH = 771

time_file = time.localtime()
nameFile = 'data' + str(time_file.tm_mday) + "_" + str(time_file.tm_mon) + "_" + str(time_file.tm_hour) + "_" + str(time_file.tm_min)

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
    """Function to check if the safety button on the controller is pressed. Y buttoon for corrent joystick"""

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
inference_ready = threading.Event()  # Event to signal new inference results
stop_threads = False  # Flag to stop threads gracefully

# Initialize as recoverable
is_rec = [True]
V_safe = [1.]
V_safe_save = []
joint_des_save = []
time_save = []
joint_pos_save = []
joint_vel_save = []
imu_quat_save = []
imu_vel_save = []

nominal = [True] # Use the nominal policy at the beginning

trot_in_place = True

N_backup = 400 # Maximum number of iterations for the backup policy to be applied
i_backup = 0   # Counter for the number of times the backup policy has been applied

def compute_observation(state, scaling_factors, nominal):
    """
    Compute the observation vector from the robot's state.
    Legs are swapped to match the order of the neural network input.
    SDK order = [FR, FL, RR, RL]
    nn order = [FL, FR, RL, RR]
    """
    #commands = get_commands() # The stopping condition here is not evaluated
    #print(commands)
    #commands = np.array([-0.89997253,  0.,         -1.38256714])
    #commands = np.array([0., 0.1, 0.])
    #commands = np.array([-0.15466003, 0.1, 0.0])
    if trot_in_place:
        if nominal:
            commands = np.array([0.2, 0.2, 0.])
        else:
            commands = np.array([-0.75, 0.3, 0.0])
    else:
        commands = np.array([0.2, 0.2, 0.])
    

    #'''
    #commands = np.array([-0.35, 0.2, 0.0])
    #commands = np.array([-0.2, 0.15, -0.01])
    #commands = np.array([-0.75, 0.3, 0.0])
    #commands = np.array([0.2, 0.2, 0.])
    #commands = np.array([-0.75, 0.25, 0.02])
    
    

    imu = state.imu
    body_quat = np.array([imu.quaternion[1], imu.quaternion[2], imu.quaternion[3], imu.quaternion[0]])
    body_vel = np.array([imu.gyroscope[0], imu.gyroscope[1], imu.gyroscope[2]])
    joint_angles1 = [state.motorState[i].q for i in range(12)]
    joint_angles = swap_legs(joint_angles1)
    joint_velocities1 = [state.motorState[i].dq for i in range(12)]
    joint_velocities = swap_legs(joint_velocities1)

    joint_pos_save.append(joint_angles1)
    joint_vel_save.append(joint_velocities1)
    imu_quat_save.append([imu.quaternion[1], imu.quaternion[2], imu.quaternion[3], imu.quaternion[0]])
    imu_vel_save.append([imu.gyroscope[0], imu.gyroscope[1], imu.gyroscope[2]])

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

def compute_actions(state, scaling_factors, nominal, is_rec):
    """
    Inference ont he nn to retrive actions from observations.
    Legs are swapped to match the order of the neural network input.
    SDK order = [FR, FL, RR, RL]
    nn order = [FL, FR, RL, RR]
    """
    global latest_actions, previous_actions, stop_threads, i_backup, last_action_backup, joint_pos_save, joint_vel_save, imu_quat_save, imu_vel_save, trot_in_place
    while not stop_threads:
        start_time = time.time()
        
        inference_ready.wait()  # Wait for signal from the main thread
        inference_ready.clear()

        if is_rec[0]:
            is_rec[0], V_safe[0] = mps.is_rec_single(state)
            V_safe_save.append([V_safe[0].tolist()])
            #is_rec[0] = True
            if is_rec[0] or trot_in_place:
                obs = compute_observation(state, scaling_factors, is_rec[0])
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                obs_normalized = actor_network.norm_obs(obs_tensor)

                with torch.no_grad():
                    new_actions1 = actor_network(obs_normalized).numpy()

                # Swap the actions to the correct order for SDK
                new_actions = swap_legs(new_actions1)

                with lock:
                    previous_actions[:] = latest_actions  # Store current actions as previous
                    latest_actions[:] = new_actions  # Update latest actions

            else:
                imu = state.imu.accelerometer
                #imu[2] = -imu[2]
                #imu[1] = -imu[1]
                joint_angles = [state.motorState[i].q for i in range(12)]
                joint_velocities = [state.motorState[i].dq for i in range(12)]
                #last_action_backup = (orderPosition(joint_angles) / 0.8) - backup_nn.joint_def.detach().cpu().numpy()
                last_action_backup = computeBackup(joint_angles, joint_velocities, backup_nn, imu, last_action_backup)
                with lock:
                    previous_actions[:] = latest_actions  # Store current actions as previous
                    latest_actions[:] = last_action_backup  # Update latest actions
        
        else:
            i_backup += 1
            if trot_in_place:
                obs = compute_observation(state, scaling_factors, is_rec[0])
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                obs_normalized = actor_network.norm_obs(obs_tensor)

                with torch.no_grad():
                    new_actions1 = actor_network(obs_normalized).numpy()

                # Swap the actions to the correct order for SDK
                new_actions = swap_legs(new_actions1)

                with lock:
                    previous_actions[:] = latest_actions  # Store current actions as previous
                    latest_actions[:] = new_actions  # Update latest actions
            else:
                imu = state.imu.accelerometer
                #imu[2] = -imu[2]
                #imu[1] = -imu[1]
                joint_angles = [state.motorState[i].q for i in range(12)]
                joint_velocities = [state.motorState[i].dq for i in range(12)]
                #if i_backup == 1:
                #    print('1')
                #    last_action_backup = (orderPosition(joint_angles) / 0.8) - backup_nn.joint_def.detach().cpu().numpy()
                last_action_backup = computeBackup(joint_angles, joint_velocities, backup_nn, imu, last_action_backup)
                with lock:
                        previous_actions[:] = latest_actions  # Store current actions as previous
                        latest_actions[:] = last_action_backup  # Update latest actions
        

        #'''
        if i_backup == N_backup:
            i_backup = 0
            is_rec[0] = True
            nominal[0] = True#'''
            if not trot_in_place:
                previous_actions[:] = np.zeros(12) 
                latest_actions[:] = np.zeros(12)
    
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
    
    rate_count = 0

    # PD tuning parameters
    Kp = [100, 100, 100]
    Kd = [3, 3, 3]

    actions = torch.zeros(12, dtype=torch.float32)

    # Decimation factor to reduce the policy update frequency - Number of control action updates @ sim DT per policy DT
    decimation = 4
    if trot_in_place:
        mps = mps_code.MPS('../nn/value_function_vel_python_38.pkl',0.6)
    else:
        mps = mps_code.MPS('../nn/VF_safe_stop_B.pkl',0.1)

    # Initialize the UDP connection
    udp = sdk.UDP(LOCAL_PORT, TARGET_IP, TARGET_PORT, LOW_CMD_LENGTH, LOW_STATE_LENGTH, -1)
    safe = sdk.Safety(sdk.LeggedType.Aliengo)
    # Initialize the command and state objects
    cmd = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)
    cmd.levelFlag = LOWLEVEL

    motiontime = 0

    disable_torques = False  # Flag to disable torques if inclination exceeds threshold or safety button is pressed

    
    # Start the inference thread
    threading.Thread(target=compute_actions, args=(state, scaling_factors, nominal, is_rec), daemon=True).start()

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
      #  print('0-2',state.motorState[0].temperature,state.motorState[1].temperature,state.motorState[2].temperature)
       # print('3-5',state.motorState[3].temperature,state.motorState[4].temperature,state.motorState[5].temperature)
       # print('6-8',state.motorState[6].temperature,state.motorState[7].temperature,state.motorState[8].temperature)
       # print('9-11',state.motorState[9].temperature,state.motorState[10].temperature,state.motorState[11].temperature)

        # Check base inclination and modify Kp, Kd if needed - to disable control torques
        if check_safety_stops(state):  # Using qpos to check inclination
            print("Safety condition triggered, disabling control gains")
            # Set Kp, Kd to 0 (disable control) for safety
            Kp = [0, 0, 0]  # Set Kp to 0 for all joints
            Kd = [0, 0, 0]  # Set Kd to 0 for all joints
            name_save = nameFile + "_VF.csv"
            with open(name_save, 'a', encoding="ISO-8859-1", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(V_safe_save)
            myfile.close()

            name_save = nameFile + "_qDes.csv"
            with open(name_save, 'a', encoding="ISO-8859-1", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(joint_des_save)
            myfile.close()

            name_save = nameFile + "_qDes.csv"
            with open(name_save, 'a', encoding="ISO-8859-1", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(joint_des_save)
            myfile.close()

            name_save = nameFile + "_qDes.csv"
            with open(name_save, 'a', encoding="ISO-8859-1", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(joint_des_save)
            myfile.close()

            name_save = nameFile + "_joint_pos_save.csv"
            with open(name_save, 'a', encoding="ISO-8859-1", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(joint_pos_save)
            myfile.close()

            name_save = nameFile + "_joint_vel_save.csv"
            with open(name_save, 'a', encoding="ISO-8859-1", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(joint_vel_save)
            myfile.close()
            
            name_save = nameFile + "_imu_quat_save.csv"
            with open(name_save, 'a', encoding="ISO-8859-1", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(imu_quat_save)
            myfile.close()
            

            name_save = nameFile + "_imu_vel_save.csv"
            with open(name_save, 'a', encoding="ISO-8859-1", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(imu_vel_save)
            myfile.close()
            

            exit()

        # First, record initial position
        if( motiontime >= 0 and motiontime < 1*(1/dt)):
            # Extract qInit values using dictionary keys
            qInit = [state.motorState[d[key]].q for key in d]

        # second, move to the origin point of a sine movement with Kp Kd
        elif( motiontime >= 1*(1/dt) and motiontime < 7*(1/dt)):
            #exit()
            rate_count += 1
            rate = rate_count / (5*(1/dt))

            # Here I don't switch the legs because the default joint angles are simmetric
            #qDes = [jointLinearInterpolation(qInit[i], default_joint_angles[i], rate) for i in range(12)]
            qDes = [jointLinearInterpolation(qInit[i], sin_mid_q[i], rate) for i in range(12)]
        
        elif( motiontime >= 7*(1/dt)):

            #if motiontime >= 10*(1/dt) and motiontime < 11*(1/dt):
            #    is_rec[0] = False
                #Kp = [25, 25, 25]
                #Kd = [0.5, 0.5, 0.5]
            # Trigger inference every `decimation` steps
            if motiontime % decimation == 0:
                inference_ready.set()

            # Get the latest available actions
            with lock:  
                current_actions = np.copy(latest_actions)

            #print(current_actions)
            if is_rec[0]:
                qDes = 0.5 * current_actions + np.array(default_joint_angles)
                torque_values = [-1.6, 0.0, 0.0]
            else:
                qDes = orderBackup(current_actions) * 0.8 + orderBackup(backup_nn.joint_def)
                torque_values = [0.0, 0.0, 0.0]

        # Clip the joint angles to the joint limits
        for i in range(4):
            qDes[i*3] = np.clip(qDes[i*3], -1.22, 1.22) # Hip joint
            qDes[i*3+1] = np.clip(qDes[i*3+1], 0.0, 1.8) # Thigh joint
            qDes[i*3+2] = np.clip(qDes[i*3+2], -2.78, -0.65) # Calf joint

        if motiontime >= 1*(1/dt):
            joint_des_save.append(np.concatenate([[time.time()],qDes[0:3]]))
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
           
        """ Safety checks"""
        #'''
        safe.PowerProtect(cmd, state, 10)
        safe.PositionLimit(cmd)

        if motiontime > 5*(1/dt):
            safe.PositionProtect(cmd, state, 0.087)#'''

        udp.SetSend(cmd)
        udp.Send()

        # Temporize the loop to maintain the desired frequency
        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        
        # elapsed_time = time.time() - step_start  # Time taken for the loop iteration
        # print(f"Loop took: {elapsed_time:.6f} seconds ({1/elapsed_time:.2f} Hz)")
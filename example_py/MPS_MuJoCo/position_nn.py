#!/usr/bin/python

import numpy as np
import torch
import mps_code
import copy
import rospy
import roslaunch
import publish_subscribe
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
import time

# Neural network and configuration imports
from config_loader import load_config, load_actor_network
from utils import scale_axis, quat_rotate_inverse, swap_legs
import pygame

import threading
import os
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

### Configuration and setup of neural networks

config_path = "config.yaml"
config = load_config(config_path)

# Use simulator or real robot
simulator = config['controller']['robot']['simulator']

# Trot policy
actor_network = load_actor_network(config['policy']['paths']['checkpoint_path'])
scaling_factors = config['policy']['scaling']
default_joint_angles = config['policy']['robot']['default_joint_angles']

max_pos = config['policy']['robot']['max_pos']
min_pos = config['policy']['robot']['min_pos']
lim_tau = config['policy']['robot']['lim_tau']
torque_values_n = config['policy']['robot']['torque_values']
scaling_qdes = scaling_factors['factor']

xml_path = config['controller']['robot']['model']

# Value function
#config_value = mps_code.load_value(config['policy']['paths']['value_function'])

## Shared variables
# For actions
lock = threading.Lock()
latest_actions = np.zeros(12)  # Store the latest actions safely across threads
current_actions = np.zeros(12)  # Store the current actions to be sent to the robot
previous_actions = np.zeros(12)  # Store the previous actions
inference_ready = threading.Event()  # Event to signal new inference results
stop_threads = False  # Flag to stop threads gracefully

# For ROS messages from the robot
torque_values = [np.zeros(12)]
imu_acc = [np.zeros(3)]
imu_quat = [np.zeros(4)]
imu_gyro = [np.zeros(3)]
joint_pos = [np.zeros(12)]
joint_vel = [np.zeros(12)]
pose = [np.zeros(7)]
twist = [np.zeros(6)]

meas_time = []
meas_time2 = []

# Initialize as recoverable
is_rec = [True]
V_safe = [1.]

nominal = [True] # Use the nominal policy at the beginning

N_backup = 400 # Maximum number of iterations for the backup policy to be applied
i_backup = 0   # Counter for the number of times the backup policy has been applied

motiontime = 0

joystick_commands = config['controller']['robot']['joystick_commands']


class ProntoThread(threading.Thread):
    def __init__(self, group = None, target = None, name = None, args = ..., kwargs = None, *, daemon = None):
        super().__init__(group, target, name, args, kwargs, daemon=daemon)

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        self.launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/aliengo_ws/catkin_ws/src/git/pronto_aliengo/pronto_aliengo/launch/pronto_aliengo.launch"])
        self.launch = roslaunch.scriptapi.ROSLaunch()

    def run(self):
        print("HERE WE GO")
        self.launch.start()        

    def join(self):
        print("TIME TO DIE")
        self.launch.stop()


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

def compute_observation(state, scaling_factors, nominal) -> np.ndarray:
    """
    Compute the observation vector from the robot's state.
    Legs are swapped to match the order of the neural network input.
    SDK order = [FR, FL, RR, RL]
    nn order = [FL, FR, RL, RR]
    """    
    # Send specific commands instead of using the controller
    if joystick_commands:
        commands = get_commands()
    else:    
        if nominal:
            commands = np.array([0., 0.1, 0.])
        else:
            commands = np.array([-0.15466003, 0.1, 0.0])#np.array([-0.15466003, 0.15466003, 0.0])

    commands = np.array([0., 0.1, 0.])
    #commands = get_commands()

    #commands = np.array([-0.15466003, 0.1, 0.0])

    imu_quat = state[1]
    imu_gyro = state[2]
    joint_pos = state[3]
    joint_vel = state[4]
    
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

def compute_actions(scaling_factors, nominal, is_rec) -> np.ndarray:    
    """
    Inference on the NN to retrive actions from observations.
    Legs are swapped to match the order of the neural network input.
    SDK order = [FR, FL, RR, RL]
    nn order = [FL, FR, RL, RR]
    """
    global latest_actions, previous_actions, stop_threads, i_backup

    #print('[ ', motiontime, '] first time in compute actions ... ',data_new[3])
    
    num_thread = 0
    while not stop_threads:
    #    print('[ ', motiontime, '] while loop in compute actions before wait ... ',data_new[3])#,data_new[3])
        inference_ready.wait()  # Wait for signal from the main thread
        inference_ready.clear()
     #   print('[ ', motiontime, '] while loop in compute actions after wait ... ')#,data_new[3])
        # MPS check only if the backup has not been activated
        #'''
        if is_rec[0]:
            # MPS
            data_use = copy.copy(data_new)
            is_rec[0], V_safe[0] = mps.is_rec_single(data_use)
            # Compute torque using nominal policy using a copy of the network not to affect the original one
            #actor_network_copy = copy.deepcopy(actor_network)
            #strt_time = time.time()
            
            is_rec[0] = True
            obs = compute_observation(data_use, scaling_factors, is_rec[0])
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            obs_normalized = actor_network.norm_obs(obs_tensor)

            with torch.no_grad():
                #new_actions_numpy = actor_network_copy(obs_normalized).numpy()
                new_actions_numpy = actor_network(obs_normalized).numpy()

            # Compute qDes with the nominal policy
            #'''
            #meas_time.append(time.time()-strt_time)
            #strt_time = time.time()
            
            #meas_time2.append(time.time()-strt_time)
            # Set to true to see how its computation affects the time without
            # switching policies
            
        else:
            i_backup += 1
        
        # Compute actions with the selected policy
            data_use = copy.copy(data_new)
            obs = compute_observation(data_use, scaling_factors, False)
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            obs_normalized = actor_network.norm_obs(obs_tensor)


            with torch.no_grad():
                new_actions_numpy = actor_network(obs_normalized).numpy()

        new_actions = swap_legs(new_actions_numpy)
      #  print("latest actions in compute actions BEFORE writing to global variable:",motiontime)#, latest_actions)#'''

        with lock:
            previous_actions[:] = latest_actions  # Store current actions as previous
            latest_actions[:] = new_actions  # Update latest actions

       # print("latest actions in compute actions AFTER writing to global variable:",motiontime)#, latest_actions)

        # Switch back to the nominal policy after applying the backup for N_backup steps
        if i_backup == N_backup:
                i_backup = 0
                is_rec[0] = True
                nominal[0] = True
        

  #  print('compute actions finished')


def jointLinearInterpolation(initPos, targetPos, rate) -> np.ndarray:
    """
    Performs a linear interpolation between initial and target joint positions.
    """
    rate = np.fmin(np.fmax(rate, 0.0), 1.0)
    p = initPos*(1-rate) + targetPos*rate
    return p

def check_safety_stops(imu_quat) -> bool:
    """
    Check if the inclination of the robot base exceeds the threshold (pi/8) and checks the safety button as well.
    """
    body_quat = imu_quat  # Quaternion from qpos
    # Calculate inclination using arcsin formula
    inclination = 2 * np.arcsin(np.sqrt(body_quat[1]**2 + body_quat[2]**2))

    stop_button = get_safety_button()  # Check if the safety button is pressed

    if pygame.joystick.get_count() != 1:
        return True

    if stop_button or inclination > np.pi/8:
        return True
    else:
        return False
    
if __name__ == '__main__':
    # Initialize pygame and the joystick module if the real robot is used
    if not simulator:
        pygame.init()
        pygame.joystick.init()

        # Check if there is at least one joystick (gamepad) connected
        if pygame.joystick.get_count() == 0:
            print("No joystick connected")
        else:
            joystick = pygame.joystick.Joystick(0)  # Get the first joystick
            joystick.init()
            print(f"Detected joystick: {joystick.get_name()}")

    # ROS communication
    rospy.init_node('communicate_aliengo')
    
    pubSub = publish_subscribe.PubSub()
    pubSub.init_publisher(config['controller']['topics'])
    pubSub.init_subscribers(config['controller']['topics'])

    

    d = {'FR_0':0, 'FR_1': 1, 'FR_2': 2,
         'FL_0':3, 'FL_1': 4, 'FL_2': 5, 
         'RR_0':6, 'RR_1': 7, 'RR_2': 8, 
         'RL_0':9, 'RL_1':10, 'RL_2':11 }

    legs = ['FR', 'FL', 'RR', 'RL']
    joints = ['_0', '_1', '_2']

    # Creates a 12-element list with the default joint angles for the standup
    q0 = 4*[0.0, 0.7, -1.5]#default_joint_angles
    dt = 0.002

    # Initial joint position, typically when the robot is on the ground
    qInit = [0, 0, 0,
             0, 0, 0,
             0, 0, 0,
             0, 0, 0]
    
    # Desired joint position, will be sent to the robot's PD controller
    qDes = [0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0]
    
    rate_count = 0

    # PD tuning parameters
    Kp = config['policy']['robot']['Kp_n']
    Kd = config['policy']['robot']['Kd_n']
    
    actions = torch.zeros(12, dtype=torch.float32)

    # Decimation factor to reduce the policy update frequency - Number of control action updates @ sim DT per policy DT
    # Decimation changed to 5 to have a 100 Hz main loop, like in the simulations
    decimation = config['controller']['robot']['decimation']
    mps = mps_code.MPS(config['policy']['paths']['value_function'],config['policy']['robot']['vf_threshold'])

    

    disable_torques = False  # Flag to disable torques if inclination exceeds threshold or safety button is pressed
    # Start the inference thread
   
    n_wait = 0
    while pubSub.cmd_pub.get_num_connections() < 1:
        n_wait += 1
       
    firstTime = True
    data_new = [np.zeros(3), np.zeros(4), np.zeros(3), np.zeros(12), np.zeros(12), np.zeros(7), np.zeros(6)]
    threading.Thread(target=compute_actions, args=(scaling_factors, nominal, is_rec), daemon=True).start()
    #pronto_thread = ProntoThread()
    rate_ros = rospy.Rate(500)  # 500 Hz for dt = 0.002
    while not rospy.is_shutdown():
        """
        Keeping the dt = 0.002, we need a decimation = 10 to keep the policy update frequency to 50Hz
        The main loop for sending commands is running at 500Hz
        """
        motiontime += 1
     #   print("state in main before copying from pubsub",motiontime)#, data_new)
        # Read data from ROS messages
        data_new = [pubSub.imu_acc, pubSub.imu_quat, pubSub.imu_gyro, pubSub.joint_pos, pubSub.joint_vel, pubSub.pose, pubSub.twist]
     #   print("state in main AFTER copying from pubsub",motiontime)#, data_new)

     #   print('data_new[3]',data_new[3])
     #   print('pubSub.joint_pos',pubSub.joint_pos)
        # Check base inclination and modify Kp, Kd if needed - to disable control torques
        if not simulator and check_safety_stops(pubSub.imu_quat):  # Using qpos to check inclination
            print("Safety condition triggered, disabling control gains",motiontime)
            # Set Kp, Kd to 0 (disable control) for safety
            Kp = 10
            Kd = 1
            pubSub.publish(qDes, np.zeros(12), torque_values*4, Kp, Kd, V_safe[0])
            exit()

        # First, record initial position
        if( motiontime >= 0 and motiontime < 1*(1/dt)):
            # Extract qInit values using dictionary keys
      #     print('[ ', motiontime, '] getting qInit ... ')#,data_new[3])
            qInit = [data_new[3][i] for i in range(12) ]
       #     print('qInit',qInit)

        # second, move to the origin point of a sine movement with Kp Kd
        elif( motiontime >= 1*(1/dt) and motiontime < 7*(1/dt)):
           # exit()
            torque_values = torque_values_n
            rate_count += 1
            rate = rate_count / (5*(1/dt))
       #     print('[ ', motiontime, '] standing up  ... ')#,data_new[3])
            # Here I don't switch the legs because the default joint angles are simmetric
            qDes = [jointLinearInterpolation(qInit[i], q0[i], rate) for i in range(12)]
            qDes = np.clip(qDes, min_pos, max_pos)
       #     print('qDes', qDes)

        
        elif( motiontime >= 7*(1/dt) and motiontime < 17*(1/dt)):
            #exit()
            if firstTime:
                print('STARTING PRONTO IN SEPARATE THREAD!!')
             #   pronto_thread.run()
                firstTime = False

        elif( motiontime >= 17*(1/dt)):
            #exit()
            if motiontime % decimation == 0:
               # print('[ ', motiontime, ' ] decimation!')
                inference_ready.set()
                

        #    print("latest actions in main before deep copy:",motiontime)#, latest_actions)
        #    print("current actions in main before deep copy:",motiontime)#, current_actions)

            # Get the latest available actions 
            with lock: 
                current_actions = np.copy(latest_actions)

         #   print("current actions in main AFTER deep copy:",motiontime)#, current_actions)
                
            # Compute and clip the desired joint angles
            qDes = scaling_qdes * current_actions + np.array(default_joint_angles)
            qDes = np.clip(qDes, min_pos, max_pos)
          
        # Publish commands only after completing the phase in which the initial joint positions are collected
        if(motiontime >= 1*(1/dt)):
            pubSub.publish(qDes, np.zeros(12), torque_values*4, Kp, Kd, V_safe[0])

        # Temporize the loop to maintain the desired frequency
        '''time_rem = rate_ros.remaining().to_nsec()
        if motiontime % decimation == 0 and time_rem < 0:
            print(time_rem)'''
        rate_ros.sleep()

    rospy.spin()
   # pronto_thread.join()
    '''print('nn', sum(meas_time)/len(meas_time))
    print('mps', sum(meas_time2)/len(meas_time2))
    print('mujoco', sum(mps.time1)/len(mps.time1))
    print('value function', sum(mps.time2)/len(mps.time2))'''
    
    
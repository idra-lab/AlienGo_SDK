import yaml
import torch
import torch.nn as nn
import numpy as np
from rl_games.algos_torch.running_mean_std import RunningMeanStd
import time

# Function to load YAML configuration
def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


# Actor Network Class
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, mlp_units=[512, 256, 128], activation=nn.ELU):
        super(ActorNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for unit in mlp_units:
            layers.append(nn.Linear(prev_dim, unit))
            layers.append(activation())
            prev_dim = unit
        self.actor_mlp = nn.Sequential(*layers)
        self.mu = nn.Linear(mlp_units[-1], action_dim)
        self.running_mean_std = RunningMeanStd((input_dim,))

    def forward(self, x):
        features = self.actor_mlp(x)
        mu = self.mu(features)
        return mu

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.running_mean_std(observation)

# Function to load the actor network
def load_actor_network(config):
    input_dim = 45
    action_dim = 12
    actor_network = ActorNetwork(input_dim=input_dim, action_dim=action_dim)
    state_dict = torch.load(config['paths']['checkpoint_path'], map_location={'cuda:1': 'cuda:0'}, weights_only=False)['model']
    actor_state_dict = {k.replace('a2c_network.', ''): v for k, v in state_dict.items()
                        if k.startswith('a2c_network.actor_mlp') or k.startswith('a2c_network.mu')or k.startswith('running_mean_std.running_mean') or k.startswith('running_mean_std.running_var') or k.startswith('running_mean_std.count')}
    actor_network.load_state_dict(actor_state_dict)
    return actor_network

def compute_actions(data_muj, scaling_factors, previous_actions, nominal, actor_network, right):
 #   start = time.time()
    obs = compute_observation(data_muj, scaling_factors, previous_actions, nominal, right)
  #  print('Compute', time.time() - start)
   # start = time.time()
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    #print('Obs Tensor', time.time() - start)
#    start = time.time()
    obs_normalized = actor_network.norm_obs(obs_tensor)
 #   print('Obs normalize', time.time() - start)
  #  start = time.time()
    with torch.no_grad():
        new_actions1 = actor_network(obs_normalized).numpy()
   # print('Actions', time.time() - start)
    return new_actions1

def compute_observation(data, scaling_factors, prev_actions1, nominal, right):
    """
    Compute the observation vector from the robot's state.
    """
    imu_quat = data.sensor('Body_Quat').data.copy()
    imu_gyro = data.sensor('Body_Gyro').data.copy()

    if nominal:
        if right:
            commands = np.array([0., 0.01, -0.3])
        else:
            commands = np.array([0., 0.01, 0.3])
        #commands = np.array([-3.5, -3.5, 0.])
        #commands = np.array([5, -0.7, 0.])
    #    commands = np.array([2.5, 0.5, 0])
        #commands = np.array([2.5, -2, -0.5])
        #commands = np.array([-0.45, -0.02, 0.])
    else:
        commands = np.array([-0.45, -0.02, 0.])
       # commands = np.array([0., 0., 0.])
    
    body_quat = np.array([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]])
    body_vel = np.array([imu_gyro[0], imu_gyro[1], imu_gyro[2]])
 
    joint_angles1 = data.qpos.copy()[7:]#[state.motorState[i].q for i in range(12)]
    joint_angles = swap_legs(joint_angles1)
    joint_velocities1 = data.qvel.copy()[6:]#[state.motorState[i].dq for i in range(12)]
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

def swap_legs(array):
    """
    Swap the front and rear legs of the array based on predefined indices.
    
    The swap logic is fixed:
    - Swap front legs (indices 3:6) with (0:3)
    - Swap rear legs (indices 9:12) with (6:9)
    """
    array_copy = array.copy()  # Make a copy to avoid modifying the original array
    
    '''# Swap front legs (3:6) with (0:3)
    array_copy[0:3] = array[3:6]
    array_copy[3:6] = array[0:3]
    
    # Swap rear legs (9:12) with (6:9)
    array_copy[6:9] = array[9:12]
    array_copy[9:12] = array[6:9]
    
    return array_copy'''
    order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    return array_copy[order]

# Quaternion rotation helper
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c
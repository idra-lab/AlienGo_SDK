import yaml
import torch
import os
import torch.nn as nn
import numpy as np
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from AlienGo_SDK.example_py.MPS_robot_sensors.utils import *


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
def load_backup_network(config, backup_policy):
    input_dim = 45
    action_dim = 12
    actor_network = ActorNetwork(input_dim=input_dim, action_dim=action_dim)
    backup_path = os.environ["LOCOSIM_DIR"] + '/robot_control/AlienGo_SDK/example_py/' + \
                   config['networks']['paths']['trot_backup' + str(backup_policy)]

    state_dict = torch.load(backup_path,
                            map_location=torch.device(config['networks']['device']), weights_only=False)['model']
    actor_state_dict = {k.replace('a2c_network.', ''): v for k, v in state_dict.items()
                        if k.startswith('a2c_network.actor_mlp') or k.startswith('a2c_network.mu') or k.startswith(
            'running_mean_std.running_mean') or k.startswith('running_mean_std.running_var') or k.startswith(
            'running_mean_std.count')}
    actor_network.load_state_dict(actor_state_dict)
    return actor_network


def compute_actions(device, data, scaling_factors, previous_actions, actor_network, policy):
    obs = compute_observation(device, data, scaling_factors, previous_actions, policy)
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    obs_normalized = actor_network.norm_obs(obs_tensor)
    with torch.no_grad():
        new_actions1 = actor_network(obs_normalized).numpy()

    return new_actions1


def compute_observation(device, data, scaling_factors, prev_actions1, policy):
    """
    Compute the observation vector from the robot's state.
    """

    if policy == 1:
        commands = np.array([-0.25, -0.075, 0.])
    elif policy == 2:
        commands = np.array([0., 0., 0.])
     #   commands = np.array([0.15, 0.1, 0.])
    #    commands = np.array([-0.01, 0.0, 0.])

    body_quat = data.imu_quat
    body_vel = data.imu_gyro

    joint_angles = swap_legs(data.joint_pos)
    joint_velocities = swap_legs(data.joint_vel)

    # Gravity vector in body frame
    gravity_body = quat_rotate_inverse(
        torch.tensor(body_quat, device=device, dtype=torch.double).unsqueeze(0),
        torch.tensor([[0.0, 0.0, -1.0]], device=device, dtype=torch.double)
    )
    prev_actions = swap_legs(prev_actions1)

    # Scale observations
    scaled_body_vel = body_vel * scaling_factors['body_ang_vel']
    scaled_commands = commands[:2] * scaling_factors['commands']
    scaled_commands = np.append(scaled_commands, commands[2] * scaling_factors['body_ang_vel'])
    scaled_gravity_body = gravity_body[0].cpu() * scaling_factors['gravity_body']
    scaled_joint_angles = np.array(joint_angles) * scaling_factors['joint_angles']
    scaled_joint_velocities = np.array(joint_velocities) * scaling_factors['joint_velocities']
    scaled_actions = prev_actions * scaling_factors['actions']

    # Concatenate into a single observation vector
    return np.concatenate((scaled_body_vel, scaled_commands, scaled_gravity_body, scaled_joint_angles,
                           scaled_joint_velocities, scaled_actions))
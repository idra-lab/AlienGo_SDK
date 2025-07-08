import yaml
import torch
import torch.nn as nn
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from collections import OrderedDict
import numpy as np


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
    state_dict = torch.load(config['paths']['checkpoint_path'], map_location=torch.device('cpu'), weights_only=False)['model']
    actor_state_dict = {k.replace('a2c_network.', ''): v for k, v in state_dict.items()
                        if k.startswith('a2c_network.actor_mlp') or k.startswith('a2c_network.mu')or k.startswith('running_mean_std.running_mean') or k.startswith('running_mean_std.running_var') or k.startswith('running_mean_std.count')}
    actor_network.load_state_dict(actor_state_dict)
    return actor_network

class Backup(nn.Module):
    def __init__(self,running_mean, running_variance, epsilon, clip_threshold, joint_def, device):
        super(Backup, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(42, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 12)
        ).to(device)

        self.mean = running_mean
        self.variance = running_variance
        self.epsilon = epsilon
        self.threshold = clip_threshold
        self.joint_def = joint_def
        self.device = device
        self.scale = torch.sqrt(running_variance.float()) + epsilon
    
    def forward(self, x):
        return self.layers(x)
    
def labels_state_dict(old_state_dict, old_keys, new_keys):
    new_state_dict = OrderedDict()
    new_dict = 0
    for k, v in old_state_dict.items():
        if k in old_keys:
            name = new_keys[new_dict] # remove `module.`
            new_state_dict[name] = v
            new_dict += 1
    return new_state_dict
########################    
def computeBackup(q_muj, v_muj, backup_nn, imu, last_action):

    """ velocity_command   -->   torch.tensor([0.0, 0.0, 0.0])
        IMU (gravity is considered)
        Joint_Position
        Joint_Velocity
        last_action"""
    vel_comm = np.zeros(3)
    pos_order, vel_order = orderState(q_muj, v_muj)
    state_order = np.concatenate((vel_comm, imu, pos_order, vel_order, last_action))
    state_torch = torch.from_numpy(state_order)
    state_torch = state_torch.to(backup_nn.device, torch.float32)
    state_torch[6:18] = state_torch[6:18] - backup_nn.joint_def

    scaled_state = torch.clamp((state_torch - backup_nn.mean.float()) / backup_nn.scale,
                min=-backup_nn.threshold, max=backup_nn.threshold)
    
    
    new_action = backup_nn.forward(scaled_state) 
    #pos_backup_order = orderBackup((new_action * 0.8) + backup_nn.joint_def)
    
    return new_action.detach().numpy()

def orderState(pos, vel):
    order_pos = [10, 7, 16, 13, 11, 8, 17, 14, 12, 9, 18, 15]
    order_vel = [ 9, 6, 15, 12, 10, 7, 16, 13, 11, 8, 17, 14]
    return np.array([pos[i-7] for i in order_pos]),np.array([vel[i-7] for i in order_vel])
            #pos[order_pos], vel[order_vel]

def orderBackup(pos):
    order_pos = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]
    return np.array([pos[i] for i in order_pos])#.detach().cpu().numpy()

def orderPosition(pos):
    order_pos = [10, 7, 16, 13, 11, 8, 17, 14, 12, 9, 18, 15]
    return np.array([pos[i-7] for i in order_pos])#pos[order_pos]
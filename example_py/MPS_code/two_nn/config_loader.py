import csv
import numpy as np
import yaml
import torch
import torch.nn as nn
from utils import labels_state_dict
from rl_games.algos_torch.running_mean_std import RunningMeanStd


# Function to load YAML configuration
def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


# Actor Network Class
class BackupNetwork(nn.Module):
    def __init__(self, config):
        super(BackupNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(42, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 12)
        )

        scaling_factors = config['scaling']
        self.mean = torch.tensor(scaling_factors['running_mean'], dtype=torch.float32)
        self.threshold = float(scaling_factors['clip_threshold'])
        self.joint_def = torch.tensor(scaling_factors['default_joint_angles'], dtype=torch.float32)
        running_variance = torch.tensor(scaling_factors['running_variance'], dtype=torch.float32)
        epsilon = float(scaling_factors['epsilon'])
        self.scale = torch.sqrt(running_variance.float()) + epsilon
    
    def forward(self, x):
        return self.layers(x)

    def norm_obs(self, observation):
    #    state_order = orderState(q_muj, v_muj)
    #    state_torch = torch.from_numpy(state_order)
     #   state_torch = state_torch.to(backup_nn.device, torch.float32)
        
        observation[6:18] = observation[6:18] - self.joint_def

        normalized_observation = torch.clamp((observation - self.mean.float()) / self.scale,
                    min=-self.threshold, max=self.threshold)
        
        
        #pos_backup = self.forward(scaled_state)
        #pos_backup_order = pos_backup#orderBackup(pos_backup + backup_nn.joint_def)
        
        return normalized_observation


# Function to load the backup network
def load_backup_network(config):
    backup_network = BackupNetwork(config)
    state_dict = torch.load(config['paths']['checkpoint_path'], map_location={'cuda:1': 'cuda:0'})['policy']
    new_keys = ["layers.0.weight", "layers.0.bias", "layers.2.weight",  "layers.2.bias",
                "layers.4.weight", "layers.4.bias", "layers.6.weight",  "layers.6.bias"]
    old_keys = ["net.0.weight",    "net.0.bias",    "net.2.weight",      "net.2.bias",
                "net.4.weight",    "net.4.bias",    "mean_layer.weight", "mean_layer.bias"] 

    backup_state_dict = labels_state_dict(state_dict, old_keys, new_keys)
    backup_network.load_state_dict(backup_state_dict)
    return backup_network

# Nominal Network Class
class NominalNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, mlp_units=[512, 256, 128], activation=nn.ELU):
        super(NominalNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for unit in mlp_units:
            layers.append(nn.Linear(prev_dim, unit))
            layers.append(activation())
            prev_dim = unit
        self.nominal_mlp = nn.Sequential(*layers)
        self.mu = nn.Linear(mlp_units[-1], action_dim)
        self.running_mean_std = RunningMeanStd((input_dim,))

    def forward(self, x):
        features = self.nominal_mlp(x)
        mu = self.mu(features)
        return mu

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.running_mean_std(observation)
        
# Function to load the nominal network
def load_nominal_network(config):
    input_dim = 45
    action_dim = 12
    nominal_network = NominalNetwork(input_dim=input_dim, action_dim=action_dim)
    state_dict = torch.load(config['paths']['checkpoint_path'], map_location={'cuda:1': 'cuda:0'})['model']
    nominal_state_dict = {k.replace('a2c_network.', ''): v for k, v in state_dict.items()
                        if k.startswith('a2c_network.actor_mlp') or k.startswith('a2c_network.mu')or k.startswith('running_mean_std.running_mean') or k.startswith('running_mean_std.running_var') or k.startswith('running_mean_std.count')}
    nominal_network.load_state_dict(nominal_state_dict)
    return nominal_network
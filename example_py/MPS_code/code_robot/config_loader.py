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
class ActorNetwork(nn.Module):
    def __init__(self, config):
        super(ActorNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(37, 256),
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
        observation[13:25] = observation[13:25] - self.joint_def

        scaled_state = torch.clamp((observation - self.mean.float()) / self.scale,
                    min=-self.threshold, max=self.threshold)
        
        
        #pos_backup = self.forward(scaled_state)
        #pos_backup_order = pos_backup#orderBackup(pos_backup + backup_nn.joint_def)
        
        return scaled_state


# Function to load the actor network
def load_actor_network(config):
    actor_network = ActorNetwork(config)
    state_dict = torch.load(config['paths']['checkpoint_path'], map_location={'cuda:1': 'cuda:0'})['policy']
    new_keys = ["layers.0.weight", "layers.0.bias", "layers.2.weight", "layers.2.bias",
                "layers.4.weight", "layers.4.bias", "layers.6.weight", "layers.6.bias"]
    old_keys = ["net.0.weight", "net.0.bias",      "net.2.weight",      "net.2.bias",
                "net.4.weight", "net.4.bias", "mean_layer.weight", "mean_layer.bias"] 

    actor_state_dict = labels_state_dict(state_dict, old_keys, new_keys)
    actor_network.load_state_dict(actor_state_dict)
    return actor_network


def load_mpc_data(config):
    torques = []
    with open(config['paths']['torque_path'],'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for row in lines:
            torques.append(np.array([float(row[0]), float(row[1]),  float(row[2]), float(row[3]),
                                     float(row[4]), float(row[5]),  float(row[6]), float(row[7]),
                                     float(row[8]), float(row[9]), float(row[10]), float(row[11])]))

    vels = []
    with open(config['paths']['velocity_path'],'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for row in lines:
            vels.append(np.array([ float(row[6]),  float(row[7]),  float(row[8]),  float(row[9]),
                                  float(row[10]), float(row[11]), float(row[12]), float(row[13]),
                                  float(row[14]), float(row[15]), float(row[16]), float(row[17])]))


    pos = []
    with open(config['paths']['position_path'],'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for row in lines:
            pos.append(np.array([  float(row[7]),  float(row[8]),  float(row[9]), float(row[10]),
                                  float(row[11]), float(row[12]), float(row[13]), float(row[14]),
                                  float(row[15]), float(row[16]), float(row[17]), float(row[18])]))

    return torques, vels, pos
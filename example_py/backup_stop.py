import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

# Backup policy
class Backup(nn.Module):
    def __init__(self, config):
        """ Neural network to stop the robot using only sensor data """
        super(Backup, self).__init__()
        
        # Data to preprocess state before sending it to the network
        self.device = config['networks']['device']
        self.mean = torch.tensor(config['stop']['scaling']['running_mean'], device=torch.device(self.device), dtype=torch.float64)
        self.threshold = float(config['stop']['scaling']['clip_threshold'])
        self.joint_def = torch.tensor(config['stop']['scaling']['default_joint_angles'], device=torch.device(self.device), dtype=torch.float64)
        self.scaling_factor = float(config['stop']['scaling']['factor'])
        epsilon = float(config['stop']['scaling']['epsilon'])
        running_variance = torch.tensor(config['stop']['scaling']['running_variance'], device=torch.device(self.device), dtype=torch.float64)
        self.scale = torch.sqrt(running_variance.float()) + epsilon

        # Change directory keys
        PATH = config['networks']['paths']['stop']
        old_keys = config['stop']['dictionary']['old_keys']
        new_keys = config['stop']['dictionary']['new_keys']
        dict_policy = torch.load(PATH, map_location=torch.device(self.device), weights_only=True)['policy']
        policy_dict = self.labels_policy_dict(dict_policy, old_keys, new_keys)
        #self.load_state_dict(policy_dict)

        # Define network architechture
        self.layers = nn.Sequential(
            nn.Linear(42, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 12)
        ).to(self.device)

        self.load_state_dict(policy_dict)

    def forward(self, x):
        """ Neural network inference """
        return self.layers(x)
    
    def labels_policy_dict(self, old_state_dict, old_keys, new_keys):
        """ Rename the dictionary keys for the required ones """
        new_policy_dict = OrderedDict()
        new_dict = 0
        for k, v in old_state_dict.items():
            if k in old_keys:
                name = new_keys[new_dict]
                new_policy_dict[name] = v
                new_dict += 1
        return new_policy_dict

    def computeBackup(self, q_muj, v_muj, imu, last_action):
        """ Scale input vector before inference and compute the desired joint positions,
            the network receives the following:
            * Velocity_command   -->   torch.tensor([0.0, 0.0, 0.0])
            * IMU (gravity is considered)
            * Joint_Position
            * Joint_Velocity
            * Last_action """
        
        # Network inputs
        vel_comm = np.zeros(3)
        pos_order = self.orderData(q_muj)
        vel_order = self.orderData(v_muj)
        state_order = np.concatenate((vel_comm, imu, pos_order, vel_order, last_action))
        state_torch = torch.from_numpy(state_order)
        state_torch = state_torch.to(self.device, torch.float32)

        # Offset joint positions as required by the network
        state_torch[6:18] = state_torch[6:18] - self.joint_def

        # Input scaling
        scaled_state = torch.clamp((state_torch - self.mean.float()) / self.scale,
                    min=-self.threshold, max=self.threshold)
        
        # Inference
        new_action = self.forward(scaled_state) 

        # Compute desired joint positions
        pos_backup_order = self.orderBackup((new_action * self.scaling_factor) + self.joint_def)
        
        return pos_backup_order, new_action.detach().cpu().numpy()

    def orderData(self, data):
        """ Change order of joint data from the one given by MuJoCo 
            to the one considered by the network and vice versa"""
        order = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
        order_final = np.zeros(12)
        for i in range(12):
            order_final[i] = data[order[i]]
        return order_final
    
    def orderBackup(self, action):
        order_action = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]
        order_final = np.zeros(12)
        for i in range(12):
            order_final[i] = action[order_action[i]]
        return order_final
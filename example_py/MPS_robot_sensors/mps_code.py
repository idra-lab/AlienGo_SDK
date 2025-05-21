import numpy as np
import mujoco
import torch
import torch.nn as nn
from collections import OrderedDict

def modelData():
    # MuJoCo robot model
    desc_dir = '/aliengo_models'
    xml = desc_dir + '/xml/aliengo.xml'
    spec = mujoco.MjSpec()
    spec.from_file(xml)
    model_muj = spec.compile()
    data_muj = mujoco.MjData(model_muj)

    return model_muj, data_muj

def load_backup_nn(config_b, device):
    """
        Load backup policy and change the dictionary labels.
    """
    PATH = 'backup.pt'
    dict_policy = torch.load(PATH, map_location=torch.device(device))['policy']

    old_keys = ["net.0.weight", "net.0.bias",      "net.2.weight",      "net.2.bias",
                "net.4.weight", "net.4.bias", "mean_layer.weight", "mean_layer.bias"] 
    new_keys = ["layers.0.weight", "layers.0.bias", "layers.2.weight", "layers.2.bias",
                "layers.4.weight", "layers.4.bias", "layers.6.weight", "layers.6.bias"]

    new_policy_dict = labels_state_dict(dict_policy, old_keys, new_keys)
    backup_nn = Backup(config_b, device)
    backup_nn.load_state_dict(new_policy_dict)
    return backup_nn

class Backup(nn.Module):
    def __init__(self, config_b, device):
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

        self.mean = config_b['scaling']['running_mean']
        self.threshold = config_b['scaling']['clip_threshold']
        self.joint_def = config_b['scaling']['joint_def']
        self.scaling_factor = config_b['scaling']['factor']
        running_variance = config_b['scaling']['running_variance']
        epsilon = config_b['scaling']['epsilon']
        self.scale = torch.sqrt(running_variance.float()) + epsilon
        self.device = device

    def forward(self, x):
        return self.layers(x)
    
def labels_state_dict(old_state_dict, old_keys, new_keys):
    """
        Rename the dictionary keys.
    """
    new_state_dict = OrderedDict()
    new_dict = 0
    for k, v in old_state_dict.items():
        if k in old_keys:
            name = new_keys[new_dict]
            new_state_dict[name] = v
            new_dict += 1
    return new_state_dict

def computeBackup(q_data, v_data, backup_nn, imu_acc, last_action):
    """ 
        Use the backup policy to compute the desired joint positions
        to stop the robot.
        The network requires the following inputs:
        * Velocity commands   -->   torch.tensor([0.0, 0.0, 0.0])
        * IMU (gravity is considered)
        * Joint positions
        * Joint velocities
        * Last actions

        The network returns the desired joint positions and the actions
        that generated them.
    """
    vel_comm = np.zeros(3)
    pos_order, vel_order = orderState(q_data, v_data)
    state_order = np.concatenate((vel_comm, imu_acc, pos_order, vel_order, last_action))
    state_torch = torch.from_numpy(state_order)
    state_torch = state_torch.to(backup_nn.device, torch.float32)
    state_torch[6:18] = state_torch[6:18] - backup_nn.joint_def

    scaled_state = torch.clamp((state_torch - backup_nn.mean.float()) / backup_nn.scale,
                min=-backup_nn.threshold, max=backup_nn.threshold)
    
    
    new_action = backup_nn.forward(scaled_state) 
    pos_backup_order = orderBackup((new_action * backup_nn.scaling_factor) + backup_nn.joint_def)
    
    return pos_backup_order, new_action.detach().cpu().numpy()


def orderState(pos, vel):
    """
        Change the order of the joint position and velocity values
        from the one used for the robot and MuJoCo to the one
        required by the network.
    """
    new_order = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
    return pos[new_order], vel[new_order]

def orderBackup(pos):
    """
        Change the order of the joint position values
        from the one the network outputs to the one required by
        the robot and MuJoCo.
    """
    order_pos = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]
    return pos[order_pos].detach().cpu().numpy()

class MPS:
    def __init__(self, decimation, max_pos, min_pos, torque_values, Kp_n, Kd_n, config_b):
        # Compute the number of MuJoCo iterations to use each network output 
        self.iter_ctrl = decimation

        # Parameters for the nominal policy
        self.max_pos = max_pos
        self.min_pos = min_pos
        self.torque_values = torque_values
        self.Kp_n = Kp_n
        self.Kd_n = Kd_n

        # Limits and conditions for the MPS
        self.jmax_compare = np.array([max_pos[0],max_pos[2],max_pos[3],max_pos[5],max_pos[6],max_pos[8],max_pos[9],max_pos[11]])
        self.jmin_compare = np.array([min_pos[0],min_pos[2],min_pos[3],min_pos[5],min_pos[6],min_pos[8],min_pos[9],min_pos[11]])
        self.X_safe = 0.1 # Minimum height not to consider a fall for trunk and hips
        self.N_mps = 200 # Number of MPS simulation steps to decide which policy to use
        self.X_inv = 10e-2 # Maximum velocity to consider the robot has stopped

        # Read parameters for the backup policy from the configuration file
        self.lim_tau = config_b['robot']['torque_limit']
        self.lim_vel = config_b['robot']['vel_limit']
        self.Kp_b = config_b['robot']['Kp_b']
        self.Kd_b = config_b['robot']['Kd_b']

        # Model robot using MuJoCo for the MPS loop
        self.model, self.data = modelData()

        # Load backup network
        device = torch.device('cpu')
        if torch.cuda.is_available():
                device = torch.device('cuda')
        self.backup_nn = load_backup_nn(config_b, device)

    def isRecSingle(self, qDes):
        iter_mps = 0

        data = self.data
        data.qacc_warmstart = 0
        
        # Define initial data for simulation
        data.qpos =
        data.qvel =
        data.qacc = 

        ## Trunk, hip and knee positions
        z_coordinates = np.array([data.body('trunk').xpos[2], data.body('FL_hip').xpos[2], data.body('FR_hip').xpos[2],
                                data.body('RL_hip').xpos[2], data.body('RR_hip').xpos[2]])
        
        data_compare = np.array([data.qpos[7],data.qpos[9],data.qpos[10],data.qpos[12],data.qpos[13],data.qpos[15],data.qpos[16],data.qpos[18]])
        check_pos = (np.any(data_compare > self.jmax_compare) or np.any(data_compare < self.jmin_compare))

        if np.any(z_coordinates < self.X_safe) or np.any(np.abs(data.qvel[6:]) > self.lim_vel) or check_pos: ## x is not in X_safe
            return False, iter_mps
        
        ## Simulate x with pi_hat
        j = 0
        q_muj = data.qpos.copy()
        v_muj = data.qvel.copy()
        torque_values = 4*self.torque_values
      
        while j < self.iter_ctrl:
            u_nominal = self.Kd_n * (- v_muj[6:]) + self.Kp_n * (qDes - q_muj[7:]) +  torque_values
            data.ctrl = np.clip(u_nominal, -self.lim_tau, self.lim_tau)
            j += 1
            mujoco.mj_step(self.model, data)
            q_muj = data.qpos.copy()
            v_muj = data.qvel.copy()
        
        for i in range(0, self.N_mps): ##simulated steps
            z_coordinates = np.array([data.body('trunk').xpos[2], data.body('FL_hip').xpos[2], data.body('FR_hip').xpos[2],
                                    data.body('RL_hip').xpos[2], data.body('RR_hip').xpos[2]])
            data_compare = np.array([data.qpos[7],data.qpos[9],data.qpos[10],data.qpos[12],data.qpos[13],data.qpos[15],data.qpos[16],data.qpos[18]])
            check_pos = (np.any(data_compare > self.jmax_compare) or np.any(data_compare < self.jmin_compare))
            
            if np.all(np.abs(data.qvel) <= self.X_inv): ## x is in X_inv
                return True, iter_mps
        
            elif np.any(z_coordinates < self.X_safe) or np.any(np.abs(data.qvel[6:]) > self.lim_vel) or check_pos: ## x is not in X_safe
                return False, iter_mps
            
            # Simulate x with pi_rec
            imu_acc = data.sensor('Body_Acc').data.copy()
            imu_acc[2] = -imu_acc[2]
            imu_acc[1] = -imu_acc[1]

            pos_backup_order, last_action = computeBackup(q_muj[7:], v_muj[6:], self.backup_nn, imu_acc, last_action)
            iter_mps += 1
            
            j = 0
            while j < self.iter_ctrl:
                j += 1
                u_backup = self.Kd_b * (- v_muj[6:]) + self.Kp_b * (pos_backup_order - q_muj[7:])
                data.ctrl = np.clip(u_backup, -self.lim_tau, self.lim_tau)

                mujoco.mj_step(self.model, data)
                q_muj = data.qpos.copy()
                v_muj = data.qvel.copy()


        return False, iter_mps

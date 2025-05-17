import numpy as np
import mujoco
from walk_sim import swap_legs, compute_actions_sim
from matplotlib.path import Path
import matplotlib
from shapely.geometry import Polygon
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
    pos_backup_order = orderBackup((new_action * 0.8) + backup_nn.joint_def)
    
    return pos_backup_order, new_action.detach().cpu().numpy()


def orderPosition(pos):
    order_pos = [10, 7, 16, 13, 11, 8, 17, 14, 12, 9, 18, 15]
    return pos[order_pos]

def orderState(pos, vel):
    order_pos = [10, 7, 16, 13, 11, 8, 17, 14, 12, 9, 18, 15]
    order_vel = [ 9, 6, 15, 12, 10, 7, 16, 13, 11, 8, 17, 14]
    return pos[order_pos], vel[order_vel]

def orderBackup(pos):
    order_pos = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]
    return pos[order_pos].detach().cpu().numpy()

class MPS:
    def __init__(self, X_safe, iter_ctrl, lim_tau, N_mps, step_rand, force_body, jmax_compare, jmin_compare, lim_vel, default_joint_angles, X_inv):
        self.X_safe = X_safe
        self.iter_ctrl = iter_ctrl
        self.lim_tau = lim_tau
        self.N_mps = N_mps
        self.step_rand = step_rand
        self.force_body = force_body
        self.jmax_compare = jmax_compare
        self.jmin_compare = jmin_compare
        self.lim_vel = lim_vel

        self.default_joint_angles = default_joint_angles
        self.X_inv = X_inv

    def isRecSingle(self, lim_tau, data, curr_step, backup_nn, current_actions):
        iter_mps = 0

        ## Trunk, hip and knee positions
        z_coordinates = np.array([data.body('trunk').xpos[2], data.body('FL_hip').xpos[2], data.body('FR_hip').xpos[2],
                                data.body('RL_hip').xpos[2], data.body('RR_hip').xpos[2]])
        
        data_compare = np.array([data.qpos[7],data.qpos[9],data.qpos[10],data.qpos[12],data.qpos[13],data.qpos[15],data.qpos[16],data.qpos[18]])
        check_pos = (np.any(data_compare > self.jmax_compare) or np.any(data_compare < self.jmin_compare))

        if np.any(z_coordinates < self.X_safe) or np.any(np.abs(data.qvel[6:]) > self.lim_vel) or check_pos: ## x is not in X_safe
            return False, iter_mps
        
        ## Simulate x with pi_hat
        j = 0
        curr_step += 1
        if np.any(data.xfrc_applied[self.force_body][:3]!= np.array([0,0,0])):
            force = True
        else:
            force  = False
        if not (curr_step >= self.step_rand and curr_step < self.step_rand + 20) and force:
            data.xfrc_applied[self.force_body][:3] = np.array([0,0,0])
        q_muj = data.qpos.copy()
        v_muj = data.qvel.copy()
        torque_values = 4*[-1.6, 0.0, 0.0]

        qDes = 0.5 * current_actions + np.array(self.default_joint_angles)
        qDes = np.clip(qDes, [-1.22,0.,-2.78,-1.22,0.,-2.78,-1.22,0.,-2.78,-1.22,0.,-2.78],
                            [1.22,1.8,-0.65,1.22,1.8,-0.65,1.22,1.8,-0.65,1.22,1.8,-0.65])
      
        while j < self.iter_ctrl:
            u_nominal = 3 * (- v_muj[6:]) + 100 * (qDes - q_muj[7:]) +  torque_values
            data.ctrl = np.clip(u_nominal, -lim_tau, lim_tau)
            j += 1
            mujoco.mj_step(data.model, data)
            q_muj = data.qpos.copy()
            v_muj = data.qvel.copy()

        curr_step += 1
        
        for i in range(0, self.N_mps): ##simulated steps
            if not (curr_step >= self.step_rand and curr_step < self.step_rand + 20) and force:# and loop == 1:
                data.xfrc_applied[self.force_body][:3] = np.array([0,0,0])
            z_coordinates = np.array([data.body('trunk').xpos[2], data.body('FL_hip').xpos[2], data.body('FR_hip').xpos[2],
                                    data.body('RL_hip').xpos[2], data.body('RR_hip').xpos[2]])
            data_compare = np.array([data.qpos[7],data.qpos[9],data.qpos[10],data.qpos[12],data.qpos[13],data.qpos[15],data.qpos[16],data.qpos[18]])
            check_pos = (np.any(data_compare > self.jmax_compare) or np.any(data_compare < self.jmin_compare))

            xy_coords = []
            j = 0
            
            if np.all(np.abs(data.qvel) <= self.X_inv): ## x is in X_inv
                return True, iter_mps
        
            elif np.any(z_coordinates < self.X_safe) or np.any(np.abs(data.qvel[6:]) > self.lim_vel) or check_pos: ## x is not in X_safe
                return False, iter_mps
            
            # Simulate x with pi_rec
            imu = data.sensor('Body_Acc').data.copy()
     
            imu[2] = -imu[2]
            imu[1] = -imu[1]
      #  print('imu2', imu)
            pos_backup_order, last_action = computeBackup(q_muj, v_muj, backup_nn, imu, last_action)
            iter_mps += 1
        #print('backup', time.time()-start)
        
            
            j = 0
            while j < self.iter_ctrl:
                j += 1
                u_backup = Kd_nn * (- v_muj[6:]) + Kp_nn * (pos_backup_order - q_muj[7:])
                data.ctrl = np.clip(u_backup, -lim_tau, lim_tau)

                mujoco.mj_step(data.model, data)
                q_muj = data.qpos.copy()
                v_muj = data.qvel.copy()

            curr_step += 1

        return False, iter_mps

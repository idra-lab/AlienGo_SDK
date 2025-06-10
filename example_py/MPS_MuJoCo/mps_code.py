import numpy as np
import mujoco
import torch
import torch.nn as nn
from collections import OrderedDict
import os
from utils import quat_rotate_inverse
import matplotlib
from matplotlib.path import Path
from shapely.geometry import Polygon
import time

def modelData():
    # MuJoCo robot model
    xml = '../MPS_robot_sensors/aliengo_models/xml/aliengo.xml'
    spec = mujoco.MjSpec()
    spec.from_file(xml)
    model_muj = spec.compile()
    data_muj = mujoco.MjData(model_muj)

    return model_muj, data_muj

def load_backup_nn(config, device):
    """
        Load backup policy and change the dictionary labels.
    """
    PATH = os.environ["LOCOSIM_DIR"] + '/robot_control/AlienGo_SDK/example_py/MPS_robot_sensors/' + \
                       config['backup']['paths']['checkpoint_path']
    dict_policy = torch.load(PATH, map_location=torch.device(device))['policy']

    old_keys = ["net.0.weight", "net.0.bias",      "net.2.weight",      "net.2.bias",
                "net.4.weight", "net.4.bias", "mean_layer.weight", "mean_layer.bias"] 
    new_keys = ["layers.0.weight", "layers.0.bias", "layers.2.weight", "layers.2.bias",
                "layers.4.weight", "layers.4.bias", "layers.6.weight", "layers.6.bias"]

    new_policy_dict = labels_state_dict(dict_policy, old_keys, new_keys)
    backup_nn = Backup(config, device)
    backup_nn.load_state_dict(new_policy_dict)
    return backup_nn

class Backup(nn.Module):
    def __init__(self, config, device):
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

        self.mean =  torch.tensor(config['backup']['scaling']['running_mean'], device=torch.device('cpu'), dtype=torch.float64)
        self.joint_def = torch.tensor(config['backup']['scaling']['default_joint_angles'], device=torch.device('cpu'), dtype=torch.float64)
        running_variance = torch.tensor(config['backup']['scaling']['running_variance'], device=torch.device('cpu'), dtype=torch.float64)
        self.scaling_factor = float(config['backup']['scaling']['factor'])
        self.threshold = float(config['backup']['scaling']['clip_threshold'])
        epsilon = float(config['backup']['scaling']['epsilon'])
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

def computeBackup(pose, twist, joint_pos, joint_vel, backup_nn, imu_acc, last_action):
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
    pos_order, vel_order = orderState(np.concatenate((pose, joint_pos)), np.concatenate((twist, joint_vel)))
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
    def __init__(self, decimation, max_pos, min_pos, torque_values, Kp_n, Kd_n, scaling_qdes, default_joint_angles, scaling_factors):
        # Compute the number of MuJoCo iterations to use each network output 
        self.iter_ctrl = 5#decimation
        

        # Parameters for the nominal policy
        self.max_pos = max_pos
        self.min_pos = min_pos
        self.torque_values = 4*torque_values
        self.Kp_n = Kp_n
        self.Kd_n = Kd_n
        self.scaling_qdes = scaling_qdes
        self.default_joint_angles = default_joint_angles
        self.scaling_factors = scaling_factors

        # Limits and conditions for the MPS
        self.jmax_compare = np.array([max_pos[0], max_pos[2], max_pos[3], max_pos[5], max_pos[6], max_pos[8],
                                      max_pos[9], max_pos[11]])
        self.jmin_compare = np.array([min_pos[0], min_pos[2], min_pos[3], min_pos[5], min_pos[6], min_pos[8],
                                      min_pos[9], min_pos[11]])
        self.X_safe = 0.1 # Minimum height not to consider a fall for trunk and hips
        self.N_mps = 50 # Number of MPS simulation steps to decide which policy to use
        self.X_inv = 10e-2 # Maximum velocity to consider the robot has stopped
        self.lim_tau = 40
        self.lim_vel = 26.5
        self.contact_height = 0.03
        self.feet_geom = [12, 20, 36, 28]

        # Model robot using MuJoCo for the MPS loop
        self.model, self.data = modelData()

    def is_rec_single(self, qDes, pose, twist, joint_pos, joint_vel, sim_nn, previous_actions, current_actions):
        iter_mps = 0
       # time_while = 0
        times_checks = 0
        time_nn = 0
        
        start_check = time.time()
        # Define initial data for simulation
        self.data.qpos = np.concatenate((pose, self.swap_legs(joint_pos)))
        self.data.qvel = np.concatenate((twist, self.swap_legs(joint_vel)))
        mujoco.mj_forward(self.model, self.data)

        ## Trunk, hip and knee positions
        z_coordinates = np.array([self.data.body('trunk').xpos[2], self.data.body('FL_hip').xpos[2], self.data.body('FR_hip').xpos[2],
                                self.data.body('RL_hip').xpos[2], self.data.body('RR_hip').xpos[2]])
        
        data_compare = np.array([self.data.qpos[7], self.data.qpos[9], self.data.qpos[10], self.data.qpos[12],
                                 self.data.qpos[13], self.data.qpos[15], self.data.qpos[16], self.data.qpos[18]])
        check_pos = (np.any(data_compare > self.jmax_compare) or np.any(data_compare < self.jmin_compare))

        if np.any(z_coordinates < self.X_safe) or np.any(np.abs(self.data.qvel[6:]) > self.lim_vel) or check_pos: ## x is not in X_safe
            return False, iter_mps
        
        ## Simulate x with pi_hat
        j = 0
        q_muj = self.data.qpos.copy()
        v_muj = self.data.qvel.copy()
        times_checks += (time.time()-start_check)
        #start_while = time.time()
        while j < self.iter_ctrl:
            u_nominal = self.Kd_n * (- v_muj[6:]) + self.Kp_n * (qDes - q_muj[7:]) + self.torque_values
            self.data.ctrl = np.clip(u_nominal, -self.lim_tau, self.lim_tau)
            j += 1
            mujoco.mj_step(self.model, self.data)
            q_muj = self.data.qpos.copy()
            v_muj = self.data.qvel.copy()
        #time_while += (time.time()-start_while)
        nominal = False
        for i in range(0, self.N_mps): ##simulated steps
            start_check = time.time()
            z_coordinates = np.array([self.data.body('trunk').xpos[2], self.data.body('FL_hip').xpos[2], self.data.body('FR_hip').xpos[2],
                                      self.data.body('RL_hip').xpos[2], self.data.body('RR_hip').xpos[2]])
            data_compare = np.array([self.data.qpos[7], self.data.qpos[9], self.data.qpos[10], self.data.qpos[12],
                                     self.data.qpos[13], self.data.qpos[15], self.data.qpos[16], self.data.qpos[18]])
            check_pos = (np.any(data_compare > self.jmax_compare) or np.any(data_compare < self.jmin_compare))

            xy_coords = []

            phase_all = np.all(np.array(
                [self.data.geom(self.feet_geom[0]).xpos[2], self.data.geom(self.feet_geom[1]).xpos[2],
                 self.data.geom(self.feet_geom[2]).xpos[2],
                 self.data.geom(self.feet_geom[3]).xpos[2]]) < self.contact_height)

            if phase_all:  # Check that the four feet are in contact with the floor
                for k in self.feet_geom:
                    xy_coords.append([self.data.geom(k).xpos[0], self.data.geom(k).xpos[1]])

                if self.capture_point_check(self.data, xy_coords):
                    return True, iter_mps

        
            if np.any(z_coordinates < self.X_safe) or np.any(np.abs(self.data.qvel[6:]) > self.lim_vel) or check_pos: ## x is not in X_safe
                return False, iter_mps
            times_checks += (time.time()-start_check)

            start_nn = time.time()
            ## Simulate x with pi_rec
            new_actions1 = self.compute_actions(previous_actions, sim_nn)

            previous_actions = current_actions
            current_actions = self.swap_legs(new_actions1)
            qDes = self.scaling_qdes * current_actions + np.array(self.default_joint_angles)
            qDes = np.clip(qDes, self.min_pos, self.max_pos)
            iter_mps += 1
            time_nn += (time.time()-start_nn)
            print('start_nn', time_nn)
            j = 0
            

            #start_while = time.time()
            while j < self.iter_ctrl:
                j += 1
                u_nominal = self.Kd_n * (- v_muj[6:]) + self.Kp_n * (qDes - q_muj[7:]) + self.torque_values
                self.data.ctrl = np.clip(u_nominal, -self.lim_tau, self.lim_tau)
                mujoco.mj_step(self.model, self.data)
                q_muj = self.data.qpos.copy()
                v_muj = self.data.qvel.copy()
            #print('time_while',time_while)
            print('times_checks',times_checks)
        return False, iter_mps

    def swap_legs(self, array):
        """
        Swap the front and rear legs of the array based on predefined indices.

        The swap logic is fixed:
        - Swap front legs (indices 3:6) with (0:3)
        - Swap rear legs (indices 9:12) with (6:9)
        """
        array_copy = array.copy()  # Make a copy to avoid modifying the original array
        order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        return array_copy[order]

    def compute_observation(self, prev_actions1):
        """
        Compute the observation vector from the robot's state.
        """
        imu_quat = self.data.sensor('Body_Quat').data.copy()
        imu_gyro = self.data.sensor('Body_Gyro').data.copy()

        commands = np.array([-0.45, -0.02, 0.])

        body_quat = np.array([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]])
        body_vel = np.array([imu_gyro[0], imu_gyro[1], imu_gyro[2]])

        joint_angles1 = self.data.qpos.copy()[7:]
        joint_angles = self.swap_legs(joint_angles1)
        joint_velocities1 = self.data.qvel.copy()[6:]
        joint_velocities = self.swap_legs(joint_velocities1)

        # Gravity vector in body frame
        gravity_body = quat_rotate_inverse(
            torch.tensor(body_quat, dtype=torch.float32).unsqueeze(0),
            torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
        ).squeeze().numpy()

        prev_actions = self.swap_legs(prev_actions1)

        # Scale observations
        scaled_body_vel = body_vel * self.scaling_factors['body_ang_vel']
        scaled_commands = commands[:2] * self.scaling_factors['commands']
        scaled_commands = np.append(scaled_commands, commands[2] * self.scaling_factors['body_ang_vel'])
        scaled_gravity_body = gravity_body * self.scaling_factors['gravity_body']
        scaled_joint_angles = np.array(joint_angles) * self.scaling_factors['joint_angles']
        scaled_joint_velocities = np.array(joint_velocities) * self.scaling_factors['joint_velocities']
        scaled_actions = prev_actions * self.scaling_factors['actions']

        # Concatenate into a single observation vector
        return np.concatenate((scaled_body_vel, scaled_commands, scaled_gravity_body, scaled_joint_angles,
                               scaled_joint_velocities, scaled_actions))

    def compute_actions(self, previous_actions, sim_nn):
        obs = self.compute_observation(previous_actions)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        obs_normalized = sim_nn.norm_obs(obs_tensor)
        with torch.no_grad():
            new_actions1 = sim_nn(obs_normalized).numpy()
        return new_actions1

    def capture_point_check(self, data_muj, xy_coords, factor=0.7):
        # Compute the capture point coordinates
        com_coordinates = data_muj.body('trunk').subtree_com
        mujoco.mj_subtreeVel(data_muj.model, data_muj)
        com_velocities = data_muj.body('trunk').subtree_linvel
        omega = np.sqrt(abs(data_muj.model.opt.gravity[2]) / com_coordinates[2])
        cp_x = com_coordinates[0] + (com_velocities[0] / omega)
        cp_y = com_coordinates[1] + (com_velocities[1] / omega)

        # Define convex hull and shrink it
        hull_path = Path(xy_coords)
        polygon = Polygon(hull_path.vertices)
        hull_path = hull_path.transformed(matplotlib.transforms.Affine2D().scale(factor))
        shrinked_polygon = Polygon(hull_path.vertices)

        translate_x = polygon.centroid.x - shrinked_polygon.centroid.x
        translate_y = polygon.centroid.y - shrinked_polygon.centroid.y
        hull_path = hull_path.transformed(matplotlib.transforms.Affine2D().translate(translate_x, translate_y))

        return hull_path.contains_point((cp_x, cp_y))

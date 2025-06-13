import numpy as np
import mujoco
import torch
import torch.nn as nn
from collections import OrderedDict
import os
from utils import quat_rotate_inverse
import time

# Value function network load
import flax.linen as nn_flax
import pickle
import jax
from jax import numpy as jnp
import os
from functools import partial

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

class CriticNetwork(nn_flax.Module):
    @nn_flax.compact
    def __call__(self, x):
        x = nn_flax.Dense(512)(x)
        x = nn_flax.LayerNorm()(x)
        x = nn_flax.elu(x)
        x = nn_flax.Dense(256)(x)
        x = nn_flax.LayerNorm()(x)
        x = nn_flax.elu(x)
        x = nn_flax.Dense(128)(x)
        x = nn_flax.LayerNorm()(x)
        x = nn_flax.elu(x)
        # Output initialized to 1 (probability of survival)
        x = nn_flax.Dense(1, kernel_init=nn_flax.initializers.zeros, bias_init=nn_flax.initializers.ones)(x)
        # x = nn.Dense(1)(x)
        return x.squeeze(-1)
    
class CriticEvaluator:
    def __init__(self, model, params):
        self.apply_fn = model.apply
        self.params = params

def normalize_inputs(obss, mean, std):
    return (obss - mean) / (std + 1e-8)

#@jax.jit
@partial(jax.jit, static_argnames=['critic_model'])
def critic_inference(critic_model, params, obs):
    return critic_model.apply(params, obs)

def modelData():
    # MuJoCo robot model
    xml = '../MPS_robot_sensors/aliengo_models/xml/aliengo.xml'
    model_muj = mujoco.MjModel.from_xml_path(xml)
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
    def __init__(self, decimation, max_pos, min_pos, torque_values, Kp_n, Kd_n, scaling_qdes, default_joint_angles, scaling_factors, config_value):
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
        self.data_value = config_value

        # Limits and conditions for the MPS
        self.jmax_compare = np.array([max_pos[0], max_pos[2], max_pos[3], max_pos[5], max_pos[6], max_pos[8],
                                      max_pos[9], max_pos[11]])
        self.jmin_compare = np.array([min_pos[0], min_pos[2], min_pos[3], min_pos[5], min_pos[6], min_pos[8],
                                      min_pos[9], min_pos[11]])
        self.X_safe = 0.1 # Minimum height not to consider a fall for trunk and hips
        self.N_mps = 5 # Number of MPS simulation steps to decide which policy to use
        self.X_inv = 10e-2 # Maximum velocity to consider the robot has stopped
        self.lim_tau = 40
        self.lim_vel = 26.5
        self.contact_height = 0.03
        self.feet_geom = [12, 20, 36, 28]
        self.grav_tens = torch.tensor([[0., 0., -1.]], device='cpu', dtype=torch.double)

        # Model robot using MuJoCo for the MPS loop
        self.model, self.data = modelData()
        self.setup_value_function()

    def setup_value_function(self):
        # Setup per la critic network
        self.critic_model = CriticNetwork()
        self.params, self.mean, self.std = self.data_value['model_params'], self.data_value['mean'], self.data_value['std']

        # Creiamo l'oggetto per eseguire la valutazione della critic network
        self.critic_network = CriticEvaluator(self.critic_model, self.params)

    def is_rec_single(self, qDes, pose, twist, joint_pos, joint_vel, sim_nn, previous_actions, current_actions):
        value_fnc_result = np.zeros(self.N_mps+1)

        # Define initial data for simulation
        self.data.qpos = np.concatenate((pose, joint_pos))
        self.data.qvel = np.concatenate((twist, joint_vel))
        mujoco.mj_forward(self.model, self.data)

        ## Simulate x with pi_hat
        q_muj = self.data.qpos.copy()
        v_muj = self.data.qvel.copy()
        for j in range(self.iter_ctrl):
            u_nominal = self.Kd_n * (- v_muj[6:]) + self.Kp_n * (qDes - q_muj[7:]) + self.torque_values
            self.data.ctrl = np.clip(u_nominal, -self.lim_tau, self.lim_tau)
            j += 1
            mujoco.mj_step(self.model, self.data)
            q_muj = self.data.qpos.copy()
            v_muj = self.data.qvel.copy()
        nominal = False
        for i in range(0, self.N_mps): ##simulated steps
            value_fnc_result[i] = self.computeValueFnc()

            ## Simulate x with pi_rec
            new_actions1 = self.compute_actions(previous_actions, sim_nn)

            previous_actions = current_actions
            current_actions = self.swap_legs(new_actions1)
            qDes = self.scaling_qdes * current_actions + np.array(self.default_joint_angles)
            qDes = np.clip(qDes, self.min_pos, self.max_pos)
            
            for j in range(self.iter_ctrl):
                u_nominal = self.Kd_n * (- v_muj[6:]) + self.Kp_n * (qDes - q_muj[7:]) + self.torque_values
                self.data.ctrl = np.clip(u_nominal, -self.lim_tau, self.lim_tau)
                mujoco.mj_step(self.model, self.data)
                q_muj = self.data.qpos.copy()
                v_muj = self.data.qvel.copy()

        '''if value_fnc_result[-1] == 1:
            return True
        else:
            return False'''
        value_fnc_result[-1] = self.computeValueFnc()
        if np.any(value_fnc_result == 0):
            return False
        else:
            return True

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
        body_quat_tensor = torch.tensor(body_quat, device='cpu', dtype=torch.double).unsqueeze(0)
        gravity_body = quat_rotate_inverse(body_quat_tensor, self.grav_tens)
        prev_actions = self.swap_legs(prev_actions1)

        # Scale observations
        scaled_body_vel = body_vel * self.scaling_factors['body_ang_vel']
        scaled_commands = commands[:2] * self.scaling_factors['commands']
        scaled_commands = np.append(scaled_commands, commands[2] * self.scaling_factors['body_ang_vel'])
        scaled_gravity_body = gravity_body[0].cpu() * self.scaling_factors['gravity_body']
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
    
    def computeValueFnc(self):
        body_quat_reordered = np.array([self.data.qpos[4], self.data.qpos[5], self.data.qpos[6], self.data.qpos[3]])
        tensor_quat = torch.tensor(body_quat_reordered, device='cpu', dtype=torch.double).unsqueeze(0)
        gravity_body = quat_rotate_inverse(tensor_quat, self.grav_tens)[0].cpu().numpy()
        body_lin_vel_global = self.data.qvel[:3].copy()
        body_ang_vel_global = self.data.qvel[3:6].copy()
        body_lin_vel_tensor = torch.tensor(body_lin_vel_global[None], device='cpu', dtype=torch.double)
        body_lin_vel_local = quat_rotate_inverse(tensor_quat, body_lin_vel_tensor)[0].cpu().numpy()
        vel_tp1 = np.concatenate([body_lin_vel_local, body_ang_vel_global])
        joint_pos_tp1 = self.swap_legs(self.data.qpos[7:].copy())
        joint_vel_tp1 = self.swap_legs(self.data.qvel[6:].copy())
        z_after = self.data.qpos[2]

        obs_flax_np = np.concatenate((
            np.array([z_after], dtype=np.float32),
            gravity_body.astype(np.float32),
            vel_tp1.astype(np.float32),
            joint_pos_tp1.astype(np.float32),
            joint_vel_tp1.astype(np.float32)
        ))

        obs_flax = jnp.array(obs_flax_np)  # <-- qui è jax.numpy array

        # Normalize the observation
        obs_flax = normalize_inputs(obs_flax, self.mean, self.std)

        # V_safe = critic_network.apply_fn(critic_network.params, obs_flax)
        V_safe = critic_inference(self.critic_model, self.critic_network.params, obs_flax)

        if V_safe > 0.9:
            #print(f"\033[92mV_safe: {V_safe:.4f}\033[0m")
            return 1
        else:
            #print(f"\033[91mV_safe: {V_safe:.4f}\033[0m")
            return 0

import numpy as np
import mujoco
import torch
from utils import quat_rotate_inverse
import time

# Value function network load
import flax.linen as nn_flax
import jax
from jax import numpy as jnp
from functools import partial
import pickle

def load_value(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

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

def modelData(xml):
    # MuJoCo robot model
    model_muj = mujoco.MjModel.from_xml_path(xml)
    data_muj = mujoco.MjData(model_muj)

    return model_muj, data_muj

class MPS:
    def __init__(self, decimation, torque_values, Kp_n, Kd_n, config_value, xml_path, lim_tau):
        # Compute the number of MuJoCo iterations to use each network output        
        self.iter_ctrl = decimation

        # Parameters for the nominal policy
        self.torque_values = 4*torque_values
        self.Kp_n = Kp_n
        self.Kd_n = Kd_n
        self.data_value = config_value

        self.lim_tau = lim_tau
        self.grav_tens = torch.tensor([[0., 0., -1.]], device='cpu', dtype=torch.double)

        # Model robot using MuJoCo for the MPS loop
        self.model, self.data = modelData(xml_path)
        self.setup_value_function()
        value_fnc_result = self.computeValueFnc(0.3)
        self.time1 = []
        self.time2 = []
 
    def setup_value_function(self):
        # Setup value function network
        self.critic_model = CriticNetwork()
        self.params, self.mean, self.std = self.data_value['model_params'], self.data_value['mean'], self.data_value['std']

        # Create object to evaluate the value function
        self.critic_network = CriticEvaluator(self.critic_model, self.params)

    def is_rec_single(self, qDes, pose, twist, joint_pos, joint_vel):
        start_time = time.time()
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
            mujoco.mj_step(self.model, self.data)
            q_muj = self.data.qpos.copy()
            v_muj = self.data.qvel.copy()
        self.time1.append(time.time()-start_time)
        start_time = time.time()
        threshold = 0.3
        #time_fnc = time.time()
        value_fnc_result = self.computeValueFnc(threshold)
        self.time2.append(time.time()-start_time)
        return value_fnc_result

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
    
    def computeValueFnc(self, threshold):
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

        obs_flax = jnp.array(obs_flax_np)  # jax.numpy array

        # Normalize the observation
        obs_flax = normalize_inputs(obs_flax, self.mean, self.std)

        V_safe = critic_inference(self.critic_model, self.critic_network.params, obs_flax)

        if V_safe > threshold:
          #  print(f"\033[92mV_safe: {V_safe:.4f}\033[0m")
            return True
        else:
          #  print(f"\033[91mV_safe: {V_safe:.4f}\033[0m")
            return False

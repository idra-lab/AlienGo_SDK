import numpy as np
import mujoco
import torch
import torch.nn as nn

# Value function network load
import flax.linen as nn_flax
import pickle
import jax
from jax import numpy as jnp
import os
from functools import partial

import flax.linen as nn
import pickle
import jax
import jax.numpy as jnp
import yaml
from pathlib import Path
from utils import quat_rotate_inverse, swap_legs


class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(128)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.ones)(x)
        return x.squeeze(-1)


class FlaxCritic:
    def __init__(self, config):

        checkpoint_path = config#["paths"]["safety_vf_path"]
        self._load_model(checkpoint_path)

    def _load_model(self, model_path: str):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        self.params = data['model_params']
        self.mean = jnp.array(data['mean'])
        self.std = jnp.array(data['std'])
        self.model = CriticNetwork()
        self._inference_fn = jax.jit(self._evaluate)

    def _evaluate(self, params, obs):
        normalized_obs = (obs - self.mean) / (self.std + 1e-8)
        return self.model.apply(params, normalized_obs)

    def evaluate(self, obs):
        obs_jnp = jnp.array(obs)
        return self._inference_fn(self.params, obs_jnp)

    def is_safe(self, obs, threshold=0.0):
        return self.evaluate(obs) >= threshold

class MPS:
    def __init__(self, vf_path, threshold):
        self.grav_tens = torch.tensor([[0., 0., -1.]], device='cpu', dtype=torch.double)
        self.vf_path = vf_path
        self.threshold = threshold
        # Model robot using MuJoCo for the MPS loop
        self.setup_value_function()

        # Max pos considering angles insetad of quaternions

    def setup_value_function(self):
        # Setup per la critic network
        self.critic = FlaxCritic(self.vf_path)

    def is_rec_single(self, state):
        #return True, 0.9
        imu_quat = state[1]
        imu_gyro = state[2]
        joint_pos = state[3]
        joint_vel = state[4]
        
        is_rec, value_fnc_result = self.computeValueFnc(joint_pos, joint_vel, imu_quat, imu_gyro)

        return is_rec, value_fnc_result

    
    def computeValueFnc(self, joint_pos, joint_vel, imu_quat, imu_gyro):

        body_quat_reordered = np.array([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]])
        tensor_quat = torch.tensor(body_quat_reordered, device='cpu', dtype=torch.double).unsqueeze(0)
        gravity_body = quat_rotate_inverse(tensor_quat, self.grav_tens)[0].cpu().numpy()

        body_ang_vel = np.array([imu_gyro[0], imu_gyro[1], imu_gyro[2]])
        
        # -------------------------------
        # Legs swap to match network order (see documentation)
        # -------------------------------
        joint_pos = swap_legs([joint_pos[i] for i in range(12)])
        joint_vel = swap_legs([joint_vel[i] for i in range(12)])

        obs_flax_np = np.concatenate((
            gravity_body.astype(np.float32),
            body_ang_vel,
            joint_pos.astype(np.float32),
            joint_vel.astype(np.float32)
        ))

        obs_flax = jnp.array(obs_flax_np)

        V_safe = self.critic.evaluate(obs_flax)

        if V_safe > self.threshold:
            #print(f"\033[92mV_safe: {V_safe:.4f}\033[0m")
            return True, V_safe
        else:
            #print(f"\033[91mV_safe: {V_safe:.4f}\033[0m")
            return False, V_safe
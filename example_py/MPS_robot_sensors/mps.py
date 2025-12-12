import torch
import mujoco
import numpy as np
import jax
import flax.linen as nn_flax
from jax import numpy as jnp
from functools import partial
import pickle
from AlienGo_SDK.example_py.MPS_robot_sensors.utils import *
import os


## Value function network
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
        return x.squeeze(-1)


## State based value function network
class CriticEvaluator:
    def __init__(self, model, params):
        self.apply_fn = model.apply
        self.params = params


def normalize_inputs(obss, mean, std):
    return (obss - mean) / (std + 1e-8)


@partial(jax.jit, static_argnames=['critic_model'])
def critic_inference(critic_model, params, obs):
    return critic_model.apply(params, obs)


## Sensor based value function
class FlaxCritic:
    def __init__(self, config):
        checkpoint_path = config
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
    def __init__(self, config, data=None):

        # Parameters from config file
        self.estimate_lipschitz = config['settings']['tests']['estimate_lipschitz']
        self.device = config['networks']['device']

        self.grav_tens = torch.tensor([[0., 0., -1.]], device=self.device, dtype=torch.double)

        # Variables to save data to estimate Lipschitz constant
        self.dif_vf_min = None
        self.dif_vf_input_min = None
        self.dif_vf_max = None
        self.dif_vf_input_max = None
        self.L_estimate = None
        self.prev_vf = None
        self.prev_vf_input = None
        self.save_L = []

        self.threshold = 10
        self.switch = 0
        self.switch_min = 2
        self.vf_additional_term = config['settings']['tests']['additional_term_vf']
        self.critic = FlaxCritic(config['networks']['paths']['vf_sensors_in_place'])
        if data != None:
            self.computeValueFncSensor(data)
        else:
            self.model, self.data = modelMuJoCo(config['settings']['paths'])
            # self.vf_additional_term = 0.018

            self.computeValueFncSensor()

        self.threshold = config['settings']['tests']['threshold_vf']

    # Function to load value function
    def load_value(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def setup_value_function(self):
        # Setup per la critic network
        self.critic_model = CriticNetwork()
        self.params, self.mean, self.std = self.data_value['model_params'], self.data_value['mean'], self.data_value[
            'std']

        # Creiamo l'oggetto per eseguire la valutazione della critic network
        self.critic_network = CriticEvaluator(self.critic_model, self.params)

    def isRecSingle(self, data_qpos=None, data_qvel=None, data=None):
        if data != None:
            if self.computeValueFncSensor():
                self.switch = 0
            else:
                self.switch += 1
            if self.switch == self.switch_min and not self.estimate_lipschitz:
                return False
            else:
                return True
        else:
            self.data.qpos = data_qpos
            self.data.qvel = data_qvel

            mujoco.mj_forward(self.model, self.data)

            if self.computeValueFncSensor():
                self.switch = 0
            else:
                self.switch += 1
            if self.switch == self.switch_min and not self.estimate_lipschitz:
                return False
            else:
                return True

    def computeValueFncSensor(self, data=None):
        if data != None:
            body_quat = data.imu_quat
            tensor_quat = torch.tensor(body_quat, device=self.device, dtype=torch.double).unsqueeze(0)

            body_ang_vel = data.imu_gyro

            # -------------------------------
            # Legs swap to match network order
            # -------------------------------
            joint_pos = swap_legs(data.joint_pos)
            joint_vel = swap_legs(data.joint_vel)
        else:
            imu_quat = self.data.qpos[3:7].copy()
            body_quat_reordered = np.array([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]])
            tensor_quat = torch.tensor(body_quat_reordered, device=self.device, dtype=torch.double).unsqueeze(0)

            body_ang_vel = self.data.qvel[3:6].copy()
            # -------------------------------
            # Legs swap to match network order (see documentation)
            # -------------------------------
            joint_pos = swap_legs(self.data.qpos[7:].copy())
            joint_vel = swap_legs(self.data.qvel[6:].copy())

        gravity_body = quat_rotate_inverse(tensor_quat, self.grav_tens)[0].cpu().numpy()

        obs_flax_np = np.concatenate((
            gravity_body.astype(np.float32),  # 3
            body_ang_vel,  # 3
            joint_pos.astype(np.float32),  # 12
            joint_vel.astype(np.float32)  # 12
        ))

        obs_flax = jnp.array(obs_flax_np)

        V_safe = self.critic.evaluate(obs_flax)

        if self.threshold < 1 and self.estimate_lipschitz:
            self.lipschitz_constant_estimation(V_safe, obs_flax_np)
            # if V_safe <0.9:
        # print(V_safe)
        # return True
        if V_safe - self.vf_additional_term > self.threshold:
            return True
        else:
            return False

    def lipschitz_constant_estimation(self, V_safe, obs_flax_np):
        if self.prev_vf != None:
            dif_check = abs(obs_flax_np - self.prev_vf_input)
            dif_vf_input = np.linalg.norm(obs_flax_np - self.prev_vf_input)
            dif_vf = np.linalg.norm(V_safe - self.prev_vf)
            L_estimate = dif_vf / dif_vf_input
            if self.dif_vf_max == None:
                self.dif_vf_input_min = dif_vf_input
                self.dif_vf_input_max = dif_vf_input

                self.dif_vf_min = dif_vf
                self.dif_vf_max = dif_vf
                self.L_estimate = np.concatenate(
                    ([L_estimate], [dif_vf_input], [dif_vf], [V_safe], obs_flax_np, [self.prev_vf], self.prev_vf_input))

            else:
                if dif_vf_input < self.dif_vf_input_min:
                    self.dif_vf_input_min = dif_vf_input
                elif dif_vf_input > self.dif_vf_input_max:
                    self.dif_vf_input_max = dif_vf_input

                if dif_vf < self.dif_vf_min:
                    self.dif_vf_min = dif_vf
                elif dif_vf > self.dif_vf_max:
                    self.dif_vf_max = dif_vf

                if L_estimate > self.L_estimate[0]:
                    self.L_estimate = np.concatenate(([L_estimate], [dif_vf_input], [dif_vf], [V_safe], obs_flax_np,
                                                      [self.prev_vf], self.prev_vf_input))

            self.save_L.append(np.concatenate(([self.dif_vf_min], [self.dif_vf_max], [self.dif_vf_input_min],
                                               [self.dif_vf_input_max], [dif_vf_input], [dif_vf], [L_estimate],
                                               [V_safe], obs_flax_np, [np.argmax(dif_check)],
                                               [dif_check[np.argmax(dif_check)]])))
        else:
            self.save_L.append(np.concatenate(([0], [0], [0], [0], [0], [0], [0], [V_safe], obs_flax_np, [0], [0])))
        self.prev_vf_input = obs_flax_np
        self.prev_vf = V_safe
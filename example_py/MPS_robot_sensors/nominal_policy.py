import numpy as np
import copy
import torch
import torch.nn as nn
from AlienGo_SDK.example_py.MPS_robot_sensors.utils  import quat_rotate_inverse
import os

# Trot policy
class Actor(nn.Module):
    def __init__(self, device):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(260, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 12)
        ).to(device)

    def forward(self, x):
        return self.layers(x)


class Estimator(nn.Module):
    def __init__(self, device):
        super(Estimator, self).__init__()
        self.fc1 = nn.Linear(260, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        self.device = device

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NominalPolicy():
    def __init__(self, config):

        self.q_def = np.array(config['robot']['nominal']['default_joint_angles'])
        self.action_scale = config['env_nominal']['action_scale']
        self.order = np.array(config['env_nominal']['order'])
        self.RL_FREQ = 1. / (config['env_nominal']['dt'] * config['env_nominal']['decimation'])
        self.step_freq = 1.4
        self.phase_signal = np.array([0.0, 0.5, 0.5, 0.0])
        self.desired_clip_actions = config['env_nominal']['desired_clip_actions']
        self.device = config['networks']['device']
        self.alpha = 0.8

        self.model_trot = self.load_nominal_network(config)
        self.state_estimator = self.load_state_estimator(config)

        # Variables to save history for nominal policy
        self.history_obs = np.zeros((5, 52), dtype=np.float32)
        self.history_est = np.zeros((5, 52), dtype=np.float32)

    # Functions to load networks
    def load_nominal_network(self, config):
        nominal_path = os.environ["LOCOSIM_DIR"] + '/robot_control/AlienGo_SDK/example_py/nn/' + \
                      config['networks']['paths']['nominal']
        state_dict = torch.load(nominal_path, map_location=torch.device(self.device), weights_only=True)[
            'model_state_dict']
        actor_state_dict = {k.replace('actor.', 'layers.'): v for k, v in state_dict.items()
                            if k.startswith('actor.')}
        actor_network = Actor(torch.device(self.device))
        actor_network.load_state_dict(actor_state_dict)
        actor_network.eval()
        return actor_network

    def load_state_estimator(self, config):
        estimator_path = os.environ["LOCOSIM_DIR"] + '/robot_control/AlienGo_SDK/example_py/nn/' + \
                         config['networks']['paths']['state_estimator']
        state_dict = torch.load(estimator_path, map_location=torch.device(self.device),
                                weights_only=True)['model_state_dict']
        estimator_network = Estimator(torch.device(self.device))
        estimator_network.load_state_dict(state_dict)
        estimator_network.to(torch.device(self.device))
        estimator_network.eval()
        return estimator_network

    def compute_qdes(self, data, prev_action, math_utils):
        velocity_cmd = np.array([0.1, 0, 0.])

        ref_base_lin_vel = np.array([velocity_cmd[0], velocity_cmd[1], 0.])
        ref_base_ang_vel = np.array([0., 0., velocity_cmd[2]])
        h_R_b = math_utils.eul2Rot(np.array([data.euler[0], data.euler[1], 0.]))

        ref_base_lin_vel_h = h_R_b @ ref_base_lin_vel

        quaternion_muj = data.imu_quat
        #body_quat_reordered = np.array([quaternion_muj[1], quaternion_muj[2], quaternion_muj[3], quaternion_muj[0]])
        tensor_quat = torch.tensor(quaternion_muj, device=self.device, dtype=torch.double).unsqueeze(0)
        base_projected_gravity = \
        quat_rotate_inverse(tensor_quat, torch.tensor([[0.0, 0.0, -1.0]], device=self.device, dtype=torch.double))[
            0].cpu().numpy()
        base_vel = data.imu_acc
        base_ang_vel = data.imu_gyro

        joint_angles = data.joint_pos
        joint_velocities = data.joint_vel
        joints_pos_delta = joint_angles - self.q_def

        # fix order Locosim -> Isaac
        joints_pos_delta_ord = np.zeros(12)
        for i in range(12):
            joints_pos_delta_ord[i] = joints_pos_delta[self.order[i]]

        joints_vel = np.zeros(12)
        for i in range(12):
            joints_vel[i] = joint_velocities[self.order[i]]
        obs = np.concatenate([
            base_vel,
            base_ang_vel,
            base_projected_gravity,
            ref_base_lin_vel_h[0:2],
            [ref_base_ang_vel[2]],
            joints_pos_delta_ord,
            joints_vel,
            prev_action.copy()
        ])

        self.phase_signal += self.step_freq * (1 / self.RL_FREQ)
        self.phase_signal = self.phase_signal % 1.0
        obs = np.concatenate((obs, self.phase_signal), axis=0)
        commands = np.array([ref_base_lin_vel_h[0], ref_base_lin_vel_h[1], ref_base_ang_vel[2]], dtype=np.float32)
        if (np.linalg.norm(commands) < 0.01):
            obs[48:52] = -1.0

        prev_est = self.history_est[1:, :]
        self.history_est = np.vstack((prev_est, copy.deepcopy(obs)))
        obs_est = self.history_est.flatten()
        # QUERY THE NETWORK
        base_lin_vel_predicted = self.state_estimator(
            torch.tensor(obs_est, dtype=torch.float32).unsqueeze(0)).detach().numpy().squeeze()
        obs[0:3] = base_lin_vel_predicted
        prev_obs = self.history_obs[1:, :]
        self.history_obs = np.vstack((prev_obs, copy.deepcopy(obs)))
        obs = self.history_obs.flatten()

        obs = obs.reshape(1, -1)
        obs = obs.astype(np.float32)
        action_1 = self.model_trot.forward(torch.from_numpy(obs)).detach().cpu().numpy()[0]
        action_1 = np.clip(action_1, -self.desired_clip_actions, self.desired_clip_actions)

        prev_action_1 = prev_action.copy()
        prev_action = action_1.copy()
        action = self.alpha * action_1 + (1 - self.alpha) * prev_action_1

        # fix order Isaac->Locosim
        action_ord = np.zeros(12)
        for i in range(12):
            action_ord[self.order[i]] = action[i]

        qDes = self.q_def + (self.action_scale * action_ord)
        return qDes, prev_action
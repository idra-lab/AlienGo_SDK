import torch
import numpy as np
from utils import quat_rotate_inverse, swap_legs

def compute_actions_sim(data_muj, scaling_factors, previous_actions, nominal, actor_network):
    obs = compute_observation_sim(data_muj, scaling_factors, previous_actions, nominal)
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    obs_normalized = actor_network.norm_obs(obs_tensor)

    with torch.no_grad():
        new_actions1 = actor_network(obs_normalized).numpy()
        
    return new_actions1

def compute_observation_sim(data, scaling_factors, prev_actions1, nominal):
    """
    Compute the observation vector from the robot's state.
    Using MuJoCo instead of the real robot
    """
    imu_quat = data.sensor('Body_Quat').data.copy()
    imu_gyro = data.sensor('Body_Gyro').data.copy()

    if nominal:
        commands = np.array([0., 0., 0.])
    else:
        commands = np.array([-0.45, -0.02, 0.])
    
    body_quat = np.array([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]])
    body_vel = np.array([imu_gyro[0], imu_gyro[1], imu_gyro[2]])
 
    joint_angles1 = data.qpos.copy()[7:]
    joint_angles = swap_legs(joint_angles1)
    joint_velocities1 = data.qvel.copy()[6:]
    joint_velocities = swap_legs(joint_velocities1)

    # Gravity vector in body frame
    gravity_body = quat_rotate_inverse(
        torch.tensor(body_quat, dtype=torch.float32).unsqueeze(0),
        torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    ).squeeze().numpy()

    prev_actions = swap_legs(prev_actions1)

    # Scale observations
    scaled_body_vel = body_vel * scaling_factors['body_ang_vel']
    scaled_commands = commands[:2] * scaling_factors['commands']
    scaled_commands = np.append(scaled_commands, commands[2] * scaling_factors['body_ang_vel'])
    scaled_gravity_body = gravity_body * scaling_factors['gravity_body']
    scaled_joint_angles = np.array(joint_angles) * scaling_factors['joint_angles']
    scaled_joint_velocities = np.array(joint_velocities) * scaling_factors['joint_velocities']
    scaled_actions = prev_actions * scaling_factors['actions']

    # Concatenate into a single observation vector
    return np.concatenate((scaled_body_vel, scaled_commands, scaled_gravity_body, scaled_joint_angles, scaled_joint_velocities, scaled_actions))
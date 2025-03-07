import numpy as np
from utils import orderState, orderPositionBackup, orderBackup
import mujoco
import torch

def compute_observation_backup(state, latest_actions, default_joint_angles_backup):
    """
    Compute the observation vector from the robot's state.
    Legs are swapped to match the order of the neural network input.
    SDK order = [FR, FL, RR, RL]
    """
    imu = state.imu
    body_acc = np.array([imu.accelerometer[0], imu.accelerometer[1], imu.accelerometer[2]])
    joint_angles1 = [state.motorState[i].q for i in range(12)]
    joint_velocities1 = [state.motorState[i].dq for i in range(12)]

    # Scale observations
    vel_comm = np.zeros(3)
    pos_order, vel_order = orderState(joint_angles1, joint_velocities1)
    latest_actions_scaled = (orderPositionBackup(latest_actions) - np.array(default_joint_angles_backup)) / 0.8

    # Concatenate into a single observation vector
    return np.concatenate((vel_comm, body_acc, pos_order, vel_order, latest_actions_scaled))

def computeBackupSimulation(q_muj, v_muj, imu, last_action, backup_network):
    vel_comm = np.zeros(3)
    pos_order, vel_order = orderState(q_muj, v_muj)
    state_order = np.concatenate((vel_comm, imu, pos_order, vel_order, last_action))
    state_torch = torch.from_numpy(state_order)
    #state_torch = state_torch.to(backup_nn.device, torch.float32)
    state_torch[6:18] = state_torch[6:18] - backup_network.joint_def

    scaled_state = torch.clamp((state_torch - backup_network.mean.float()) / backup_network.scale,
                min=-backup_network.threshold, max=backup_network.threshold)
    
    
    new_action = backup_network.forward(scaled_state) 
    pos_backup_order = orderBackup((new_action * 0.8) + backup_network.joint_def)
    
    return pos_backup_order, new_action.detach().cpu().numpy()

def isRecSingle(min_height, qDes, iter_ctrl, lim_tau, model, data, N_mps, backup_network, Kp_nn, Kd_nn, X_inv, curr_step, step_rand,force_body,jmax_compare,jmin_compare,lim_vel):
    ## Position limits
    X_safe = min_height

    ## Trunk, hip and knee positions
    z_coordinates = np.array([data.body('trunk').xpos[2], data.body('FL_hip').xpos[2], data.body('FR_hip').xpos[2],
                              data.body('RL_hip').xpos[2], data.body('RR_hip').xpos[2]])
     
    data_compare = np.array([data.qpos[7],data.qpos[9],data.qpos[10],data.qpos[12],data.qpos[13],data.qpos[15],data.qpos[16],data.qpos[18]])
    check_pos = (np.any(data_compare > jmax_compare) or np.any(data_compare < jmin_compare))

    if np.any(z_coordinates < X_safe) or np.any(np.abs(data.qvel[6:]) > lim_vel) or check_pos: ## x is not in X_safe
        return False
     
	## Simulate x with pi_hat
    j = 0
    curr_step += 1
    if np.any(data.xfrc_applied[force_body][:3]!= np.array([0,0,0])):
        force = True
    else:
        force  = False
    if not (curr_step >= step_rand and curr_step < step_rand + 20) and force:
        data.xfrc_applied[force_body][:3] = np.array([0,0,0])#'''
    q_muj = data.qpos.copy()
    v_muj = data.qvel.copy()
    torque_values = 4*[-1.6, 0.0, 0.0]
    while j < iter_ctrl:
        u_nominal = 3 * (- v_muj[6:]) + 100 * (qDes - q_muj[7:]) +  torque_values
        data.ctrl = np.clip(u_nominal, -lim_tau, lim_tau)
        j += 1
        mujoco.mj_step(model, data)
        q_muj = data.qpos.copy()
        v_muj = data.qvel.copy()

    curr_step += 1
    last_action = (orderPositionBackup(q_muj)  - backup_network.joint_def.detach().cpu().numpy()) / 0.8
    for i in range(0, N_mps): ##simulated steps
        #mujoco.mj_forward(model, data)
        if not (curr_step >= step_rand and curr_step < step_rand + 20) and force:
            data.xfrc_applied[force_body][:3] = np.array([0,0,0])#'''
        z_coordinates = np.array([data.body('trunk').xpos[2], data.body('FL_hip').xpos[2], data.body('FR_hip').xpos[2],
                                data.body('RL_hip').xpos[2], data.body('RR_hip').xpos[2]])
        data_compare = np.array([data.qpos[7],data.qpos[9],data.qpos[10],data.qpos[12],data.qpos[13],data.qpos[15],data.qpos[16],data.qpos[18]])
        check_pos = (np.any(data_compare > jmax_compare) or np.any(data_compare < jmin_compare))

        if np.all(np.abs(data.qvel) <= X_inv): ## x is in X_inv
            return True
        
        elif np.any(z_coordinates < X_safe) or np.any(np.abs(data.qvel[6:]) > lim_vel) or check_pos: ## x is not in X_safe
            return False
        
        ## Simulate x with pi_rec
        imu = data.sensor('Body_Acc').data.copy() 
     
        imu[2] = -imu[2]
        imu[1] = -imu[1]
        
        pos_backup_order, last_action = computeBackupSimulation(q_muj, v_muj, imu, last_action, backup_network)
        
        j = 0
        
        while j < iter_ctrl:
            j += 1
            u_backup = Kd_nn * (- v_muj[6:]) + Kp_nn * (pos_backup_order - q_muj[7:])
            data.ctrl = np.clip(u_backup, -lim_tau, lim_tau)

            mujoco.mj_step(model, data)
            q_muj = data.qpos.copy()
            v_muj = data.qvel.copy()

        curr_step += 1
    return False
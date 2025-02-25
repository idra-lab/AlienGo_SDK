import numpy as np
import mujoco
import torch
import torch.nn as nn
from collections import OrderedDict
import time

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
########################    
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

def cPD(Kp, Kd, e, e_dot):
    u_k = Kd * e_dot + Kp * e
    return u_k

def isRecSingle(min_height, u_nominal, iter_ctrl, lim_tau, model, data, N_mps, backup_nn, Kp_nn, Kd_nn, X_inv, curr_step, step_rand,force_body,jmax_compare,jmin_compare):
     ## Position limits
     #mujoco.mj_forward(model,data)
     X_safe = min_height
     iter_mps = 0
     curr_step_mujoco = curr_step

     ## Trunk, hip and knee positions
     z_coordinates = np.array([data.body('trunk').xpos[2], data.body('FL_hip').xpos[2], data.body('FR_hip').xpos[2],
                                data.body('RL_hip').xpos[2], data.body('RR_hip').xpos[2]])
     
     z_knees = np.array([data.body('FL_calf').xpos[2], data.body('FR_calf').xpos[2],
                              data.body('RL_calf').xpos[2], data.body('RR_calf').xpos[2]])
     data_compare = np.array([data.qpos[7],data.qpos[9],data.qpos[10],data.qpos[12],data.qpos[13],data.qpos[15],data.qpos[16],data.qpos[18]])
     check_pos = (np.any(data_compare > jmax_compare) or np.any(data_compare < jmin_compare))
     #print('z1',z_coordinates)
     if np.any(z_coordinates < X_safe) or np.any(np.abs(data.qvel[6:]) > 21) or check_pos: ## x is not in X_safe
          #print('z_coordinates',z_coordinates)
          #print('data.qvel[6:]',data.qvel[6:])
          return False, iter_mps
     
	 ## Simulate x with pi_hat
     data.ctrl = u_nominal
     j = 0
     #'''
     #print('u_nominal',u_nominal)
     #print('v_mujA', data.qvel.copy())
     curr_step += 1
     if np.any(data.xfrc_applied[force_body][:3]!= np.array([0,0,0])):
         force = True
     else:
         force  = False
     if not (curr_step >= step_rand and curr_step < step_rand + 20) and force:# and loop == 1:
        data.xfrc_applied[force_body][:3] = np.array([0,0,0])#'''
     while j < iter_ctrl:
        j += 1
        mujoco.mj_step(model, data)
     q_muj = data.qpos.copy()
     v_muj = data.qvel.copy()
     #print('v_mujB', v_muj)

     curr_step += 1
     '''z_coordinates = np.array([data.body('trunk').xpos[2], data.body('FL_hip').xpos[2], data.body('FR_hip').xpos[2],
                                data.body('RL_hip').xpos[2], data.body('RR_hip').xpos[2]])
     print('z2',z_coordinates)'''
     #print('ssssssssssssssssssssssssssssssssssssssssssssss')
     #force_body = model.body('trunk').id
     #last_action = (orderPosition(q_muj) / 0.8) - backup_nn.joint_def.detach().cpu().numpy()
     last_action = (orderPosition(q_muj)  - backup_nn.joint_def.detach().cpu().numpy())/ 0.8#np.zeros(12)
     for i in range(0, N_mps): ##simulated steps
        #mujoco.mj_forward(model, data)
        '''z_coordinates = np.array([data.body('trunk').xpos[2], data.body('FL_hip').xpos[2], data.body('FR_hip').xpos[2],
                                data.body('RL_hip').xpos[2], data.body('RR_hip').xpos[2]])
        print('z3',z_coordinates)'''
        #'''
        if not (curr_step >= step_rand and curr_step < step_rand + 20) and force:# and loop == 1:
            data.xfrc_applied[force_body][:3] = np.array([0,0,0])#'''
        z_coordinates = np.array([data.body('trunk').xpos[2], data.body('FL_hip').xpos[2], data.body('FR_hip').xpos[2],
                                data.body('RL_hip').xpos[2], data.body('RR_hip').xpos[2]])
        z_knees = np.array([data.body('FL_calf').xpos[2], data.body('FR_calf').xpos[2],
                              data.body('RL_calf').xpos[2], data.body('RR_calf').xpos[2]])
        data_compare = np.array([data.qpos[7],data.qpos[9],data.qpos[10],data.qpos[12],data.qpos[13],data.qpos[15],data.qpos[16],data.qpos[18]])
        check_pos = (np.any(data_compare > jmax_compare) or np.any(data_compare < jmin_compare))

        '''print('z4',z_coordinates)'''
        if np.all(np.abs(data.qvel) <= X_inv): ## x is in X_inv
            #print("stopped")
            return True, iter_mps
        
        elif np.any(z_coordinates < X_safe) or np.any(np.abs(data.qvel[6:]) > 21) or check_pos: ## x is not in X_safe
            #print('fall')
            return False, iter_mps
        
        ## Simulate x with pi_rec
        start = time.time()
        imu_init = data.sensor('Body_Acc').data.copy()
        imu_quat = data.sensor('Body_Quat').data.copy()
        imu_mat = np.zeros([9,1])
        imu_quatB = imu_quat.reshape(4,1)
        mujoco.mju_quat2Mat(imu_mat, imu_quatB)
        gravity_projection = imu_mat.reshape(3,3) * data.model.opt.gravity.copy()
        #print(gravity_projection[:,2], data.model.opt.gravity.copy())
        imu = imu_init #+ gravity_projection[:,2] #data.model.opt.gravity.copy()#
    #    print('imu_init',imu_init)
    #    print('gravity_projection[:,2]',gravity_projection[:,2])
     #   print('imu1', imu)
        #imu[2] -= 9.81 ##########works
     #   print('imu2', imu)
        imu[2] = -imu[2]
        imu[1] = -imu[1]
        pos_backup_order, last_action = computeBackup(q_muj, v_muj, backup_nn, imu, last_action)
        iter_mps += 1
        #print('backup', time.time()-start)
        
        j = 0
        start = time.time()
        
        #print(iter_ctrl, 'iter')
        while j < iter_ctrl:
            j += 1
            u_backup = Kd_nn * (- v_muj[6:]) + Kp_nn * (pos_backup_order - q_muj[7:])
            data.ctrl = np.clip(u_backup, -lim_tau, lim_tau)

            mujoco.mj_step(model, data)
            q_muj = data.qpos.copy()
            v_muj = data.qvel.copy()
            #print(data.xfrc_applied[force_body][:3])
        #print('while', time.time()-start)

        curr_step += 1
     return False, iter_mps

def vis(joint_res):
    #'''
    import time
    import aligator_fnc
    from panda3d_viewer import Viewer
    import pinocchio as pin
    from pinocchio.robot_wrapper import RobotWrapper
    from pinocchio.visualize.panda3d_visualizer import Panda3dVisualizer
    dt = 0.01
    desc_dir = '../../aliengo_models'
    shared_urdf = desc_dir + '/urdf/aliengo.urdf'
    model_pin, collision, visual = pin.buildModelsFromUrdf(shared_urdf, desc_dir)
    rdata = RobotWrapper(model_pin, collision, visual)
    userview = input("Enter Y or y to view: ")
    while userview == 'Y' or userview == 'y' or userview == "F" or userview == "f" or userview == "S" or userview == "s":

        #input('press key to continue: ')
        visualize = joint_res
        # Open a Panda3D GUI window
        viewer = Viewer(window_title="aliengo")
        # Attach the robot to the viewer scene
        rdata.setVisualizer(Panda3dVisualizer())
        rdata.initViewer(viewer=viewer)
        rdata.loadViewerModel(group_name=rdata.model.name)
        rdata.viewer.show_floor(True)
        if userview == "F" or userview == "f":
            viewer.reset_camera((5, 0, 0), look_at=(0, 0, 0))
        elif userview == "S" or userview == "s":
            viewer.reset_camera((0, 5, 0), look_at=(0, 0, 0))
       # print(range(len(visualize)))

        for j in range(len(visualize)):
            step_start = time.time()
            time_until_next_step = dt - (time.time() - step_start)
            #time_until_next_step = 0.2 - (time.time() - step_start)
            if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
            # Display configuration
            x_disp = visualize[j]
            q0_init = np.concatenate((x_disp[:3],x_disp[4:7],np.array([x_disp[3]]), aligator_fnc.orderAligatorMujoco(x_disp[7:])))
            #print(i, x_disp)
            rdata.display(q0_init)


        viewer.join()
        userview = input("Enter Y or y to view: ")#'''
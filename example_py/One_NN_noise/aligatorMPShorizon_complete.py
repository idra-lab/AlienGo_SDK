import mujoco
from mujoco import viewer
import time
import numpy as np
import pinocchio as pin
#from aligator import manifolds, dynamics, constraints
#from pinocchio.robot_wrapper import RobotWrapper
#from pinocchio.visualize.panda3d_visualizer import Panda3dVisualizer
import torch
from mps import MPS, capture_point_check
#from collections import OrderedDict
import copy
#from panda3d_viewer import Viewer
import matplotlib.pyplot as plt
import os
from walk import load_config, load_actor_network, swap_legs, compute_observation, compute_actions

#from memory_profiler import profile 
def orderAligatorMujoco(data):
    return np.concatenate((data[3:6], data[:3], data[9:12], data[6:9]))

def modelData():
     # MuJoCo robot model with obstacle
     desc_dir = '../../aliengo_models'
     xml = desc_dir + '/xml/aliengo.xml'
     spec = mujoco.MjSpec()
     spec.from_file(xml)
     model_muj = spec.compile()
    # data_muj = mujoco.MjData(model_muj)
     '''body1 = spec.worldbody.add_body()
     geom = body1.add_geom()
     geom.type = mujoco.mjtGeom.mjGEOM_BOX
     geom.size[0] = 0.1
     geom.size[1] = 1
     geom.size[2] = 0.06
     geom.pos = [2, 0, 1.41]#[10, 10., 10]#[1, 0., 0.47]'''
     #print('a')
     model_muj = spec.compile()
     data_muj = mujoco.MjData(model_muj)
     
     #'''
     
     #print('b')
     

     '''tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
     print('tot_mE',tot_m)
     print('used_mE',used_m)
     print('free_mE',free_m)'''
     # Pinocchio robot model   
     shared_urdf = desc_dir + '/urdf/aliengo.urdf'
     model_pin, collision, visual = pin.buildModelsFromUrdf(shared_urdf, desc_dir)
     data_pin = model_pin.createData()
     return model_muj, data_muj, model_pin, data_pin

########################
def viewMuJoCo(model_muj, data_muj, torques, forces, force_body, userview):

     '''print(data_muj.qpos)
     print(data_muj.qvel)
     print(data_muj.qacc)
     print(data_muj.ctrl)
     print(data_muj.xfrc_applied[force_body])'''
     renderer = mujoco.Renderer(model_muj)
     '''data_muj.qpos = qpos[0]
     data_muj.qvel = qvel[0]
     data_muj.qacc = qacc[0]
     mujoco.mj_forward(model_muj,data_muj)'''

     '''print(data_muj.qpos)
     print(data_muj.qvel)
     print(data_muj.qacc)
     print(data_muj.ctrl)
     print(data_muj.xfrc_applied[force_body])'''

     
     i = 0
     N = len(torques)
     with mujoco.viewer.launch_passive(model_muj, data_muj) as viewer:
          viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
          viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
          viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
          viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
          data_muj.qacc_warmstart = 0
          if userview == "S" or userview == "s":
               viewer.cam.azimuth = 90
               viewer.cam.elevation=0
          elif userview == "F" or userview == "f":
               viewer.cam.azimuth = -180
               viewer.cam.elevation=0
          elif userview == "T" or userview == "t":
               viewer.cam.elevation=270
               viewer.cam.azimuth=0
          elif userview == "X" or userview == "x":
               viewer.cam.azimuth = -180
               viewer.cam.elevation=0
               viewer.cam.lookat=np.array([-0.78903609,  1.29131991,  0.41741882])


          viewer.cam.distance = 2
          while i < N:
               step_start = time.time()
               data_muj.ctrl = torques[i]
               data_muj.xfrc_applied[force_body] = forces[i]
               mujoco.mj_step(model_muj, data_muj)
               renderer.update_scene(data_muj)
               viewer.sync()

               

               time_until_next_step = model_muj.opt.timestep - (time.time() - step_start)
               '''print(i)
               print(data_muj.qpos)'''
               if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
               i += 1
     '''mujoco.mj_step(model_muj, data_muj)
     renderer.update_scene(data_muj)
     viewer.sync()'''
     renderer.close()
     return 

#@profile
def testForces(i_rand, force_rand, force_mag, test_num, model_muj, data_muj, model_pin, data_pin):
     # Save data
     data_pos = []
     data_vel = []
     data_torque = []
     data_fall = []
     data_knee = []
     data_feet = []

     data_mps = []
     data_mps_force = []
     data_mps_no_force = []

     dt = 0.01
     N_mps = 100 # Number of MPS simulation steps to decide which policy to use
     contact_height = 0.03#4#
     lim_tau = 40#35.5#
     lim_vel = 26.5#21.0#

     # Original limits
    # lim_tau = 33.5
    # lim_vel = 21.0
     min_height = 0.1
     min_height_2 = 0.05
     torque_values = 4*[-1.6, 0.0, 0.0]
     
     Kp = 100
     Kd = 3

     # Contact phases and walk parameters
     T_fs = 5
     T_ss = 40

     # Initial configuration
     th_q0 = 0.75
     q0 = np.array([0.,    0.,  0.39175, 0., 0., 0., 1., # trunk
   #  th_q0 = 0.7
   #  q0 = np.array([0.,    0.,  0.39125, 0., 0., 0., 1., # trunk
                    0., th_q0,  -1.5,                    # FL
                    0., th_q0,  -1.5,                    # FR
                    0., th_q0,  -1.5,                    # RL
                    0., th_q0,  -1.5])                   # RR

     jmin = model_pin.lowerPositionLimit
     jmax = model_pin.upperPositionLimit
     
     jmin_muj = orderAligatorMujoco(jmin[7:])
     jmax_muj = orderAligatorMujoco(jmax[7:])

     FR_id = model_pin.getFrameId("FR_foot")
     FL_id = model_pin.getFrameId("FL_foot")
     RR_id = model_pin.getFrameId("RR_foot")
     RL_id = model_pin.getFrameId("RL_foot")

     ## FR, FL, RL, RR
     feet_geom = [12, 20, 36, 28]

     jmax_compare = np.array([jmax_muj[0],jmax_muj[2],jmax_muj[3],jmax_muj[5],jmax_muj[6],jmax_muj[8],jmax_muj[9],jmax_muj[11]])
     jmin_compare = np.array([jmin_muj[0],jmin_muj[2],jmin_muj[3],jmin_muj[5],jmin_muj[6],jmin_muj[8],jmin_muj[9],jmin_muj[11]])
     compare_pos_i =  np.array([0,2,3,5,6,8,9,11])

     # Forward dynamics
     qpos0 = np.concatenate((q0[:3],np.array([q0[6]]),q0[3:6], orderAligatorMujoco(q0[7:])))
     qvel0 = np.zeros(model_pin.nv)
     qacc0 = np.zeros(model_pin.nv)
     data_muj.qpos = qpos0
     data_muj.qvel = qvel0
     data_muj.qacc = qacc0
     data_muj.ctrl = np.zeros(12)#np.clip(torque_values, -lim_tau, lim_tau)

     ######### NN walk
     # Config and neural network setup
     config_path = "config.yaml"
     config = load_config(config_path)
     actor_network = load_actor_network(config)
     scaling_factors = config['scaling']
     default_joint_angles = config['robot']['default_joint_angles']
     
     q_muj = data_muj.qpos.copy()
     v_muj = data_muj.qvel.copy()

     # Number of times the OCP is solved
     N_mpc = 2400*2#4000
     iter_ctrl = int(dt / model_muj.opt.timestep)

     step_rand = (2*T_fs + T_ss + int(T_ss/2) + i_rand)
     #step_rand = i_rand + 80
     loop = 1
     loop_max = 1#2#
     last_i = 0

     # Check which policy should be used  
     start = time.time()
     force_body = model_muj.body('trunk').id
     data_sim = copy.deepcopy(data_muj)
     copy_data = copy.deepcopy(data_muj)
     mujoco.mj_forward(model_muj,copy_data)
     previous_actions = np.zeros(12)
     current_actions = np.zeros(12)
     nominal = True
     actor_network_copy = copy.deepcopy(actor_network)
     right = True
     new_actions1 = compute_actions(data_muj, scaling_factors, previous_actions, nominal, actor_network_copy, right)
     #start_rec = time.time()
     no_noise_all = []
     with_noise_all = []
     mps = MPS(min_height, iter_ctrl, lim_tau, N_mps, step_rand, force_body, jmax_compare, jmin_compare, lim_vel, default_joint_angles, scaling_factors, contact_height, feet_geom)
     is_rec, iter_mps, no_noise, with_noise = mps.isRecSingle(lim_tau, copy_data, 0, actor_network_copy, np.copy(previous_actions), swap_legs(new_actions1))
 #    print('Time rec', time.time() - start_rec)
     data_mps.append([iter_mps])
     '''no_noise_all.append([no_noise])
     with_noise_all.append([with_noise])'''

     if not is_rec:
          backup_used = True
          nominal = False

     new_actions1 = compute_actions(data_muj, scaling_factors, previous_actions, nominal, actor_network, right)
     previous_actions = current_actions
     current_actions = swap_legs(new_actions1)
     qDes = 0.5 * current_actions + np.array(default_joint_angles)
     qDes = np.clip(qDes, [-1.22,0.,-2.78,-1.22,0.,-2.78,-1.22,0.,-2.78,-1.22,0.,-2.78],
                             [1.22,1.8,-0.65,1.22,1.8,-0.65,1.22,1.8,-0.65,1.22,1.8,-0.65])

     torques = []
     qpos = []
     qvel = []
     qacc = []
     forces = []
     forces_all = []

     x_trunk = []
     y_trunk = []
     lin_vel = []

     qpos.append(data_muj.qpos.copy())
     qvel.append(data_muj.qvel.copy())
     qacc.append(data_muj.qacc.copy())

     i = 1
     j = 0

     backup_used = False
     robot_stopped = False
     
     

     max_pos = np.zeros(12)
     min_pos = np.zeros(12)
     max_vel = np.zeros(12)
     min_vel = np.zeros(12)

     
     #####################################
     while i < N_mpc and loop <= loop_max:
          if (i >= step_rand and i < step_rand + 20) and loop == 1:
               data_muj.xfrc_applied[force_body][:3] = force_rand
          else:
               data_muj.xfrc_applied[force_body][:3] = np.array([0,0,0])
          while j < iter_ctrl:
               j += 1
               u = Kd * (- v_muj[6:]) + Kp * (qDes - q_muj[7:]) + torque_values
               data_muj.ctrl = np.clip(u, -lim_tau, lim_tau)
               torques.append(data_muj.ctrl.copy())
               forces.append(data_muj.xfrc_applied[force_body].copy())
               forces_all.append(data_muj.xfrc_applied.copy())
               mujoco.mj_step(model_muj, data_muj)
     
               q_muj = data_muj.qpos.copy()
               v_muj = data_muj.qvel.copy()
               
               if np.any(np.abs(data_muj.ctrl) > lim_tau) and len(data_torque) == 0:
                    data_torque.append([data_muj.ctrl.copy(), force_mag, test_num])             
               
               ## Trunk and hip positions
               coordinates = np.array([data_muj.body('trunk').xpos, data_muj.body('FL_hip').xpos, data_muj.body('FR_hip').xpos,
                              data_muj.body('RL_hip').xpos, data_muj.body('RR_hip').xpos])
               z_coordinates = np.array([coordinates[0][2], coordinates[1][2], coordinates[2][2], coordinates[3][2], coordinates[4][2]])
               if np.any(z_coordinates < min_height) and len(data_fall) == 0:
                    data_fall.append([force_mag, test_num])

               ## Knee positions
               coordinates = np.array([data_muj.body('FL_calf').xpos, data_muj.body('FR_calf').xpos,
                              data_muj.body('RL_calf').xpos, data_muj.body('RR_calf').xpos])
               z_coordinates = np.array([coordinates[0][2], coordinates[1][2], coordinates[2][2], coordinates[3][2]])
               if np.any(z_coordinates < min_height_2) and len(data_knee) == 0:
                    data_knee.append([force_mag, test_num])
               
               qpos.append(q_muj)
               qvel.append(v_muj)
               qacc.append(data_muj.qacc.copy())

               if i == step_rand and loop == 1 and j == 1:
                    q0_init = np.concatenate((q_muj[:3],q_muj[4:7],np.array([q_muj[3]]), orderAligatorMujoco(q_muj[7:])))
                    pin.forwardKinematics(model_pin, data_pin, q0_init)
                    pin.updateFramePlacements(model_pin, data_pin)

                    FR_placement = data_pin.oMf[FR_id]
                    FL_placement = data_pin.oMf[FL_id]
                    RR_placement = data_pin.oMf[RR_id]
                    RL_placement = data_pin.oMf[RL_id]
                    if np.all(np.array([data_muj.geom(feet_geom[0]).xpos[2],data_muj.geom(feet_geom[1]).xpos[2],data_muj.geom(feet_geom[2]).xpos[2],data_muj.geom(feet_geom[3]).xpos[2]]) < contact_height):
                         curr_stage = "ALL"
                    elif np.any(np.array([data_muj.geom(feet_geom[0]).xpos[2],data_muj.geom(feet_geom[1]).xpos[2],data_muj.geom(feet_geom[2]).xpos[2],data_muj.geom(feet_geom[3]).xpos[2]]) < contact_height):
                         curr_stage = ''
                         if data_muj.geom(feet_geom[0]).xpos[2] < contact_height:
                              curr_stage += 'FR_'
                         if data_muj.geom(feet_geom[1]).xpos[2] < contact_height:
                              curr_stage += 'FL_'
                         if data_muj.geom(feet_geom[2]).xpos[2] < contact_height:
                              curr_stage += 'RL_'
                         if data_muj.geom(feet_geom[3]).xpos[2] < contact_height:
                              curr_stage += 'RR'
                         if curr_stage[-1] == '_':
                              curr_stage = curr_stage[:-1]
                    else:
                         curr_stage = 'NONE'

                    data_feet = [curr_stage, FL_placement.translation[2].copy(), FR_placement.translation[2].copy(),
                              RL_placement.translation[2].copy(), RR_placement.translation[2].copy()]

          x_trunk.append(data_muj.body('trunk').xpos[0])
          y_trunk.append(data_muj.body('trunk').xpos[1])
          
          lin_vel.append(np.sqrt(v_muj[0]**2 + v_muj[1]**2))
          
          data_compare = np.array([data_muj.qpos[7],data_muj.qpos[9],data_muj.qpos[10],data_muj.qpos[12],data_muj.qpos[13],data_muj.qpos[15],data_muj.qpos[16],data_muj.qpos[18]])

          check_pos = (np.any(data_compare > jmax_compare) or np.any(data_compare < jmin_compare))
          if check_pos and len(data_pos) == 0:
               data_pos.append([data_muj.qpos[7:].copy(), force_mag, test_num])

          if check_pos:
               k = 0
               for l in data_compare:
                    if l > jmax_compare[k] and l > max_pos[compare_pos_i[k]]:
                         max_pos[compare_pos_i[k]] = l
                    elif l < jmin_compare[k] and l < min_pos[compare_pos_i[k]]:
                         min_pos[compare_pos_i[k]] = l
                    elif l > jmax_compare[k] and max_pos[compare_pos_i[k]] == 0:
                         max_pos[compare_pos_i[k]] = l
                    k += 1

          check_vel = np.any(np.abs(data_muj.qvel[6:]) > lim_vel)
          if check_vel and len(data_vel) == 0:
               data_vel.append([data_muj.qvel[6:].copy(), force_mag, test_num])
          if check_vel:
               k = 0
               for l in data_muj.qvel[6:]:
                    if l > lim_vel and l > max_vel[k]:
                         max_vel[k] = l
                    elif l < -lim_vel and l < min_vel[k]:
                         min_vel[k] = l
                    k += 1
          xy_coords = []
          j = 0
          
          #phase_all = False
          #if np.all(np.array([data_muj.geom(feet_geom[0]).xpos[2],data_muj.geom(feet_geom[1]).xpos[2],data_muj.geom(feet_geom[2]).xpos[2],data_muj.geom(feet_geom[3]).xpos[2]]) < contact_height):
          phase_all = np.all(np.array([data_muj.geom(feet_geom[0]).xpos[2],data_muj.geom(feet_geom[1]).xpos[2],data_muj.geom(feet_geom[2]).xpos[2],data_muj.geom(feet_geom[3]).xpos[2]]) < contact_height)
          #if not is_rec:
           #    print(i, 'step_rand', step_rand)
            #   loop += 1
          if phase_all and not is_rec:# Check that the four feet are in contact with the floor
               for k in feet_geom:
                    xy_coords.append([data_muj.geom(k).xpos[0],data_muj.geom(k).xpos[1]])
               robot_stopped_check = capture_point_check(data_muj, xy_coords)

               actor_network_copy = copy.deepcopy(actor_network)
               new_actions1 = compute_actions(data_muj, scaling_factors, previous_actions, True, actor_network_copy, right)

               # Compute torque using nominal policy       
               # Check which policy should be used
               copy_data = copy.deepcopy(data_muj)
               is_rec_rev, iter_mps, no_noise, with_noise = mps.isRecSingle(lim_tau, copy_data, i, actor_network_copy, np.copy(current_actions), swap_legs(new_actions1))
             #  print('is_rec_rev', is_rec_rev)
               '''no_noise_all.append([no_noise])
               with_noise_all.append([with_noise])'''

               if robot_stopped_check and is_rec_rev:
                   robot_stopped = True
                   is_rec = True
                   last_i = i
               #    print('Stop: i',i,'loop',loop)
               #    print('step_rand', step_rand)
               #    print(np.abs(v_muj))
                   loop += 1
                   i = 0
                   nominal = True
               
          j = 0
          if i > N_mpc / 2:
               right = False
          if is_rec and loop <= loop_max:
               actor_network_copy = copy.deepcopy(actor_network)
               
               new_actions1 = compute_actions(data_muj, scaling_factors, previous_actions, nominal, actor_network_copy, right)

               # Compute torque using nominal policy       
               # Check which policy should be used
               copy_data = copy.deepcopy(data_muj)
               #print('i', i)
               is_rec, iter_mps, no_noise, with_noise = mps.isRecSingle(lim_tau, copy_data, i, actor_network_copy, np.copy(current_actions), swap_legs(new_actions1))
               '''no_noise_all.append([no_noise])
               with_noise_all.append([with_noise])'''
               if not is_rec:
                    print('i',i)
               data_mps.append([iter_mps])
               if i >= step_rand:
                    data_mps_force.append([iter_mps])
               else:
                    data_mps_no_force.append([iter_mps])
               '''if not is_rec:
                    print('no is_rec', i, step_rand)'''
          i += 1

          if not is_rec:
               backup_used = True
               nominal = False

          new_actions1 = compute_actions(data_muj, scaling_factors, previous_actions, nominal, actor_network, right)
          previous_actions = current_actions
          current_actions = swap_legs(new_actions1)
          qDes = 0.5 * current_actions + np.array(default_joint_angles)
          qDes = np.clip(qDes, [-1.22,0.,-2.78,-1.22,0.,-2.78,-1.22,0.,-2.78,-1.22,0.,-2.78],
                             [1.22,1.8,-0.65,1.22,1.8,-0.65,1.22,1.8,-0.65,1.22,1.8,-0.65])
  

     #'''
     userview = input("Enter N or n to stop, T or t for top, F or f for front, S or s side: ")
     while userview != 'N' and userview != 'n':
          print('len(torques)',len(torques))
          copy_data = copy.deepcopy(data_sim)
          viewMuJoCo(model_muj, copy_data, torques, forces, force_body, userview)
          userview = input("Enter N or n to stop, T or t for top, F or f for front, S or s side: ")

     #'''

     print('vel',np.mean(lin_vel))
     circle1 = plt.Circle( (0, -1), 1, fill = False )
     circle2 = plt.Circle( (0, 1), 1, fill = False )
     traj, axs_traj = plt.subplots(1, 1, squeeze=False)
     traj.suptitle('Robot rajectory', fontsize=16)
     axs_traj[0,0].add_artist( circle1 )
     axs_traj[0,0].add_artist( circle2 )
     axs_traj[0,0].plot(x_trunk, y_trunk)

     '''no_noise_all_np = np.array(no_noise_all)
     with_noise_all_np = np.array(with_noise_all)

     trunkP, axs_trunkP = plt.subplots(3, 1, squeeze=False)
     trunkP.suptitle('trunk position', fontsize=16)
     trunkO, axs_trunkO = plt.subplots(3, 1, squeeze=False)
     trunkO.suptitle('trunk orientation', fontsize=16)
     trunkLV, axs_trunkLV = plt.subplots(3, 1, squeeze=False)
     trunkLV.suptitle('trunk linear velocity', fontsize=16)
     trunkAV, axs_trunkAV = plt.subplots(3, 1, squeeze=False)
     trunkAV.suptitle('trunk angular velocity', fontsize=16)
     
     FR_V, axs_FR_V = plt.subplots(3, 1, squeeze=False)
     FR_V.suptitle('FR', fontsize=16)
     FL_V, axs_FL_V = plt.subplots(3, 1, squeeze=False)
     FL_V.suptitle('FL', fontsize=16)
     RR_V, axs_RR_V = plt.subplots(3, 1, squeeze=False)
     RR_V.suptitle('RR', fontsize=16)
     RL_V, axs_RL_V = plt.subplots(3, 1, squeeze=False)
     RL_V.suptitle('RL', fontsize=16)
     for i in range(3):
            axs_trunkP[i,0].plot(no_noise_all_np[:,0+i], linewidth=1)
            axs_trunkP[i,0].plot(with_noise_all_np[:,0+i], linewidth=1)

            axs_trunkO[i,0].plot(no_noise_all_np[:,3+i], linewidth=1)
            axs_trunkO[i,0].plot(with_noise_all_np[:,3+i], linewidth=1)

            axs_trunkLV[i,0].plot(no_noise_all_np[:,6+i], linewidth=1)
            axs_trunkLV[i,0].plot(with_noise_all_np[:,6+i], linewidth=1)

            axs_trunkAV[i,0].plot(no_noise_all_np[:,9+i], linewidth=1)
            axs_trunkAV[i,0].plot(with_noise_all_np[:,9+i], linewidth=1)

            axs_FR_V[i,0].plot(no_noise_all_np[:,12+i], linewidth=1)
            axs_FR_V[i,0].plot(with_noise_all_np[:,12+i], linewidth=1)

            axs_FL_V[i,0].plot(no_noise_all_np[:,15+i], linewidth=1)
            axs_FL_V[i,0].plot(with_noise_all_np[:,15+i], linewidth=1)

            axs_RR_V[i,0].plot(no_noise_all_np[:,18+i], linewidth=1)
            axs_RR_V[i,0].plot(with_noise_all_np[:,18+i], linewidth=1)

            axs_RL_V[i,0].plot(no_noise_all_np[:,21+i], linewidth=1)
            axs_RL_V[i,0].plot(with_noise_all_np[:,21+i], linewidth=1)'''
     

     plt.show()
     

     final_q = data_muj.qpos.copy()
     final_dq = data_muj.qvel.copy()
     '''coordinates = np.array([data_muj.body('trunk').xpos, data_muj.body('FL_hip').xpos, data_muj.body('FR_hip').xpos,
                              data_muj.body('RL_hip').xpos, data_muj.body('RR_hip').xpos])
     z_coordinates = np.array([coordinates[0][2], coordinates[1][2], coordinates[2][2], coordinates[3][2], coordinates[4][2]])
     print(z_coordinates)'''

     return data_pos, data_vel, data_torque, data_fall, data_knee, backup_used, data_feet, robot_stopped, final_q, final_dq, last_i, max_pos, min_pos, max_vel, min_vel, data_mps, data_mps_force, data_mps_no_force

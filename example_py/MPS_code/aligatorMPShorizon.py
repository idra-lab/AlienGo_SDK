import aligator_fnc
import mujoco
from mujoco import viewer
import time
import numpy as np
import pinocchio as pin
#from aligator import manifolds, dynamics, constraints
#from pinocchio.robot_wrapper import RobotWrapper
#from pinocchio.visualize.panda3d_visualizer import Panda3dVisualizer
#import torch
import mps
#from collections import OrderedDict
import copy
#from panda3d_viewer import Viewer
import matplotlib.pyplot as plt
import os

#from memory_profiler import profile 

def modelData():
     # MuJoCo robot model with obstacle
     desc_dir = 'models/aliengo_models'
     xml = desc_dir + '/xml/aliengo.xml'
     spec = mujoco.MjSpec()
     spec.from_file(xml)
     model_muj = spec.compile()
     body1 = spec.worldbody.add_body()
     geom = body1.add_geom()
     geom.type = mujoco.mjtGeom.mjGEOM_BOX
     geom.size[0] = 0.1
     geom.size[1] = 1
     geom.size[2] = 0.06
     geom.pos = [2, 0, 1.41]
     model_muj = spec.compile()
     data_muj = mujoco.MjData(model_muj)
     
     # Pinocchio robot model   
     shared_urdf = desc_dir + '/urdf/aliengo.urdf'
     model_pin, collision, visual = pin.buildModelsFromUrdf(shared_urdf, desc_dir)
     data_pin = model_pin.createData()
     return model_muj, data_muj, model_pin, data_pin, collision, visual

########################
def viewMuJoCo2(model_muj, data_muj, qpos, qvel, qacc, torques, forces, force_body, userview):

     
     renderer = mujoco.Renderer(model_muj)
     data_muj.qpos = qpos[0]
     data_muj.qvel = qvel[0]
     data_muj.qacc = qacc[0]
     mujoco.mj_forward(model_muj,data_muj)
     dt = 0.01
     iter_ctrl = int(dt / model_muj.opt.timestep)
     i = 1
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


          viewer.cam.distance = 2
          j = 0
          while i < N:
               step_start = time.time()
               
               
               while j < iter_ctrl:
                    data_muj.ctrl = torques[i-1]
                    data_muj.xfrc_applied[force_body] = forces[i-1]
                    j += 1
                    mujoco.mj_step(model_muj, data_muj)
                    renderer.update_scene(data_muj)
                    viewer.sync()

               i += 1
               j = 0
               
               print(data_muj.qpos)
               time_until_next_step = dt - (time.time() - step_start)
               if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
               

     renderer.close()
     return 

def viewMuJoCo(model_muj, data_muj, qpos, qvel, qacc, torques, forces, force_body, userview):

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

def uMuJoCo(Kp, Kd, q_al, q_muj, v_al, v_muj, eff_al, lim_tau):
     pd  = Kd * (v_al - v_muj) + Kp * (q_al - q_muj)
     u_muj = eff_al + pd
     u_muj = np.clip(u_muj, -lim_tau, lim_tau)
     return u_muj, pd

def testForces(i_rand, force_rand, force_mag, test_num, backup_nn, model_muj, data_muj_70, model_pin, data_pin, collision, visual, x_warmstart, u_warmstart, ctrl_al, is_rec, prev_stage, qpos_dt, qvel_dt, qacc_dt, torques_dt, forces, view_MuJoCo, data_muj_init):
     # Save data
     data_pos = []
     data_vel = []
     data_torque = []
     data_fall = []
     data_knee = []
     data_feet = []

     data_mps = []
     data_mps_force = []

     obstacle = 100
     dt = 0.01
     N_mps = 200 # Number of MPS simulation steps to decide which policy to use
     lim_tau = 33.5
     lim_vel = 21.0
     min_height = 0.1
     min_height_2 = 0.05

     KpH_nn = 25
     KpT_nn = 25
     KpC_nn = 25
     Kp_nn = np.array([KpH_nn, KpT_nn, KpC_nn,
                         KpH_nn, KpT_nn, KpC_nn,
                         KpH_nn, KpT_nn, KpC_nn,
                         KpH_nn, KpT_nn, KpC_nn])

     KdH_nn = 0.5
     KdT_nn = 0.5
     KdC_nn = 0.5
     Kd_nn = np.array([KdH_nn, KdT_nn, KdC_nn,
                         KdH_nn, KdT_nn, KdC_nn,
                         KdH_nn, KdT_nn, KdC_nn,
                         KdH_nn, KdT_nn, KdC_nn])

     # Contact phases and walk parameters
     T_fs = 5
     T_ss = 40
     x_fwd = 0.1*1
     swing_apex =  0.1

     X_inv = 10e-2

     # Initial configuration
     th_q0 = 0.75
     q0_target = np.array([0.,    0.,  0.39175, 0., 0., 0., 1., # trunk
                    0., th_q0,  -1.5,                    # FL
                    0., th_q0,  -1.5,                    # FR
                    0., th_q0,  -1.5,                    # RL
                    0., th_q0,  -1.5])                   # RR
    
     data_muj = copy.deepcopy(data_muj_70)

     jmin = model_pin.lowerPositionLimit
     jmax = model_pin.upperPositionLimit

     FR_id = model_pin.getFrameId("FR_foot")
     FL_id = model_pin.getFrameId("FL_foot")
     RR_id = model_pin.getFrameId("RR_foot")
     RL_id = model_pin.getFrameId("RL_foot")

     
     jmin_muj = aligator_fnc.orderAligatorMujoco(jmin[7:])
     jmax_muj = aligator_fnc.orderAligatorMujoco(jmax[7:])

     jmax_compare = np.array([jmax_muj[0],jmax_muj[2],jmax_muj[3],jmax_muj[5],jmax_muj[6],jmax_muj[8],jmax_muj[9],jmax_muj[11]])
     jmin_compare = np.array([jmin_muj[0],jmin_muj[2],jmin_muj[3],jmin_muj[5],jmin_muj[6],jmin_muj[8],jmin_muj[9],jmin_muj[11]])
     compare_pos_i =  np.array([0,2,3,5,6,8,9,11])
     q0 = data_muj.qpos.copy()
     dq0 = data_muj.qvel.copy()

     q0_init = np.concatenate((q0[:3],q0[4:7],np.array([q0[3]]), aligator_fnc.orderAligatorMujoco(q0[7:])))
     dq0_init = np.concatenate((dq0[:6], aligator_fnc.orderAligatorMujoco(dq0[6:])))

     ctrl_al.control_step(q0_init, dq0_init, prev_stage)
     add_obstacle = False
     remove_obstacle = False
     align = False
     stages = ctrl_al.phaseDefinition(T_fs, T_ss, x_fwd, swing_apex, 2*T_fs + T_ss + int(T_ss/2), add_obstacle, remove_obstacle, q0_target, align)

     initial_state = np.concatenate((q0_init, dq0_init))
     start = time.time()

     dist_plot = []
     x_plot = []
     z_plot = []
     conv_plot = []
     conv_color = []
     setup_time_all = []
     run_time_all = []

     solve_time_all = []
     isrec_time_all = []

     start = time.time()
     joint_res, joint_vels, x_res, eff_res, prev_stage, curr_stage, conv, setup_time, run_time = ctrl_al.solveAligator(initial_state, stages, x_warmstart, u_warmstart)
     solve_time_all.append(time.time()-start)
     setup_time_all.append(setup_time)
     run_time_all.append(run_time)
     conv_T = 0
     conv_F = 0
     if conv:
          conv_color.append('blue')
          conv_plot.append(1)
          conv_T +=1
     else:
          conv_color.append('red')
          conv_plot.append(0)
          conv_F +=1

     # Forward dynamics
     eff_al = aligator_fnc.orderAligatorMujoco(eff_res[0])

     q_muj = data_muj.qpos.copy()
     v_muj = data_muj.qvel.copy()

     # Number of times the OCP is solved
     N_mpc = 2000
     u = np.clip(eff_al, -lim_tau, lim_tau)
     iter_ctrl = int(dt / model_muj.opt.timestep)
     
     force_body = model_muj.body('trunk').id
     data_sim = copy.deepcopy(data_muj)
     copy_data = copy.deepcopy(data_muj)
     
     is_rec, iter_mps = mps.isRecSingle(min_height, u, iter_ctrl, lim_tau, model_muj, copy_data, N_mps, backup_nn, Kp_nn, Kd_nn, X_inv,0,0,force_body)
     data_mps.append([iter_mps])
   #  print('iter_mps', iter_mps)
     #print('isRec',is_rec)
     #isrec_time_all.append(time.time()-start)
     del copy_data

     if not is_rec:#False:#
        pos_backup_order = mps.computeBackup(q_muj, v_muj, backup_nn)
     
     torques = []
     qpos = []
     qvel = []
     qacc = []
     forcesB = []
     forces_all = []

     qpos.append(data_muj.qpos.copy())
     qvel.append(data_muj.qvel.copy())
     qacc.append(data_muj.qacc.copy())

     i = 2*T_fs + T_ss + int(T_ss/2) + 1
     j = 0

     backup_used = False
     robot_stopped = False

     step_rand = (2*T_fs + T_ss + int(T_ss/2) + i_rand) 
     loop = 1
     loop_max = 1
     last_i = 0

     max_pos = np.zeros(12)
     min_pos = np.zeros(12)
     max_vel = np.zeros(12)
     min_vel = np.zeros(12)

     while i < N_mpc and loop <= loop_max:
          qpos_dt.append(q_muj)
          qvel_dt.append(v_muj)
          qacc_dt.append(data_muj.qacc.copy())
          if (i >= step_rand and i < step_rand + 20) and loop == 1:
               data_muj.xfrc_applied[force_body][:3] = force_rand
          else:
               data_muj.xfrc_applied[force_body][:3] = np.array([0,0,0])
          forces.append(data_muj.xfrc_applied[force_body].copy())
          while j < iter_ctrl:
               
               j += 1
               if not is_rec:
                    u_backup = Kd_nn * (- v_muj[6:]) + Kp_nn * (pos_backup_order - q_muj[7:])
                    u = np.clip(u_backup, -lim_tau, lim_tau)

               data_muj.ctrl = u
               torques.append(data_muj.ctrl.copy())
               forcesB.append(data_muj.xfrc_applied[force_body].copy())
               forces_all.append(data_muj.xfrc_applied.copy())
               mujoco.mj_step(model_muj, data_muj)

               q_muj = data_muj.qpos.copy()
               v_muj = data_muj.qvel.copy()
               
               if np.any(np.abs(data_muj.ctrl) > lim_tau) and len(data_torque) == 0:
                    data_torque.append([data_muj.ctrl.copy(), force_mag, test_num])
               data_compare = np.array([data_muj.qpos[7],data_muj.qpos[9],data_muj.qpos[10],data_muj.qpos[12],data_muj.qpos[13],data_muj.qpos[15],data_muj.qpos[16],data_muj.qpos[18]])

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
                    q0_init = np.concatenate((q_muj[:3],q_muj[4:7],np.array([q_muj[3]]), aligator_fnc.orderAligatorMujoco(q_muj[7:])))
                    pin.forwardKinematics(model_pin, data_pin, q0_init)#, dq0, acc0)
                    pin.updateFramePlacements(model_pin, data_pin)

                    FR_placement = data_pin.oMf[FR_id]
                    FL_placement = data_pin.oMf[FL_id]
                    RR_placement = data_pin.oMf[RR_id]
                    RL_placement = data_pin.oMf[RL_id]

                    data_feet = [curr_stage, FL_placement.translation[2].copy(), FR_placement.translation[2].copy(),
                              RL_placement.translation[2].copy(), RR_placement.translation[2].copy()]

          torques_dt.append(data_muj.ctrl.copy())

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

          if np.all(np.abs(v_muj) <= X_inv) and not is_rec:
               is_rec = True
               last_i = i
               loop += 1

               align = True
               i = 0
               prev_stage = 'ALL'
               robot_stopped = True
          j = 0
          if is_rec and loop <= loop_max:

               q0_init = np.concatenate((q_muj[:3],q_muj[4:7],np.array([q_muj[3]]), aligator_fnc.orderAligatorMujoco(q_muj[7:])))
               dq0_init = np.concatenate((v_muj[:6], aligator_fnc.orderAligatorMujoco(v_muj[6:])))
               
               initial_state = np.concatenate((q0_init, dq0_init))

               x_warmstart = x_res[1:] + [x_res[-1]]
               x_warmstart[0] = initial_state
               u_warmstart = eff_res[1:] + [eff_res[-1]]
               u_warmstart[0] = aligator_fnc.orderAligatorMujoco(u)
               if i == 0:
                    x_warmstart = 0
                    u_warmstart = 0
               start = time.time()
               ctrl_al.control_step(q0_init, dq0_init, prev_stage)
               start = time.time()
               stages = ctrl_al.phaseDefinition(T_fs, T_ss, x_fwd, swing_apex, i, add_obstacle, remove_obstacle, q0_target, align)
               start = time.time()
               joint_res, joint_vels, x_res, eff_res, prev_stage, curr_stage, conv, setup_time, run_time = ctrl_al.solveAligator(initial_state, stages, x_warmstart, u_warmstart)
               
               solve_time_all.append(time.time()-start)
               setup_time_all.append(setup_time)
               run_time_all.append(run_time)
               
               if conv:
                    conv_color.append('blue')
                    conv_plot.append(1)
                    conv_T +=1
               else:
                    conv_color.append('red')
                    conv_plot.append(0)
                    conv_F +=1

               # Compute torque using nominal policy       
               # Check which policy should be used
               eff_al = aligator_fnc.orderAligatorMujoco(eff_res[0])
               u = np.clip(eff_al, -lim_tau, lim_tau)
               copy_data = copy.deepcopy(data_muj)
               start = time.time()
               is_rec, iter_mps = mps.isRecSingle(min_height, u, iter_ctrl, lim_tau, model_muj, copy_data, N_mps, backup_nn, Kp_nn, Kd_nn, X_inv,i, step_rand,force_body)
               data_mps.append([iter_mps])
               if i >= step_rand:
                    data_mps_force.append([iter_mps])
               isrec_time_all.append(time.time()-start)
               del copy_data

               pin.forwardKinematics(model_pin, data_pin, q0_init)
               pin.updateFramePlacements(model_pin, data_pin)

               FR_placement = data_pin.oMf[FR_id]
               FL_placement = data_pin.oMf[FL_id]
               RR_placement = data_pin.oMf[RR_id]
               RL_placement = data_pin.oMf[RL_id]

               dist1 = FL_placement.translation[1] - FR_placement.translation[1]
               dist2 = RL_placement.translation[1] - RR_placement.translation[1]
               dist_plot.append([dist1, dist2])

               x_plot.append([FL_placement.translation[0], FR_placement.translation[0],
                              RL_placement.translation[0], RR_placement.translation[0]])
               
               z_plot.append([FL_placement.translation[2], FR_placement.translation[2],
                              RL_placement.translation[2], RR_placement.translation[2]])#'''
               
          i += 1

          
          if not is_rec:
               pos_backup_order = mps.computeBackup(q_muj, v_muj, backup_nn)
               backup_used = True

     '''
     import csv

     path_save = 'data'

     nameFile = path_save + "/torques.csv"
     with open(nameFile, 'w', encoding="ISO-8859-1", newline='') as myfile:
          wr = csv.writer(myfile)
          wr.writerows(torques_dt)
     myfile.close()

     nameFile = path_save + "/vels.csv"
     with open(nameFile, 'w', encoding="ISO-8859-1", newline='') as myfile:
          wr = csv.writer(myfile)
          wr.writerows(qvel_dt)
     myfile.close()

     nameFile = path_save + "/pos.csv"
     with open(nameFile, 'w', encoding="ISO-8859-1", newline='') as myfile:
          wr = csv.writer(myfile)
          wr.writerows(qpos_dt)
     myfile.close()

     nameFile = path_save + "/acc.csv"
     with open(nameFile, 'w', encoding="ISO-8859-1", newline='') as myfile:
          wr = csv.writer(myfile)
          wr.writerows(qacc_dt)
     myfile.close()
     #'''

     if view_MuJoCo:
          userview = input("Enter N or n to stop, T or t for top, F or f for front, S or s side: ")
          while userview != 'N' and userview != 'n':
               copy_data = copy.deepcopy(data_sim)
               viewMuJoCo(model_muj, copy_data, qpos, qvel, qacc, torques, forcesB, force_body, userview)
               userview = input("Enter N or n to stop, T or t for top, F or f for front, S or s side: ")

     
     final_q = data_muj.qpos[7:].copy()
     ctrl_al = None
     data_muj = None
     torques = None
     qpos = None
     qvel = None
     qacc = None
     forces = None
     forces_all = None

     return data_pos, data_vel, data_torque, data_fall, data_knee, backup_used, data_feet, robot_stopped, final_q, last_i, max_pos, min_pos, max_vel, min_vel, data_mps, data_mps_force


def testForcesInit(backup_nn, model_muj, data_muj_init, model_pin, data_pin, collision, visual):
     data_mps = []
     obstacle = 100#0.37#
     dt = 0.01
     N_mps = 200 # Number of MPS simulation steps to decide which policy to use
     lim_tau = 33.5
     min_height = 0.1

     lim_tau = 33.5
     lim_vel = 21.0
     min_height = 0.1
     min_height_2 = 0.05

     KpH_nn = 25
     KpT_nn = 25
     KpC_nn = 25
     Kp_nn = np.array([KpH_nn, KpT_nn, KpC_nn,
                         KpH_nn, KpT_nn, KpC_nn,
                         KpH_nn, KpT_nn, KpC_nn,
                         KpH_nn, KpT_nn, KpC_nn])

     KdH_nn = 0.5
     KdT_nn = 0.5
     KdC_nn = 0.5
     Kd_nn = np.array([KdH_nn, KdT_nn, KdC_nn,
                         KdH_nn, KdT_nn, KdC_nn,
                         KdH_nn, KdT_nn, KdC_nn,
                         KdH_nn, KdT_nn, KdC_nn])

     # Contact phases and walk parameters
     T_fs = 5
     T_ss = 40
     x_fwd = 0.1*1
     swing_apex =  0.1

     X_inv = 10e-2

     # Initial configuration
     th_q0 = 0.75
     q0 = np.array([0.,    0.,  0.39175, 0., 0., 0., 1., # trunk
                    0., th_q0,  -1.5,                    # FL
                    0., th_q0,  -1.5,                    # FR
                    0., th_q0,  -1.5,                    # RL
                    0., th_q0,  -1.5])                   # RR

     data_muj = copy.deepcopy(data_muj_init)

     dq0 = np.zeros(model_pin.nv)

     prev_stage = 'ALL'
     ctrl_al = aligator_fnc.aligatorControl(model_pin, collision, visual, obstacle, dt, q0, dq0, prev_stage)

     add_obstacle = False
     remove_obstacle = False
     align = False
     stages = ctrl_al.phaseDefinition(T_fs, T_ss, x_fwd, swing_apex, 0, add_obstacle, remove_obstacle, q0, align)

     initial_state = np.concatenate((q0, dq0))
     x_warmstart = 0
     u_warmstart = 0

     joint_res, joint_vels, x_res, eff_res, prev_stage, curr_stage, conv, setup_time, run_time = ctrl_al.solveAligator(initial_state, stages, x_warmstart, u_warmstart)

     # Forward dynamics
     eff_al = aligator_fnc.orderAligatorMujoco(eff_res[0])

     qpos0 = np.concatenate((joint_res[0][:3],np.array([joint_res[0][6]]),joint_res[0][3:6], aligator_fnc.orderAligatorMujoco(joint_res[0][7:])))
     qvel0 = np.concatenate((joint_vels[0][:6], aligator_fnc.orderAligatorMujoco(joint_vels[0][6:])))
     qacc0 = 0
     data_muj.qpos = qpos0
     data_muj.qvel = qvel0
     data_muj.qacc = qacc0

     q_muj = data_muj.qpos.copy()
     v_muj = data_muj.qvel.copy()

     jmin = model_pin.lowerPositionLimit
     jmax = model_pin.upperPositionLimit
     
     jmin_muj = aligator_fnc.orderAligatorMujoco(jmin[7:])
     jmax_muj = aligator_fnc.orderAligatorMujoco(jmax[7:])

     jmax_compare = np.array([jmax_muj[0],jmax_muj[2],jmax_muj[3],jmax_muj[5],jmax_muj[6],jmax_muj[8],jmax_muj[9],jmax_muj[11]])
     jmin_compare = np.array([jmin_muj[0],jmin_muj[2],jmin_muj[3],jmin_muj[5],jmin_muj[6],jmin_muj[8],jmin_muj[9],jmin_muj[11]])

     
     #model_muj.opt.timestep = 0.004
     # Number of times the OCP is solved
     N_mpc = 2*T_fs + T_ss + int(T_ss/2)
     u = np.clip(eff_al, -lim_tau, lim_tau)
     iter_ctrl = int(dt / model_muj.opt.timestep)
     
     force_body = model_muj.body('trunk').id
     copy_data = copy.deepcopy(data_muj)
     mujoco.mj_forward(model_muj,copy_data)
     
     is_rec, iter_mps = mps.isRecSingle(min_height, u, iter_ctrl, lim_tau, model_muj, copy_data, N_mps, backup_nn, Kp_nn, Kd_nn, X_inv,0,0,force_body)
     data_mps.append([iter_mps])
    # print('iter_mps', iter_mps)

     del copy_data

     if not is_rec:
        pos_backup_order = mps.computeBackup(q_muj, v_muj, backup_nn)

     torques_dt = []
     qpos_dt = []
     qvel_dt = []
     qacc_dt = []
     forces = []

     i = 1
     j = 0

     max_pos = np.zeros(12)
     min_pos = np.zeros(12)
     max_vel = np.zeros(12)
     min_vel = np.zeros(12)
     while i <= N_mpc:
          qpos_dt.append(q_muj)
          qvel_dt.append(v_muj)
          qacc_dt.append(data_muj.qacc.copy())
          forces.append(data_muj.xfrc_applied[force_body].copy())
          while j < iter_ctrl:
               j += 1
               if not is_rec:
                    u_backup = Kd_nn * (- v_muj[6:]) + Kp_nn * (pos_backup_order - q_muj[7:])
                    u = np.clip(u_backup, -lim_tau, lim_tau)

               data_muj.ctrl = u
               #forces.append(data_muj.xfrc_applied[force_body].copy())
               mujoco.mj_step(model_muj, data_muj)
               q_muj = data_muj.qpos.copy()
               v_muj = data_muj.qvel.copy()

               if np.any(np.abs(data_muj.ctrl) > lim_tau):
                    print('ctrl')
                    print(np.abs(data_muj.ctrl), '>', lim_tau)
               data_compare = np.array([data_muj.qpos[7],data_muj.qpos[9],data_muj.qpos[10],data_muj.qpos[12],data_muj.qpos[13],data_muj.qpos[15],data_muj.qpos[16],data_muj.qpos[18]])
               

               ## Trunk and hip positions
               coordinates = np.array([data_muj.body('trunk').xpos, data_muj.body('FL_hip').xpos, data_muj.body('FR_hip').xpos,
                              data_muj.body('RL_hip').xpos, data_muj.body('RR_hip').xpos])
               z_coordinates = np.array([coordinates[0][2], coordinates[1][2], coordinates[2][2], coordinates[3][2], coordinates[4][2]])
               if np.any(z_coordinates < min_height):
                    print('Fall')

               ## Knee positions
               coordinates = np.array([data_muj.body('FL_calf').xpos, data_muj.body('FR_calf').xpos,
                              data_muj.body('RL_calf').xpos, data_muj.body('RR_calf').xpos])
               z_coordinates = np.array([coordinates[0][2], coordinates[1][2], coordinates[2][2], coordinates[3][2]])
               if np.any(z_coordinates < min_height_2):
                    print('Knee')

          torques_dt.append(data_muj.ctrl.copy())

          check_vel = np.any(np.abs(data_muj.qvel[6:]) > lim_vel)
          if check_vel:
               k = 0
               for l in data_muj.qvel[6:]:
                    if l > lim_vel and l > max_vel[k]:
                         max_vel[k] = l
                    elif l < -lim_vel and l < min_vel[k]:
                         min_vel[k] = l
                    k += 1

          check_pos = (np.any(data_compare > jmax_compare) or np.any(data_compare < jmin_compare))
          if check_pos:
               print('pos i', i)
               print(jmin_muj,'>',data_muj.qpos[7:], '>', jmax_muj, i)

          j = 0
          if is_rec:

               q0_init = np.concatenate((q_muj[:3],q_muj[4:7],np.array([q_muj[3]]), aligator_fnc.orderAligatorMujoco(q_muj[7:])))
               dq0_init = np.concatenate((v_muj[:6], aligator_fnc.orderAligatorMujoco(v_muj[6:])))
               
               initial_state = np.concatenate((q0_init, dq0_init))

               x_warmstart = x_res[1:] + [x_res[-1]]
               x_warmstart[0] = initial_state
               u_warmstart = eff_res[1:] + [eff_res[-1]]
               u_warmstart[0] = aligator_fnc.orderAligatorMujoco(u)
               if i == 0:
                    x_warmstart = 0
                    u_warmstart = 0
               ctrl_al.control_step(q0_init, dq0_init, prev_stage)
               prev_stage_ant = prev_stage
               stages = ctrl_al.phaseDefinition(T_fs, T_ss, x_fwd, swing_apex, i, add_obstacle, remove_obstacle, q0, align)
               joint_res, joint_vels, x_res, eff_res, prev_stage, curr_stage, conv, setup_time, run_time = ctrl_al.solveAligator(initial_state, stages, x_warmstart, u_warmstart)

               # Compute torque using nominal policy       
               # Check which policy should be used
               eff_al = aligator_fnc.orderAligatorMujoco(eff_res[0])
               u = np.clip(eff_al, -lim_tau, lim_tau)
               
               copy_data = copy.deepcopy(data_muj)
               is_rec, iter_mps = mps.isRecSingle(min_height, u, iter_ctrl, lim_tau, model_muj, copy_data, N_mps, backup_nn, Kp_nn, Kd_nn, X_inv,i, 0,force_body)
               if i + 1 <= N_mpc:
                    data_mps.append([iter_mps])
               del copy_data
               
          i += 1

          
          if not is_rec:
               pos_backup_order = mps.computeBackup(q_muj, v_muj, backup_nn)
     
     
     return data_muj, x_warmstart, u_warmstart, ctrl_al, is_rec, prev_stage_ant, data_mps, qpos_dt, qvel_dt, qacc_dt, torques_dt, forces
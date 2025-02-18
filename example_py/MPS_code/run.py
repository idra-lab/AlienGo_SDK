# Order of data (position, velocity, acceleration, torques)
# FR_hip, FR_thigh, FR_calf
# FL_hip, FL_thigh, FL_calf
# RR_hip, RR_thigh, RR_calf
# RL_hip, RL_thigh, RL_calf

import aligatorMPShorizon #forceTest
import numpy as np
import csv
import torch
import mps

import os
import gc


# Data to preprocess state before sending it to the network
device = torch.device('cpu')
if torch.cuda.is_available():
        device = torch.device('cuda')

running_mean = torch.tensor([  4.8700e-02, -8.6040e-04,  2.4586e-01,  9.6818e-01, -1.2454e-02,
                                -6.7572e-03, -2.1485e-02,  1.4759e-02, -2.9173e-04, -3.7350e-02,
                                -8.5417e-03, -2.7321e-05, -1.6754e-02,  2.1093e-01, -1.4264e-01,
                                2.4056e-01, -2.3169e-01,  2.7045e-01,  2.6174e-01,  3.5283e-01,
                                2.4183e-01, -3.4884e-01, -2.9622e-01, -2.7078e-01, -3.0096e-01,
                                3.2596e-02, -2.4975e-02,  4.9519e-02, -4.6313e-02,  1.7437e-01,
                                1.4159e-01,  1.0916e-01,  8.8281e-02, -2.1362e-01, -1.6740e-01,
                                -9.8866e-02, -1.2485e-01], device=device, dtype=torch.float64)

running_variance = torch.tensor([1.8488e-02, 1.0488e-02, 4.0845e-03, 2.0256e-02, 2.5170e-02, 8.4353e-03,
                                8.6522e-03, 3.5092e-02, 1.9456e-02, 5.4424e-02, 2.4965e+00, 4.8901e-01,
                                2.2312e-01, 7.2991e-02, 1.5376e-01, 1.1025e-01, 1.0565e-01, 8.9872e-02,
                                7.2449e-02, 8.9041e-02, 8.8463e-02, 6.4336e-02, 9.9900e-02, 8.3476e-02,
                                8.3678e-02, 6.6343e+00, 6.7514e+00, 9.8686e+00, 9.3465e+00, 4.8276e+00,
                                4.9296e+00, 7.1135e+00, 6.5952e+00, 8.0300e+00, 8.3643e+00, 1.2187e+01,
                                1.1027e+01], device=device, dtype=torch.float64)

epsilon = 1e-8

clip_threshold = 5.0

joint_def = torch.tensor([ 0.1000, -0.1000,  0.1000,
                        -0.1000,  0.8000,  0.8000,
                        1.0000,  1.0000, -1.5000,
                        -1.5000, -1.5000, -1.5000], device=device, dtype=torch.float64)

# Load neural network for backup policy
PATH = 'models/FULL_STATE__NN_v3.pt'
dict_policy = torch.load(PATH, map_location=torch.device(device), weights_only=True)

new_keys = ["layers.0.weight", "layers.0.bias", "layers.2.weight", "layers.2.bias",
                "layers.4.weight", "layers.4.bias", "layers.6.weight", "layers.6.bias"]
old_keys = ["net.0.weight", "net.0.bias",      "net.2.weight",      "net.2.bias",
                "net.4.weight", "net.4.bias", "mean_layer.weight", "mean_layer.bias"] 

new_policy_dict = mps.labels_state_dict(dict_policy, old_keys, new_keys)
backup_nn = mps.Backup(running_mean, running_variance, epsilon, clip_threshold, joint_def, device)
backup_nn.load_state_dict(new_policy_dict)

# MuJoCo and Pinocchio models

model_muj, data_muj_init, model_pin, data_pin, collision, visual = aligatorMPShorizon.modelData()

# False to run all the tests for one Force, True to only run one test
test_code = True#False#

if test_code:

        force_mag = 95
        force_applied = np.array([0.6584730894598917,-0.5965196811390472,0.458887197981067]) * force_mag
        iter_application = 22
        data_muj_70, x_warmstart, u_warmstart, ctrl_al, is_rec, prev_stage_ant, data_mps, qpos_dt, qvel_dt, qacc_dt, torques_dt, forces = aligatorMPShorizon.testForcesInit(backup_nn, model_muj, data_muj_init, model_pin, data_pin, collision, visual)
        data_pos, data_vel, data_torque, data_fall, data_knee, backup_used, data_feet, robot_stopped, final_q, last_iteration, max_pos, min_pos, max_vel, min_vel, data_mpsB, save_data_mps_force = aligatorMPShorizon.testForces(iter_application, force_applied, force_mag, 0, backup_nn, model_muj, data_muj_70, model_pin, data_pin, collision, visual, x_warmstart, u_warmstart, ctrl_al, is_rec, prev_stage_ant, qpos_dt.copy(), qvel_dt.copy(), qacc_dt.copy(), torques_dt.copy(), forces.copy(), True, data_muj_init)


else:
        tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])

        directions = []
        test_force = '95'

        test_force_int = int(test_force)
        test_num_last = 0
        path_init = "force_data"
        dir_path = path_init + "/sphere_points_full.csv"

        i_rand = []
        iter_path = path_init + "/aplication_iteration.csv"

        force_mag = [test_force_int]


        data_muj_70, x_warmstart, u_warmstart, ctrl_al, is_rec, prev_stage_ant, data_mps, qpos_dt, qvel_dt, qacc_dt, torques_dt = aligatorMPShorizon.testForcesInit(backup_nn, model_muj, data_muj_init, model_pin, data_pin, collision, visual)
        test_num = 1

        rng = np.random.default_rng()
        tests_array = rng.choice(10001, 10, replace=False)

        for k in force_mag:
            dir_file = open(dir_path)
            dir_text = dir_file.readline().rstrip().split(',')
            while dir_text[0] != '':
                j = np.array([float(dir_text[0]),float(dir_text[1]),float(dir_text[2])])
                data_sim = []
                save_pos = []
                save_vel = []
                save_torque = []
                save_fall = []
                save_knee = []
                save_backup = []
                save_feet = []
                save_stop = []
                save_stop_backup = []
                

                force_applied = j * k

                iter_file = open(iter_path)
                iter_text = iter_file.readline().rstrip().split(',')
                while iter_text[0] != '':
                    i = int(iter_text[0])
                    
                    if  test_num > test_num_last:
                        print('test_num',test_num)

                        save_data_mps = data_mps.copy()

                        data_pos, data_vel, data_torque, data_fall, data_knee, backup_used, data_feet, robot_stopped, final_q, last_iteration, max_pos, min_pos, max_vel, min_vel, data_mpsB, save_data_mps_force = aligatorMPShorizon.testForces(i, force_applied, k, test_num, backup_nn, model_muj, data_muj_70, model_pin, data_pin, collision, visual, x_warmstart, u_warmstart, ctrl_al, is_rec, prev_stage_ant, qpos_dt.copy(), qvel_dt.copy(), qacc_dt.copy(), torques_dt.copy(), False, data_muj_init)
                        save_data_mps += data_mpsB

                        file_data_mps_complete = []
                        for iter_mps in save_data_mps:
                                file_data_mps_complete.append([iter_mps[0], k, test_num])
                        file_data_mps = []
                        for iter_mps in data_mpsB:
                                file_data_mps.append([iter_mps[0], k, test_num])
                        file_data_mps_force = []
                        for iter_mps in save_data_mps_force:
                                file_data_mps_force.append([iter_mps[0], k, test_num])

                        data_sim.append([i, j, k, test_num, backup_used, data_feet[0], data_feet[1:], final_q, last_iteration, max_pos, min_pos, max_vel, min_vel])
                              
                        if len(data_pos) > 0:
                                save_pos.append([data_pos[0][0], data_pos[0][1], data_pos[0][2], max_pos, min_pos])
                        if len(data_vel) > 0:
                                save_vel.append([data_vel[0][0], data_vel[0][1], data_vel[0][2], max_vel, min_vel])
                        if len(data_torque) > 0:
                                save_torque.append([data_torque[0][0], data_torque[0][1], data_torque[0][2]])
                        if len(data_fall) > 0:
                                save_fall.append([data_fall[0][0],data_fall[0][1]])
                        if len(data_knee) > 0:
                                save_knee.append([data_knee[0][0],data_knee[0][1]])
                        if not backup_used:
                                save_backup.append([k, test_num])
                        if not robot_stopped:
                                save_stop.append([k, test_num])
                        if not backup_used and not robot_stopped:
                                save_stop_backup.append([k, test_num])


                        # Save data about number of iterations 
                        '''
                        path_init = ".."
                        # With first 70 iterations
                        nameFile = path_init + "/Results/iterations_New/save_iterations_complete" + test_force + "_New.csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(file_data_mps_complete)
                        myfile.close()

                        # No first 70 iterations
                        nameFile = path_init + "/Results/iterations_New/save_iterations" + test_force + "_New.csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(file_data_mps)
                        myfile.close()

                        # Only since the force is applied
                        nameFile = path_init + "/Results/iterations_New/save_iterations_force" + test_force + "_New.csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(file_data_mps_force)
                        myfile.close()#'''

                    test_num += 1
                    
                    
                    
                    data_pos = None
                    data_vel = None
                    data_torque = None
                    data_fall = None
                    data_knee = None
                    backup_used = None
                    data_feet = None
                    robot_stopped = None
                    gc.collect()

                    iter_text = iter_file.readline().rstrip().split(',')

                iter_file.close()
                
                # Save data from tests
                '''if len(save_pos) > 0:
                        nameFile = path_init + "/Results/save_pos" + test_force + "_New.csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_pos)
                        myfile.close()

                if len(save_vel) > 0:
                        nameFile = path_init + "/Results/save_vel" + test_force + "_New.csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_vel)
                        myfile.close()'''
                '''
                if len(save_torque) > 0:
                        nameFile = path_init + "/Results/save_torque" + test_force + ".csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_torque)
                        myfile.close()

                if len(save_fall) > 0:
                        nameFile = path_init + "/Results/save_fall" + test_force + ".csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_fall)
                        myfile.close()

                if len(save_knee) > 0:
                        nameFile = path_init + "/Results/save_knee" + test_force + ".csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_knee)
                        myfile.close()

                if len(save_backup) > 0:
                        nameFile = path_init + "/Results/save_backup" + test_force + ".csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_backup)
                        myfile.close()

                if len(save_stop) > 0:
                        nameFile = path_init + "/Results/save_stop" + test_force + ".csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_stop)
                        myfile.close()

                if len(save_stop_backup) > 0:
                        nameFile = path_init + "/Results/save_stop_backup" + test_force + ".csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_stop_backup)
                        myfile.close()#'''
                        
                '''if len(data_sim) > 0:
                        nameFile = path_init + "/Results/data_sim" + test_force + "_New.csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(data_sim)
                        myfile.close()

                        print('test_num',test_num)
                        tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
                        print('free_m',free_m)'''
                        

                data_sim = None
                save_pos = None
                save_vel = None
                save_torque = None
                save_fall = None
                save_knee = None
                save_backup = None
                save_feet = None
                save_stop = None
                save_stop_backup = None
                gc.collect()
                for o in gc.garbage:
                         print('Retained: {} 0x{:x}'.format(o, id(o)))
                dir_text = dir_file.readline().rstrip().split(',')
            dir_file.close()

        
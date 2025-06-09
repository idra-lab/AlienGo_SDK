from aligatorMPShorizon_complete import modelData, testForces
import numpy as np
import csv
import time
import copy
from numpy.random import default_rng

# MuJoCo and Pinocchio models

model_muj, data_muj, model_pin, data_pin = modelData()

test_code = True#False#

if test_code:
        #start = time.time()
        force_mag = 0*200#240#200#250#100#70#50#0#
        force_applied = np.array([ -0.975369117929369, 0.201025291270696,  -0.0907960134544247 ]) * force_mag
        
        force_applied = np.array([ -0.598635159600919, 0.525490957279855,  -0.604561989796483 ]) * force_mag

        force_mag = 70 #135 with 0.04m, 65 with 0.03
        force_applied = np.array([0.6584730894598917,-0.5965196811390472,0.458887197981067]) * force_mag
        #force_applied = np.array([-0.6671558632641383,0.01812370024266151,-0.7446976471037066]) * force_mag

        iter_force = 25
        force_mag = 190*0

        iter_force = 100#0
        force_mag = 0*200
        force_applied = np.array([0.2000716943600823,0.20900583365070158,0.9572292717086439]) * force_mag
        data_pos, data_vel, data_torque, data_fall, data_knee, backup_used, data_feet, robot_stopped, final_q, final_dq, last_iteration, max_pos, min_pos, max_vel, min_vel, data_mps, data_mps_force, data_mps_no_force = testForces(iter_force, force_applied, force_mag, 1, model_muj, data_muj, model_pin, data_pin)
        print('final_q')
        print(final_q)
        print('final_dq')
        print(final_dq)
        print(data_fall)
        print(data_knee)
        print(data_pos)
        print(data_vel)
        '''print(time.time() - start)
        file_data_mps_complete = []
        for iter_mps in data_mps:
                file_data_mps_complete.append([iter_mps[0]])
        # With first 70 iterations
        nameFile = "save_iterations_complete.csv"
        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(file_data_mps_complete)'''
else:
        directions = []
        test_force = '200'
        test_force_int = int(test_force)
        test_num_last = 0
        path_init = ".."
        dir_path = path_init + "/sphere_points_full.csv"

        i_rand = []
        iter_path = path_init + "/aplication_iteration.csv"

        force_mag = [test_force_int]
        
        test_num = 1
        for k in force_mag:
            '''
            if test_num == 11:#31:
                break#'''
            
            dir_file = open(dir_path)
            dir_text = dir_file.readline().rstrip().split(',')
            rng = default_rng()
            sph_run = rng.choice(1000, size=100, replace=False)
            sph_iter = 0
            while dir_text[0] != '':
                '''
                if test_num == 11:
                        break#'''
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
                    
                    if  test_num > test_num_last and sph_iter in sph_run:
                        data_muj_copy = copy.deepcopy(data_muj)
                        data_pos, data_vel, data_torque, data_fall, data_knee, backup_used, data_feet, robot_stopped, final_q, final_dq, last_iteration, max_pos, min_pos, max_vel, min_vel, data_mps, data_mps_force, data_mps_no_force = testForces(i, force_applied, k, test_num, model_muj, data_muj_copy, model_pin, data_pin)

                        file_data_mps_complete = []
                        for iter_mps in data_mps:
                                file_data_mps_complete.append([iter_mps[0], k, test_num])
                        file_data_mps_force = []
                        for iter_mps in data_mps_force:
                                file_data_mps_force.append([iter_mps[0], k, test_num])
                        file_data_mps_no_force = []
                        for iter_mps in data_mps_no_force:
                                file_data_mps_no_force.append([iter_mps[0], k, test_num])

                        data_sim.append([i, j, k, test_num, backup_used, data_feet[0], data_feet[1:], final_q, final_dq, last_iteration, max_pos, min_pos, max_vel, min_vel])

                        if len(data_pos) > 0:
                                save_pos.append([data_pos[0][0], data_pos[0][1], data_pos[0][2]])
                        if len(data_vel) > 0:
                                save_vel.append([data_vel[0][0], data_vel[0][1], data_vel[0][2]])
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

                        #'''
                        # All
                        nameFile = "../Results_NN/iterations/save_iterations_complete" + test_force + ".csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(file_data_mps_complete)
                        myfile.close()

                        # Before force
                        nameFile = "../Results_NN/iterations/save_iterations_no_force" + test_force + ".csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(file_data_mps_no_force)
                        myfile.close()

                        # Only since the force is applied
                        nameFile = "../Results_NN/iterations/save_iterations_force" + test_force + ".csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(file_data_mps_force)
                        myfile.close()#'''

                    test_num += 1
                    '''
                    if test_num  == 11:#31:
                        break#'''
                    

                    iter_text = iter_file.readline().rstrip().split(',')
                

                iter_file.close()

                if len(save_pos) > 0:
                        nameFile = path_init + "/Results_NN/" + test_force + "/save_pos" + test_force + ".csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_pos)
                        myfile.close()

                if len(save_vel) > 0:
                        nameFile = path_init + "/Results_NN/" + test_force + "/save_vel" + test_force + ".csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_vel)
                        myfile.close()

                if len(save_torque) > 0:
                        nameFile = path_init + "/Results_NN/" + test_force + "/save_torque" + test_force + ".csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_torque)
                        myfile.close()

                if len(save_fall) > 0:
                        nameFile = path_init + "/Results_NN/" + test_force + "/save_fall" + test_force + ".csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_fall)
                        myfile.close()

                if len(save_knee) > 0:
                        nameFile = path_init + "/Results_NN/" + test_force + "/save_knee" + test_force + ".csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_knee)
                        myfile.close()

                if len(save_backup) > 0:
                        nameFile = path_init + "/Results_NN/" + test_force + "/save_backup" + test_force + ".csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_backup)
                        myfile.close()

                if len(save_stop) > 0:
                        nameFile = path_init + "/Results_NN/" + test_force + "/save_stop" + test_force + ".csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_stop)
                        myfile.close()

                if len(save_stop_backup) > 0:
                        nameFile = path_init + "/Results_NN/" + test_force + "/save_stop_backup" + test_force + ".csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_stop_backup)
                        myfile.close()
                        
                if len(data_sim) > 0:
                        nameFile = path_init + "/Results_NN/" + test_force + "/data_sim" + test_force + ".csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(data_sim)
                        myfile.close()
                        
                #'''

                dir_text = dir_file.readline().rstrip().split(',')
                sph_iter += 1
            dir_file.close()

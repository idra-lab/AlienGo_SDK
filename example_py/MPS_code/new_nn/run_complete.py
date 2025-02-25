import aligatorMPShorizon_complete #forceTest
import numpy as np
import csv
import torch
import mps

import os
#from guppy import hpy
#import objgraph 
import gc

'''
flags = (gc.DEBUG_COLLECTABLE |
         gc.DEBUG_UNCOLLECTABLE |
         gc.DEBUG_SAVEALL
         )

gc.set_debug(flags)
'''

# Data to preprocess state before sending it to the network
device = torch.device('cpu')
#if torch.cuda.is_available():
#        device = torch.device('cuda')

running_mean = torch.tensor([ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.21205050e+00,
                             -8.06230644e-03, -1.42328272e+00,  1.00789203e-01, -1.14620532e-01,
                              6.92671321e-02, -2.20188718e-01,  2.29067680e-01,  2.09180188e-01,
                              2.06069022e-01,  1.45354481e-01, -4.06120459e-01, -4.48609024e-01,
                             -3.88464357e-01, -3.97695643e-01, -6.33672667e-03, -1.95075319e-03,
                             -3.96507459e-03, -4.42201867e-02,  1.14105612e-01,  1.03970752e-01,
                              5.92749748e-02,  4.45349231e-02, -1.63696425e-01, -1.79439128e-01,
                             -1.44850614e-01, -1.38541725e-01,  3.80852309e-02, -1.01241193e-01,
                             -1.90169735e-02, -1.39184525e-01,  1.36771685e-01,  1.38808689e-01,
                              1.64408062e-01,  1.45937922e-01,  4.67878538e-02, -5.74250520e-04,
                             -6.52443376e-02, -1.06306087e-02], device=device, dtype=torch.float64)

running_variance = torch.tensor([1.45321798e-08, 1.45321798e-08, 1.45321798e-08, 1.02176822e+01,
                                 1.08386133e+01, 2.91442128e+01, 6.12955153e-02, 7.91401819e-02,
                                 6.40535954e-02, 3.81343809e-02, 8.86714353e-02, 6.60074706e-02,
                                 5.39578640e-02, 3.24022101e-02, 5.45107210e-02, 5.66611032e-02,
                                 6.45901655e-02, 6.92300715e-02, 6.07200216e+00, 5.75513999e+00,
                                 8.04183510e+00, 8.15766779e+00, 4.10619267e+00, 4.09391329e+00,
                                 5.59485546e+00, 5.44709528e+00, 6.09915664e+00, 6.84910827e+00,
                                 9.67759842e+00, 8.53798207e+00, 7.66597364e-02, 7.27123376e-02,
                                 8.60715622e-02, 6.56937354e-02, 8.89956898e-02, 8.91989478e-02,
                                 8.37767829e-02, 6.97930652e-02, 1.43025920e-01, 1.22080731e-01,
                                 1.49352419e-01, 1.33925065e-01], device=device, dtype=torch.float64)

epsilon = 1e-8

clip_threshold = 5.0

joint_def = torch.tensor([ 0.1000, -0.1000,  0.1000,
                        -0.1000,  0.8000,  0.8000,
                        1.0000,  1.0000, -1.5000,
                        -1.5000, -1.5000, -1.5000], device=device, dtype=torch.float64)

# Load neural network for backup policy
#PATH = '/home/jessica/SMPS/MuJoCo_Aligator_Full/backup_policy/test/FULL_STATE__NN_v3.pt'
PATH = 'best_agent.pt'
dict_policy = torch.load(PATH, map_location=torch.device(device))['policy']

new_keys = ["layers.0.weight", "layers.0.bias", "layers.2.weight", "layers.2.bias",
                "layers.4.weight", "layers.4.bias", "layers.6.weight", "layers.6.bias"]
old_keys = ["net.0.weight", "net.0.bias",      "net.2.weight",      "net.2.bias",
                "net.4.weight", "net.4.bias", "mean_layer.weight", "mean_layer.bias"] 

new_policy_dict = mps.labels_state_dict(dict_policy, old_keys, new_keys)
backup_nn = mps.Backup(running_mean, running_variance, epsilon, clip_threshold, joint_def, device)
backup_nn.load_state_dict(new_policy_dict)

# MuJoCo and Pinocchio models

model_muj, data_muj_init, model_pin, data_pin, collision, visual = aligatorMPShorizon_complete.modelData()

test_code = False#True#

if test_code:
       # hp = hpy()

        force_mag = 1*100#240#200#250#100#70#50#0#
        force_applied = np.array([0,1,0]) * force_mag

        # Example sliding
        force_mag = 0*90#240#200#250#100#70#50#0#
        force_applied = np.array([ -0.975369117929369, 0.201025291270696,  -0.0907960134544247 ]) * force_mag
        
        #force_applied = np.array([ -0.598635159600919, 0.525490957279855,  -0.604561989796483 ]) * force_mag

        #tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        '''print('tot_m',tot_m)
        print('used_m',used_m)
        print('free_m',free_m)'''

        '''
        force_mag = 500#250#100#70#50
        force_rand = np.random.uniform(-1,1,3)
        force_applied = (force_rand / np.linalg.norm(force_rand)) * force_mag
        print('force_applied',force_applied)#'''
        #data_pos, data_vel, data_torque, data_fall, data_knee, backup_used = mujoco_aligatorMPSforceTest.testForces(force_applied, 0, 0)
        
        #start = hp.heap() 
        data_pos, data_vel, data_torque, data_fall, data_knee, backup_used, data_feet, robot_stopped, final_q, last_iteration, max_pos, min_pos, max_vel, min_vel = aligatorMPShorizon_complete.testForces(74, force_applied, force_mag, 1, backup_nn, model_muj, data_muj_init, model_pin, data_pin, collision, visual)
     #   end = hp.heap() 
        #used = end - start 
        #print(start)
     #   print(end)
        #print(used)
       # data_pos, data_vel, data_torque, data_fall, data_knee, backup_used, data_feet = aligatorMPShorizon.testForces(130,force_applied, force_mag,1)
        #'''
        '''print('data_pos')
        print(data_pos)
        print('data_vel')
        print(data_vel)
        print('data_torque')
        print(data_torque)
        print('data_fall')
        print(data_fall)
        print('data_knee')
        print(data_knee)
        print('backup_used')
        print(backup_used)
        print('data_feet')
        print(data_feet)
        print('robot_stopped')
        print(robot_stopped)
        print('final_q')
        print(final_q)
        print('last_iteration')
        print(last_iteration)
        print('max_pos')
        print(max_pos)

        print('min_pos')
        print(min_pos)
        print('max_vel')
        print(max_vel)
        print('min_vel')
        print(min_vel)'''



else:
        tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        '''print('tot_mA',tot_m)
        print('used_mA',used_m)
        print('free_mA',free_m)'''

        directions = []
        test_force = '90'#input("Enter force for test: ")
        test_force_int = int(test_force)
        #test_num_last = input("Enter number for last test in previous run: ")
        test_num_last = 1#int(test_num_last)
        path_init = "/home/jessica/SMPS/random_forces"
        path_init = "../force_data"
        dir_path = path_init + "/sphere_points_full.csv"
        '''
        with open(file_path,'r') as csvfile:
            lines = csv.reader(csvfile, delimiter=',')
            for row in lines:
                directions.append(np.array([float(row[0]),float(row[1]),float(row[2])]))
        csvfile.close()#'''

        i_rand = []
        iter_path = path_init + "/aplication_iteration.csv"
        '''
        with open(file_path,'r') as csvfile:
            lines = csv.reader(csvfile, delimiter=',')
            for row in lines:
                i_rand.append(int(row[0]))
        csvfile.close() #'''   

        
        
        force_mag = [60,65,70,75,80,85,90,95,100,105]#[290, 300, 310, 320, 330, 340, 350]#[250, 260, 270, 280, 290, 300]#[150,160,170,180,190,200,210,220,230,240] #150,160,170,180,190,200,210,220,230,240

        force_mag = [test_force_int]

        
        '''tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        print('tot_mB',tot_m)
        print('used_mB',used_m)
        print('free_mB',free_m)'''

        
        test_num = 1
        #hp = hpy()
        #'''
        for k in force_mag:#[force_mag[0]]:#
            #'''
            if test_num == 11:#31:
                break#'''
            
            dir_file = open(dir_path)
            dir_text = dir_file.readline().rstrip().split(',')
            while dir_text[0] != '':# and test_num < 4:#for j in [directions[0]]:#directions:#if True:# 
                #'''
                if test_num == 11:#31:
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
                while iter_text[0] != '':# and test_num < 4:#for i in i_rand:
                    
                    #print('iter_text',iter_text[0])
                    i = int(iter_text[0])
                  #  hp.setrelheap()
                 #   objgraph.show_most_common_types()
                 #   objgraph.show_growth()
                    
                    if  test_num == test_num_last:#True:#test_num == 498:#test_num == 1 or test_num == 2 or test_num == 965:#test_num >= 1 or test_num <= 50:# or test_num == 228 or test_num == 5000:#True:#
                        #print('test_num',test_num)
                        '''tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
                      #  print('tot_m',tot_m)
                      #  print('used_m',used_m)
                        print('free_m',free_m)'''

                        
            #j=np.array([0,1,0])
                    
                      #  print(i, j)
                        
                     #   start = hp.heap() 
                     #   hp.setrelheap()
                        print('i',i)
                        data_pos, data_vel, data_torque, data_fall, data_knee, backup_used, data_feet, robot_stopped, final_q, last_iteration, max_pos, min_pos, max_vel, min_vel = aligatorMPShorizon_complete.testForces(i, force_applied, k, test_num, backup_nn, model_muj, data_muj_init, model_pin, data_pin, collision, visual)
                     #   print('dir()')
                     #   print(dir())
                     #   end = hp.heap() 
                        #used = end - start 
                     #   print(start)
                     #   print(end)
                        #print(used)

                    #data_pos, data_vel, data_torque, data_fall, data_knee, backup_used, data_feet = aligatorMPShorizon.testForces(0, force_applied, force_mag, test_num)
                    
                        '''
                        print(data_pos)
                        print(data_vel)
                        print(data_torque)
                        print(data_fall)
                        print(data_knee)
                        print(backup_used)
                        print(data_feet)
                        print(final_q)#'''

                        data_sim.append([i, j, k, test_num, backup_used, data_feet[0], data_feet[1:], final_q, last_iteration, max_pos, min_pos, max_vel, min_vel])
                        #'''        
                        print('data_pos',data_pos)
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
                        print('robot_stopped',robot_stopped)
                        if not robot_stopped:
                                save_stop.append([k, test_num])
                        if not backup_used and not robot_stopped:
                                save_stop_backup.append([k, test_num])#'''

                    test_num += 1
                    #'''
                    if test_num  == 11:#31:
                        break#'''
                    
                    #'''
                    data_pos = None
                    data_vel = None
                    data_torque = None
                    data_fall = None
                    data_knee = None
                    backup_used = None
                    data_feet = None
                    robot_stopped = None
                    gc.collect()#'''

                    iter_text = iter_file.readline().rstrip().split(',')
                

                iter_file.close()

                '''
                

                if len(save_pos) > 0:
                        nameFile = path_init + "/Results/save_pos" + test_force + "B.csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_pos)
                        myfile.close()

                if len(save_vel) > 0:
                        nameFile = path_init + "/Results/save_vel" + test_force + "B.csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_vel)
                        myfile.close()

                if len(save_torque) > 0:
                        nameFile = path_init + "/Results/save_torque" + test_force + "B.csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_torque)
                        myfile.close()

                if len(save_fall) > 0:
                        nameFile = path_init + "/Results/save_fall" + test_force + "B.csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_fall)
                        myfile.close()

                if len(save_knee) > 0:
                        nameFile = path_init + "/Results/save_knee" + test_force + "B.csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_knee)
                        myfile.close()

                if len(save_backup) > 0:
                        nameFile = path_init + "/Results/save_backup" + test_force + "B.csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_backup)
                        myfile.close()

                if len(save_stop) > 0:
                        nameFile = path_init + "/Results/save_stop" + test_force + "B.csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_stop)
                        myfile.close()

                if len(save_stop_backup) > 0:
                        nameFile = path_init + "/Results/save_stop_backup" + test_force + "B.csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(save_stop_backup)
                        myfile.close()
                        
                if len(data_sim) > 0:
                        nameFile = path_init + "/Results/data_sim" + test_force + "B.csv"
                        with open(nameFile, 'a', encoding="ISO-8859-1", newline='') as myfile:
                                wr = csv.writer(myfile)
                                wr.writerows(data_sim)
                        myfile.close()

                        print('test_num',test_num)
                        tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
                        print('free_m',free_m)
                        
                #'''

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
    #if isinstance(o, Graph):
                         print('Retained: {} 0x{:x}'.format(o, id(o)))
                dir_text = dir_file.readline().rstrip().split(',')
            dir_file.close()

        '''
        nameFile = path_init + "/Results/data_sim_" + str(int(k)) + ".csv"
        with open(nameFile, 'w', encoding="ISO-8859-1", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(data_sim)
        myfile.close()

        nameFile = path_init + "/Results/save_pos_" + str(int(k)) + ".csv"
        with open(nameFile, 'w', encoding="ISO-8859-1", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(save_pos)
        myfile.close()

        nameFile = path_init + "/Results/save_vel_" + str(int(k)) + ".csv"
        with open(nameFile, 'w', encoding="ISO-8859-1", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(save_vel)
        myfile.close()

        nameFile = path_init + "/Results/save_torque_" + str(int(k)) + ".csv"
        with open(nameFile, 'w', encoding="ISO-8859-1", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(save_torque)
        myfile.close()

        nameFile = path_init + "/Results/save_fall_" + str(int(k)) + ".csv"
        with open(nameFile, 'w', encoding="ISO-8859-1", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(save_fall)
        myfile.close()

        nameFile = path_init + "/Results/save_knee_" + str(int(k)) + ".csv"
        with open(nameFile, 'w', encoding="ISO-8859-1", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(save_knee)
        myfile.close()

        nameFile = path_init + "/Results/save_backup_" + str(int(k)) + ".csv"
        with open(nameFile, 'w', encoding="ISO-8859-1", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(save_backup)
        myfile.close()#'''

        '''i = i_rand[0]
        j = directions[0]

        force_applied = j * force_mag
        #print(force_applied)
        #    print(len(force_applied))
        print(i, j)
        data_pos, data_vel, data_torque, data_fall, data_knee, backup_used, data_feet = aligatorMPShorizon.testForces(i, force_applied, force_mag, test_num)

        #data_pos, data_vel, data_torque, data_fall, data_knee, backup_used, data_feet = aligatorMPShorizon.testForces(0, force_applied, force_mag, test_num)


        print(data_pos)
        print(data_vel)
        print(data_torque)
        print(data_fall)
        print(data_knee)
        print(backup_used)
        print(data_feet)#'''
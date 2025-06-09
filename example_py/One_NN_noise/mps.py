import numpy as np
import mujoco
from walk import swap_legs, compute_actions
from matplotlib.path import Path
import matplotlib
from shapely.geometry import Polygon
import time
import copy

# extract joint states from the generalised position vector returned by the policy
def orderPosition(pos):
    order_pos = [10, 7, 16, 13, 11, 8, 17, 14, 12, 9, 18, 15]
    return pos[order_pos]

# extract joint position and velocity from the generalised position/velocity vector returned by the policy
def orderState(pos, vel):
    order_pos = orderPosition(pos)
    order_vel = [ 9, 6, 15, 12, 10, 7, 16, 13, 11, 8, 17, 14]
    return pos[order_pos], vel[order_vel]

def orderBackup(pos):
    order_pos = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]
    return pos[order_pos].detach().cpu().numpy()

def capture_point_check(data_muj, xy_coords, factor = 0.7):
    # Compute the capture point coordinates
    com_coordinates = data_muj.body('trunk').subtree_com
    mujoco.mj_subtreeVel(data_muj.model, data_muj)
    com_velocities = data_muj.body('trunk').subtree_linvel
    omega = np.sqrt(abs(data_muj.model.opt.gravity[2]) / com_coordinates[2])
    cp_x = com_coordinates[0] + (com_velocities[0]/omega)
    cp_y = com_coordinates[1] + (com_velocities[1]/omega)

    # Define convex hull and shrink it
    hull_path = Path(xy_coords)
    polygon = Polygon(hull_path.vertices)
    hull_path = hull_path.transformed(matplotlib.transforms.Affine2D().scale(factor))
    shrinked_polygon = Polygon(hull_path.vertices)
    
    translate_x = polygon.centroid.x - shrinked_polygon.centroid.x
    translate_y = polygon.centroid.y - shrinked_polygon.centroid.y
    hull_path = hull_path.transformed(matplotlib.transforms.Affine2D().translate(translate_x, translate_y))

    return hull_path.contains_point((cp_x,cp_y))

def modelData():
     # MuJoCo robot model with obstacle
     desc_dir = '../../aliengo_models'
     xml = desc_dir + '/xml/aliengoB.xml'
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
     return data_muj

def viewMuJoCo(model_muj, data_muj, torques, pos, force_body, userview):

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
               #data_muj.qpos = pos[i]
               mujoco.mj_step(model_muj, data_muj)
               renderer.update_scene(data_muj)
               viewer.sync()

               

               time_until_next_step = model_muj.opt.timestep - (time.time() - step_start)
               '''print(i)
               print(data_muj.qpos)'''
               if time_until_next_step > 0:
                    time.sleep(0.1)#time_until_next_step)
               i += 1
     '''mujoco.mj_step(model_muj, data_muj)
     renderer.update_scene(data_muj)
     viewer.sync()'''
     renderer.close()
     return 

class MPS:
    def __init__(self, X_safe, iter_ctrl, lim_tau, N_mps, step_rand, force_body, jmax_compare, jmin_compare, lim_vel, default_joint_angles, scaling_factors, contact_height, feet_geom):
        self.X_safe = X_safe
        self.iter_ctrl = iter_ctrl
        self.lim_tau = lim_tau
        self.N_mps = N_mps
        self.step_rand = step_rand
        self.force_body = force_body
        self.jmax_compare = jmax_compare
        self.jmin_compare = jmin_compare
        self.lim_vel = lim_vel

        self.default_joint_angles = default_joint_angles
        self.scaling_factors = scaling_factors
        self.contact_height = contact_height
        self.feet_geom = feet_geom

        self.percentage_noise = 0.01
        # Max pos considering angles insetad of quaternions
        self.max_pos = [0.5, 0.5, 0.5] + [0, 0, 0]*4
        #self.max_pos = [0., 0., 0.] + [0, 0, 0]*4
        self.max_vel = [5, 5, 5, 0.3, 0.3, 0.3] + [0.1]*12
        self.max_rot = 5*np.pi/180

        self.data = modelData()

    def isRecSingle(self, lim_tau, data_muj, curr_step, actor_network, previous_actions, current_actions):
        iter_mps = 0
        curr_step_i = copy.copy(curr_step)
        #print('data_muj',data_muj.qpos, data_muj.qvel)
        
        #data = modelData()#self.data
        #data = self.data
        
        torques = []
        pos = []
        force_body = self.data.model.body('trunk').id
        quat_no_noise = data_muj.qpos[3:7].copy() #+ np.random.normal(0,self.percentage_noise,1)*self.max_pos[3:7]

        quat_q0 = quat_no_noise[0]
        quat_q1 = quat_no_noise[1]
        quat_q2 = quat_no_noise[2]
        quat_q3 = quat_no_noise[3]
        noise = np.random.normal(0,self.percentage_noise,3)*self.max_rot
        roll = np.arctan2(2*(quat_q0*quat_q1 + quat_q2*quat_q3), 1 - 2*(quat_q1**2+quat_q2**2)) 
        pitch = np.arcsin(2*(quat_q0*quat_q2 - quat_q3*quat_q1)) 
        yaw = np.arctan2(2*(quat_q0*quat_q3 + quat_q1*quat_q2), 1 - 2*(quat_q2**2+quat_q3**2)) #'''

        '''quat_mat = np.zeros(9)
        mujoco.mju_quat2Mat(quat_mat, quat_no_noise)



        mat_quat = np.zeros(4)
        mujoco.mju_mat2Quat(mat_quat,quat_mat)'''
        #print('quat1', quat_no_noise, 'quat2',mat_quat)
        quat_noise = np.zeros(4)
        mujoco.mju_euler2Quat(quat_noise, np.array([roll + noise[0], pitch + noise[1], yaw + noise[2]]), 'xyz')
        #print('quat1', quat_no_noise, 'quat2', quat_noise)

        self.data.qpos = np.concatenate((data_muj.qpos[:3].copy() + np.random.normal(0,self.percentage_noise,3)*self.max_pos[:3],
                                    quat_noise,
                                    data_muj.qpos[7:].copy() + np.random.normal(0,self.percentage_noise,12)*self.max_pos[3:]))


        self.data.qvel = data_muj.qvel.copy() + np.random.normal(0,self.percentage_noise,18)*self.max_vel
        #data.qacc = data_muj.qacc.copy() #+ np.random.normal(0,self.percentage_noise,18)*self.max_acc
       # print('data.qpos',data.qpos)
        #data.qacc_warmstart = data.qacc.copy()#0
        #print('dataA',data.qpos, data.qvel)
        mujoco.mj_forward(self.data.model,self.data)
        no_noise = [data_muj.qpos[0],data_muj.qpos[1], data_muj.qpos[2], roll, pitch, yaw, data_muj.qvel]
        with_noise = [self.data.qpos[0], self.data.qpos[1], self.data.qpos[2],roll + noise[0], pitch + noise[1], yaw + noise[2], self.data.qvel]
        #print('data',data.qpos, data.qvel)
        data_sim = copy.deepcopy(self.data)
        ## Trunk, hip and knee positions
        z_coordinates = np.array([self.data.body('trunk').xpos[2], self.data.body('FL_hip').xpos[2], self.data.body('FR_hip').xpos[2],
                                self.data.body('RL_hip').xpos[2], self.data.body('RR_hip').xpos[2]])
        
        data_compare = np.array([self.data.qpos[7],self.data.qpos[9],self.data.qpos[10],self.data.qpos[12],self.data.qpos[13],self.data.qpos[15],self.data.qpos[16],self.data.qpos[18]])
        check_pos = (np.any(data_compare > self.jmax_compare) or np.any(data_compare < self.jmin_compare))

        if np.any(z_coordinates < self.X_safe) or np.any(np.abs(self.data.qvel[6:]) > self.lim_vel) or check_pos: ## x is not in X_safe
            return False, iter_mps, no_noise, with_noise
        
        ## Simulate x with pi_hat
        j = 0
        curr_step += 1
        '''if np.any(data.xfrc_applied[self.force_body][:3]!= np.array([0,0,0])):
            force = True
        else:
            force  = False
        if not (curr_step >= self.step_rand and curr_step < self.step_rand + 20) and force:
            data.xfrc_applied[self.force_body][:3] = np.array([0,0,0])'''
        q_muj = self.data.qpos.copy()
        v_muj = self.data.qvel.copy()
        torque_values = 4*[-1.6, 0.0, 0.0]

        qDes = 0.5 * current_actions + np.array(self.default_joint_angles)
        qDes = np.clip(qDes, [-1.22,0.,-2.78,-1.22,0.,-2.78,-1.22,0.,-2.78,-1.22,0.,-2.78],
                            [1.22,1.8,-0.65,1.22,1.8,-0.65,1.22,1.8,-0.65,1.22,1.8,-0.65])
      
        while j < self.iter_ctrl:
            u_nominal = 3 * (- v_muj[6:]) + 100 * (qDes - q_muj[7:]) +  torque_values
            self.data.ctrl = np.clip(u_nominal, -lim_tau, lim_tau)
            torques.append(self.data.ctrl.copy())
            pos.append(self.data.qpos.copy())
            j += 1
            mujoco.mj_step(self.data.model, self.data)
            q_muj = self.data.qpos.copy()
            v_muj = self.data.qvel.copy()
        curr_step += 1
        
        nominal = False#True#
        for i in range(0, self.N_mps): ##simulated steps
            '''if not (curr_step >= self.step_rand and curr_step < self.step_rand + 20) and force:# and loop == 1:
                data.xfrc_applied[self.force_body][:3] = np.array([0,0,0])'''
            z_coordinates = np.array([self.data.body('trunk').xpos[2], self.data.body('FL_hip').xpos[2], self.data.body('FR_hip').xpos[2],
                                    self.data.body('RL_hip').xpos[2], self.data.body('RR_hip').xpos[2]])
            data_compare = np.array([self.data.qpos[7],self.data.qpos[9],self.data.qpos[10],self.data.qpos[12],self.data.qpos[13],self.data.qpos[15],self.data.qpos[16],self.data.qpos[18]])
            check_pos = (np.any(data_compare > self.jmax_compare) or np.any(data_compare < self.jmin_compare))

            xy_coords = []
            j = 0
            
            #phase_all = False
            #if np.all(np.array([data.geom(self.feet_geom[0]).xpos[2],data.geom(self.feet_geom[1]).xpos[2],data.geom(self.feet_geom[2]).xpos[2],data.geom(self.feet_geom[3]).xpos[2]]) < self.contact_height):
                #phase_all = True
            phase_all = np.all(np.array([self.data.geom(self.feet_geom[0]).xpos[2],self.data.geom(self.feet_geom[1]).xpos[2],self.data.geom(self.feet_geom[2]).xpos[2],self.data.geom(self.feet_geom[3]).xpos[2]]) < self.contact_height)
            if phase_all:# Check that the four feet are in contact with the floor
                for k in self.feet_geom:
                    #print('all feet')
                    xy_coords.append([self.data.geom(k).xpos[0],self.data.geom(k).xpos[1]])
                
                if capture_point_check(self.data, xy_coords):
                    return True, iter_mps, no_noise, with_noise
            if np.any(z_coordinates < self.X_safe) or np.any(np.abs(self.data.qvel[6:]) > self.lim_vel) or check_pos: ## x is not in X_safe
                '''print('z_coordinates')
                print(z_coordinates)
                print('np.abs(data.qvel[6:])',self.lim_vel)
                print(np.abs(data.qvel[6:]))
                print('check_pos')
                print(check_pos)'''
                userview = input("B Enter N or n to stop, T or t for top, F or f for front, S or s side: ")
                while userview != 'N' and userview != 'n':
                    print('len(torques)',len(torques))
                    copy_data = copy.deepcopy(data_sim)
                    viewMuJoCo(self.data.model, copy_data, torques, pos, force_body, userview)
                    userview = input("Enter N or n to stop, T or t for top, F or f for front, S or s side: ")


                return False, iter_mps, no_noise, with_noise
            
            ## Simulate x with pi_rec
            new_actions1 = compute_actions(self.data, self.scaling_factors, previous_actions, nominal, actor_network, True)
            previous_actions = current_actions
            current_actions = swap_legs(new_actions1)
            qDes = 0.5 * current_actions + np.array(self.default_joint_angles)
            qDes = np.clip(qDes, [-1.22,0.,-2.78,-1.22,0.,-2.78,-1.22,0.,-2.78,-1.22,0.,-2.78],
                                [1.22,1.8,-0.65,1.22,1.8,-0.65,1.22,1.8,-0.65,1.22,1.8,-0.65])
            
            iter_mps += 1
            
            j = 0
            while j < self.iter_ctrl:
                j += 1
                u_nominal = 3 * (- v_muj[6:]) + 100 * (qDes - q_muj[7:]) +  torque_values
                self.data.ctrl = np.clip(u_nominal, -lim_tau, lim_tau)
                torques.append(self.data.ctrl.copy())
                pos.append(self.data.qpos.copy())
                mujoco.mj_step(self.data.model, self.data)
                q_muj = self.data.qpos.copy()
                v_muj = self.data.qvel.copy()

            curr_step += 1
      #  print('no stop')
        userview = input("A Enter N or n to stop, T or t for top, F or f for front, S or s side: ")
        while userview != 'N' and userview != 'n':
            print('len(torques)',len(torques))
            copy_data = copy.deepcopy(data_sim)
            viewMuJoCo(self.data.model, copy_data, torques, pos, force_body, userview)
            userview = input("Enter N or n to stop, T or t for top, F or f for front, S or s side: ")

        return False, iter_mps, no_noise, with_noise

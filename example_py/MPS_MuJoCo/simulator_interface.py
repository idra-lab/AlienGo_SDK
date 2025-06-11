import os
import time
import mujoco
from mujoco import viewer
import mujoco.gl_context
import numpy as np
import rospy
import publish_subscribe_sim
from config_loader import load_config, load_actor_network

class MujocoSim():
    def __init__(self):

        self.model, self.data = self.modelData()
        self.force_body = self.model.body('trunk').id
        self.lim_tau = 40.
        self.lim_vel = 26.5
        self.iter_ctrl = 5
        self.qDes = np.zeros(12)#None

    def modelData(self):
        # MuJoCo robot model
        xml = '../MPS_robot_sensors/aliengo_models/xml/aliengo.xml'
        model_muj = mujoco.MjModel.from_xml_path(xml)
        data_muj = mujoco.MjData(model_muj)
        
        self.q0 = np.array([0, 0,  0.07,
                            1,   0, 0,  0,
                            -0.7, 1.2, -2.7,
                             0.7, 1.2, -2.7,
                            -0.7, 1.2, -2.7,   
                             0.7, 1.2, -2.7])       
        data_muj.qpos = self.q0
        data_muj.qvel = np.zeros(18)
        mujoco.mj_forward(model_muj,data_muj)
        return model_muj, data_muj
    
    def simulate(self, Kp, Kd, viewer_muj, renderer, iter_loop):
        
        joint_pos, _, forward_torques = pubSub.wait_for_all_messages()
        self.qDes = joint_pos
        #self.qDes = self.joint_pos
        #print(iter_loop)
        # Add force for 200 iterations
        if iter_loop >= 5000 and iter_loop < 5200:
           self.data.xfrc_applied[self.force_body][2] = 180
        else:
            self.data.xfrc_applied[self.force_body][2] = 0

        step_start = time.time()
        #sim_time_start = self.data.time
        if np.any(self.qDes != np.zeros(12)):
            q_muj = self.data.qpos.copy()
            v_muj = self.data.qvel.copy()
            #while j < self.iter_ctrl:
            
            u_nominal = Kd * (- v_muj[6:]) + Kp * (self.qDes - q_muj[7:]) +  forward_torques
            self.data.ctrl = np.clip(u_nominal, -self.lim_tau, self.lim_tau)
            mujoco.mj_step(self.model, self.data)
            q_muj = self.data.qpos.copy()
            v_muj = self.data.qvel.copy()
            renderer.update_scene(self.data)
            viewer_muj.sync()
        #time.sleep(0.02)
        pubSub.publisher(self.data.qpos[7:].copy(), self.data.qvel[6:].copy(),
                         self.data.sensor('Body_Gyro').data.copy(),
                         self.data.sensor('Body_Quat').data.copy(), 
                         self.data.sensor('Body_Acc').data.copy(),
                         self.data.qpos[:7], self.data.qvel[:6])
        #sim_time_elapsed = self.data.time - sim_time_start
        #real_time_elapsed = time.time() - step_start
        #if real_time_elapsed > 0:
        #    realtime_factor = sim_time_elapsed / real_time_elapsed
        time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        


if __name__ == '__main__':
    try:
        config_path = 'config.yaml'
        config = load_config(config_path)
        pubSub = publish_subscribe_sim.PubSub()
        pubSub.init_publishers()
        pubSub.init_subscribers()
        mujoco_sim = MujocoSim()
        renderer = mujoco.Renderer(mujoco_sim.model)
        viewer_muj = mujoco.viewer.launch_passive(mujoco_sim.model, mujoco_sim.data)
        while pubSub.imu_data_pub.get_num_connections() < 1 or pubSub.odom_data_pub.get_num_connections() < 1 or pubSub.joint_state_pub.get_num_connections() < 1:
            pass#print('Waiting for subscribers')
        iter_loop = 0
        while not rospy.is_shutdown():
            mujoco_sim.simulate(100, 3, viewer_muj, renderer, iter_loop)
            #print("mujoco_sim.data.sensor('Body_Gyro').data.copy()",mujoco_sim.data.sensor('Body_Gyro').data.copy())
            iter_loop += 1
    except Exception as e:
        print(e)
        rospy.signal_shutdown("killed")
        renderer.close()

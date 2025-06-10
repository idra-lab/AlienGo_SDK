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
        self.lim_tau = 40.
        self.lim_vel = 26.5
        self.iter_ctrl = 5
        self.qDes = np.zeros(12)#None

    def modelData(self):
        # MuJoCo robot model
        xml = '../MPS_robot_sensors/aliengo_models/xml/aliengo.xml'
        model_muj = mujoco.MjModel.from_xml_path(xml)
        data_muj = mujoco.MjData(model_muj)
        th_q0 = 0.75
        self.q0 = np.array([0, 0,  0.07,
                            1,   0, 0,  0,
                            -0.7, 1.2, -2.7,
                             0.7, 1.2, -2.7,
                            -0.7, 1.2, -2.7,   
                             0.7, 1.2, -2.7])                   # RR
        print('data_muj.qpos',data_muj.qpos)
        print('data_muj.qvel',data_muj.qvel)
        data_muj.qpos = self.q0
        data_muj.qvel = np.zeros(18)
        mujoco.mj_forward(model_muj,data_muj)
        return model_muj, data_muj
    
    def simulate(self, Kp, Kd, viewer_muj, renderer):
        '''th_q0 = 0.75
        q0 = np.array([
                        0., th_q0,  -1.5,                    # FL
                        0., th_q0,  -1.5,                    # FR
                        0., th_q0,  -1.5,                    # RL
                        0., th_q0,  -1.5])                   # RR'''
        
        print ('qDes',self.qDes)
       # if np.any(self.qDes != None) and np.any(self.qDes != np.zeros(12)):
        print('++++++++++++++++ before wait in sim')
        joint_pos, _, forward_torques = pubSub.wait_for_all_messages()
        self.qDes = joint_pos
        print('++++++++++++++++ after wait in sim')
        #self.qDes = self.joint_pos

        step_start = time.time()
        sim_time_start = self.data.time
        if np.any(self.qDes != np.zeros(12)):
            q_muj = self.data.qpos.copy()
            v_muj = self.data.qvel.copy()
            #while j < self.iter_ctrl:
            
            u_nominal = Kd * (- v_muj[6:]) + Kp * (self.qDes - q_muj[7:]) +  forward_torques
            self.data.ctrl = np.clip(u_nominal, -self.lim_tau, self.lim_tau)
            print('self.data.ctrl',self.data.ctrl)
            mujoco.mj_step(self.model, self.data)
            q_muj = self.data.qpos.copy()
            v_muj = self.data.qvel.copy()
            renderer.update_scene(self.data)
            viewer_muj.sync()
        time.sleep(0.02)
        print('before publish')
        pubSub.publisher(self.data.qpos[7:].copy(), self.data.qvel[6:].copy(),
                            self.data.sensor('Body_Gyro').data.copy(), 
                            self.data.sensor('Body_Quat').data.copy(), 
                            self.data.sensor('Body_Acc').data.copy(),
                            self.data.qpos[:7], self.data.qvel[:6])
        print('after publish')
        
        print('after wait')
        sim_time_elapsed = self.data.time - sim_time_start
        real_time_elapsed = time.time() - step_start
        if real_time_elapsed > 0:
            realtime_factor = sim_time_elapsed / real_time_elapsed
            print(f"Real-time factor: {realtime_factor:.3f}")
        time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
        print('time_until_next_step',time_until_next_step)
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
            print('Waiting for subscribers')
        while not rospy.is_shutdown():
            mujoco_sim.simulate(100, 3, viewer_muj, renderer)
            #print("mujoco_sim.data.sensor('Body_Gyro').data.copy()",mujoco_sim.data.sensor('Body_Gyro').data.copy())
            
    except Exception as e:
        print(e)
        rospy.signal_shutdown("killed")
        renderer.close()

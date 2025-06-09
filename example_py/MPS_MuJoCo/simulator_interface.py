import os
import time
import mujoco
from mujoco import viewer
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

    def modelData(self):
        # MuJoCo robot model
        xml = '../MPS_robot_sensors/aliengo_models/xml/aliengo.xml'
        spec = mujoco.MjSpec()
        spec.from_file(xml)
        model_muj = spec.compile()
        data_muj = mujoco.MjData(model_muj)
        th_q0 = 0.75
        q0 = np.array([0.,    0.,  0.39175, 1., 0., 0., 0., # trunk
                        0., th_q0,  -1.5,                    # FL
                        0., th_q0,  -1.5,                    # FR
                        0., th_q0,  -1.5,                    # RL
                        0., th_q0,  -1.5])                   # RR
        
        data_muj.qpos = q0
        data_muj.qvel = np.zeros(18)
        mujoco.mj_forward(model_muj,data_muj)
        return model_muj, data_muj
    
    def simulate(self, Kp, Kd, viewer_muj, renderer):
        th_q0 = 0.75
        q0 = np.array([
                        0., th_q0,  -1.5,                    # FL
                        0., th_q0,  -1.5,                    # FR
                        0., th_q0,  -1.5,                    # RL
                        0., th_q0,  -1.5])                   # RR
        if np.all(pubSub.joint_pos != np.zeros(12)):
            qDes = pubSub.joint_pos
        else:
            qDes = q0
        forward_torques = [-1.6,0,0]*4#pubSub.joint_eff

        j = 0
        q_muj = self.data.qpos.copy()
        v_muj = self.data.qvel.copy()
        while j < self.iter_ctrl:
            
            step_start = time.time()
            u_nominal = Kd * (- v_muj[6:]) + Kp * (qDes - q_muj[7:]) +  forward_torques
            self.data.ctrl = np.clip(u_nominal, -self.lim_tau, self.lim_tau)
            j += 1
            mujoco.mj_step(self.model, self.data)
            q_muj = self.data.qpos.copy()
            v_muj = self.data.qvel.copy()
            renderer.update_scene(self.data)
            viewer_muj.sync()

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
        while not rospy.is_shutdown():
            mujoco_sim.simulate(100, 3, viewer_muj, renderer)
            print("mujoco_sim.data.sensor('Body_Gyro').data.copy()",mujoco_sim.data.sensor('Body_Gyro').data.copy())
            pubSub.publisher(mujoco_sim.data.qpos[7:], mujoco_sim.data.qvel[6:],
                             mujoco_sim.data.sensor('Body_Gyro').data.copy(), 
                             mujoco_sim.data.sensor('Body_Quat').data.copy(), 
                             mujoco_sim.data.sensor('Body_Acc').data.copy(),
                             mujoco_sim.data.qpos[:7], mujoco_sim.data.qvel[:6])
    except:
        rospy.signal_shutdown("killed")
        renderer.close()

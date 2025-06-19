import time
import mujoco
from mujoco import viewer
import numpy as np
import rospy
import publish_subscribe_sim
from config_loader import load_config

class MujocoSim():
    def __init__(self, config):

        self.model, self.data = self.modelData(config['controller']['robot']['model'])
        self.force_body = self.model.body('trunk').id
        self.lim_tau = config['policy']['robot']['lim_tau']
        self.lim_vel = config['policy']['robot']['lim_vel']
        self.iter_ctrl = config['controller']['robot']['decimation']
        self.qDes = np.zeros(12)

    def modelData(self, path):
        # MuJoCo robot model
        model_muj = mujoco.MjModel.from_xml_path(path)
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
        # Save data from ROS messages
        self.qDes = pubSub.joint_pos
        forward_torques = pubSub.joint_eff

        # Apply force for 400 iterations
        if iter_loop >= 10000 and iter_loop < 10400:
           self.data.xfrc_applied[self.force_body][2] = 0*160
        else:
            self.data.xfrc_applied[self.force_body][2] = 0

        step_start = time.time()
        
        if np.any(self.qDes != np.zeros(12)):
            
            q_muj = self.data.qpos.copy()
            v_muj = self.data.qvel.copy()
            
            # Compute and apply torques for one simulation time step
            u_nominal = Kd * (- v_muj[6:]) + Kp * (self.qDes - q_muj[7:]) +  forward_torques
            self.data.ctrl = np.clip(u_nominal, -self.lim_tau, self.lim_tau)
            mujoco.mj_step(self.model, self.data)
            q_muj = self.data.qpos.copy()
            v_muj = self.data.qvel.copy()
            renderer.update_scene(self.data)
            viewer_muj.sync()

        # Publish data for controller
        pubSub.publisher(self.data.qpos[7:].copy(), self.data.qvel[6:].copy(),
                         self.data.sensor('Body_Gyro').data.copy(),
                         self.data.sensor('Body_Quat').data.copy(), 
                         self.data.sensor('Body_Acc').data.copy(),
                         self.data.qpos[:7], self.data.qvel[:6])
        
        # Wait for the time the control action was simulated
        time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        


if __name__ == '__main__':
    try:
        # Load values for the simulator
        config_path = 'config.yaml'
        config = load_config(config_path)

        # Initialize publishers and subscribers
        pubSub = publish_subscribe_sim.PubSub()
        pubSub.init_publishers()
        pubSub.init_subscribers()

        # Initialize MuJoCo
        mujoco_sim = MujocoSim(config)
        renderer = mujoco.Renderer(mujoco_sim.model)
        viewer_muj = mujoco.viewer.launch_passive(mujoco_sim.model, mujoco_sim.data)

        # Wait for subscribers
        while pubSub.imu_data_pub.get_num_connections() < 1 or pubSub.odom_data_pub.get_num_connections() < 1 or pubSub.joint_state_pub.get_num_connections() < 1:
            pass
        
        iter_loop = 0
        while not rospy.is_shutdown():
            mujoco_sim.simulate(100, 3, viewer_muj, renderer, iter_loop)
            iter_loop += 1
    except Exception as e:
        print(e)
        rospy.signal_shutdown("killed")
        renderer.close()

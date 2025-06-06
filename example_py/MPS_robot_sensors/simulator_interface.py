import mujoco
from mujoco import viewer
import numpy as np
import rospy

class MujocoSim():
    def __init__(self):

        self.model, self.data = self.modelData()
        self.lim_tau
        self.iter_ctrl

    def modelData(self):
        # MuJoCo robot model
        xml = '/aliengo_models/xml/aliengo.xml'
        spec = mujoco.MjSpec()
        spec.from_file(xml)
        model_muj = spec.compile()
        data_muj = mujoco.MjData(model_muj)

        return model_muj, data_muj
    
    def simulate(self, Kp, Kd, forward_torques):
        j = 0
        while j < self.iter_ctrl:
            u_nominal = Kd * (- v_muj[6:]) + Kp * (qDes - q_muj[7:]) +  forward_torques
            self.data.ctrl = np.clip(u_nominal, -self.lim_tau, self.lim_tau)
            j += 1
            mujoco.mj_step(self.data.model, self.data)
            q_muj = self.data.qpos.copy()
            v_muj = self.data.qvel.copy()

if __name__ == '__main__':
    mujoco_sim = MujocoSim()
    while not rospy.is_shutdown():
        mujoco_sim.simulate(Kp, Kd, forward_torques)

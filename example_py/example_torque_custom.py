#!/usr/bin/python

import sys
import time
import math
import numpy as np

sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# low cmd
TARGET_PORT = 8007
LOCAL_PORT = 8082
TARGET_IP = "192.168.123.10"   # target IP address

LOW_CMD_LENGTH = 610
LOW_STATE_LENGTH = 771

def jointLinearInterpolation(initPos, targetPos, rate):

    #rate = np.fmin(np.fmax(rate, 0.0), 1.0)
    if rate > 1.0:
        rate = 1.0
    elif rate < 0.0:
        rate = 0.0

    p = initPos*(1-rate) + targetPos*rate
    return p


if __name__ == '__main__':

    d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
         'FL_0':3, 'FL_1':4, 'FL_2':5, 
         'RR_0':6, 'RR_1':7, 'RR_2':8, 
         'RL_0':9, 'RL_1':10, 'RL_2':11 }
    PosStopF  = math.pow(10,9)
    VelStopF  = 16000.0
    HIGHLEVEL = 0x00
    LOWLEVEL  = 0xff
    sin_mid_q = [0.0, 0.9, -1.7]
    dt = 0.002
    qInit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    qDes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sin_count = 0
    rate_count = 0
    Kp = [0, 0, 0]
    Kd = [0, 0, 0]

    udp = sdk.UDP(LOCAL_PORT, TARGET_IP, TARGET_PORT, LOW_CMD_LENGTH, LOW_STATE_LENGTH, -1)
    #udp = sdk.UDP(8082, "192.168.123.10", 8007, 610, 771)
    safe = sdk.Safety(sdk.LeggedType.Aliengo)

    
    cmd = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)
    cmd.levelFlag = LOWLEVEL

    #print(dir(state))


    Tpi = 0
    motiontime = 0
    while True:
        time.sleep(0.002)
        motiontime += 1


        #print(motiontime)
        #print(state.imu.rpy)
        
        
        udp.Recv()
        udp.GetRecv(state)
        
        if( motiontime >= 0):

            # first, get record initial position
            if( motiontime >= 0 and motiontime < 10):
                qInit[0] = state.motorState[d['FR_0']].q
                qInit[1] = state.motorState[d['FR_1']].q
                qInit[2] = state.motorState[d['FR_2']].q

                qInit[3] = state.motorState[d['FL_0']].q
                qInit[4] = state.motorState[d['FL_1']].q
                qInit[5] = state.motorState[d['FL_2']].q

                qInit[6] = state.motorState[d['RR_0']].q
                qInit[7] = state.motorState[d['RR_1']].q
                qInit[8] = state.motorState[d['RR_2']].q

                qInit[9] = state.motorState[d['RL_0']].q
                qInit[10] = state.motorState[d['RL_1']].q
                qInit[11] = state.motorState[d['RL_2']].q

                #qDes = qInit

                #print("qInit: ", qInit)

                #print(qInit)

            
            # second, move to the origin point of a sine movement with Kp Kd
            if( motiontime >= 10 and motiontime < 400.0):
                rate_count += 1
                rate = rate_count/400.0                       # needs count to 200
                # Kp = [5, 5, 5]
                # Kd = [1, 1, 1]
                Kp = [20, 20, 20]
                Kd = [1, 1, 1]
                
                qDes[0] = jointLinearInterpolation(qInit[0], sin_mid_q[0], rate)
                qDes[1] = jointLinearInterpolation(qInit[1], sin_mid_q[1], rate)
                qDes[2] = jointLinearInterpolation(qInit[2], sin_mid_q[2], rate)

                qDes[3] = jointLinearInterpolation(qInit[3], sin_mid_q[0], rate)
                qDes[4] = jointLinearInterpolation(qInit[4], sin_mid_q[1], rate)
                qDes[5] = jointLinearInterpolation(qInit[5], sin_mid_q[2], rate)

                qDes[6] = jointLinearInterpolation(qInit[6], sin_mid_q[0], rate)
                qDes[7] = jointLinearInterpolation(qInit[7], sin_mid_q[1], rate)
                qDes[8] = jointLinearInterpolation(qInit[8], sin_mid_q[2], rate)

                qDes[9] = jointLinearInterpolation(qInit[9], sin_mid_q[0], rate)
                qDes[10] = jointLinearInterpolation(qInit[10], sin_mid_q[1], rate)
                qDes[11] = jointLinearInterpolation(qInit[11], sin_mid_q[2], rate)
            
            # last, do sine wave
            freq_Hz = 0.5
            # freq_Hz = 5
            freq_rad = freq_Hz * 2* math.pi
            t = dt*sin_count

            if( motiontime >= 600.0):

                """ Kp = [100, 100, 100]
                Kd = [3, 3, 3] """
                Kp = [20, 20, 20]
                Kd = [1, 1, 1]

                sin_count += 1
                # sin_joint1 = 0.6 * sin(3*M_PI*sin_count/1000.0)
                # sin_joint2 = -0.9 * sin(3*M_PI*sin_count/1000.0)
                sin_joint1 = 0.2 * math.sin(t*freq_rad)
                sin_joint2 = -0.3 * math.sin(t*freq_rad)
                qDes[0] = sin_mid_q[0]
                qDes[1] = sin_mid_q[1] + sin_joint1
                qDes[2] = sin_mid_q[2] + sin_joint2

                qDes[3] = sin_mid_q[0]
                qDes[4] = sin_mid_q[1] + sin_joint1
                qDes[5] = sin_mid_q[2] + sin_joint2

                qDes[6] = sin_mid_q[0]
                qDes[7] = sin_mid_q[1] + sin_joint1
                qDes[8] = sin_mid_q[2] + sin_joint2

                qDes[9] = sin_mid_q[0]
                qDes[10] = sin_mid_q[1] + sin_joint1
                qDes[11] = sin_mid_q[2] + sin_joint2
                # qDes[2] = sin_mid_q[2]


            # Calculate the torques to apply
            q_pos = []
            q_vel = []
            for ds in d:
                q_pos.append(state.motorState[d[ds]].q)
                q_vel.append(state.motorState[d[ds]].dq)

            q_pos = np.array(q_pos)
            q_vel = np.array(q_vel)

            kp_custom = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
            kd_custom = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

            torques1 = (kp_custom * 
                    (
                        qDes
                        - q_pos
                    ) - kd_custom
                    * q_vel
                )
            torques = torques1.copy()

            for i in range(0, len(torques1), 3):  # Step by 3 to handle each group
                torques[i] = np.clip(torques1[i], -25.0, 25.0)         # First element in group
                torques[i + 1] = np.clip(torques1[i + 1], -25.0, 25.0)   # Second element in group
                torques[i + 2] = np.clip(torques1[i + 2], -25.0, 25.0)   # Third element in group
            
            #print("torques: ", torques)

            cmd.motorCmd[d['FR_0']].q = PosStopF
            cmd.motorCmd[d['FR_0']].dq = VelStopF
            cmd.motorCmd[d['FR_0']].Kp = 0
            cmd.motorCmd[d['FR_0']].Kd = 0
            cmd.motorCmd[d['FR_0']].tau = torques[0]

            cmd.motorCmd[d['FR_1']].q = PosStopF
            cmd.motorCmd[d['FR_1']].dq = VelStopF
            cmd.motorCmd[d['FR_1']].Kp = 0
            cmd.motorCmd[d['FR_1']].Kd = 0
            cmd.motorCmd[d['FR_1']].tau = torques[1]

            cmd.motorCmd[d['FR_2']].q = PosStopF
            cmd.motorCmd[d['FR_2']].dq = VelStopF
            cmd.motorCmd[d['FR_2']].Kp = 0
            cmd.motorCmd[d['FR_2']].Kd = 0
            cmd.motorCmd[d['FR_2']].tau = torques[2]


            cmd.motorCmd[d['FL_0']].q = PosStopF
            cmd.motorCmd[d['FL_0']].dq = VelStopF
            cmd.motorCmd[d['FL_0']].Kp = 0
            cmd.motorCmd[d['FL_0']].Kd = 0
            cmd.motorCmd[d['FL_0']].tau = torques[3]

            cmd.motorCmd[d['FL_1']].q = PosStopF
            cmd.motorCmd[d['FL_1']].dq = VelStopF
            cmd.motorCmd[d['FL_1']].Kp = 0
            cmd.motorCmd[d['FL_1']].Kd = 0
            cmd.motorCmd[d['FL_1']].tau = torques[4]

            cmd.motorCmd[d['FL_2']].q =  PosStopF
            cmd.motorCmd[d['FL_2']].dq = VelStopF
            cmd.motorCmd[d['FL_2']].Kp = 0
            cmd.motorCmd[d['FL_2']].Kd = 0
            cmd.motorCmd[d['FL_2']].tau = torques[5]


            cmd.motorCmd[d['RR_0']].q = PosStopF
            cmd.motorCmd[d['RR_0']].dq = VelStopF
            cmd.motorCmd[d['RR_0']].Kp = 0
            cmd.motorCmd[d['RR_0']].Kd = 0
            cmd.motorCmd[d['RR_0']].tau = torques[6]

            cmd.motorCmd[d['RR_1']].q = PosStopF
            cmd.motorCmd[d['RR_1']].dq = VelStopF
            cmd.motorCmd[d['RR_1']].Kp = 0
            cmd.motorCmd[d['RR_1']].Kd = 0
            cmd.motorCmd[d['RR_1']].tau = torques[7]

            cmd.motorCmd[d['RR_2']].q =  PosStopF
            cmd.motorCmd[d['RR_2']].dq = VelStopF
            cmd.motorCmd[d['RR_2']].Kp = 0
            cmd.motorCmd[d['RR_2']].Kd = 0
            cmd.motorCmd[d['RR_2']].tau = torques[8]


            cmd.motorCmd[d['RL_0']].q = PosStopF
            cmd.motorCmd[d['RL_0']].dq = VelStopF
            cmd.motorCmd[d['RL_0']].Kp = 0
            cmd.motorCmd[d['RL_0']].Kd = 0
            cmd.motorCmd[d['RL_0']].tau = torques[9]

            cmd.motorCmd[d['RL_1']].q = PosStopF
            cmd.motorCmd[d['RL_1']].dq = VelStopF
            cmd.motorCmd[d['RL_1']].Kp = 0
            cmd.motorCmd[d['RL_1']].Kd = 0
            cmd.motorCmd[d['RL_1']].tau = torques[10]

            cmd.motorCmd[d['RL_2']].q =  PosStopF
            cmd.motorCmd[d['RL_2']].dq = VelStopF
            cmd.motorCmd[d['RL_2']].Kp = 0
            cmd.motorCmd[d['RL_2']].Kd = 0
            cmd.motorCmd[d['RL_2']].tau = torques[11]
            # cmd.motorCmd[d['FR_2']].tau = 2 * sin(t*freq_rad)

            """ if motiontime > 100:
                exit() """


        # if(motiontime > 10):
        #     safe.PowerProtect(cmd, state, 1)


        udp.SetSend(cmd)
        udp.Send()

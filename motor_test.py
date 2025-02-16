import sys
import time
import math
import numpy as np

sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# low cmd
TARGET_PORT = 8007
LOCAL_PORT = 8091
TARGET_IP = "192.168.123.10"   # target IP address

LOW_CMD_LENGTH = 610
LOW_STATE_LENGTH = 771

if __name__ == '__main__':

    d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
         'FL_0':3, 'FL_1':4, 'FL_2':5, 
         'RR_0':6, 'RR_1':7, 'RR_2':8, 
         'RL_0':9, 'RL_1':10, 'RL_2':11 }
    HIGHLEVEL = 0x00
    LOWLEVEL  = 0xff

    udp = sdk.UDP(LOCAL_PORT, TARGET_IP, TARGET_PORT, LOW_CMD_LENGTH, LOW_STATE_LENGTH, -1)
    #udp = sdk.UDP(8082, "192.168.123.10", 8007, 610, 771)
    safe = sdk.Safety(sdk.LeggedType.Aliengo)

    
    cmd = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)
    cmd.levelFlag = LOWLEVEL

    print(state.footForce)
    #print(dir(state))
    #print(cmd.robotID)
    #print(state.motorState[d['FR_0']].q)
    #print(state)
    #print(state.motorState[0].q)

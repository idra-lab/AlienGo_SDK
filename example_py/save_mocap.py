"""
    Minimal usage example
    Connects to QTM and streams 3D data forever
    (start QTM first, load file, Play->Play with Real-Time output)
"""

import asyncio
import qtm_rt
import numpy as np
import time
import time
import csv
import sys
import signal

def signal_handler(signal, frame):
    name_save = nameFile + "_pos.csv"
    with open(name_save, 'a', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(data_pos)
    myfile.close()
            

    name_save = nameFile + "_rot.csv"
    with open(name_save, 'a', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(data_rot)
    myfile.close()
    print('Data saved')
    sys.exit(0)

def on_packet(packet):
    global last_pos
    global last_rot
    """ Callback function that is called everytime a data packet arrives from QTM """
    _, bodies = packet.get_6d()
    for i, body in enumerate(bodies):
        pos, rot = body
        x, y, z = pos
        rot_elements = [j for j in rot[0]]
        if i == 0:
            last_pos = np.array([x,y,z])
            last_rot = np.array(rot_elements)


async def setup():
    """ Main function """
    connection = await qtm_rt.connect("192.168.225.1")
    if connection is None:
        return

    await connection.stream_frames(components=["6d"], on_packet=on_packet)


async def main():
    asyncio.create_task(setup())

    await asyncio.sleep(1)
    global data_pos = []
    global data_rot = []
    time_file = time.localtime()
    global nameFile = 'data' + str(time_file.tm_mday) + "_" + str(time_file.tm_mon) + "_" + str(time_file.tm_hour) + "_" + str(time_file.tm_min)

    current_timestamp = time.time_ns()
    while True: 
        current_timestamp = time.time_ns()
        if last_pos is not None:
            data_pos.append(np.append(current_timestamp,last_pos.copy()))
        if last_rot is not None:
            data_rot.append(np.append(current_timestamp,last_rot.copy()))
        await asyncio.sleep(0.01)


signal.signal(signal.SIGINT, signal_handler)
asyncio.run(main())
print('Finish acquisition')

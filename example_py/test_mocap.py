"""
    Minimal usage example
    Connects to QTM and streams 3D data forever
    (start QTM first, load file, Play->Play with Real-Time output)
"""

import asyncio
import qtm_rt
import numpy as np
import time

def on_packet(packet):
    global last_pos
    global last_rot
    """ Callback function that is called everytime a data packet arrives from QTM """
    # print("Framenumber: {}".format(packet.framenumber))
    # header, markers = packet.get_3d_markers()
    # print("Component info: {}".format(header))
    # for marker in markers:
    #     print("\t", marker)
    _, bodies = packet.get_6d()
    for i, body in enumerate(bodies):
        pos, rot = body
        x, y, z = pos
        print(f'Pos: {pos}, Rot {rot}')
        print('Pos', x,y,z)

        rot_elements = [j for j in rot[0]]

        print('pos', x,y,z)
        print('rot', rot_elements)

        '''print('Rot1', rot[0][0],rot[0][1],rot[0][2])
        print('Rot2', rot[0][3],rot[0][4],rot[0][5])
        print('Rot3', rot[0][6],rot[0][7],rot[0][8])'''

        
        if i == 0:
            last_pos = np.array([x,y,z])

            last_rot = np.array(rot_elements)
            #if last_meas is not None:
             #   print('last_meas',last_meas)


async def setup():
    """ Main function """
    connection = await qtm_rt.connect("192.168.225.1")
    if connection is None:
        return

    await connection.stream_frames(components=["6d"], on_packet=on_packet)


async def main():
    asyncio.create_task(setup())

    await asyncio.sleep(1)
    data = []
    data2 = []

    current_timestamp = time.time_ns()
    for _ in range(1000): 
        if last_pos is not None:
            data.append(np.append(current_timestamp,last_pos.copy()))
        if last_rot is not None:
            data2.append(np.append(current_timestamp,last_rot.copy()))
        await asyncio.sleep(0.01)
        # time.sleep(0.01)

    np.savez_compressed('data/mocap_example.npz', pos=np.asarray(data),  rot=np.asarray(data2))
    print('Data saved')

# if __name__ == "__main__":
#     asyncio.ensure_future(setup())
#     asyncio.get_event_loop().run_forever()

asyncio.run(main())
print('Finish acquisition')

#import matplotlib.pyplot as plt
data = np.load('data/mocap_example.npz')

#print(data['pos'].shape)

'''ax = plt.figure().add_subplot(projection='3d')
ax.plot(data['pos'][:, 0], data['pos'][:, 1], data['pos'][:, 2])

ax.set_xlabel('x')
ax.set_ylabel('y')

plt.show()'''
for i in range(len(data['pos'])):
    print(data['pos'][i])
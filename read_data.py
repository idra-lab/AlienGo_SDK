import numpy as np
import csv

name_file = "data23_4_16_38_robot.csv"
timestamps = []
tick = []
motion = []
with open(name_file,'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines: 
        timestamps.append(int(row[0]))
        tick.append(int(row[1]))
        motion.append(int(row[2]))
csvfile.close()


tick_array = np.array(tick)

tick_diff = tick_array[1:] - tick_array[0:-1]

tick_mean = np.mean(tick_diff)

tick_std = np.std(tick_diff)

tick_var = np.var(tick_diff)

print(tick_mean)
print(tick_std)
print(tick_var)
print('max',np.max(tick_diff), np.argmax(tick_diff))
print(tick_array[1:][np.argmax(tick_diff)])
print(tick_array[0:-1][np.argmax(tick_diff)])

timestamp_array = np.array(timestamps)
timestamp_diff = timestamp_array[1:] - timestamp_array[0:-1]
print(timestamp_array[1:][np.argmax(tick_diff)])
print(timestamp_array[0:-1][np.argmax(tick_diff)])

motion_array = np.array(motion)
print(motion_array[1:][np.argmax(tick_diff)])
print(motion_array[0:-1][np.argmax(tick_diff)])

print('diff')
motion_diff = motion_array[1:] - motion_array[0:-1]
for i in motion_diff:
    if i != 2:
        print(i)


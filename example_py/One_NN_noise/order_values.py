import csv
import numpy as np

file_path = "save_iterations_complete.csv"
values = []
with open(file_path,'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        values.append(int(row[0]))

'''file_path = "save_iterations_complete5.csv"
values2 = []
with open(file_path,'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        values2.append(int(row[0]))

for i in range(len(values)):
    if values[i] != values2[i]:
        print('diff')'''

#print(values)
values.sort()
print(values)
print(np.max(values))
i = 0
j = 0
for n in values:
    if n > 50:
        i += 1
    else:
        j += 1

print(i)
print(j)
print(len(values))
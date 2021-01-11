import os
import time
from math import pi
from numpy.linalg import pinv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trajectory_cubic as traj
from torque_calc import torque_calc as tc, torque_calc_friction as tcf

test_num = range(10)
test_times = 10
data = []
for num in test_num:
    for i in range(test_times):
        name = "\\20190125_test" + str(num+1) + "_" + str(i+1)
        filepath = "C:\\Users\\WU-TW YANG\\Desktop\\Experiment\\Dynamics Model\\2019_01_25\collect data"
        # name = "\\2018066_test" + str(num+1) + "_" + str(i+1)
        # filepath = "C:\\Users\\WU-TW YANG\\Desktop\\Manager\\Biorola\\Progress\\Transfer\\collect_data\\20180606"
        filename_data = filepath + name + ".csv"
        filename_fig = filepath + name + ".png"
        data_temp = pd.read_csv(filename_data, header=None, skipfooter=1, engine='python')
        data_temp = data_temp.values

        if data == []:
            data = data_temp.copy()
        else:
            data = np.vstack((data, data_temp))

# column 1 : on/off
# column 2~7 : XYZABC
# column 8~13 : joint position
# column 14~19 : joint velocity
# column 20~25 : joint torque (actual)
# column 26~31 : joint torque (command)
# column 32 : timestamp

# XYZ : mm -> m
for i in range(1, 4, 1):
    data[:, i] = data[:, i] / 1000.0
# torque : mNm -> Nm
# timestamp : ms -> s
for i in range(19, 32, 1):
    data[:, i] = data[:, i] / 1000.0

length = data.shape[0]
# reference time
t = data[:, 31] - data[0, 31]

# unit of x, v, a: deg -> rad
pos_ori = data[:, 7:10] / 180 * pi
vel_ori = data[:, 13:16] / 180 * pi

acc_ori = np.zeros((length, 3))
for i in range(2, length-1, 1):
    for j in range(3):
        acc_ori[i, j] = ( vel_ori[i, j] - vel_ori[i-1, j] ) / ( t[i] - t[i-1] )

# unit of Inertia: kg*mm^2 -> kg*m^2
# unit of mass: kg
# unit of length from origin(frame i-1) to mass center: mm -> m
I_1 = np.array([11600, 8800, 12000], dtype=float) / (10**6)
I_2 = np.array([23600, 204600, 198300], dtype=float) / (10**6)
I_3 = np.array([3400, 34300, 33800], dtype=float) / (10**6)
m = np.array([4.0, 8.6, 2.28, 3.56], dtype=float)
d1 = 145.1 / 1000
a2 = 329.0/1000
a3 = 311.5/1000 + 106.0/1000
L1 = d1 - 1.7/1000
L2 = a2 - 150.0/1000
L3 = a3 - 74.0/1000
# a2 = 429.0 / 1000
# a3 = 411.5 / 1000 + 106.0 / 1000
# L1 = d1 - 1.7 / 1000
# L2 = a2 - 204 / 1000
# L3 = a3 - 99 / 1000
g = 9.81

torque_sim = np.zeros((length, 3))
for i in range(length):
    torque_sim[i, :] = tc(pos_ori[i, :], vel_ori[i, :], acc_ori[i, :])
    # torque_sim_with_f[i, :] = tcf(pos_ori[i, :], vel_ori[i, :], acc_ori[i, :])

torque_actual = data[:, 19:22]
torque_diff = torque_actual - torque_sim

A1 = np.zeros((length, 3));
for i in range(length):
    A1[i,0] = acc_ori[i,0]
    A1[i,1] = vel_ori[i,0]
    A1[i,2] = np.sign(vel_ori[i,0])
X1 = np.dot(pinv(A1), torque_diff[:,0])    # [Bm1, Cm1, fc1]

A2 = np.zeros((length, 3));
for i in range(length):
    A2[i,0] = acc_ori[i,1]
    A2[i,1] = vel_ori[i,1]
    A2[i,2] = np.sign(vel_ori[i,1])
X2 = np.dot(pinv(A2), torque_diff[:,1])    # [Bm2, Cm2, fc2]

A3 = np.zeros((length, 3));
for i in range(length):
    A3[i,0] = acc_ori[i,2]
    A3[i,1] = vel_ori[i,2]
    A3[i,2] = np.sign(vel_ori[i,2])
X3 = np.dot(pinv(A3), torque_diff[:,2])    # [Bm3, Cm3, fc3]

Bm = np.array([X1[0], X2[0], X3[0]])
Cm = np.array([X1[1], X2[1], X3[1]])
fc = np.array([X1[2], X2[2], X3[2]])

print("Bm = ", Bm)
print("Cm = ", Cm)
print("fc = ", fc)

# print("X1 = ", X1)
# print("X2 = ", X2)
# print("X3 = ", X3)

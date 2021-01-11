import os
import time
from math import pi
import math
import numpy as np
import pandas as pd
import keras as K
import matplotlib.pyplot as plt
import trajectory_cubic as traj
from torque_calc import torque_calc as tc, torque_calc_friction as tcf
from module import six_DOF as six

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

def main(z, plot_func=True):
    plot_func = False

    # filepath = "C:\\Users\\YungHsiu\\Desktop\\code\\collect_data\\20180606"
    # name = "\\2018066_test10_1" #+ str(z)
    # filepath = "C:\\Users\\YungHsiu\\Desktop\\code\\collect_data"
    # name = "\\20180625_test_1"
    # name = "\\20180626_exp_2"
    # filepath = "C:\\Users\\YungHsiu\\Desktop\\code\\collect_data\\20180628_test_NN"
    # name = "\\20180628_test5_3"
    filepath = "C:\\Users\\WU-TW YANG\\Desktop\\Manager\\Biorola\\Progress\\Transfer\\collect_data\\20180705_online_training"
    name = "\\20180705_ep_4"
    filename_data = filepath + name + ".csv"
    filename_fig = filepath + name + "_traj" + ".png"
    data = pd.read_csv(filename_data, header=None, skipfooter=1, engine='python')
    data = data.values
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

    torque_sim = np.zeros((length, 3))
    torque_sim_with_f = np.zeros((length, 3))
    for i in range(length):
        torque_sim[i, :] = tc(pos_ori[i, :], vel_ori[i, :], acc_ori[i, :])
        torque_sim_with_f[i, :] = tcf(pos_ori[i, :], vel_ori[i, :], acc_ori[i, :])
    
    six.draw_traj(pos_ori/pi*180)
    
    # model = K.models.load_model('dynamic_model_1_tanh.h5')
    model = K.models.load_model('dynamic_model.h5')
    print(filename_data)
    torque_actual = data[:, 19:22]
    energy = traj.total_energy(torque_actual, vel_ori, t)
    print("Energy of [J1, J2, J3] (Actual): ", energy, "[Joule]")
    print("Total Energy: ", np.sum(energy))
    
    torque_diff = torque_actual - torque_sim_with_f
    temp_J1 = np.sum(torque_diff[:, 0]**2)
    temp_J2 = np.sum(torque_diff[:, 1]**2)
    temp_J3 = np.sum(torque_diff[:, 2]**2)
    temp = temp_J1 + temp_J2 + temp_J3
    RMSE_before_J1 = math.sqrt(temp_J1 / (3*torque_diff.shape[0]) )
    RMSE_before_J2 = math.sqrt(temp_J2 / (3*torque_diff.shape[0]) )
    RMSE_before_J3 = math.sqrt(temp_J3 / (3*torque_diff.shape[0]) )
    RMSE_before = math.sqrt(temp / (3*torque_diff.shape[0]) )
    print("---without NN---")
    print("RMSE of J1, J2, J3: %.4f %.4f %.4f" % (RMSE_before_J1, RMSE_before_J2, RMSE_before_J3))
    print("RMSE: %.4f" % RMSE_before)
    energy = traj.total_energy(torque_sim_with_f, vel_ori, t)
    print("Energy of [J1, J2, J3] (Simulation): ", energy, "[Joule]")
    print("Total Energy: ", np.sum(energy))
    
    torque_after_training = np.zeros((length, 3))
    Xtest = np.hstack((pos_ori, vel_ori, acc_ori))
    for i in range(length):
        comp = model.predict( np.array([ Xtest[i] ]) )
        torque_after_training[i] = tcf(pos_ori[i, :], vel_ori[i, :], acc_ori[i, :]) + comp
    torque_diff = torque_actual - torque_after_training
    temp_J1 = np.sum(torque_diff[:, 0]**2)
    temp_J2 = np.sum(torque_diff[:, 1]**2)
    temp_J3 = np.sum(torque_diff[:, 2]**2)
    temp = temp_J1 + temp_J2 + temp_J3
    RMSE_after_J1 = math.sqrt(temp_J1 / (3*torque_diff.shape[0]) )
    RMSE_after_J2 = math.sqrt(temp_J2 / (3*torque_diff.shape[0]) )
    RMSE_after_J3 = math.sqrt(temp_J3 / (3*torque_diff.shape[0]) )
    RMSE_after = math.sqrt(temp / (3*torque_diff.shape[0]) )
    print("---with NN---")
    print("RMSE of J1, J2, J3: %.4f %.4f %.4f" % (RMSE_after_J1, RMSE_after_J2, RMSE_after_J3))
    print("RMSE: %.4f" % RMSE_after)
    energy = traj.total_energy(torque_after_training, vel_ori, t)
    print("Energy of [J1, J2, J3] (Simulation with NN): ", energy, "[Joule]")
    print("Total Energy: ", np.sum(energy))

    print("----------")
    print("----------")
    
    if plot_func is True:
        # plot figure
        # plt.figure(figsize=(12.8, 9.6), dpi=100)
        plt.figure(figsize=(12.8, 6.4), dpi=100)
        '''
        # X, Y, Z
        plt.subplot(3, 3, 1)
        plt.plot(t, data[:,1], 'r', label="X")
        plt.plot(t, data[:,2], 'g', label="Y")
        plt.plot(t, data[:,3], 'b', label="Z")
        plt.legend(loc='upper right')
        plt.axis([0, t[-1], -1, 1])
        plt.xticks(np.arange(0, t[-1], 0.5))
        plt.xlabel("time [sec]")
        plt.ylabel("position [m]")
        plt.title("Position in Cartesian Space")
        # Rx, Ry, Rz
        plt.subplot(3, 3, 2)
        plt.plot(t, data[:,4], 'r', label="Rx")
        plt.plot(t, data[:,5], 'g', label="Ry")
        plt.plot(t, data[:,6], 'b', label="Rz")
        plt.legend(loc='upper right')
        plt.axis([0, t[-1], -360, 360])
        plt.xticks(np.arange(0, t[-1], 0.5))
        plt.xlabel("time [sec]")
        plt.ylabel(r"${\theta}$ [deg]")
        plt.title("Rotation in Cartesian Space")
        '''
        #####
        # joint position
        plt.subplot(2, 3, 1)
        plt.plot(t, data[:,7], 'r', label="$J_1$")
        plt.plot(t, data[:,8], 'g', label="$J_2$")
        plt.plot(t, data[:,9], 'b', label="$J_3$")
        plt.legend(loc='upper right')
        plt.axis([0, t[-1], -270, 270])
        plt.xticks(np.arange(0, t[-1], 0.5))
        plt.xlabel("time [sec]")
        plt.ylabel(r"${\theta}$ [deg]")
        plt.title("Joint Position")
        # joint velocity
        plt.subplot(2, 3, 2)
        plt.plot(t, data[:,13], 'r', label="$J_1$")
        plt.plot(t, data[:,14], 'g', label="$J_2$")
        plt.plot(t, data[:,15], 'b', label="$J_3$")
        plt.legend(loc='upper right')
        plt.axis([0, t[-1], -180, 180])
        plt.xticks(np.arange(0, t[-1], 0.5))
        plt.xlabel("time [sec]")
        plt.ylabel(r"${\dot\theta}$ [deg/s]")
        plt.title("Joint Velocity")
        # joint acceleration
        plt.subplot(2, 3, 3)
        plt.plot(t, acc_ori[:,0]/pi*180, 'r', label="$J_1$")
        plt.plot(t, acc_ori[:,1]/pi*180, 'g', label="$J_2$")
        plt.plot(t, acc_ori[:,2]/pi*180, 'b', label="$J_3$")
        plt.legend(loc='upper right')
        plt.axis([0, t[-1], -360, 360])
        plt.xticks(np.arange(0, t[-1], 0.5))
        plt.xlabel("time [sec]")
        plt.ylabel(r"${\ddot\theta}$ ${[deg/s^2]}$")
        plt.title("Joint Acceleration")
        #####
        # joint torque (actual)
        plt.subplot(2, 3, 4)
        plt.plot(t, data[:,19], 'r', label="$J_1$")
        plt.plot(t, data[:,20], 'g', label="$J_2$")
        plt.plot(t, data[:,21], 'b', label="$J_3$")
        plt.legend(loc='upper right')
        plt.axis([0, t[-1], -150, 150])
        plt.xticks(np.arange(0, t[-1], 0.5))
        plt.xlabel("time [sec]")
        plt.ylabel(r"${\tau}$ [Nm]")
        plt.title("Torque (Actual)")
        # joint torque (Simulation)
        plt.subplot(2, 3, 5)
        plt.plot(t, torque_sim_with_f[:,0], 'r', label="$J_1$")
        plt.plot(t, torque_sim_with_f[:,1], 'g', label="$J_2$")
        plt.plot(t, torque_sim_with_f[:,2], 'b', label="$J_3$")
        plt.legend(loc='upper right')
        plt.axis([0, t[-1], -150, 150])
        plt.xticks(np.arange(0, t[-1], 0.5))
        plt.xlabel("time [sec]")
        plt.ylabel(r"${\tau}$ [Nm]")
        plt.title(r"Torque (Simulation)")
        # joint torque (Simulation with NN)
        plt.subplot(2, 3, 6)
        plt.plot(t, torque_after_training[:, 0], 'r', label="$J_1$")
        plt.plot(t, torque_after_training[:, 1], 'g', label="$J_2$")
        plt.plot(t, torque_after_training[:, 2], 'b', label="$J_3$")
        plt.legend(loc='upper right')
        plt.axis([0, t[-1], -150, 150])
        plt.xticks(np.arange(0, t[-1], 0.5))
        plt.xlabel("time [sec]")
        plt.ylabel(r"${\tau}$ [Nm]")
        plt.title(r"Torque (Simulation with NN)")
        
        plt.tight_layout()
        plt.savefig(filename_fig)
        # print("figure saved successfully!")
        # plt.show()
    
    #####
    plt.figure(figsize=(12.8, 3.2), dpi=100)
    #####
    # joint torque (actual)
    plt.subplot(1, 3, 1)
    plt.plot(t, data[:,19], 'r', label="$J_1$")
    plt.plot(t, data[:,20], 'g', label="$J_2$")
    plt.plot(t, data[:,21], 'b', label="$J_3$")
    plt.legend(loc='upper right')
    plt.axis([0, t[-1], -150, 150])
    plt.xticks(np.arange(0, t[-1], 0.5))
    plt.xlabel("time [sec]")
    plt.ylabel(r"${\tau}$ [Nm]")
    plt.title("Torque (Actual)")
    # joint torque (Simulation)
    plt.subplot(1, 3, 2)
    plt.plot(t, torque_sim_with_f[:,0], 'r', label="$J_1$")
    plt.plot(t, torque_sim_with_f[:,1], 'g', label="$J_2$")
    plt.plot(t, torque_sim_with_f[:,2], 'b', label="$J_3$")
    plt.legend(loc='upper right')
    plt.axis([0, t[-1], -150, 150])
    plt.xticks(np.arange(0, t[-1], 0.5))
    plt.xlabel("time [sec]")
    plt.ylabel(r"${\tau}$ [Nm]")
    plt.title(r"Torque (Simulation)")
    # joint torque (Simulation with NN)
    plt.subplot(1, 3, 3)
    plt.plot(t, torque_after_training[:, 0], 'r', label="$J_1$")
    plt.plot(t, torque_after_training[:, 1], 'g', label="$J_2$")
    plt.plot(t, torque_after_training[:, 2], 'b', label="$J_3$")
    plt.legend(loc='upper right')
    plt.axis([0, t[-1], -150, 150])
    plt.xticks(np.arange(0, t[-1], 0.5))
    plt.xlabel("time [sec]")
    plt.ylabel(r"${\tau}$ [Nm]")
    plt.title(r"Torque (Simulation with NN)")
    
    plt.tight_layout()
    plt.savefig(filename_fig)
    # print("figure saved successfully!")
    # plt.show()
    



def validation():
    model = K.models.load_model('dynamic_model.h5')
    test_num = range(10)
    test_times = 10
    # test_times = [11, 9, 9, 11]*2
    data = []
    energy_error = []    # J1, J2, J3, Total
    for num in test_num:
        for z in range(test_times):
            name = "\\2018066_test" + str(num+1) + "_" + str(z+1)
            filepath = "C:\\Users\\WU-TW YANG\\Desktop\\Manager\\Biorola\\Progress\\Transfer\\collect_data\\20180606"
            filename_data = filepath + name + ".csv"
            filename_fig = filepath + name + ".png"
            data = pd.read_csv(filename_data, header=None, skipfooter=1, engine='python')
            data = data.values
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

            torque_sim_with_f = np.zeros((length, 3))
            for i in range(length):
                torque_sim_with_f[i, :] = tcf(pos_ori[i, :], vel_ori[i, :], acc_ori[i, :])

            torque_actual = data[:, 19:22]
            energy_act = traj.total_energy(torque_actual, vel_ori, t)
            # print("Energy of [J1, J2, J3] (Actual): ", energy, "[Joule]")
            # print("Total Energy: ", np.sum(energy))

            energy_sim = traj.total_energy(torque_sim_with_f, vel_ori, t)
            # print("Energy of [J1, J2, J3] (Simulation): ", energy, "[Joule]")
            # print("Total Energy: ", np.sum(energy))
            
            torque_after_training = np.zeros((length, 3))
            Xtest = np.hstack((pos_ori, vel_ori, acc_ori))
            # Xtrain_plus = np.array([ vel_ori[:,0]**2, vel_ori[:,1]**2, vel_ori[:,2]**2, 
            #                                 vel_ori[:,0]*vel_ori[:,1], vel_ori[:,0]*vel_ori[:,2], vel_ori[:,1]*vel_ori[:,2] ])
            # Xtest = np.hstack((Xtest, np.transpose(Xtrain_plus)))
            for i in range(length):
                comp = model.predict( np.array([ Xtest[i] ]) )
                torque_after_training[i] = comp
                # torque_after_training[i] = tcf(pos_ori[i, :], vel_ori[i, :], acc_ori[i, :]) + comp
            energy_sim = traj.total_energy(torque_after_training, vel_ori, t)
            
            error_each = energy_sim - energy_act
            error_total = np.sum(energy_sim) - np.sum(energy_act)
            if energy_error == []:
                energy_error = np.hstack((error_each, error_total))
            else:
                temp = np.hstack((error_each, error_total))
                energy_error = np.vstack((energy_error, temp))
    print("----------")
    print("joint 1")
    print("average: (J)", np.mean(energy_error[:,0]))
    print("std: ", np.std(energy_error[:,0]))
    print("joint 2")
    print("average: (J)", np.mean(energy_error[:,1]))
    print("std: ", np.std(energy_error[:,1]))
    print("joint 3")
    print("average: (J)", np.mean(energy_error[:,2]))
    print("std: ", np.std(energy_error[:,2]))

    print("total")
    print("average: (J)", np.mean(energy_error[:,3]))
    print("std: ", np.std(energy_error[:,3]))
    print("----------")
    x = range(energy_error.shape[0])
    plt.plot(x, energy_error[:,0], 'r', label="$J_1$")
    plt.plot(x, energy_error[:,1], 'g', label="$J_2$")
    plt.plot(x, energy_error[:,2], 'b', label="$J_3$")
    plt.plot(x, energy_error[:,3], 'k', label="Total")
    plt.title("Energy Error")
    plt.axis([0, x[-1], -16, 4])
    plt.legend(loc='upper right')
    plt.xlabel("data #")
    plt.ylabel(r"Error [J]")
    plt.tight_layout()
    plt.show()

def validation2():
    model = K.models.load_model('dynamic_model.h5')
    test_num = range(10)
    test_times = 10

    data = []
    energy_error = []    # J1, J2, J3, Total
    for num in test_num:
        for z in range(test_times):
            name = "\\2018066_test" + str(num+1) + "_" + str(z+1)
            filepath = "C:\\Users\\WU-TW YANG\\Desktop\\Manager\\Biorola\\Progress\\Transfer\\collect_data\\20180606"
            filename_data = filepath + name + ".csv"
            filename_fig = filepath + name + ".png"
            data = pd.read_csv(filename_data, header=None, skipfooter=1, engine='python')
            data = data.values
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

            torque_sim_with_f = np.zeros((length, 3))
            for i in range(length):
                torque_sim_with_f[i, :] = tcf(pos_ori[i, :], vel_ori[i, :], acc_ori[i, :])

            torque_actual = data[:, 19:22]
            energy_act = traj.total_energy(torque_actual, vel_ori, t)
            # print("Energy of [J1, J2, J3] (Actual): ", energy, "[Joule]")
            # print("Total Energy: ", np.sum(energy))

            energy_sim = traj.total_energy(torque_sim_with_f, vel_ori, t)
            # print("Energy of [J1, J2, J3] (Simulation): ", energy, "[Joule]")
            # print("Total Energy: ", np.sum(energy))

            torque_after_training = np.zeros((length, 3))
            Xtest = np.hstack((pos_ori, vel_ori, acc_ori))
            # Xtrain_plus = np.array([ vel_ori[:,0]**2, vel_ori[:,1]**2, vel_ori[:,2]**2, 
            #                                 vel_ori[:,0]*vel_ori[:,1], vel_ori[:,0]*vel_ori[:,2], vel_ori[:,1]*vel_ori[:,2] ])
            # Xtest = np.hstack((Xtest, np.transpose(Xtrain_plus)))
            for i in range(length):
                comp = model.predict( np.array([ Xtest[i] ]) )
                torque_after_training[i] = comp
                # torque_after_training[i] = tcf(pos_ori[i, :], vel_ori[i, :], acc_ori[i, :]) + comp
            energy_sim = traj.total_energy(torque_after_training, vel_ori, t)
            
            error_each = energy_sim - energy_act
            error_total = (np.sum(energy_sim) - np.sum(energy_act))/np.sum(energy_act)
            if energy_error == []:
                energy_error = np.hstack((error_each, error_total))
            else:
                temp = np.hstack((error_each, error_total))
                energy_error = np.vstack((energy_error, temp))
    print("----------")
    print("average: (%)", np.mean(energy_error[:,3])*100)
    print("std: ", np.std(energy_error[:,3]))
    print("----------")
    x = range(energy_error.shape[0])
    plt.plot(x, energy_error[:,3]*100, 'k', label="Total")
    plt.title("Energy Error")
    plt.axis([0, x[-1], -40, 10])
    plt.legend(loc='upper right')
    plt.xlabel("data #")
    plt.ylabel(r"Error [%]")
    plt.tight_layout()
    plt.show()

def validation3():
    model = K.models.load_model('dynamic_model_main_half_2.h5')
    # test_num = range(3)
    test_num = [3,4]
    test_times = 3
    num = len(test_num)*test_times
    energy_list_act = np.zeros(num)
    energy_list_sim = np.zeros(num)
    energy_list_NN = np.zeros(num)
    RMSE_list_before = np.zeros(num)
    RMSE_list_after = np.zeros(num)
    idx = 0
    for num in test_num:
        for z in range(test_times):
            filepath = "C:\\Users\\WU-TW YANG\\Desktop\\Manager\\Biorola\\Progress\\Transfer\\collect_data\\20180628_test_NN"
            name = "\\20180628_test" + str(num+1) + "_" + str(z+1)
            filename_data = filepath + name + ".csv"
            filename_fig = filepath + name + ".png"
            data = pd.read_csv(filename_data, header=None, skipfooter=1, engine='python')
            data = data.values
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

            torque_sim_with_f = np.zeros((length, 3))
            for i in range(length):
                torque_sim_with_f[i, :] = tcf(pos_ori[i, :], vel_ori[i, :], acc_ori[i, :])

            torque_actual = data[:, 19:22]
            energy_act = traj.total_energy(torque_actual, vel_ori, t)
            energy_list_act[idx] = np.sum(energy_act)
            # print("Energy of [J1, J2, J3] (Actual): ", energy, "[Joule]")
            # print("Total Energy: ", np.sum(energy))

            energy_sim = traj.total_energy(torque_sim_with_f, vel_ori, t)
            energy_list_sim[idx] = np.sum(energy_sim)
            torque_diff = torque_actual - torque_sim_with_f
            temp = np.sum(torque_diff**2)
            RMSE_before = math.sqrt(temp / (3*torque_diff.shape[0]) )
            RMSE_list_before[idx] = RMSE_before
            # print("Energy of [J1, J2, J3] (Simulation): ", energy, "[Joule]")
            # print("Total Energy: ", np.sum(energy))

            torque_after_training = np.zeros((length, 3))
            Xtest = np.hstack((pos_ori, vel_ori, acc_ori))
            # Xtrain_plus = np.array([ vel_ori[:,0]**2, vel_ori[:,1]**2, vel_ori[:,2]**2, 
            #                                 vel_ori[:,0]*vel_ori[:,1], vel_ori[:,0]*vel_ori[:,2], vel_ori[:,1]*vel_ori[:,2] ])
            # Xtest = np.hstack((Xtest, np.transpose(Xtrain_plus)))
            for i in range(length):
                comp = model.predict( np.array([ Xtest[i] ]) )
                torque_after_training[i] = comp
                # torque_after_training[i] = torque_sim_with_f[i] + comp
            energy_sim_NN = traj.total_energy(torque_after_training, vel_ori, t)
            energy_list_NN[idx] = np.sum(energy_sim_NN)
            torque_diff = torque_actual - torque_after_training
            temp = np.sum(torque_diff**2)
            RMSE_after = math.sqrt(temp / (3*torque_diff.shape[0]) )
            RMSE_list_after[idx] = RMSE_after

            idx += 1
    energy_error_sim = -(energy_list_act - energy_list_sim)
    sim_mean = np.mean(energy_error_sim)
    sim_std = np.std(energy_error_sim)
    print("energy error (sim)", sim_mean, sim_std)
    energy_error_NN = -(energy_list_act - energy_list_NN)
    NN_mean = np.mean(energy_error_NN)
    NN_std = np.std(energy_error_NN)
    print("energy error (sim with NN)", NN_mean, NN_std)

    RMSE_sim_mean = np.mean(RMSE_list_before)
    RMSE_sim_std = np.std(RMSE_list_before)
    print("RMSE (sim)", RMSE_sim_mean, RMSE_sim_std)
    RMSE_NN_mean = np.mean(RMSE_list_after)
    RMSE_NN_std = np.std(RMSE_list_after)
    print("RMSE (sim with NN)", RMSE_NN_mean, RMSE_NN_std)

    x = [0.3,0.7]
    width = 0.25
    label = ('Simulation', 'Simulation with NN')
    plt.figure()
    plt.bar(x[0], sim_mean, width, yerr=sim_std)
    plt.bar(x[1], NN_mean, width, yerr=NN_std)
    # plt.xlabel('data #')
    plt.xticks(x, label)
    plt.ylabel('energy consumption error [J]')
    # plt.legend(loc='upper right')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=len(energy_list_act))
    # plt.ti0.2h0.8_layout()
    plt.show()

    plt.figure()
    plt.bar(x[0], RMSE_sim_mean, width, yerr=RMSE_sim_std)
    plt.bar(x[1], RMSE_NN_mean, width, yerr=RMSE_NN_std)
    # plt.xlabel('data #')
    plt.xticks(x, label)
    plt.ylabel('RMSE of torque [Nm]')
    # plt.legend(loc='upper right')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=len(RMSE_list_before))
    # plt.tight_layout()
    plt.show()

def validation3_barplot():
    # test1, 2, 3
    energy_mean_sim = [-4.1141, -5.2072, -2.4744]
    energy_std_sim = [2.2302, 1.7495, 1.8337]
    RMSE_mean_sim = [4.6048, 5.0377, 3.9556]
    RMSE_std_sim = [1.0948, 1.1227, 0.6352]
    # test[1,2,3]
    model1_energy_mean = [-1.3686, -1.6940, -0.8804]
    model1_energy_std = [1.0592, 0.8853, 1.1099]
    model1_RMSE_mean = [4.2426, 4.1782, 4.3392]
    model1_RMSE_std = [0.3774, 0.3111, 0.4419]

    model2_energy_mean = [-0.4378, -0.8929, 0.2449]
    model2_energy_std = [1.1050, 0.9493, 0.9612]
    model2_RMSE_mean = [4.1022, 4.1844, 3.9788]
    model2_RMSE_std = [0.3958, 0.3363, 0.4434]

    model3_energy_mean = [-0.4920, -0.7152, -0.1573]
    model3_energy_std = [0.8210, 0.6014, 0.9777]
    model3_RMSE_mean = [3.9825, 3.9551, 4.0235]
    model3_RMSE_std = [0.3094, 0.2958, 0.3245]

    x = np.arange(4)
    width = 0.35
    x_label = ('Simulation', 'model1', 'model2', 'model3')
    for i in range(3):
        plt.figure()
        plt.bar(x[0], energy_mean_sim[i], width, yerr=energy_std_sim[i])
        plt.bar(x[1], model1_energy_mean[i], width, yerr=model1_energy_std[i])
        plt.bar(x[2], model2_energy_mean[i], width, yerr=model2_energy_std[i])
        plt.bar(x[3], model3_energy_mean[i], width, yerr=model3_energy_std[i])
        plt.xticks(x, x_label)
        plt.ylabel('energy consumption error [J]')
        plt.show()
    
    for i in range(3):
        plt.figure()
        plt.bar(x[0], RMSE_mean_sim[i], width, yerr=RMSE_std_sim[i])
        plt.bar(x[1], model1_RMSE_mean[i], width, yerr=model1_RMSE_std[i])
        plt.bar(x[2], model2_RMSE_mean[i], width, yerr=model2_RMSE_std[i])
        plt.bar(x[3], model3_RMSE_mean[i], width, yerr=model3_RMSE_std[i])
        plt.xticks(x, x_label)
        plt.ylabel('RMSE of torque [Nm]')
        plt.show()

def energy_exp():
    # filepath = "C:\\Users\\YungHsiu\\Desktop\\code\\collect_data\\20180705_online_training"
    # name = "\\20180705_ep_102"
    # filepath = "C:\\Users\\YungHsiu\\Desktop\\code\\collect_data\\20180628_exp_RL_result"
    # name = "\\20180628_test_5"
    filepath = "C:\\Users\\WU-TW YANG\\Desktop\\Manager\\Biorola\\Progress\\Transfer\\collect_data\\CH5-2"
    name = "\\CH5-2_exp_5"
    filename_data = filepath + name + ".csv"
    filename_fig = filepath + name + "_traj" + ".png"
    data = pd.read_csv(filename_data, header=None, skipfooter=1, engine='python')
    data = data.values
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

    torque_actual = data[:, 19:22]
    energy = traj.total_energy(torque_actual, vel_ori, t)
    print("Energy of [J1, J2, J3] (Actual): ", energy, "[Joule]")
    print("Total Energy: ", np.sum(energy))

def time_exp():
    filepath = "C:\\Users\\WU-TW YANG\\Desktop\\Manager\\Biorola\\Progress\\Transfer\\collect_data\\CH5-3_auto"
    name = "\\CH5-3_exp_5"
    filename_data = filepath + name + ".csv"
    filename_fig = filepath + name + "_traj" + ".png"
    data = pd.read_csv(filename_data, header=None, skipfooter=1, engine='python')
    data = data.values
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

    torque_actual = data[:, 19:22]
    traj.plot_torque(torque_actual, t, hlines=True)

    vel_ = np.zeros((data.shape[0], 3))
    for i in range(data.shape[0]):
        if i==0 or i==data.shape[0]-1: continue
        else:
            vel_[i, :] = (pos_ori[i+1, :]-pos_ori[i-1, :]) / (t[i+1]-t[i-1]) * 180/math.pi

    start = np.array( [[ 64, -43, -48, 0 ]], dtype=float )
    end = np.array( [[ 94, -26, -70,  0.9999]], dtype=float )
    num_of_via = 2
    via = np.zeros((num_of_via, 4))
    for i in range(num_of_via):
        via[i, :] = start + (i+1) / (num_of_via+1) * (end-start)
    via = np.array( [[ 72.1000, -37.4000, -44.8000,  0.3803],
                     [ 91.1000, -37.8000, -44.7000,  0.6154]] )
    x = np.concatenate((start, via, end))
    traj_pos, traj_vel, traj_acc, t = traj.traj_planning_ALL(x)

    '''
    print("error:", traj_pos[18,0]-data[18,7])
    #####
    plt.figure(figsize=(12.8, 3.2), dpi=100)
    #####
    # joint position
    aaa=t.shape[0]
    # t=t[:aaa]
    plt.subplot(1, 3, 1)
    plt.plot(t, data[:aaa,7], 'r', label="$J_1$")
    plt.plot(t, data[:aaa,8], 'g', label="$J_2$")
    plt.plot(t, data[:aaa,9], 'b', label="$J_3$")
    plt.plot(t, traj_pos[:aaa,0], 'r--', label="$J_1$")
    plt.plot(t, traj_pos[:aaa,1], 'g--', label="$J_2$")
    plt.plot(t, traj_pos[:aaa,2], 'b--', label="$J_3$")
    plt.vlines(0.5, -270, 270, color='k', linestyles='dashdot')
    plt.legend(loc='upper right')
    plt.axis([0, t[-1], 75, 85])
    # plt.axis([0, t[-1], -270, 270])
    plt.xticks(np.arange(0, t[-1], 0.5))
    plt.xlabel("time [sec]")
    plt.ylabel(r"${\theta}$ [deg]")
    plt.title("Joint Position")
    # joint velocity
    plt.subplot(1, 3, 2)
    plt.plot(t, data[:aaa,13], 'r', label="$J_1$")
    plt.plot(t, data[:aaa,14], 'g', label="$J_2$")
    plt.plot(t, data[:aaa,15], 'b', label="$J_3$")
    plt.plot(t, vel_[:aaa,0], 'y', label="$J_1$")
    plt.plot(t, vel_[:aaa,1], 'm', label="$J_2$")
    plt.plot(t, vel_[:aaa,2], 'c', label="$J_3$")
    plt.plot(t, traj_vel[:aaa,0], 'r--', label="$J_1$")
    plt.plot(t, traj_vel[:aaa,1], 'g--', label="$J_2$")
    plt.plot(t, traj_vel[:aaa,2], 'b--', label="$J_3$")
    plt.vlines(0.5, -180, 180, color='k', linestyles='dashdot')
    plt.legend(loc='upper right')
    plt.axis([0, t[-1], -180, 180])
    plt.xticks(np.arange(0, t[-1], 0.5))
    plt.xlabel("time [sec]")
    plt.ylabel(r"${\dot\theta}$ [deg/s]")
    plt.title("Joint Velocity")
    # joint acceleration
    plt.subplot(1, 3, 3)
    plt.plot(t, acc_ori[:aaa,0]/pi*180, 'r', label="$J_1$")
    plt.plot(t, acc_ori[:aaa,1]/pi*180, 'g', label="$J_2$")
    plt.plot(t, acc_ori[:aaa,2]/pi*180, 'b', label="$J_3$")
    plt.plot(t, traj_acc[:aaa,0], 'r--', label="$J_1$")
    plt.plot(t, traj_acc[:aaa,1], 'g--', label="$J_2$")
    plt.plot(t, traj_acc[:aaa,2], 'b--', label="$J_3$")
    plt.vlines(0.5, -720, 720, color='k', linestyles='dashdot')
    plt.legend(loc='upper right')
    plt.axis([0, t[-1], -720, 720])
    plt.xticks(np.arange(0, t[-1], 0.5))
    plt.xlabel("time [sec]")
    plt.ylabel(r"${\ddot\theta}$ ${[deg/s^2]}$")
    plt.title("Joint Acceleration")

    plt.tight_layout()
    plt.show()
    '''

    '''
    torque = np.zeros((t.shape[0],3))
    for i in range(t.shape[0]):
        torque[i, :] = tcf(traj_pos[i, :]/180*pi, traj_vel[i, :]/180*pi, traj_acc[i, :]/180*pi)
    #####
    plt.figure(figsize=(12.8/3, 3.2), dpi=100)
    #####
    aaa=t.shape[0]
    # t=t[:aaa]
    plt.plot(t, data[:aaa,19], 'r', label="$J_1$")
    plt.plot(t, data[:aaa,20], 'g', label="$J_2$")
    plt.plot(t, data[:aaa,21], 'b', label="$J_3$")
    # plt.plot(t, data[:aaa,22], 'y', label="$J_4$")
    # plt.plot(t, data[:aaa,23], 'm', label="$J_5$")
    # plt.plot(t, data[:aaa,24], 'c', label="$J_6$")
    plt.plot(t, torque[:aaa,0], 'r--', label="$J_1$")
    plt.plot(t, torque[:aaa,1], 'g--', label="$J_2$")
    plt.plot(t, torque[:aaa,2], 'b--', label="$J_3$")
    plt.vlines(0.5, -270, 270, color='k', linestyles='dashdot')
    # plt.legend(loc='upper right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=6)
    # plt.axis([0, t[-1], -270, 270])
    plt.axis([0, t[-1], -270, 270])
    plt.xticks(np.arange(0, t[-1], 0.5))
    plt.xlabel("time [sec]")
    plt.ylabel(r"${\theta}$ [deg]")
    plt.title("Torque")

    plt.tight_layout()
    plt.show()
    '''


if __name__ == "__main__":
    '''
    for i in range(5):
        main(i+1)
    '''
    # main(2)
    # validation()
    # validation2()
    # validation3()
    # validation3_barplot()
    # energy_exp()
    time_exp()

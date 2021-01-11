import os
import time
import math
import random
from math import pi
import numpy as np
import pandas as pd
import keras as K
import matplotlib.pyplot as plt
import trajectory_cubic as traj
from torque_calc import torque_calc as tc, torque_calc_friction as tcf, torque_calc_onlyfriction as tcof

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

def train():
    test_num = range(28)
    # test_num1 = [30,31,32,33]
    # test_num = np.hstack((test_num, test_num1))
    # test_num = [1,2,5,6,8]
    test_times = 10

    data = []
    for num in test_num:
        for i in range(test_times):
            name = "\\2019021_test" + str(num+1) + "_" + str(i+1)
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
    
    # test_num = range(12)
    # test_times = 10
    # for num in test_num:
    #     for i in range(test_times):
    #         name = "\\2019021_test" + str(num+1) + "_" + str(i+1)
    #         filepath = "C:\\Users\\WU-TW YANG\\Desktop\\Experiment\\Dynamics Model\\2019_01_25\collect data"
    #         # name = "\\2018066_test" + str(num+1) + "_" + str(i+1)
    #         # filepath = "C:\\Users\\WU-TW YANG\\Desktop\\Manager\\Biorola\\Progress\\Transfer\\collect_data\\20180606"
    #         filename_data = filepath + name + ".csv"
    #         filename_fig = filepath + name + ".png"
    #         data_temp = pd.read_csv(filename_data, header=None, skipfooter=1, engine='python')
    #         data_temp = data_temp.values
            
    #         if data == []:
    #             data = data_temp.copy()
    #         else:
    #             data = np.vstack((data, data_temp))

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
    torque_sim_only_f = np.zeros((length, 3))
    for i in range(length):
        torque_sim[i, :] = tc(pos_ori[i, :], vel_ori[i, :], acc_ori[i, :])
        torque_sim_with_f[i, :] = tcf(pos_ori[i, :], vel_ori[i, :], acc_ori[i, :])
        torque_sim_only_f[i, :] = tcof(pos_ori[i, :], vel_ori[i, :], acc_ori[i, :])

    # energy = traj.total_energy(torque_sim_with_f, vel_ori, t)
    # print("Total Energy of [J1, J2, J3]: ", energy, "[Joule]")

    torque_actual = data[:, 19:22]
    # torque_comp = torque_actual - torque_sim_with_f
    # torque_comp = torque_actual - torque_sim
    # torque_comp = torque_actual - torque_sim_only_f
    torque_comp = torque_actual - 0
    random.seed(87)
    # a = random.sample(range(length), 2000)
    a = range(length)
    # state: angular pos, vel, acc of joint 1~3
    Xtrain = np.zeros((len(a), 9))
    # Xtrain = np.zeros((len(a), 6))
    # Xtrain_plus = np.zeros((len(a), 6))
    # Xtrain_ = np.zeros((len(a), 15))
    Ytrain = np.zeros((len(a), 3))
    for i in range(Xtrain.shape[0]):
        idx = a[i]
        Xtrain[i] = np.hstack((pos_ori[idx, :], vel_ori[idx, :], acc_ori[idx, :]))
        # Xtrain[i] = np.hstack((pos_ori[idx, :], vel_ori[idx, :]))
        # Xtrain_plus[i] = np.array([ vel_ori[idx,0]**2, vel_ori[idx,1]**2, vel_ori[idx,2]**2, 
        #                             vel_ori[idx,0]*vel_ori[idx,1], vel_ori[idx,0]*vel_ori[idx,2], vel_ori[idx,1]*vel_ori[idx,2] ])
        # Xtrain_[i] = np.hstack((Xtrain[i], Xtrain_plus[i]))
        Ytrain[i] = torque_comp[idx, :]

    from keras.models import Sequential
    from keras.layers import Dense, Dropout, BatchNormalization
    from keras.callbacks import EarlyStopping

    model = Sequential()
    # model.add(Dense(units=512, activation='tanh', input_dim=9))
    # model.add(Dropout(0.5))
    # model.add(Dense(units=512, activation='tanh'))
    # model.add(Dropout(0.5))
    # model.add(Dense(units=512, activation='tanh'))
    # model.add(Dropout(0.5))
    # model.add(Dense(units=512, activation='tanh'))
    # model.add(Dropout(0.5))
    # model.add(Dense(units=512, activation='tanh'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=512, activation='tanh', input_dim=9))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(units=1024, activation='tanh'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(units=1024, activation='tanh'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(units=1024, activation='tanh'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(units=1024, activation='tanh'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(units=1024, activation='tanh'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(units=512, activation='tanh'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(units=3, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.compile(loss='mean_absolute_error', optimizer='adam')
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 20, verbose = 1,mode = 'min')
    history = model.fit(Xtrain, Ytrain, epochs=100, batch_size=128, validation_split=0.2, shuffle=True, callbacks=[early_stopping])

    model.save('dynamic_model_700.h2')
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


def test():
    # filepath = "C:\\Users\\WU-TW YANG\\Desktop\\Manager\\Biorola\\Progress\\Transfer\\collect_data\\20180606"
    filepath = "C:\\Users\\WU-TW YANG\\Desktop\\Experiment\\Dynamics Model\\2019_01_25\collect data"
    # filepath = "C:\\Users\\YungHsiu\\Desktop\\code\\collect_data"
    # name = "\\2018066_test4_10"
    # name = "\\2018066_example_1"
    # name = "\\20180515_test2"
    name = "\\20190125_test33_6"
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

    torque_sim = np.zeros((length, 3))
    torque_sim_with_f = np.zeros((length, 3))
    for i in range(length):
        torque_sim[i, :] = tc(pos_ori[i, :], vel_ori[i, :], acc_ori[i, :])
        torque_sim_with_f[i, :] = tcf(pos_ori[i, :], vel_ori[i, :], acc_ori[i, :])
    
    torque_actual = data[:, 19:22]
    # torque_diff = torque_actual - torque_sim_with_f
    torque_diff = torque_actual - torque_sim

    # RMSE_before = 0
    # temp = 0
    # for i in range(torque_diff.shape[0]):
    #     x = np.sum(torque_diff**2)
    #     temp += x
    # RMSE_before = math.sqrt(temp / torque_diff.shape[0])

    RMSE_before_j1 = 0
    RMSE_before_j2 = 0
    RMSE_before_j3 = 0
    temp_j1 = 0
    temp_j2 = 0
    temp_j3 = 0
    for i in range(torque_diff.shape[0]):
        j1 = np.sum(torque_diff[i, 0]**2)
        j2 = np.sum(torque_diff[i, 1]**2)
        j3 = np.sum(torque_diff[i, 2]**2)
        temp_j1 += j1
        temp_j2 += j2
        temp_j3 += j3
    RMSE_before_j1 = math.sqrt(temp_j1/torque_diff.shape[0])
    RMSE_before_j2 = math.sqrt(temp_j2/torque_diff.shape[0])
    RMSE_before_j3 = math.sqrt(temp_j3/torque_diff.shape[0])

    # model = K.models.load_model('dynamic_model_1_tanh.h5')
    # model = K.models.load_model('dynamic_model.h3_1')
    model = K.models.load_model('dynamic_model_700.h2_1')

    torque_after_training = np.zeros((length, 3))
    Xtest = np.hstack((pos_ori, vel_ori, acc_ori))
    # Xtrain_plus = np.array([ vel_ori[:,0]**2, vel_ori[:,1]**2, vel_ori[:,2]**2, 
    #                                 vel_ori[:,0]*vel_ori[:,1], vel_ori[:,0]*vel_ori[:,2], vel_ori[:,1]*vel_ori[:,2] ])
    # Xtest = np.hstack((Xtest, np.transpose(Xtrain_plus)))
    for i in range(length):
        comp = model.predict( np.array([ Xtest[i] ]) )
        # comp1 = model1.predict( np.array([ Xtest[i] ]) )
        torque_after_training[i] = comp   #h2
        # torque_after_training[i] = tc(pos_ori[i, :], vel_ori[i, :], acc_ori[i, :]) + comp   #h4
        # torque_after_training[i] = tcf(pos_ori[i, :], vel_ori[i, :], acc_ori[i, :]) + comp   #h5
        # torque_after_training[i] = tcof(pos_ori[i, :], vel_ori[i, :], acc_ori[i, :]) + comp   #h3
    
    torque_diff = torque_actual - torque_after_training
    # RMSE_after = 0
    # temp = 0
    # for i in range(torque_diff.shape[0]):
    #     x = np.sum(torque_diff**2)
    #     temp += x
    # RMSE_after = math.sqrt(temp / torque_diff.shape[0])

    RMSE_after_j1 = 0
    RMSE_after_j2 = 0
    RMSE_after_j3 = 0
    temp_j1 = 0
    temp_j2 = 0
    temp_j3 = 0
    for i in range(torque_diff.shape[0]):
        j1 = np.sum(torque_diff[i, 0]**2)
        j2 = np.sum(torque_diff[i, 1]**2)
        j3 = np.sum(torque_diff[i, 2]**2)
        temp_j1 += j1
        temp_j2 += j2
        temp_j3 += j3
    RMSE_after_j1 = math.sqrt(temp_j1/torque_diff.shape[0])
    RMSE_after_j2 = math.sqrt(temp_j2/torque_diff.shape[0])
    RMSE_after_j3 = math.sqrt(temp_j3/torque_diff.shape[0])

    # compare
    energy = traj.total_energy(torque_actual, vel_ori, t)
    print("Energy of [J1, J2, J3] (Actual): ", energy, "[Joule]")
    print("Total Energy: ", np.sum(energy))

    print("---before training---")
    print("RMSE_j1: ", RMSE_before_j1)
    print("RMSE_j2: ", RMSE_before_j2)
    print("RMSE_j3: ", RMSE_before_j3)
    # energy = traj.total_energy(torque_sim_with_f, vel_ori, t)
    energy = traj.total_energy(torque_sim, vel_ori, t)
    print("Energy of [J1, J2, J3] (Simulation): ", energy, "[Joule]")
    print("Total Energy: ", np.sum(energy))

    print("---after training---")
    print("RMSE_j1: ", RMSE_after_j1)
    print("RMSE_j2: ", RMSE_after_j2)
    print("RMSE_j3: ", RMSE_after_j3)
    energy = traj.total_energy(torque_after_training, vel_ori, t)
    print("Energy of [J1, J2, J3] (Simulation with NN): ", energy, "[Joule]")
    print("Total Energy: ", np.sum(energy))

    # plt.figure(figsize=(12.8, 9.6), dpi=100)
    plt.figure(figsize=(12.8, 3.2), dpi=100)
    

    plt.subplot(1, 3, 1)
    plt.plot(t, torque_actual[:, 0], 'r', label="$actual$")
    plt.plot(t, torque_sim[:, 0], 'g', label="$sim$")
    plt.plot(t, torque_after_training[:, 0], 'b', label="$trained$")
    plt.legend(loc='upper right')
    plt.axis([0, t[-1], -100, 100])
    plt.xticks(np.arange(0, t[-1], 0.5))
    plt.xlabel("time [sec]")
    plt.ylabel(r"${\tau}$ [Nm]")
    plt.title(r"Joint 1")
    # plt.title(r"Actual Torque")


    plt.subplot(1, 3, 2)
    # plt.plot(t, torque_sim_with_f[:, 0], 'r', label="$J_1$")
    # plt.plot(t, torque_sim_with_f[:, 1], 'g', label="$J_2$")
    # plt.plot(t, torque_sim_with_f[:, 2], 'b', label="$J_3$")
    plt.plot(t, torque_actual[:, 1], 'r', label="$actual$")
    plt.plot(t, torque_sim[:, 1], 'g', label="$sim$")
    plt.plot(t, torque_after_training[:, 1], 'b', label="$trained$")
    plt.legend(loc='upper right')
    plt.axis([0, t[-1], -100, 100])
    plt.xticks(np.arange(0, t[-1], 0.5))
    plt.xlabel("time [sec]")
    plt.ylabel(r"${\tau}$ [Nm]")
    plt.title(r'Joint 2')
    # plt.title(r"Simulated Torque (with ${\tau_{diff}}$)")


    plt.subplot(1, 3, 3)
    plt.plot(t, torque_actual[:, 2], 'r', label="$actual$")
    plt.plot(t, torque_sim[:, 2], 'g', label="$sim$")
    plt.plot(t, torque_after_training[:, 2], 'b', label="$trained$")
    plt.legend(loc='upper right')
    plt.axis([0, t[-1], -100, 100])
    plt.xticks(np.arange(0, t[-1], 0.5))
    plt.xlabel("time [sec]")
    plt.ylabel(r"${\tau}$ [Nm]")
    plt.title(r'Joint 3')
    # plt.title(r"Simulated Torque (with NN)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    selection = input("0 for 'training' / 1 for testing, [0/1]: ")
    selection = int(selection)
    if selection == 0:
        train()
    elif selection == 1:
        test()

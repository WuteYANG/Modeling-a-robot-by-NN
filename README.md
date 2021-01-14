# Modeling-a-robot-by-NN

1 torque_calc_compensate file-converts units of data and includes information like lengths of links and inertia of links of robots
2 torque_calc - contains reduced-order model, and friction model
3 torque_calc_main- import collected torques of robots and compare them with reduced-order model and friction model(from torque_calc) with plots
4 model_comp_NN_new - Neural network model with training setting
5 model_comp_NN - Neural network model(old)

model_comp_NN_new will use functions of files in 1-3. The training data is included in the folder(Collect dat). There are 34 paths. In each path, robot was controlled to move back and forth with 5 different speed. Thus, each path has 10 files (2x5). Path 1 to 28 were used as training data and path 29 to 34 were the testing data.

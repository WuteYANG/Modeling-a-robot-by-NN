import numpy as np
from math import sin, cos, pi

def torque_calc(x, v, a):
    """
    input: theta(3), omega(3), alpha(3) in [radian]
    """
    # x, v, a (radian)

    # unit of Inertia: kg*mm**2 -> kg*m**2
    # unit of mass: kg
    # unit of length from origin(frame i-1) to mass center: mm -> m
    I_1 = np.array( [11600, 8800, 12000], float ) / 10**6
    I_2 = np.array( [23600, 204600, 198300], float ) / 10**6
    I_3 = np.array( [3400, 34300, 33800], float ) / 10**6
    m = np.array( [4.0, 8.6, 2.28, 3.56] )
    d1 = 145.1 / 1000
    a2 = 329.0/1000
    a3 = 311.5/1000 + 106.0/1000
    L1 = d1 - 1.7/1000
    L2 = a2 - 150.0/1000
    L3 = a3 - 74.0/1000
    # a2 = 429.0 / 1000
    # a3 = 411.5 / 1000 + 106.0 / 1000
    # L1 = d1 - 1.7 / 1000
    # L2 = a2 - 204.0 / 1000
    # L3 = a3 - 99.0 / 1000
    g = 9.81
    

    Inertial_1_1 = I_1[1] + I_2[0]*cos(x[1])**2 + I_2[1]*sin(x[1])**2 + I_3[0]*cos(x[1]+x[2])**3 + I_3[1]*sin(x[1]+x[2]) + m[1]*L2**2*sin(x[1])**2 + m[2]*a2**2*sin(x[1])**2 + m[3]*a2**2*sin(x[1])**2 + m[2]*L3**2*sin(x[1]+x[2])**2 + m[3]*a3**2*sin(x[1]+x[2])**2 + 2*L3*a2*m[2]*sin(x[1])*sin(x[1]+x[2]) + 2*a3*a2*m[3]*sin(x[1])*sin(x[1]+x[2])
    Coriolis_1_12 = -2*I_3[0]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*I_3[1]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -2*I_2[0]*cos(x[1])*sin(x[2]) + 2*I_2[1]*cos(x[1])*sin(x[2]) + 2*L3**2*m[2]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*a3**2*m[3]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*L2**2*m[1]*cos(x[1])*sin(x[1]) + 2*a2**2*m[2]*cos(x[1])*sin(x[1]) + 2*a2**2*m[3]*cos(x[1])*sin(x[1]) + 2*L3*a2*m[2]*cos(x[1])*sin(x[1]+x[2]) + 2*L3*a2*m[2]*sin(x[1])*cos(x[1]+x[2]) + 2*a3*a2*m[3]*cos(x[1])*sin(x[1]+x[2]) + 2*a3*a2*m[3]*sin(x[1])*cos(x[1]+x[2])
    Coriolis_1_13 = -2*I_3[0]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*I_3[1]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*L3**2*m[2]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*a3**2*m[3]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*L3*a2*m[2]*cos(x[1])*sin(x[1]+x[2]) + 2*a3*a2*m[3]*sin(x[1])*cos(x[1]+x[2])
    tao_1 = a[0]*Inertial_1_1 + v[0]*( v[1]*Coriolis_1_12 + v[2]*Coriolis_1_13 )

    Inertial_2_2 = I_2[2] + I_3[2] + L2**2*m[1] + L3**2*m[2] + a3**2*m[3] + a2**2*m[2]*cos(x[2])**2 + a2**2*m[3]*cos(x[2])**2 + a2**2*m[2]*sin(x[2])**2 + a2**2*m[3]*sin(x[2])**2 + 2*a2*a3*m[3]*cos(x[2]) + 2*a2*L3*m[2]*cos(x[2])
    Inertial_2_3 = I_3[2] + L3**2*m[2] + a3**2*m[3] + a2*a3*m[3]*cos(x[2]) + a2*L3*m[2]*cos(x[2])
    Centrifugal_2_1 = I_2[0]*cos(x[1])*sin(x[1]) + -I_2[1]*cos(x[1])*sin(x[1]) + I_3[0]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -I_3[1]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -L2**2*m[1]*cos(x[1])*sin(x[1]) + -a2**2*m[2]*cos(x[1])*sin(x[1]) + -a2**2*m[3]*cos(x[1])*sin(x[1]) + -L3**2*m[2]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -a3**2*m[3]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -L3*a2*m[2]*cos(x[1])*sin(x[1]+x[2]) + -L3*a2*m[2]*sin(x[1])*cos(x[1]+x[2]) + -a3*a2*m[3]*cos(x[1])*sin(x[1]+x[2]) + -a3*a2*m[3]*sin(x[1])*cos(x[1]+x[2])
    Centrifugal_2_3 = -a2*a3*m[3]*sin(x[2]) + -a2*L3*m[2]*sin(x[2])
    Coriolis_2_23 = -2*L3*a2*m[2]*sin(x[2]) + -2*a3*a2*m[3]*sin(x[2])
    Potential_2 = -g*( L3*m[2]*sin(x[1]+x[2]) + a3*m[3]*sin(x[1]+x[2]) + L2*m[1]*sin(x[1]) + a2*m[2]*sin(x[1]) + a2*m[3]*sin(x[1]))
    tao_2 = a[1]*Inertial_2_2 + a[2]*Inertial_2_3 + v[0]**2*Centrifugal_2_1 + v[2]**2*Centrifugal_2_3 + v[1]*v[2]*Coriolis_2_23 + Potential_2

    Inertial_3_2 = I_3[2] + L3**2*m[2] + a3**2*m[3] + a2*a3*m[3]*cos(x[2]) + L3*a2*m[2]*cos(x[2])
    Inertial_3_3 = I_3[2] + L3**2*m[2] + a3**2*m[3]
    Centrifugal_3_1 = I_3[0]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -I_3[1]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -L3**2*m[2]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -a3**2*m[3]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -L3*a2*m[2]*sin(x[1])*cos(x[1]+x[2]) + -a3*a2*m[3]*sin(x[1])*cos(x[1]+x[2])
    Centrifugal_3_2 = a2*a3*m[3]*sin(x[2]) + L3*a2*m[2]*sin(x[2])
    Potential_3 = -g*( L3*m[2]*sin(x[1]+x[2]) + a3*m[3]*sin(x[1]+x[2]) )
    tao_3 = a[1]*Inertial_3_2 + a[2]*Inertial_3_3 + v[0]**2*Centrifugal_3_1 + v[1]**2*Centrifugal_3_2 + Potential_3

    torque = np.array( [tao_1, tao_2, tao_3] ) 

    return torque

def torque_calc_friction(x, v, a):
    """
    input: theta(3), omega(3), alpha(3) in [radian]
    """
    # x, v, a (radian)

    # unit of Inertia: kg*mm**2 -> kg*m**2
    # unit of mass: kg
    # unit of length from origin(frame i-1) to mass center: mm -> m
    I_1 = np.array( [11600, 8800, 12000], float ) / 10**6
    I_2 = np.array( [23600, 204600, 198300], float ) / 10**6
    I_3 = np.array( [3400, 34300, 33800], float ) / 10**6
    m = np.array( [4.0, 8.6, 2.28, 3.56] )
    d1 = 145.1 / 1000
    a2 = 329.0/1000
    a3 = 311.5/1000 + 106.0/1000
    L1 = d1 - 1.7/1000
    L2 = a2 - 150.0/1000
    L3 = a3 - 74.0/1000
    # a2 = 429.0 / 1000
    # a3 = 411.5 / 1000 + 106.0 / 1000
    # L1 = d1 - 1.7 / 1000
    # L2 = a2 - 204.0 / 1000
    # L3 = a3 - 99.0 / 1000
    g = 9.81
    # driver system compensation

    # 20180606
    # a3 = 411.5 / 1000
    # Bm = np.array([ 3.7536,  3.4174,  2.2944 ])
    # Cm = np.array([ 8.2665,  10.0073,  8.1045 ])
    # fc = np.array([ 7.0170,  7.1101,  6.2530 ])

    # 20180606
    # test1~4
    # a3 = 411.5 / 1000 + 106.0 / 1000
    # Bm = np.array([ 3.4633,  3.6206,  2.0072 ])
    # Cm = np.array([ 8.2692,  9.9479,  8.1159 ])
    # fc = np.array([ 7.0147,  7.1186,  6.2468 ])
    
    # 20180606
    # test1~10
    # a3 = 411.5 / 1000 + 106.0 / 1000
    Bm = np.array([ 3.0,  3.0,  3.0 ])
    Cm = np.array([ 4.6,  4.6,  4.6 ])
    fc = np.array([ 6.0,  6.0,  6.0 ])
    # Bm =  np.array([  2.1329,  1.7535,  2.8223])
    # Cm =  np.array([ 11.1971,  5.1426,  6.3069])
    # fc =  np.array([  6.0590,  6.6673,  5.8601])
    
    Inertial_1_1 = I_1[1] + I_2[0]*cos(x[1])**2 + I_2[1]*sin(x[1])**2 + I_3[0]*cos(x[1]+x[2])**3 + I_3[1]*sin(x[1]+x[2]) + m[1]*L2**2*sin(x[1])**2 + m[2]*a2**2*sin(x[1])**2 + m[3]*a2**2*sin(x[1])**2 + m[2]*L3**2*sin(x[1]+x[2])**2 + m[3]*a3**2*sin(x[1]+x[2])**2 + 2*L3*a2*m[2]*sin(x[1])*sin(x[1]+x[2]) + 2*a3*a2*m[3]*sin(x[1])*sin(x[1]+x[2])
    Coriolis_1_12 = -2*I_3[0]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*I_3[1]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -2*I_2[0]*cos(x[1])*sin(x[2]) + 2*I_2[1]*cos(x[1])*sin(x[2]) + 2*L3**2*m[2]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*a3**2*m[3]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*L2**2*m[1]*cos(x[1])*sin(x[1]) + 2*a2**2*m[2]*cos(x[1])*sin(x[1]) + 2*a2**2*m[3]*cos(x[1])*sin(x[1]) + 2*L3*a2*m[2]*cos(x[1])*sin(x[1]+x[2]) + 2*L3*a2*m[2]*sin(x[1])*cos(x[1]+x[2]) + 2*a3*a2*m[3]*cos(x[1])*sin(x[1]+x[2]) + 2*a3*a2*m[3]*sin(x[1])*cos(x[1]+x[2])
    Coriolis_1_13 = -2*I_3[0]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*I_3[1]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*L3**2*m[2]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*a3**2*m[3]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*L3*a2*m[2]*cos(x[1])*sin(x[1]+x[2]) + 2*a3*a2*m[3]*sin(x[1])*cos(x[1]+x[2])
    comp1 = Bm[0]*a[0] + Cm[0]*v[0] + fc[0]*np.sign(v[0])
    tao_1 = a[0]*Inertial_1_1 + v[0]*( v[1]*Coriolis_1_12 + v[2]*Coriolis_1_13 ) + comp1

    Inertial_2_2 = I_2[2] + I_3[2] + L2**2*m[1] + L3**2*m[2] + a3**2*m[3] + a2**2*m[2]*cos(x[2])**2 + a2**2*m[3]*cos(x[2])**2 + a2**2*m[2]*sin(x[2])**2 + a2**2*m[3]*sin(x[2])**2 + 2*a2*a3*m[3]*cos(x[2]) + 2*a2*L3*m[2]*cos(x[2])
    Inertial_2_3 = I_3[2] + L3**2*m[2] + a3**2*m[3] + a2*a3*m[3]*cos(x[2]) + a2*L3*m[2]*cos(x[2])
    Centrifugal_2_1 = I_2[0]*cos(x[1])*sin(x[1]) + -I_2[1]*cos(x[1])*sin(x[1]) + I_3[0]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -I_3[1]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -L2**2*m[1]*cos(x[1])*sin(x[1]) + -a2**2*m[2]*cos(x[1])*sin(x[1]) + -a2**2*m[3]*cos(x[1])*sin(x[1]) + -L3**2*m[2]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -a3**2*m[3]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -L3*a2*m[2]*cos(x[1])*sin(x[1]+x[2]) + -L3*a2*m[2]*sin(x[1])*cos(x[1]+x[2]) + -a3*a2*m[3]*cos(x[1])*sin(x[1]+x[2]) + -a3*a2*m[3]*sin(x[1])*cos(x[1]+x[2])
    Centrifugal_2_3 = -a2*a3*m[3]*sin(x[2]) + -a2*L3*m[2]*sin(x[2])
    Coriolis_2_23 = -2*L3*a2*m[2]*sin(x[2]) + -2*a3*a2*m[3]*sin(x[2])
    Potential_2 = -g*( L3*m[2]*sin(x[1]+x[2]) + a3*m[3]*sin(x[1]+x[2]) + L2*m[1]*sin(x[1]) + a2*m[2]*sin(x[1]) + a2*m[3]*sin(x[1]))
    comp2 = Bm[1]*a[1] + Cm[1]*v[1] + fc[1]*np.sign(v[1])
    tao_2 = a[1]*Inertial_2_2 + a[2]*Inertial_2_3 + v[0]**2*Centrifugal_2_1 + v[2]**2*Centrifugal_2_3 + v[1]*v[2]*Coriolis_2_23 + Potential_2 + comp2

    Inertial_3_2 = I_3[2] + L3**2*m[2] + a3**2*m[3] + a2*a3*m[3]*cos(x[2]) + L3*a2*m[2]*cos(x[2])
    Inertial_3_3 = I_3[2] + L3**2*m[2] + a3**2*m[3]
    Centrifugal_3_1 = I_3[0]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -I_3[1]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -L3**2*m[2]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -a3**2*m[3]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -L3*a2*m[2]*sin(x[1])*cos(x[1]+x[2]) + -a3*a2*m[3]*sin(x[1])*cos(x[1]+x[2])
    Centrifugal_3_2 = a2*a3*m[3]*sin(x[2]) + L3*a2*m[2]*sin(x[2])
    Potential_3 = -g*( L3*m[2]*sin(x[1]+x[2]) + a3*m[3]*sin(x[1]+x[2]) )
    comp3 = Bm[2]*a[2] + Cm[2]*v[2] + fc[2]*np.sign(v[2])
    tao_3 = a[1]*Inertial_3_2 + a[2]*Inertial_3_3 + v[0]**2*Centrifugal_3_1 + v[1]**2*Centrifugal_3_2 + Potential_3 + comp3

    torque = np.array( [tao_1, tao_2, tao_3] ) 

    return torque

def torque_calc_onlyfriction(x, v, a):
    """
    input: theta(3), omega(3), alpha(3) in [radian]
    """
    # x, v, a (radian)

    # unit of Inertia: kg*mm**2 -> kg*m**2
    # unit of mass: kg
    # unit of length from origin(frame i-1) to mass center: mm -> m
    I_1 = np.array( [11600, 8800, 12000], float ) / 10**6
    I_2 = np.array( [23600, 204600, 198300], float ) / 10**6
    I_3 = np.array( [3400, 34300, 33800], float ) / 10**6
    m = np.array( [4.0, 8.6, 2.28, 3.56] )
    d1 = 145.1 / 1000
    a2 = 329.0/1000
    a3 = 311.5/1000 + 106.0/1000
    L1 = d1 - 1.7/1000
    L2 = a2 - 150.0/1000
    L3 = a3 - 74.0/1000
    # a2 = 429.0 / 1000
    # a3 = 411.5 / 1000 + 106.0 / 1000
    # L1 = d1 - 1.7 / 1000
    # L2 = a2 - 204.0 / 1000
    # L3 = a3 - 99.0 / 1000
    g = 9.81
    # driver system compensation

    # 20180606
    # a3 = 411.5 / 1000
    # Bm = np.array([ 3.7536,  3.4174,  2.2944 ])
    # Cm = np.array([ 8.2665,  10.0073,  8.1045 ])
    # fc = np.array([ 7.0170,  7.1101,  6.2530 ])

    # 20180606
    # test1~4
    # a3 = 411.5 / 1000 + 106.0 / 1000
    # Bm = np.array([ 3.4633,  3.6206,  2.0072 ])
    # Cm = np.array([ 8.2692,  9.9479,  8.1159 ])
    # fc = np.array([ 7.0147,  7.1186,  6.2468 ])
    
    # 20180606
    # test1~10
    # a3 = 411.5 / 1000 + 106.0 / 1000
    Bm = np.array([ 3.0,  3.0,  3.0 ])
    Cm = np.array([ 4.6,  4.6,  4.6 ])
    fc = np.array([ 6.0,  6.0,  6.0 ])
    # Bm =  np.array([  2.1329,  1.7535,  2.8223])
    # Cm =  np.array([ 11.1971,  5.1426,  6.3069])
    # fc =  np.array([  6.0590,  6.6673,  5.8601])
    
    Inertial_1_1 = I_1[1] + I_2[0]*cos(x[1])**2 + I_2[1]*sin(x[1])**2 + I_3[0]*cos(x[1]+x[2])**3 + I_3[1]*sin(x[1]+x[2]) + m[1]*L2**2*sin(x[1])**2 + m[2]*a2**2*sin(x[1])**2 + m[3]*a2**2*sin(x[1])**2 + m[2]*L3**2*sin(x[1]+x[2])**2 + m[3]*a3**2*sin(x[1]+x[2])**2 + 2*L3*a2*m[2]*sin(x[1])*sin(x[1]+x[2]) + 2*a3*a2*m[3]*sin(x[1])*sin(x[1]+x[2])
    Coriolis_1_12 = -2*I_3[0]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*I_3[1]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -2*I_2[0]*cos(x[1])*sin(x[2]) + 2*I_2[1]*cos(x[1])*sin(x[2]) + 2*L3**2*m[2]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*a3**2*m[3]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*L2**2*m[1]*cos(x[1])*sin(x[1]) + 2*a2**2*m[2]*cos(x[1])*sin(x[1]) + 2*a2**2*m[3]*cos(x[1])*sin(x[1]) + 2*L3*a2*m[2]*cos(x[1])*sin(x[1]+x[2]) + 2*L3*a2*m[2]*sin(x[1])*cos(x[1]+x[2]) + 2*a3*a2*m[3]*cos(x[1])*sin(x[1]+x[2]) + 2*a3*a2*m[3]*sin(x[1])*cos(x[1]+x[2])
    Coriolis_1_13 = -2*I_3[0]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*I_3[1]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*L3**2*m[2]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*a3**2*m[3]*cos(x[1]+x[2])*sin(x[1]+x[2]) + 2*L3*a2*m[2]*cos(x[1])*sin(x[1]+x[2]) + 2*a3*a2*m[3]*sin(x[1])*cos(x[1]+x[2])
    comp1 = Bm[0]*a[0] + Cm[0]*v[0] + fc[0]*np.sign(v[0])
    tao_1 = comp1

    Inertial_2_2 = I_2[2] + I_3[2] + L2**2*m[1] + L3**2*m[2] + a3**2*m[3] + a2**2*m[2]*cos(x[2])**2 + a2**2*m[3]*cos(x[2])**2 + a2**2*m[2]*sin(x[2])**2 + a2**2*m[3]*sin(x[2])**2 + 2*a2*a3*m[3]*cos(x[2]) + 2*a2*L3*m[2]*cos(x[2])
    Inertial_2_3 = I_3[2] + L3**2*m[2] + a3**2*m[3] + a2*a3*m[3]*cos(x[2]) + a2*L3*m[2]*cos(x[2])
    Centrifugal_2_1 = I_2[0]*cos(x[1])*sin(x[1]) + -I_2[1]*cos(x[1])*sin(x[1]) + I_3[0]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -I_3[1]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -L2**2*m[1]*cos(x[1])*sin(x[1]) + -a2**2*m[2]*cos(x[1])*sin(x[1]) + -a2**2*m[3]*cos(x[1])*sin(x[1]) + -L3**2*m[2]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -a3**2*m[3]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -L3*a2*m[2]*cos(x[1])*sin(x[1]+x[2]) + -L3*a2*m[2]*sin(x[1])*cos(x[1]+x[2]) + -a3*a2*m[3]*cos(x[1])*sin(x[1]+x[2]) + -a3*a2*m[3]*sin(x[1])*cos(x[1]+x[2])
    Centrifugal_2_3 = -a2*a3*m[3]*sin(x[2]) + -a2*L3*m[2]*sin(x[2])
    Coriolis_2_23 = -2*L3*a2*m[2]*sin(x[2]) + -2*a3*a2*m[3]*sin(x[2])
    Potential_2 = -g*( L3*m[2]*sin(x[1]+x[2]) + a3*m[3]*sin(x[1]+x[2]) + L2*m[1]*sin(x[1]) + a2*m[2]*sin(x[1]) + a2*m[3]*sin(x[1]))
    comp2 = Bm[1]*a[1] + Cm[1]*v[1] + fc[1]*np.sign(v[1])
    tao_2 = comp2

    Inertial_3_2 = I_3[2] + L3**2*m[2] + a3**2*m[3] + a2*a3*m[3]*cos(x[2]) + L3*a2*m[2]*cos(x[2])
    Inertial_3_3 = I_3[2] + L3**2*m[2] + a3**2*m[3]
    Centrifugal_3_1 = I_3[0]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -I_3[1]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -L3**2*m[2]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -a3**2*m[3]*cos(x[1]+x[2])*sin(x[1]+x[2]) + -L3*a2*m[2]*sin(x[1])*cos(x[1]+x[2]) + -a3*a2*m[3]*sin(x[1])*cos(x[1]+x[2])
    Centrifugal_3_2 = a2*a3*m[3]*sin(x[2]) + L3*a2*m[2]*sin(x[2])
    Potential_3 = -g*( L3*m[2]*sin(x[1]+x[2]) + a3*m[3]*sin(x[1]+x[2]) )
    comp3 = Bm[2]*a[2] + Cm[2]*v[2] + fc[2]*np.sign(v[2])
    tao_3 = comp3

    torque = np.array( [tao_1, tao_2, tao_3] ) 

    return torque


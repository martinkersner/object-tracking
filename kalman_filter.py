#!/usr/bin/env python

# 2D Kalman Filter
# Martin Kersner, m.kersne@gmail.com
# 2016/02/04

# Inspired by https://www.youtube.com/playlist?list=PLX2gX-ftPVXU3oUFNATxGXY90AULiqnWT

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

# Original position
x_0 = 4000
y_0 = 3000

# initial conditions
a_x_0 = 2
a_y_0 = 2

delta_t = 1

# observation errors
delta_oe_x = 25
delta_oe_y = 25
delta_oe_v_x = 6
delta_oe_v_y = 6

diag = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,0],
                 [0,0,0,1]])
H = diag
C = diag
I = diag

# process errors in process covariance matrix
delta_P_x = 20
delta_P_y = 20
delta_P_v_x = 5
delta_P_v_y = 5

# observations 
x = [4000, 4260, 4550, 4860, 5110]
y = [3000, 3260, 3550, 3860, 4110]
v_x = [280,   282,  285,  286,  290]
v_y = [280,   282,  285,  286,  290]

# noise
w_k = 0
Q_k = 0
Z_k = 0

################################################################################

A = np.array([[1, 0, delta_t,       0],
              [0, 1,       0, delta_t],
              [0, 0,       1,       0],
              [0, 0,       0,       1]])

B = np.array([[0.5*(delta_t**2),                0],
              [0,                0.5*(delta_t**2)],
              [delta_t,                         0],
              [0,                        delta_t]])

u_k = np.array([[a_x_0],
                [a_y_0]])

P_k_prev = np.array([[delta_P_x**2,          delta_P_x*delta_P_y,   delta_P_x*delta_P_v_x,     delta_P_x*delta_P_v_y],
                     [delta_P_y*delta_P_x,   delta_P_y**2,          delta_P_y*delta_P_v_x,     delta_P_y*delta_P_v_y],
                     [delta_P_v_x*delta_P_x, delta_P_v_x*delta_P_y, delta_P_v_x**2,          delta_P_v_x*delta_P_v_y],
                     [delta_P_v_y*delta_P_x, delta_P_v_y*delta_P_y, delta_P_v_y*delta_P_v_x,          delta_P_v_y**2]])

P_k_prev *= diag

X_k_prev = np.array([[x[0]   ],
                     [y[0]   ],
                     [v_x[0] ],
                     [v_y[0]]])

R = np.array([[delta_oe_x**2, 0,             0,                              0], 
              [0,             delta_oe_y**2, 0,                              0],
              [0,             0,             delta_oe_v_x**2,                0],
              [0,             0,             0,               delta_oe_v_y**2]])

for x_i, y_i, v_x_i, v_y_i in zip(x[1:], y[1:], v_x[1:], v_y[1:]):
  X_k_prev = A.dot(X_k_prev) + B.dot(u_k) + w_k
  
  # the predicted process covariance matrix
  P_k_prev = A.dot(P_k_prev).dot(A.T) * diag + Q_k
  
  # Kalman gain
  K = np.true_divide(P_k_prev.dot(H.T), H.dot(P_k_prev).dot(H.T) + R)
  K[np.isnan(K)] = 0 # remove NaN values
  
  # the new observation
  Y_k_m = np.array([[x_i  ], 
                    [y_i  ], 
                    [v_x_i], 
                    [v_y_i]])

  Y_k = C.dot(Y_k_m) + Z_k
  
  # the current state
  X_k = X_k_prev + K.dot(Y_k - H.dot(X_k_prev))
  
  # update the process covariance matrix
  P_k = (I-K.dot(H)).dot(P_k_prev)
  
  X_k_prev = X_k
  P_k_prev = P_k

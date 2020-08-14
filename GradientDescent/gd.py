# -*- coding: utf-8 -*-
"""
Multi-variable regression
"""

#set 
w = set('mexicanelephant')

# list
x = [1, 2]

# dictionary
y = {'dude': 2, 'mexican': 8}

# tuple
z = (2, 3)




"""
Dataset
"""

import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt



#dataset = []
x = np.array([1, 2, 3, 4, 5, 6, 22, 33, 44])
y = np.array([1, 2, 3, 4, 5, 6, 9, 19, 22])

theta_0 = -1
theta_1 = -3

def hypothesis (theta_0, theta_1, x):
    return theta_0 + theta_1 * x


def loss(theta_0, theta_1, x, y):
    # J = 1 / 2 m sum (h(x_i) - y_i)²

    m = len (x)
    h = hypothesis(theta_0, theta_1, x)
    e = h - y
    return 1 / (2*m) * sum( pow(e, 2) )


def gradient_descent(theta_0, theta_1, alpha, x, y):
    
    m = len (x)
    
    h = hypothesis(theta_0, theta_1, x)
    
    e = h - y
    
    J_0 = 1 / m * sum( e )
    J_1 = 1 / m * sum( e * x)
       
    # update theta by theta_j - alpha dJ / dtheta_j
    temp_0 = theta_0 - alpha * J_0
    temp_1 = theta_1 - alpha * J_1
    
    return temp_0, temp_1
    
loss_v = []

for i in np.arange(0, 20, 1):
    # training
    theta_0, theta_1 = gradient_descent(theta_0, theta_1, 0.001, x, y)
    

    loss_v.append(loss(theta_0, theta_1, x, y))    
    
    # inference
    h = hypothesis(theta_0, theta_1, x)
    
    # plot hypothesis and data set

    if i % 2 == 0:
        fig, ax = plt.subplots(1, 2)
    
    
        ax[0].plot(x, h)
        ax[0].plot(x, y)
        ax[0].set(xlabel='x', ylabel='y',
               title='training set and hypothesis')
        ax[0].grid()    
        
        ax[1].plot(np.arange(0, len(loss_v)), loss_v)
        ax[1].set(xlabel='iter', ylabel='loss',
               title='Loss')
        ax[1].grid()
        plt.show()
        time.sleep(1)
    


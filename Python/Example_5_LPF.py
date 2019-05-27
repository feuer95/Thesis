# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:58:58 2019

@author: elena
"""

from LPFMethod import longpath # Recall the long-path following function 
import numpy as np # To create vectors
import pandas as pd # Export to excel 
import matplotlib.pyplot as plt
'''
ANALYSIS OF THE SEQUENCE COMPUTED BY LPF METHOD FOR THIS EXAMPLE 
'''

# Input data
A = np.array([[1, 1, 2],[2, 0, 1],[2, 1, 3]])
c = np.array([-3, -2, -4])
b = np.array([4, 1, 7])

# Recall of the LPF method
x_l, s_l, u_l = longpath(A, b, c)

#Create a DataFrame for LPF
dfl = pd.DataFrame(u_l, columns = ['it', 'g_l', 'x_l','s_l']) 
dfl.plot(x = 'it', y = 'g_l', color = 'g', grid = True)

# Solutions at the 2nd interation
gamma = 0.001
c_A = 6
u_l[2][1] # g_1 = 15.67..
x = u_l[2][2]
s = u_l[2][3]
(x*s > gamma*np.dot(x,s)/c_A).all() # True: it is in the neighborhhod

# Create boundary of the neighborhood and value mu
mu = []
N = []
for i in range(len(u_l)):
    x = u_l[i][2]
    s = u_l[i][3]
    mi = np.dot(x,s)/ c_A
    gmi = gamma*mi
    mu.append([mi.copy()])
    N.append([gmi.copy()])
    
plt.plot(x ='it', y = 'N', color = 'g', grid = True)
plt.show()

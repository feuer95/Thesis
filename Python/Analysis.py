# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:20:16 2019

@author: elena
"""
from LPFMethod import longpath # Recall the long-path following function 
from MehrotraMethod import mehrotra # Recall mehrotra's method
from SimplexMethodIIphases import SimplexMethod # Recall the simplex method
import numpy as np # To create vectors
import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # Print plot
from input_data import input_data # Recall the examples 
'''
                                                  ===
                                            Number of iterations 
                                                  ===

Input data in canonical form: matrix A, vector b and c as np.array

Implementation of the mehrotra(A, b, c, c_form = 0, w = 0.005) 
                      longpath(A, b, c, gamma = 0.001, s_min = 0.1, s_max = 0.9, c_form = 0, w = 0.005)
                      SimplexMethod(A, b, c, max_iter = 500, rule = 0, c_form = 0)
                      
Output points x, s, list u

Plot the number of iterations in graphic.

'''

#%%

# Input data of canonical form of the example number p
p = 3
(A, b, c) = input_data(8)

# Recall the interior point methods
x_s, u_s = SimplexMethod(A, b, c)
x_l, s_l, u_l = longpath(A, b, c)
x_m, s_m, u_m = mehrotra(A, b, c)

# Plot the number of iterations
plt.figure()
plt.bar(['it_simplex', 'it_LPF','it_Mehrotra'],[len(u_s),len(u_l),len(u_m)], color='c')
plt.title('Number of iterations: example {}'.format(p))
plt.legend('it', loc = 2)
locs, labels = plt.yticks(np.arange(0, max(len(u_m),len(u_l),len(u_s)), step = 1))
plt.show()



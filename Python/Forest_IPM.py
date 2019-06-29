# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:52:17 2019

@author: Elena
"""
import numpy as np
from AffineMethod import affine 

from MehrotraMethod import mehrotra
from MehrotraMethod2 import mehrotra2

from LPFMethod import longpath1
from LPFMethod2 import longpath2
from LPFMethod_cp import longpathC
from LPFMethod_PC import longpathPC

import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # Print plot
from cent_meas import cent_meas
from SimplexMethodIIphases import SimplexMethod
# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)


'''                     ===
            FOREST_SERVICE_ALLOCATION 
                        ===

Find the MAXIMUM total NPV: the constraint set in standard form A x = b using the methods:
    
    1. affine(A, b, -c, c_form = 1)
    2. mehrotra(A, b, -c, c_form = 1) 
    3. longpath1(A, b, -c, gamma = 0.001, s_min = 0.1, s_max = 0.9, c_form = 0, w = 0.005)
    
    input data: A, b, c, c_form = 1, w = 10^{-8} default

'''

print('\n\tfirst TEST ON FOREST SERVICE ALLOCATION\n')

""" import & construct input data in standard form"""

excel_file = 'Forest.xlsx'
r = pd.read_excel('Forest.xlsx')
c = np.array(r['p'])  # Cost vector of maximum problem
T = np.array(-r['t'])
G = np.array(-r['g'])
W = np.array(-r['w'])/788

# Construct A in a standard form -> c_form = 1 in the input function
B = np.zeros((7,21))
for i in range(7):
    B[i,i*3:(i+1)*3] = np.ones(3)
Y = np.vstack((T,G,W)) # inequality constraints
r_Y, c_Y = Y.shape
r_B, c_B = B.shape 
AI = np.concatenate((B, np.zeros((r_B,r_Y))), axis = 1)  
AII = np.concatenate((Y, np.identity(r_Y)), axis = 1)

#%%
A = np.concatenate((AI, AII), axis = 0)

# Concatenate c
c = np.concatenate((c, np.zeros(r_Y)), axis = 0)

# Construct b
S = np.asarray(r['s'])
S = S[np.logical_not(np.isnan(S))]
b = np.array([-40000, -5, -70])
b = np.concatenate((S, b))

#%%

""" run the methods """

# Recall the interior point methods
# Plot dual gap e centering measure
#
#x_a, s_a , u_a = affine(A, b, -c, c_form = 1)
#dfu = cent_meas(x_a, u_a, label = 'Affine') # 29 it
#
#x_m, s_m, u_m = mehrotra(A, b, -c, c_form = 1)
#dfm = cent_meas(x_m, u_m, label = 'Mehrotra') # 9 iterations
#
#x_l, s_l, u_l = longpath(A, b, -c, c_form = 1)
#dful = cent_meas(x_l, u_l, label = 'LPF', plot = 0) # 21 iterations
#
#x_m, s_m, u_m = mehrotra2(A, b, -c, c_form = 1)
#dfm = cent_meas(x_m, u_m, label = 'Mehrotra', plot = 0) # 37 iterations

#x_c, s_c, u_c = longpath2(A, b, -c, c_form = 1) # 30 it
#dfc = cent_meas(x_c, u_c, label = 'LPF2')
#
#x_pc, s_pc, u_pc = longpathPC(A, b, -c, c_form = 1)
#cent_meas(x_pc, u_pc, label = 'LPF PC', plot = 0) # 19 iterations
#
#P1, u = SimplexMethod(A, b, -c, rule = 0, c_form = 1) # 45 iterations

#plt.show()
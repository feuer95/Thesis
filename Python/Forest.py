# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:52:17 2019

@author: Elena
"""
import numpy as np
import pandas as pd
from SimplexMethodIIphases import SimplexMethod
from MehrotraAlgorithm import mehrotra
from Long_path_following import longpath
from Long_path_following2 import longpath2
from AffineScalingMethod import affine

# Clean form of printed vectors
np.set_printoptions(precision=4, threshold=10, edgeitems=4, linewidth=120, suppress = True)

excel_file = 'sample.xlsx'
r = pd.read_excel('sample.xlsx')
c = np.array(r['p'])
T = np.array(-r['t'])
G = np.array(-r['g'])
W = np.array(-r['w'])/788
B = np.zeros((7,21))
for i in range(7):
    B[i,i*3:(i+1)*3] = np.ones(3)
    
# Concatenate A
Y = np.vstack((T,G,W)) # inequality constraints
r_Y, c_Y = Y.shape
r_B, c_B = B.shape 
AI = np.concatenate((B, np.zeros((r_B,r_Y))), axis = 1)  
AII = np.concatenate((Y, np.identity(r_Y)), axis = 1)
A = np.concatenate((AI, AII), axis = 0)


# Concatenate c
c = np.concatenate((c, np.zeros(r_Y)), axis = 0)
# Vector b
S = np.asarray(r['s'])
S = S[np.logical_not(np.isnan(S))]
b = np.array([-40000, -5, -70])

# Concatenate b
b = np.concatenate((S, b))

''' RUN THE METHOD '''

#SimplexMethod(A, 'True', b, -c, 1000, 0)

x = affine(A, 0, b, -c)
#affine(A, b, c) # MATRIX SINGULAR 

#mehrotra(A, b, c)

#longpath2(A, b, c, 0.0001, 0.1, 0.9)

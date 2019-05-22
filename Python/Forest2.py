# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:14:03 2019

@author: elena
"""
import numpy as np
import pandas as pd
from SimplexMethodIIphases import SimplexMethod
from MehrotraMethod import mehrotra
from LPFMethod import longpath
from AffineMethod import affine

# Clean form of printed vectors
np.set_printoptions(precision=4, threshold=10, edgeitems=4, linewidth=120, suppress = True)


''' FOREST_SERVICE_ALLOCATION '''

"""
Find the MAXIMUM total NPV: the constraint set in canonical form A x = b using the methods:
    
    1. affine(A, b, c, c_form = 0, w = 0.005):
    2. mehrotra(A, b, c, c_form = 0, w = 0.005)
    3. longpath(A, b, c, gamma = 0.001, s_min = 0.1, s_max = 0.9, c_form = 0, w = 0.005)
    4. SimplexMethod(A, b, c, max_iter = 500, rule = 0, c_form = 0) 
    
    input data: A, b, c ( c_form = 0, w = 0.005 default )
"""

print('\n\tsecond TEST ON FOREST SERVICE ALLOCATION\n')

""" import & construct input data """

excel_file = 'Forest.xlsx'
r = pd.read_excel('Forest.xlsx')
c = np.array(r['p'])
T = np.array(-r['t'])
G = np.array(-r['g'])
W = np.array(-r['w'])/788
    
# Concatenate A
B = np.zeros((7,21))
for i in range(7):
    B[i,i*3:(i+1)*3] = np.ones(3)
A = np.vstack((B,-B,T,G,W)) # inequality constraints

# Vector b
S = np.asarray(r['s'])
S = S[np.logical_not(np.isnan(S))]
b = np.array([-40000, -5, -70])
b = np.concatenate((S,-S, b))


""" run the methods """

#x5, y5 = mehrotra(A, b, -c, w = 0.05)
# found optimal value after 9 iterations

#x6, y6 = affine(A, b, -c, w = 0.05)
# LinAlgError: Matrix is not positive definite

#x6, y6 = affine(A, b, -c, w = 0.5)
# found optimal value after 26 iterations with tollerance 0.5

#x6, y6 = longpath(A, b, -c, w = 50)
# LinAlgError: Matrix is not positive definite

SimplexMethod(A, b, -c, max_iter = 500)
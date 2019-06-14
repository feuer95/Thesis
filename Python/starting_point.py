# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:14:11 2019

@author: elena
"""

import numpy as np

# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)


'''' STARTING POINT '''

"""
Input data: np.arrays of matrix A, cost vector c, vector b of the STANDARD LP
Output: vector x, lambda, s infeasible points
(See Mehrotra's paper)
"""

def sp(A, c, b) -> (np.array, np.array, np.array):
    
    r_A, c_A = A.shape

    V = np.linalg.inv(np.dot(A, A.T)) 
    x = np.dot(np.linalg.pinv(A), b) # initial feasible x
    la = np.dot(A, c)                                
    y = np.dot(V, la)                # initial feasible lambda
    s = c - np.dot(A.T, y)           # initial feasible s
            
    # First update 
    dx = np.min(x)
    if dx < 0:
       x += (-3/2)*dx*np.ones(c_A) 
    ds = np.min(s)
    if ds < 0:
       s += (-3/2)*ds*np.ones(c_A)
    
    # Second update 
    if not np.dot(x,s) == 0: 
        Dx = np.dot(x,s)/sum(s)
        Ds = np.dot(x,s)/sum(x)
        x += 2*Dx*np.ones(c_A) 
        s += 2*Ds*np.ones(c_A)
    return x,y,s
 
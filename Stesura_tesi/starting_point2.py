# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 08:37:08 2019

@author: elena
"""
import numpy as np
import random

'''      ================
         STARTING POINT 2
         ================
     
     
Input data: np.arrays of matrix A, cost vector c, vector b of the STANDARD LP
Output: vector x, lambda, s infeasible points

(See Mehrotra's paper)

'''

def sp2(A, c, b) -> (np.array, np.array, np.array):
    
    eps = random.random()
    r_A, c_A = A.shape
    
    x = np.ones(c_A)                 
    s = np.ones(c_A)   # xTs = c_A        
    y = np.zeros(r_A) 
    r = np.concatenate((np.dot(A,x) - b, c -s - np.dot(A.T, y)))       
    
    n = np.linalg.norm(r)
    if n / np.dot(x,s) < eps:
        return x, y, s
    else:
        et = n/(eps*c_A)
        x *= et
        s *= et
        
    return x, y, s


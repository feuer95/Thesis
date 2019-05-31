# -*- coding: utf-8 -*-
"""
Created on Wed May 29 07:53:54 2019

@author: elena
"""

import numpy as np

''' 
                         = EXAMPLES = 


 We compute some examples of a LP problem in a canonical form:

             min { c^{T} x | A x <= b, x >= 0 }

 The input data are:
     A matrix n x m with n < m
     b and c vectors respectively (n x 1) and (m x 1)

 All examples are labelled with a number, in order to be recalled easily in other editors
 
'''

def input_data(w):
    if w == 1:
        A = np.array([[3, 2], [0, 1]])
        b = np.array([4, 3])
        c = np.array([-1, -1])
    if w == 2: # Expected output is c^{T} x* = -17700 at x* = [650, 1100]
        A = np.array([[1, 0],[0, 1],[1, 1],[4, 2]])
        c = np.array([-12, -9])
        b = np.array([1000, 1500, 1750, 4800])
    if w == 3: # expected output is c^{T} x* = -88 at x* = [12, 8]
        A = np.array([[2, 1],[2, 3]])
        c = np.array([-4, -5])
        b = np.array([32, 48])
    if w == 4: #The problem is not feasible
        A = np.array([[1, 1],[-1, -1]])
        c = np.array([2, -3])
        b = np.array([8, -11])
    if w == 5: # Expected output is c^{T} x* = -8. 5 at x* = [0.5 3.5 0.  0.  0.  2.5] 
        A = np.array([[1, 1, 2],[2, 0, 1],[2, 1, 3]])
        c = np.array([-3, -2, -4])
        b = np.array([4, 1, 7])
    if w == 6: # Expected output is c^{T} x* = -25 at x* = [3, 0, 4, 0, 0]
        A = np.array([[2, 0, -1, 2, -1], [-1, 3, 1, -1, 2]])
        b = np.array([2, 1])
        c = np.array([-3, 1, -4, -1, -2])
    if w == 7: # Expected output is c^{T} x* = 85 at x* = [7.5, 0, 2.5, 0, 0]
        A = np.array([[-1, 1, -1, 1, 1], [-1, -4, 1, 3, 1]])
        b = np.array([-10, -5])
        c = np.array([9, 16, 7, -3, -1])
    if w == 8: # Expected output is c^{T} x* = -3 at x* = [2, 1, 0, 0, 0] !!!!
        A = np.array([[1, 2],[2, -1],[0, 1]])
        c = np.array([-1, -1])
        b = np.array([4, 3, 1])
    if w == 9: # Example Unlimited problem           
       A = np.array([[1, -1],[-1, 1]])
       c = np.array([-1, -1])
       b = np.array([1, 1])
    if w == 10: # Example in cycle, need Bland's rule            
       A = np.array([[0.25, -60, -0.04, 9],[0.5, -90, -0.02, 3],[0, 0, 1, 0]])
       c = np.array([-0.75, 150, -0.02, 6])
       b = np.array([0, 0, 1])
    return A, b, c


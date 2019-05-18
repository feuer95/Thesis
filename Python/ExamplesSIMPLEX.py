# -*- coding: utf-8 -*-
"""
Created on Sun May  5 15:39:31 2019

@author: Elena
"""

from SimplexMethodIIphases import SimplexMethod
import numpy as np

''' EXAMPLES'''

"""
 We compute some examples of a LP problem in a canonical form:

             min { c^{T} x | A x <= b, x >= 0 }

 The input data are:
 A matrix n x m with n < m
 b and c vectors respectively (n x 1) and (m x 1)

 The input in a canonical form is required in the implementation of the 
 simplex method
"""


#%%
    
# Input data of canonical LP:
A = np.array([[3, 2], [0, 1]])
b = np.array([4, 3])
c = np.array([-1, -1])

# Expected output is c^{T} x* = -2 at x* = [0, 2]
SimplexMethod(A, b, c, 500)

#%%
    
# Input data of canonical LP:
A = np.array([[1, 0],[0, 1],[1, 1],[4, 2]])
c = np.array([-12, -9])
b = np.array([1000, 1500, 1750, 4800])

# Expected output is c^{T} x* = -17700 
# at x* = [650, 1100]
SimplexMethod(A, b, c, 500)

#%%
    
# Input data of canonical LP:
A = np.array([[2, 1],[2, 3]])
c = np.array([-4, -5])
b = np.array([32, 48])

# Recall the function: expected output is c^{T} x* = -88 
# at x* = [12, 8]
SimplexMethod(A, b, c, 500)

#%%
    
# Input data of canonical LP:
A = np.matrix([[1, 1],[-1, -1]])
c = np.array([2, -3])
b = np.array([8, -11])

# Expected output is c^{T} x* = 3 in phase I -> 
#The problem is not feasible

SimplexMethod(A, b, c, 500)

#%%
    
# Input data of canonical LP:
A = np.matrix([[1, 1, 2],[2, 0, 1],[2, 1, 3]])
c = np.array([-3, -2, -4])
b = np.array([4, 1, 7])

# Expected output is c^{T} x* = -8. 5 at x* = [0.5 3.5 0.  0.  0.  2.5] 

SimplexMethod(A, b, c, 500)
   
#%%

# Input data of canonical LP:
A = np.matrix([[2, 0, -1, 2, -1], [-1, 3, 1, -1, 2]])
b = np.array([2, 1])
c = np.array([-3, 1, -4, -1, -2])

# Recall the function: expected output is c^{T} x* = -25 at x* = [3, 0, 4, 0, 0]

SimplexMethod(A, b, c, 500)

#%%

# Input data of canonical LP:
A = np.array([[-1, 1, -1, 1, 1], [-1, -4, 1, 3, 1]])
b = np.array([-10, -5])
c = np.array([9, 16, 7, -3, -1])

# Recall the function: expected output is c^{T} x* = 85 at x* = [7.5, 0, 2.5, 0, 0]

SimplexMethod(A, b, c, 500)

#%%

# Input data of canonical LP:
A = np.matrix([[1, 2],[2, -1],[0, 1]])
c = np.array([-1, -1])
b = np.array([4, 3, 1])

# Recall the function: expected output is c^{T} x* = -3 at x* = [2, 1, 0, 0, 0]
# SimplexMethod(A, b, c, max_iter)
SimplexMethod(A, b, c, 500)


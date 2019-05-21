# -*- coding: utf-8 -*-
"""
Created on Sat May 11 19:28:53 2019

@author: Elena
"""
import numpy as np
from AffineMethod import affine

''' EXAMPLES'''

"""
 We compute some examples of a LP problem in a canonical form:

             min { c^{T} x | A x <= b, x >= 0 }
             
A matrix n x m with n < m 
b and c vectors respectively (n x 1) and (m x 1)

 The input data of the function affine(A, b, c, c_form):
Input data: np.arrays of matrix A, cost vector c, vector b of the LP                 
            c_form: canonical problem -> 0 by default
            
 The output print the solution in abox with info : 
 x*, dual gap, c^x*, number of iterations.
 
 At each iteration: centering parameter
                    Search direction
                    max step length
                    current point
                    dual gap
                     
"""

#%%

# Input data of canonical form
A = np.array([[3, 2], [0, 1]])
b = np.array([4, 3])
c = np.array([-1, -1])
x = affine(A, b, c)    

#Output data
# After 6 iterations found optimal solution -2 at x* = [0. 2. 0. 1.]
# with s = [0.5 0.  0.5 0. ]

#%%

# Input data
A = np.array([[1, 0],[0, 1],[1, 1],[4, 2]])
c = np.array([-12, -9])
b = np.array([1000, 1500, 1750, 4800])
x = affine(A, b, c)     

#Output data
# After 9 iterations found optimal solution -17699.997 at x* = [ 649.999 1100.001]

#%%
    
# Input data of canonical LP:
A = np.array([[2, 1],[2, 3]])
c = np.array([-4, -5])
b = np.array([32, 48])
x = affine(A, b, c)     

#Output data
# After 6 iterations found optimal solution -88 at x = [11.999, 8]
# with y = [-0.5, -1.5], s = [0, 0, 0.5, 1.5] 

#%%
    
# Input data of canonical LP:
A = np.array([[1, 1],[-1, -1]])
c = np.array([2, -3])
b = np.array([8, -11])
x = affine(A, b, c)        

# IT GIVES A PROBLEM OF CHOLESKY COMPUTATION

#%%
    
# Input data of canonical LP:
A = np.array([[1, 1, 2],[2, 0, 1],[2, 1, 3]])
c = np.array([-3, -2, -4])
b = np.array([4, 1, 7])
x = affine(A, b, c)       

# Output data
# After 8 iterations found optimal sol
# x* = [0.499 3.499 0.001 0.    0.001 2.5  ]
# Optimal cost:     -8.502  

#%%

# Input
A = np.array([[2, 0, -1, 2, -1], [-1, 3, 1, -1, 2]])
b = np.array([2, 1])
c = np.array([-3, 1, -4, -1, -2])
x = affine(A, b, c)   

# Output data
# After 8 iterations found optimal sol c^x* -25
# x* = [3. 0. 4. 0. 0.]
# s* = [ 0. 34.  0.  2. 13.  7. 11.] 

#%%

# Input data of canonical LP:
A = np.array([[-1, 1, -1, 1, 1], [-1, -4, 1, 3, 1]])
b = np.array([-10, -5])
c = np.array([9, 16, 7, -3, -1])
x = affine(A, b, c)   

#Output data
# After 12 iterations found optimal solution 85 at x* = [7.5, 0, 2.5, 0, 0]
# y* = [-8, -1], 
# s* = [ 0.    19.999  0.  8.  8.  8.  1.]

#%%

# Input data of canonical LP:
A = np.array([[1, 2],[2, -1],[0, 1]])
c = np.array([-1, -1])
b = np.array([4, 3, 1])
x = affine(A, b, c)     

#Output data
# After 4 iterations found optimal solution -3 at x* = [2, 1, 0, 0, 0]
# y* = [-0.317 -0.342 -0.709] 
# s* = [0.    0.    0.317 0.342 0.709]

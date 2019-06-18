# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:25:40 2019

@author: elena
"""

from print_boxed import print_boxed # Print pretty boxes
from stdForm import stdForm # Convert in standard form
import numpy as np # Create vectors
import matplotlib.pyplot as plt # Create graphics
import pandas as pd # Export to excel 
from input_data import input_data

# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)


'''' _SIMPLEX METHOD II PHASES_ for swedish steel '''

"""
Input data: np.arrays: A, vector b, cost vector c of the model LP
            maximum no of iterations          (default 500)
            rule: if Bland's rule             (default 0)
            c_form: if canonical form         (default 0)
            
Output: vector x* optimal vector
        list u = [iterations, bases, vectors x, solutions c*x]
"""
    
def SimplexMethodI(A, b, c, max_it = 500, rule = 0, c_form = 0):
    
    """ Error checking """
        
    if not (isinstance(A, np.ndarray) or isinstance(b, np.ndarray) or isinstance(c, np.ndarray)):
        raise Exception('Inputs must be a numpy arrays')
                
    # Construction in a standard form [A | I]
    if c_form == 0:
        (A, c) = stdForm(A, c)   
    A = np.asmatrix(A)    
    r_A, c_A = A.shape

    """ Check full rank matrix """
    
    # Remove ld rows:
    if not np.linalg.matrix_rank(A) == r_A:        
        A = A[[i for i in range(r_A) if not np.array_equal(np.linalg.qr(A)[1][i, :], np.zeros(c_A))], :]
        r_A = A.shape[0]  # Update no. of rows                                                       
    B = {0, 1, 2, 3, 4, 6, 7, 10, 12, 14, 16} 
    D = list(B)
    x = np.zeros(c_A)
    x[D] = np.linalg.solve(A[:, D], b)
              
    info, x, B, z, itII, u = fun(A, c, x, B, 0, max_it, rule)
        
    # Print termination of phase II 
    
    if info == 0:
        print_boxed("Found optimal solution at x* =\n{}\n\n".format(x) +
#                    "Basis indexes: {}\n".format(B) +
#                    "Nonbasis indexes: {}\n".format(set(range(c_A)) - B) +
                    "Optimal cost: {}\n".format(z.round(decimals = 3))+
                    "Number of iterations: {}.".format(itII))
    elif info == 1:
        print("\nUnlimited problem.")
    elif info == 2:
        print('The problem is not solved after {} iterations.'.format(max_it))
    return x, u
    

"""Algorithm"""
    
def fun(A, c, x, B, it, max_it, rule) -> (float, np.array, set, float, np.array, list):
    
    r_A, c_A = np.shape(A)
    B, NB = list(B), set(range(c_A)) - B  # Basic /nonbasic index lists    
    B_inv = np.linalg.inv(A[:, B])
    z = np.dot(c, x)  # Value of obj. function
    u = []
    while it <= max_it:  # Ensure procedure terminates (for the min reduced cost rule)
        print("\t\nIteration: {}\nCurrent x: {} \nCurrent B: {}\n".format(it, x, B), end = '')
        u.append([it, B.copy(), x, z.copy()]) # Update table
        lamda = np.dot(c[B], B_inv)
        if rule == 0:  # Bland rule
            optimum = True
            for s in NB: # New reduced cost
                m = np.asscalar(c[s] - np.dot(lamda, A[:, s]))
                if m < 0: # Find d < 0
                    optimum = False
                    break
        elif rule == 1: # Withou Bland's rule
            m , s = min([(np.asscalar(c[q] - lamda * A[:, q]), q) for q in NB], key=(lambda tup: tup[0]))
#           ^ c_s and position s
            optimum = (m >= 0) 
#                true if the minimum of the cost vector is positive
        if optimum:
            info = 0
            return info, x, set(B), z, it, u
            
        """Feasible basic direction"""
        d = np.zeros(c_A) 
#       ^ vector that increments x_B and x_NB: here c_A is m + n because A is in standard form
        
        for i in range(r_A):
            d[B[i]] = np.asscalar(-B_inv[i, :] * A[:, s]) #solve B*d = A_s -> -B^-1*A_s
        d[s] = 1
        neg = [(-x[B[i]] / d[B[i]], i) for i in range(r_A) if d[B[i]] < 0]
        
        if len(neg) == 0: # If d > 0
            info = 1            
            return info, x, B, None, it, u  # info = 1 if Unlimited return
        
        theta, r = min(neg, key=(lambda t: t[0]))  # Find r
        
        x = x + theta * d
        z = z + theta * m       
        
        # Update inverse:
        
        for i in set(range(r_A)) - {r}:
            B_inv[i, :] -= d[B[i]]/d[B[r]] * B_inv[r, :]
        B_inv[r, :] /= -d[B[r]]
        
        NB = NB - {s} | {B[r]}  # Update non-basic index set
        B[r] = s                # Update basic index list       
        it += 1                 # Update iteration
    return 2, x, set(B), z, it, u # info = 2 if max_iteration

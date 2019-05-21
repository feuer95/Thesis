# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:49:50 2019

@author: Elena
"""
from stdForm import stdForm # Standard form transform
from print_boxed import print_boxed
from starting_point import sp # Find initial infeasible points
import numpy as np

# Clean form of printed vectors
np.set_printoptions(precision=4, threshold=10, edgeitems=4, linewidth=120, suppress = True)


''' AFFINE-SCALING METHOD '''

"""
Input data: np.arrays of matrix A, cost vector c, vector b of the LP
            c_form: canonical form -> 0
"""


def affine(A, b, c, c_form = 0):
        
    print('\n\tCOMPUTATION OF PRIMAL-DUAL AFFINE SCALING ALGORITHM')
    
    # Algorithm in 4 steps:  
    # 0..Input error checking
    # 1..Find the initial point with Mehrotra's method 
    # 2..obtain the search direction,
    # 3..find the largest step   
    
        
    """ Input error checking """
        
    if not (isinstance(A, np.ndarray) or isinstance(b, np.ndarray) or isinstance(c, np.ndarray)):
       raise Exception('Inputs must be a numpy arrays')
        
    # Construction in a standard form [A | I]
    if c_form == 0:
        A, c = stdForm(A, c)    
    r_A, c_A = A.shape
    
    """ Check full rank matrix """
    
    if not np.linalg.matrix_rank(A) == r_A: # Remove ld rows:
        
        A = A[[i for i in range(r_A) if not np.array_equal(np.linalg.qr(A)[1][i, :], np.zeros(c_A))], :]
        r_A = A.shape[0]  # Update no. of rows


    """ Initial points """
    
    # Initial infeasible positive (x,y,s) and Initial gap g
    (x,y,s) = sp(A, c, b)    
    g = np.dot(c,x) - np.dot(y,b)
    
    print('\nInitial primal-dual point:\n x = {} \n lambda = {} \n s = {}.\n'.format(x, y, s))    
    print('Dual initial gap: {}.\n'.format("%10.3f"%g))      
    
   #%%
        
    """ search vector direction """
    
    # Compute the search direction solving the matricial system with sigma = 0
    # Instead of solving the std system matrix it is uses AUGMENT SYSTEM 
    # with CHOL approach
    
    it = 0
    while abs(g) > 0.02:
        
        print("\tIteration: {}\n".format(it))
        S_inv = np.linalg.inv(np.diag(s))           
        W1 = S_inv*np.diag(x)                       # W1 = D = S^(-1)*X    
        W2 = np.dot(A, W1)                          # W      A*S^(-1)*X
        W  = np.dot(W2, A.T)
        L = np.linalg.cholesky(W)                   # CHOLESKY for A* D^2 *A^T
        L_inv = np.linalg.inv(L) 
        
        # RHS of the system        
        rb = b - np.dot(A, x)
        rc = c - np.dot(A.T, y) - s
        rxs = - x*s  # Newton step toward x*s = 0
        
        B = rb + np.dot(W2, rc) - np.dot(np.dot(A, S_inv), rxs) #RHS of normal equation form
        z = np.dot(L_inv, B)
        
        # SEARCH DIRECTION:        
        y1 = np.dot(L_inv.T, z)
        s1 = rc - np.dot(A.T, y1)
        x1 = np.dot(S_inv, rxs) - np.dot(W1,s1)
        print('Search direction vectors: \n delta_x = {} \n delta_lambda = {} \n delta_s = {}.\n'.format(x1.round(decimals = 3),x1.round(decimals = 3),s1.round(decimals = 3)))
        
        #%%
        
        """ largest step length """
        
        # Largest step length T such that (x, s) + T (x1, s1) is positive
        m = min([(-x[i] / x1[i], i) for i in range(c_A) if x1[i] < 0], default = [1])[0] 
        n = min([(-s[i] / s1[i], i) for i in range(c_A) if s1[i] < 0], default = [1])[0] 
        T = (0.9)*min(m, n) 

           
        # INCREMENT of the vectors and the number of iterations
        x += min(T,1)*x1
        y += min(T,1)*y1
        s += min(T,1)*s1        
        z = np.dot(c, x) # Current optimal solution
        g = z - np.dot(y, b)        
        it += 1
        
        print('Current point:\n x = {} \n lambda = {} \n s = {}.\n'.format(x, y, s))
        print('Dual next gap: {}.\n'.format("%10.3f"%g))
        
    print_boxed("Found optimal solution of the canonical problem at\n x* = {}.\n".format(x) +
                "Dual gap: {}\n".format("%10.6f"%g) +
                "Optimal cost: {}\n".format("%10.3f"%z) +
                "Number of iteration: {}".format(it))
    return x

if __name__ == "__main__":
    
#     Input data of canonical LP:
    A = np.array([[1, 0],[0, 1],[1, 1],[4, 2]])
    c = np.array([-12, -9])
    b = np.array([1000, 1500, 1750, 4800])
    affine(A, b, c, 0)
    
# optimal solution of the canonical problem at 
#  x* = [ 650. 1100.  350.  400.    0.    0.]                                                                                    |
# Dual gap:   0.002675                               
# Optimal cost:     -17699.997                                                    
#| Number of iteration: 31   
    A = np.array([[1, 1]])
    b = np.array([1])
    c = np.array([1, 0])
#    array([0.124, 0.438, 0.438]) abs g > 0.5
    
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:49:50 2019

@author: Elena
"""
from Initial_primal_feasible_point import InitFeas
from Initial_dual_feasible_point import InitFeasD
from stdForm import stdForm
import numpy as np

# Clean form of printed vectors
np.set_printoptions(precision=4, threshold=10, edgeitems=4, linewidth=120, suppress = True)

""" Info boxes """
    
def print_boxed(msg: str) -> None:

    lines = msg.splitlines()
    max_len = max(len(line) for line in lines)

    if max_len > 100:
        raise ValueError("Overfull box")

    print('-' * (max_len + 4))
    for line in lines:
        print('| ' + line + ' ' * (max_len - len(line)) + ' |')
    print('-' * (max_len + 4))    

#%%

''' AFFINE-SCALING METHOD '''

def affine(A, b, c, u, v):
        
    """ Error checking """
        
    if not (isinstance(A, np.ndarray) or isinstance(b, np.ndarray) or isinstance(c, np.ndarray)):
        info = 1
        raise Exception('Inputs must be a numpy arrays: INFO {}'.format(info))
        
    # Construction in a standard form [A | I]
    A, c = stdForm(A, c)    
    r_A, c_A = A.shape
    print('\n\tCOMPUTATION OF ALGORITHM')
    
    # The algorithm in three steps:    
    # 1..Construct the initial point with the two functions, 
    # 2..obtain the search direction, 
    # 3..find the largest step   
    
    """ Initial points """
    
    # (x, s) is positive
    if u == 0:
        x = InitFeas(A, b, 0.009, 0.008)
    if u == 1:
        x = np.ones(c_A)*0.1 
        print(x)
    if v == 0:
        (y, s) = InitFeasD(A, c, 0.5, 0.5)
    if v == 1:
        s = np.ones(c_A)
        y = np.ones(r_A)*(-1) 
    
    # Initial gap
    g = np.dot(c,x) - np.dot(y,b)
    print('Dual initial gap: {}.\n'.format("%10.3f"%g))
    
    
    
   #%%
        
    """ search vector direction """
    
    # Compute the search direction solving the matricial system
    # Instead of solving the std system matrix it is uses AUGMENT SYSTEM with CHOL approach
    it = 0
    while abs(g) > 0.005:
        print("\tIteration: {}\n".format(it), end='')
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
        
        # Largest step length such that (x, s) is positive
        m = min([(-x[i] / x1[i], i) for i in range(c_A) if x[i]/x1[i] < 0], default = [0])[0] 
        n = min([(-s[i] / s1[i], i) for i in range(c_A) if s[i]/s1[i] < 0])[0] 
        t = (0.999)*min(m, n)
        if t > 1:
           t = 1
           
        # INCREMENT of the vectors and the number of iterations
        x = x + t*x1
        y = y + t*y1
        s = s + t*s1
        
        it += 1
        print('\nCurrent point:\n x = {} \n lambda = {} \n s = {}.\n'.format(x.round(decimals = 3), y.round(decimals = 3), s.round(decimals = 3)))
        z = np.dot(c, x)
        g = z - np.dot(y,b)
        print('Dual next gap: {}.\n'.format("%10.3f"%g))
        
    print_boxed("Found optimal solution of the canonical problem at\n x* = {}.\n\n".format(x[:c_A-r_A].round(decimals = 3)) +
                "Dual gap: {}\n".format("%10.6f"%g) +
                "Optimal cost: {}\n".format("%10.3f"%z) +
                "Number of iteration: {}".format(it))
    return x

if __name__ == "__main__":
#     Input data of canonical LP:
    A = np.array([[1, 0],[0, 1],[1, 1],[4, 2]])
    c = np.array([-12, -9])
    b = np.array([1000, 1500, 1750, 4800])
    affine(A, b, c, 0, 0)
    
# optimal solution of the canonical problem at 
#  x* = [0. 2.].                                                                                        |
# Dual gap:   0.002675                               
# Optimal cost:     -2.000                           
#| Number of iteration: 4   

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:49:50 2019

@author: Elena
"""
from Initial_primal_feasible_point import InitFeas
from Initial_dual_feasible_point import InitFeasD
from stdForm import stdForm
import numpy as np
import random

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

''' LONG-PATH FOLLOWING METHOD '''

def longpath(A, b, c, gamma, s_min, s_max, u, v):
        
    """ Error checking """
        
    if not (isinstance(A, np.ndarray) or isinstance(b, np.ndarray) or isinstance(c, np.ndarray)):
        info = 1
        raise Exception('Inputs must be a numpy arrays: INFO {}'.format(info))
        
    # Construction in a standard form [A | I]
    A, c = stdForm(A, c)    
    r_A, c_A = A.shape
    E = lambda a: (s+a*s1)*(x+a*x1)-(gamma*np.dot((s+a*s1),(x+a*x1)))/c_A #Function E: set of values in N_(gamma)
    print('\n\tCOMPUTATION OF ALGORITHM with parameters:\n \t Gamma {}, [sigma_min, sigma_max] = [{}, {}]\n\n'.format(gamma,s_min,s_max))
    
    # The algorithm in three steps:
    
    # 1..Construct the initial point with the two functions, 
    # 2..obtain the search direction, 
    # 3..find the largest step   
    
    """ Initial points """
    
    if u == 0:
        x = InitFeas(A, b, 0.3, 0.8)
    if u == 1:
        x = np.ones(c_A)*0.1 
    if v == 0:
        (y,s) = InitFeasD(A, c, 0.5, 0.5)
    if v == 1:
        s = np.ones(c_A)
        y = np.ones(r_A)*(-1) 
    
    # Initial gap
    g = np.dot(c,x) - np.dot(y,b)
    print('Dual initial gap: {}.\n'.format("%10.3f"%g))
    
    # Check if (x, y, s) in neighborhood N_inf:
    
    if (x*s > gamma*np.dot(x,s)/c_A).all():
        print("Initial point is in N_inf(gamma), gamma = {}\n".format("%10.6f"%gamma))
    #%%
        
    """ search vector direction """
    
    # Compute the search direction solving the matricial system
    # Instead of solving the std system matrix it is uses AUGMENT SYSTEM with CHOL approach
    it = 0
    while abs(g) > 0.005:
        print("\tIteration: {}\n".format(it), end='')
        sigma = random.uniform(s_min, s_max) # Choose centering parameter SIGMA_k in [sigma_min , sigma_max]
        print("Centering parameter sigma:{}.\n".format("%10.3f"%sigma))

        S_inv = np.linalg.inv(np.diag(s))           
        W1 = S_inv*np.diag(x)                       # W1 = D = S^(-1)*X    
        W2 = np.dot(A, W1)                          # W      A*S^(-1)*X
        W  = np.dot(W2, A.T)
        L = np.linalg.cholesky(W)                   # CHOLESKY for A* D^2 *A^T
        L_inv = np.linalg.inv(L)
        
        # RHS of the system
        
        rb = b - np.dot(A, x)
        rc = c - np.dot(A.T, y) - s
        rxs = - x*s + sigma*(sum(x*s)/c_A)*np.ones(c_A)  # Newton step toward x*s = sigma*mi
        
        B = rb + np.dot(W2, rc) - np.dot(np.dot(A, S_inv), rxs) #RHS of normal equation form
        z = np.dot(L_inv, B)
        
        # SEARCH DIRECTION:
        
        y1 = np.dot(L_inv.T, z)
        s1 = rc - np.dot(A.T, y1)
        x1 = np.dot(S_inv, rxs) - np.dot(W1,s1)
        print('Search direction vectors: \n delta_x = {} \n delta_lambda = {} \n delta_s = {}.\n'.format(x1.round(decimals = 3),x1.round(decimals = 3),s1.round(decimals = 3)))
        
        #%%
        
        """ largest step length """
        
        v = np.arange(0, 1.001, 0.0001)
        i = len(v)-1
        while i >= 0:
            if (E(v[i]) > 0).all():
                t = v[i]
                print('Largest step length:{}'.format("%10.3f"%t))
                break
            else:
                i -= 1
        
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

#if __name__ == "__main__":
    # Input data of canonical LP:

 
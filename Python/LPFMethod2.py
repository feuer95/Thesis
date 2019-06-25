# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:33:26 2019

@author: elena
"""
from starting_point import sp # Function to find the initial infeasible point
from print_boxed import print_boxed # Print pretty info boxes
from stdForm import stdForm # Function to extend LP in a standard form
import numpy as np # To create vectors
from cent_meas import cent_meas
from input_data import input_data
from term import term # Compute the conditions of termination
import random

# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)

'''                                 ====
                LONG-PATH FOLLOWING METHOD with sigma RANDOM
                                    ====
                                    
Input data: np.arrays of matrix A, cost vector c, vector b of the LP
            neighborhood param gamma    (10^{-3} by default)
            c_form: canonical form      (0 by default)
            range of sigma random:      ([s_min = 0.1, s_max = 0.9] by default)
            max_it:                     (500 by default) 
            
The system is computed with augmented system

Output data: x, s, u
'''


def longpath2(A, b, c, gamma = 0.001, s_min = 0.1, s_max = 0.9, c_form = 0, w = 10**(-8), max_it = 500):
        
    print('\n\tCOMPUTATION OF LPF ALGORITHM with augmented system')
    
    """ Input error checking """
        
    if not (isinstance(A, np.ndarray) or isinstance(b, np.ndarray) or isinstance(c, np.ndarray)):
        info = 1
        raise Exception('Inputs must be a numpy arrays: INFO {}'.format(info))
        
    # Construction in a standard form [A | I]
    if c_form == 0:
        A, c = stdForm(A, c)
        
    r_A, c_A = A.shape
    E = lambda a: (s+a*s1)*(x+a*x1)-(gamma*np.dot((s+a*s1),(x+a*x1)))/c_A #Function E: set of values in N_(gamma)
    
    """ Check full rank matrix """
    
    if not np.linalg.matrix_rank(A) == r_A: # Remove ld rows:
        A = A[[i for i in range(r_A) if not np.array_equal(np.linalg.qr(A)[1][i, :], np.zeros(c_A))], :]
        r_A = A.shape[0]  # Update no. of rows

    """ Initial points """
    
    # Initial infeasible positive (x,y,s) and Initial gap g
    (x, y, s) = sp(A, c, b)    
    g = np.dot(c,x) - np.dot(y,b)
    
    print('\nInitial primal-dual point:\nx = {} \nlambda = {} \ns = {}'.format(x, y, s))    
    print('Dual initial gap: {}.\n'.format("%10.3f"%g))      
    
    # Check if (x, y, s) in neighborhood N_inf:
    
    if (x*s > gamma*np.dot(x,s)/c_A).all():
        print("Initial point is in N_inf(gamma), gamma = {}\n".format("%10.6f"%gamma))
        
    #%%
        
    """ search vector direction """
    
    # Compute the search direction solving the matricial system
    # Instead of solving the std system matrix it is uses AUGMENT SYSTEM with CHOL approach
    
    it = 0
    tm = term(it)
    u = []
    u.append([it, g, x, s, b - np.dot(A,x), c - np.dot(A.T, y) - s])
    while tm > w:
        
        print("\tIteration: {}".format(it+1))
        cp = random.uniform(s_min, s_max) # Choose centering parameter SIGMA_k in [sigma_min , sigma_max]
        print("Centering parameter cp:{}.\n".format("%10.3f"% cp))

        # solve search direction with AUGMENTED SYSTEM
          
        
        X_inv = np.linalg.inv(np.diag(x))           
        W1 = X_inv*np.diag(s)          # W1 = X^(-1)*S        
        T = np.concatenate((np.zeros((r_A,r_A)), A), axis = 1)
        U = np.concatenate((A.T, -W1), axis = 1)
        V = np.concatenate((T, U), axis = 0)
        
        rb = b - np.dot(A, x)
        rc = c - np.dot(A.T, y) - s
        rxs = - x*s + cp*(sum(x*s)/c_A)*np.ones(c_A)  # Newton step toward x*s = sigma*mi

        r = np.hstack((rb, rc - np.dot(X_inv, rxs)))        
        o = np.linalg.solve(V,r)
       
        y1 = o[:r_A]
        x1 = o[r_A:c_A + r_A]      
        s1 = np.dot(X_inv, rxs) - np.dot(W1, x1)
        print('Search direction vectors: \n delta_x = {} \n delta_lambda = {} \n delta_s = {}.\n'.format(x1.round(decimals = 3), y1.round(decimals = 3),s1.round(decimals = 3)))

        #%%
        
        """ largest step length """
        
        # We find the maximum alpha s. t the next iteration is in N_gamma
        v = np.arange(0, 1.000, 0.0001)
        i = len(v)-1
        while i >= 0:
            if (E(v[i]) > 0).all():
                t = v[i]
                print('Largest step length:{}'.format("%10.3f"%t))
                break
            else:
                i -= 1
        
        # Increment the points and iteration
        x += t*x1
        y += t*y1
        s += t*s1
        print('\nCurrent point:\n x = {} \n lambda = {} \n s = {}.\n'.format(x.round(decimals = 3), y.round(decimals = 3), s.round(decimals = 3)))
        z = np.dot(c, x)
        g = z - np.dot(y, b)
        u.append([it, g, x.copy(), s.copy(), rb.copy(), rc.copy()])
        
        it += 1
        if it == max_it:
            print("Iterations maxed out")
            return x, s, u
               
        # Termination elements
        tm = term(it, b, c, rb, rc, z, g)
        print('Dual next gap: {}.\n'.format("%10.3f"%g))
        
    print_boxed("Found optimal solution of the problem at\n x* = {}.\n\n".format(x.round(decimals = 3)) +
                "Dual gap: {}\n".format("%10.6f"%g) +
                "Optimal cost: {}\n".format("%10.3f"%z) +
                "Number of iteration: {}".format(it))
    return x, s, u


#%%
    
if __name__ == "__main__": 
    
#     Input data of canonical LP:
    
    (A, b, c) = input_data(18)
        
    x, s, u = longpath2(A, b, c)
    
    ul = cent_meas(x, u, 'LPF2')

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:49:50 2019

@author: Elena
"""
from starting_point import sp # Function to find the initial infeasible point
from print_boxed import print_boxed # Print pretty info boxes
from stdForm import stdForm # Function to extend LP in a standard form
import numpy as np # To create vectors
import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # Print plot
import random

# Clean form of printed vectors
np.set_printoptions(precision=4, threshold=10, edgeitems=4, linewidth=120, suppress = True)

''' LONG-PATH FOLLOWING METHOD '''

"""

Input data: np.arrays of matrix A, cost vector c, vector b of the LP
            neighborhood parameter gamma -> 10^{-3} by default
            c_form: canonical form -> 0 by default
"""

def longpath(A, b, c, gamma = 0.001, s_min = 0.1, s_max = 0.9, c_form = 0, w = 0.005, max_iter = 300):
        
    print('\n\tCOMPUTATION OF LPF ALGORITHM')
    
    # Algorithm in 4 steps:  
    # 0..Input error checking
    # 1..Find the initial point with Mehrotra's method 
    # 2..obtain the search direction,
    # 3..find the largest step          
        
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
    (x,y,s) = sp(A, c, b)    
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
    u = []
    u.append([it, g, x, s])
    while abs(g) > w:
        print("\tIteration: {}\n".format(it+1), end='')
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
        
        # We find the maximum alpha s. t the next iteration is in N_gamma
        v = np.arange(0, 1.001, 0.0001)
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
        it += 1
        if it == max_iter:
            print("Iterations maxed out")
            return x, s, u
        print('\nCurrent point:\n x = {} \n lambda = {} \n s = {}.\n'.format(x.round(decimals = 3), y.round(decimals = 3), s.round(decimals = 3)))
        z = np.dot(c, x)
        g = z - np.dot(y, b)
        u.append([it, g, x.copy(), s.copy()])
        print('Dual next gap: {}.\n'.format("%10.3f"%g))
        
    print_boxed("Found optimal solution of the problem at\n x* = {}.\n\n".format(x.round(decimals = 3)) +
                "Dual gap: {}\n".format("%10.6f"%g) +
                "Optimal cost: {}\n".format("%10.3f"%z) +
                "Number of iteration: {}".format(it))
    return x, s, u


#%%ù
    
"""Input data"""

# Input data of canonical LP:
if __name__ == "__main__":
    
     # Example in cycle
     
#    c = np.array([-0.75, 150, -0.02, 6])
#    b = np.array([0, 0, 1])
#    A = np.array([[0.25, -60, -0.04, 9],[0.5, -90, -0.02, 3],[0, 0, 1, 0]])

    # Unlimited problem
    
#    A = np.array([[1, -1],[-1, 1]])
#    c = np.array([-1, -1])
#    b = np.array([1, 1])
   
     # Example with b negative
    
#    A = np.array([[-1, 1, -1, 1, 1], [-1, -4, 1, 3, 1]])
#    b = np.array([-10, -5])
#    c = np.array([9, 16, 7, -3, -1])
     
    # Input data that works
    
    A = np.array([[1, 1, 2],[2, 0, 1],[2, 1, 3]])
    c = np.array([-3, -2, -4])
    b = np.array([4, 1, 7])

    # creo lista mu
    x, s, u = longpath(A, b, c)
    mu = []
    for i in range(len(u)):
        mu.append(np.dot(u[i][2],u[i][3])/6)
        
    # Create a dataframe and convert to excel           
    dfu = pd.DataFrame(u, columns = ['it', 'Current g', 'Current x', 'Current s'])   
    dfu['mu'] = mu
    dfu.to_excel("LPF.xlsx", index = False) 
    
    # Plot the graphic with dataframe elements
    plt.figure()
    plt.plot(dfu['it'], dfu['Current g'], label = 'Cost value', marker = '.')
    plt.grid(b = True, which = 'major')
    plt.title('Dual gap LPF')

















    
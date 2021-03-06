# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:32:34 2019

@author: elena

"""
from stdForm import stdForm # Standard form transform
from print_boxed import print_boxed # Print pretty info box
from starting_point import sp # Find initial infeasible points
from input_data import input_data
from term import term # Compute the conditions of termination
import numpy as np
from cent_meas import cent_meas # Plot dual gap and centering measure

# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 20, edgeitems = 4, linewidth = 120, suppress = True)


'''                           ===
                     AFFINE-SCALING METHOD
                              ===

Input data: np.arrays of matrix A, cost vector c, vector b of the LP
            c_form: canonical form -> default 0
            w = tollerance -> default 10^(-8)
            
Output data: x: primal solution
             s: dual solution
             u: list: iteration, dual gas, Current x, Current s, Feasibility x, Feasibility s
'''

def affine2(A, b, c, c_form = 0, w = 10**(-8), max_iter = 500):
        
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
    (x, y, s) = sp(A, c, b)    
    g = np.dot(c, x) - np.dot(y, b)
    
    print('\nInitial primal-dual point:\n x = {} \n lambda = {} \n s = {}.\n'.format(x, y, s))    
    print('Dual initial gap: {}.\n'.format("%10.3f"%g))      
    
   #%%
        
    """ search vector direction """
       
    it = 0
    tm = term(it)
    u = []
    u.append([it, g, x, s, b - np.dot(A,x), c - np.dot(A.T, y) - s])
    while tm > w:
        
        print("\tIteration: {}\n".format(it))
        # solve search direction with AUGMENTED SYSTEM
        X_inv = np.linalg.inv(np.diag(x))           
        W1 = X_inv*np.diag(s)                       # W1 = D = X^(-1)*S   
        T = np.concatenate((np.zeros((r_A,r_A)), A), axis = 1)
        U = np.concatenate((A.T, -W1), axis = 1)
        V = np.concatenate((T, U), axis = 0)
        
        rb = b -np.dot(A, x)
        rc = c -np.dot(A.T, y) -s
        rxs = -x*s  # Newton step toward x*s 
        
        r = np.hstack((rb, -np.dot(X_inv,rxs)))        
        o = np.linalg.solve(V,r)
       
        y1 = o[:r_A]
        x1 = o[r_A:c_A+r_A]
        s1 = np.dot(X_inv, rxs) - np.dot(W1, x1)
        print('Search direction vectors: \n delta_x = {} \n delta_lambda = {} \n delta_s = {}.\n'.format(x1.round(decimals = 3), y1.round(decimals = 3),s1.round(decimals = 3)))
        
        #%%
        
        """ largest step length """
        
        # Largest step length T such that (x, s) + T (x1, s1) is positive
        m = min([(-x[i] / x1[i], i) for i in range(c_A) if x1[i] < 0], default = [1])[0] 
        n = min([(-s[i] / s1[i], i) for i in range(c_A) if s1[i] < 0], default = [1])[0] 
        T = (0.9)*min(m, n) 

           
        # INCREMENT of the vectors and iterations
        x += min(T,1)*x1
        y += min(T,1)*y1
        s += min(T,1)*s1
        
        z = np.dot(c, x) # Current optimal solution
        g = z - np.dot(y, b) 
        u.append([it, g, x.copy(), s.copy(), rb.copy(), rc.copy()])       
                
        # Termination elements
        tm = term(it, b, c, rb, rc, z, g)

        it += 1
        if it == max_iter:
            raise TimeoutError("Iterations maxed out") 

        print('Current point:\n x = {} \n lambda = {} \n s = {}.\n'.format(x, y, s))
        print('Dual next gap: {}.\n'.format("%10.3f"%g))
        
    print_boxed("Found optimal solution of the problem at\n x* = {}.\n".format(x) +
                "Dual gap: {}\n".format("%10.6f"%g) +
                "Optimal cost: {}\n".format("%10.3f"%z) +
                "Number of iterations: {}".format(it))
    return x, s, u

#%%
    
if __name__ == "__main__": 
    
    # Input data of canonical LP:
#    (A, b, c) = input_data(10)
        
    x, s, u = affine2(A, b, c)
    
#    ua = cent_meas(x, u, 'Affine 2')
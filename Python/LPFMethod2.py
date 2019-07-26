# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:33:26 2019

@author: elena
"""

from starting_point import sp       # Function to find the initial infeasible point
from starting_point2 import sp2     # Function to find the initial infeasible point
from print_boxed import print_boxed # Print pretty info boxes
from stdForm import stdForm         # Function to extend LP in a standard form
import numpy as np                  # To create vectors
from input_data import input_data   # Data problems
from term import term               # Compute the conditions of termination
from cent_meas import cent_meas

# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)

'''                     ============================
                        LONG-PATH FOLLOWING METHOD B
                        ============================
                                    
Input data: np.arrays of matrix A, cost vector c, vector b of the LP
            neighborhood param gamma    (10^{-3} by default)
            c_form: canonical form      (0 by default)
            sigma:                      (min 0.1, 100*mu)
            max_it:                     (500 by default) 
            
The system is computed with augmented system
            
Output data: vector x primal solution
             vector s solution
             u: list [it, gap, x, s]

'''


def longpath2(A, b, c, gamma = 0.001, c_form = 0, w = 10**(-8), max_it = 500, info = 0, ip = 1):
    
    print('\n\tCOMPUTATION OF LPF ALGORITHM 2')    
    
    # Algorithm in 4 steps:  
    # 0..Input error checking
    # 1..Find the initial point with Mehrotra's method 
    # 2..obtain the search direction
    # 3..find the largest step   
    
    """ Input error checking & construction in a standard form """
        
    if not (isinstance(A, np.ndarray) or isinstance(b, np.ndarray) or isinstance(c, np.ndarray)):
       raise Exception('Inputs must be a numpy arrays')
        
    if c_form == 0:
        A, c = stdForm(A, c)    
    r_A, c_A = A.shape
    if not np.linalg.matrix_rank(A) == r_A: # Check full rank matrix:Remove ld rows:
        A = A[[i for i in range(r_A) if not np.array_equal(np.linalg.qr(A)[1][i, :], np.zeros(c_A))], :]
        r_A = A.shape[0]  # Update no. of rows
    
    """ Initial points: Initial infeasible positive (x,y,s) and initial gap g """
    
    if ip == 0:
        (x, y, s) = sp(A, c, b)
    else:
        (x, y, s) = sp2(A, c, b)        

    (x, y, s) = sp(A, c, b)    
    g = np.dot(c, x) - np.dot(y, b)
    
    if info == 0:
        print('\nInitial primal-dual point:\nx = {} \nlambda = {} \ns = {}'.format(x, y, s))    
        print('Dual initial g: {}.\n'.format("%10.3f"%g))      
    
    # Check if (x, y, s) in neighborhood N_inf:
    mu = np.dot(x,s)/c_A
    if not (x*s > gamma*mu).all():
        print("Initial point is not in N")
    E = lambda a: (s+a*s1)*(x+a*x1)-(gamma*np.dot((s+a*s1),(x+a*x1)))/c_A  # Function E: set of values in N_(gamma)
    
    #%%
        
    """ Search vector direction """
    
    '''Modified Newton's method with with normal equations: find the direction vector (y1, s1, x1)'''

    it = 0        # Num of iterations
    tm = term(it) # Define tollerance tm = inf
    u = []        # Construct list of info elements 
    u.append([it, g, x, s, b - np.dot(A,x), c - np.dot(A.T, y) - s])
    sig = []
    
    while tm > w:
        cp = min(0.1, 100*np.dot(x,s)/c_A)
        
        if info == 0:
            print("\tIteration: {}\n".format(it), end='')
            print("Centering parameter sigma:{}.\n".format("%10.3f"%cp))      

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
        
        if info == 0:
            print('Search direction vectors: \n delta_x = {} \n delta_lambda = {} \n delta_s = {}.\n'.format(x1.round(decimals = 3), y1.round(decimals = 3),s1.round(decimals = 3)))
        
        """ Compute the largest step length & increment of the points and the iteration"""
        
        # We find the maximum alpha s. t the next iteration is in N_gamma
        v = np.arange(0, 1.000, 0.0001)
        i = len(v)-1
        while i >= 0:
            if (E(v[i]) > 0).all():
                t = v[i]
#                print('Largest step length:{}'.format("%10.3f"%t))
                break
            else:
                i -= 1
        
        # Update step
        x += t*x1            # Current x
        y += t*y1            # Current y
        s += t*s1            # Current s
        rb = b - np.dot(A, x)
        rc = c - np.dot(A.T, y) - s

        z = np.dot(c, x)     # Current opt. solution
        g = z - np.dot(y, b) # Current gap
        it += 1
        u.append([it, g, x.copy(), s.copy(), rb.copy(), rc.copy()])
        sig.append([cp])

        # Termination elements
        m, n, q = term(it, b, c, rb, rc, z, g)
        tm = max(m, n, q)
        if it == max_it:
            raise TimeoutError("Iterations maxed out") 
        if info == 0:    
            print('\nCurrent point:\n x = {} \n lambda = {} \n s = {}.\n'.format(x.round(decimals = 3), y.round(decimals = 3), s.round(decimals = 3)))       
            print('Dual next gap: {}.\n'.format("%10.3f"%g))

#%%
        
    print_boxed("Found optimal solution of the problem at\n x* = {}.\n\n".format(x.round(decimals = 3)) +
                "Dual gap: {}\n".format("%2.2e"%g) +
                "Optimal cost: {}\n".format("%10.8E"%z) +
                "Number of iteration: {}".format(it))
    return x, s, u, sig


#%%
'''                     ===============
                        LPF METHOD test
                        ===============
'''    
if __name__ == "__main__": 
    
    # Input data of canonical LP:
    
    (A, b, c) = input_data(18)
        
    x, s, u, sig = longpath2(A, b, c, info = 1)
    
    dlpf = cent_meas(x, u, 'LPF', plot = 0)

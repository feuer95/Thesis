# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:35:51 2019

@author: Elena
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:52:35 2019

@author: Elena
"""
from stdForm import stdForm # Function of standard form transform
import numpy as np
from print_boxed import print_boxed

# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)

''' PREDICTOR-CORRECTOR MEHROTRA ALGORITHM '''

def mehrotra(A, b, c):
    if not (isinstance(A, np.ndarray) or isinstance(b, np.ndarray) or isinstance(c, np.ndarray)):
        raise Exception('Inputs must be a numpy arrays')
    
    # Input data in a standard form [A | I]:
    A, c = stdForm(A, c)    
    r_A, c_A = A.shape
    
    # Check if input data are correct
    if not (isinstance(A, np.ndarray) or isinstance(b, np.ndarray) or isinstance(c, np.ndarray)):
        raise Exception('Inputs must be a numpy arrays')
    
    print('\n\tCOMPUTATION OF MEHROTRA ALGORITHM\n')
    
    #%%
    
    """ Infeasible initial vectors"""
    
    # We search the feasible points  x and s with minimum norm: min ||x||_2, min ||s||_2 
    # then we construct positive initial infeasible points
    
    V = np.linalg.inv(np.dot(A, A.T)) 
    x = np.dot(np.linalg.pinv(A), b) # initial feasible x
    la = np.dot(A, c)                                
    y = np.dot(V, la)                # initial feasible lambda
    s = c - np.dot(A.T, y)           # initial feasible s
            
    # First update 
    dx = np.min(x)
    if dx < 0:
       x += (-3/2)*dx*np.ones(c_A) 
    ds = np.min(s)
    if ds < 0:
       s += (-3/2)*ds*np.ones(c_A)
    
    # Second update 
    Dx = np.dot(x,s)/sum(s)
    Ds = np.dot(x,s)/sum(x)
    
    x += 2*Dx*np.ones(c_A) 
    s += 2*Ds*np.ones(c_A)
    
    print('Initial infeasible vectors:\n x_0 = {} \n lambda_0 = {}\n s_0 = {}'.format(x.round(decimals = 3),y.round(decimals = 3),s.round(decimals = 3)))
    g = np.dot(c,x) - np.dot(y,b)
    print('Dual initial gap: {}.'.format("%.3f"%g))
    
    #%%
    
    """ Predictor step: compute affine direction """
    
    # Compute affine scaling direction solving Qs = R with an augmented system D^2 = S^{-1}*X 
    # with Q a large sparse matrix 
    # and R = [rb, rc, - x_0*s_0]
    
    it = 0
    while abs(sum(x*s)) > 0.5:
        print("\n\tIteration: {}\n".format(it), end='')   
        # CHOLESKY for normal equation with matrix A* D^2 *A^{T}
        S_inv = np.linalg.inv(np.diag(s))  # S^{-1}
        W1 = np.dot(S_inv, np.diag(x))     # W1 = D = S^(-1)*X    
        W2 = np.dot(A, W1)                 # W2      A*S^(-1)*X
        W  = np.dot(W2, A.T)
        L = np.linalg.cholesky(W) 
        L_inv = np.linalg.inv(L)
        
        # RHS of the system, including the minus
        
        rb = b - np.dot(A, x) 
        rc = c - np.dot(A.T, y) - s
        rxs = - x*s
        
        B = rb + np.dot(W2, rc) - np.dot(np.dot(A, S_inv), rxs) #RHS of normal equation form
        z = np.dot(L_inv, B)
        
        # SEARCH DIRECTION:
        y1 = np.dot(L_inv.T, z)
        s1 = rc - np.dot(A.T, y1)
        x1 = np.dot(S_inv, rxs) - np.dot(W1, s1)
        print("\nPREDICTOR STEP:\nAffine direction:\n({}, {}, {})\n".format(x1, y1, s1))
        
        #%%
        
        """ Centering step: compute search direction """
        
        # Find alfa1_affine and alfa2_affine : maximum steplength along affine-scaling direction
        # Find the minimum( x_{i}/x1_{i} , 1) and the minimum( s_{i}/s1_{i} , 1)
        
        h = min([(-x[i]/ x1[i], i) for i in range(c_A) if x1[i] < 0], default = [0])[0]
        k = min([(-s[i]/ s1[i], i) for i in range(c_A) if s1[i] < 0], default = [0])[0]
        
        alfa1 = min(h,1)
        alfa2 = min(k,1)
        
        # Set the centering parameter to Sigma = (mi_aff/mi)^{3}   
        mi = np.dot(x,s)/c_A # Duality measure
        mi_af = np.dot(x + alfa1*x1,s + alfa2*s1)/c_A # Average value of the incremented vectors
        Sigma = (mi_af/mi)**3
    #    print("alfa_affine = ({}, {})\nmi_affine = {} \nSet centering parameter Sigma = {}\n".format("%.3f"%alfa1, "%.3f"%alfa2, "%.3f"%mi_af, "%.3f"%Sigma))
        
        # SECOND SEARCH DIRECTION
        Rxs = - x1*s1 + Sigma*mi*np.ones((c_A)) # RHS of the New system, including minus
        
        Rb = - np.dot(np.dot(A,S_inv), Rxs) # RHS of New normal equation form
        z = np.dot(L_inv, Rb)
        
        # SEARCH DIRECTION:
        y2 = np.dot(L_inv.T, z)
        s2 = - np.dot(A.T, y2)
        x2 = np.dot(S_inv, Rxs) - np.dot(W1, s2)
    #    print("Search direction: ({}, {}, {})\n".format(x2, y2, s2))
        
        #%%
        """ Corrector step: compute (x_k+1, lambda_k+1, s_k+1) """
        
        # The steplength without eta_k
        x2 += x1
        s2 += s1
        y2 += y1
        H = min([(-x[i]/ x2[i], i) for i in range(c_A) if x2[i] < 0], default = [0])[0]
        K = min([(-s[i]/ s2[i], i) for i in range(c_A) if s2[i] < 0], default = [0])[0]
        Alfa1 = min(0.99*H, 1)
        Alfa2 = min(0.99*K, 1)
        
        # Compute the final update , using sol2 and Alfa_i 
        
        x += Alfa1*x2
        y += Alfa2*y2
        s += Alfa2*s2
        it += 1
        # Feasibility og primal-dual vector
        B = np.dot(A,x)
#        C = np.dot(A.T, y) + s
        # Dual gap c^{T}*x - b^{T}*y = x*s
        z = np.dot(c,x)
        g = z - np.dot(y,b)
        if max(B-b) < 0.05:
            feas = 'yes'
        else:
            feas = 'no'
        print('CORRECTOR STEP:\nCurrent primal-dual point: \n x_k = ',x,'\n s_k = ',s,'\nlambda_k = ',y)
        print('Current g: {}\n'.format("%.3f"%g))
    #    print('\nCost function: {}\n'.format("%.3f"%np.dot(c,x)))
        print('Feasibility:',feas)
        
        
    print_boxed("Found optimal solution of the standard problem at\n x* = {}.\n\n".format(x) +
                "Dual gap: {}\n".format("%10.6f"%g) +
                "Optimal cost: {}\n".format("%10.3f"%z) +
                "Number of iteration: {}".format(it))
    if it == 300:
           raise TimeoutError("Iterations maxed out")
    return x       


""" input data """

# Input data of canonical LP:
if __name__ == "__main__":
    
    A = np.array([[0.25, -60, -0.04, 9, 1, 0, 0],[0.5, -90, -0.02, 3, 0, 1, 0],[0, 0, 1, 0, 0, 0, 1]])   
    b = np.array([0, 0, 1])
    c = np.array([-0.75, 150, -0.02, 6, 0, 0, 0])
    mehrotra(A, b, c)
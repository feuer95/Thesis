# -*- coding: utf-8 -*-
"""
Created on Sat May  4 21:59:02 2019

@author: Elena
"""
import numpy as np

# Clean form of printed vectors
np.set_printoptions(precision=4, threshold=10, edgeitems=4, linewidth=120, suppress = True)

'''  YE AND LUSTIG VARIANT INTERIOR POINT METHOD FOR DUAL LP  '''

#                  NEW LINEAR PROGRAMMING:
# solve {min(1*csi) | A^{T} y_{+} - A^{T} y_{-} + s + csi(c - D ys_{o}) = c,  
# with csi, y_(+), y_(-), s > 0 and D =[A^{T} | - A^{T} | I ] 

# Let initial feasible point ys_{o} =(1)
# Implementation of variant interior point method to find optimal solution of this new LP
# (y*, s*, csi*) with csi* = 0 and x* initial feasible point 
# for the standard primal problem


def InitFeasD(A: np.matrix, c: np.array, epsilon, sigma):
    
    print('\tCOMPUTATION OF ALGORITHM with parameters:\n \t Epsilon {}, Sigma {}\n\n'.format(epsilon,sigma))

    # Standard form formulation of A^(T) y <= c
    N, M = np.shape(A)
    D = np.concatenate((A.T, -A.T, np.identity(M)), axis = 1)
    n, m = np.shape(D) 
    
    # INPUT DATA OF NEW LP
    Q = c - np.sum(D, axis = 1)
    #                         ^ ys_{o} = 1
    D_P = np.concatenate((D, np.asmatrix(Q).T), axis = 1)    
    c_P = np.concatenate((np.zeros(m),[1]), axis = 0) # New cost vector 
    

    r = (n*m)**(-0.5) # Method's parameter
    x = np.ones(m + 1) #initial point
    
    print('Starting initial vector x = [1...1].\n') 
    err = c - np.dot(D,x[0:m]) # Error vector of the initial point
    
    # If the || err ||_2 < toll  we Exit
    
    if np.linalg.norm(err,2) < epsilon: 
        
        print('Unit vector is optimal solution.\n')
        return(x[0:m])
        
    # Iteration algorithm such that x[m] --> 0 and x[0:m] --> a IFS
    
    while np.linalg.norm(err) > epsilon:
        
        # Construction of the vector Projection of q on the matrix projection P
        
        X = np.diag(x)
        D_P = np.dot(D_P, X)
        C = np.concatenate((D_P, -np.asmatrix(c).T), axis = 1)
        C_ = np.linalg.pinv(C)
        P = np.identity(m + 2) - np.dot(C_,C)    # Projection matrix
        
        q = np.concatenate((np.dot(X,c_P), [np.dot(-c_P,x)]))
        
        y = np.dot(P,q)   # Projection of q
        
        w = sigma*(r/np.linalg.norm(y,2))  # Direction step-length
        z = np.ones(m+2)/(m+2) - w*y
        
        x = np.dot(z[0,0:m+1], X)*((z[0,m+1])**-1) # Solution of the new LP
        x = np.array(x)[0]  # Clean the form of (x,epsilon)
        err = c - np.dot(D,x[0:m])
        
        print('Improved vector (y, s, csi) = {}\nError = {}'.format(x,err))
        print(' A^(T) y_+ - A^(T) Y_- + s = {} \n'.format(np.dot(D,x[0:m])))
        
        y = x[0:N] - x[N:N+N] # Reconstruction of y = y_(+) - y_(-)
        s = x[N+N: m]         
    return y , s


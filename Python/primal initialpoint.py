# -*- coding: utf-8 -*-
"""

Created on Sat May  4 12:32:39 2019

@author: Elena
"""

import numpy as np


#Clean the form of printing vectors
np.set_printoptions(precision=4, threshold=10, edgeitems=4, linewidth=120, suppress = True)


'''YE AND LUSTIG VARIANT INTERIOR POINT METHOD'''

#                     NEW LINEAR PROGRAMMING:
# find {min(1*epsilon) | Ax + epsilon(b - Ax_{o}) = b, epsilon >= 0}
# with initial feasible point (1)
# solution (x*, epsilon*) with epsilon* = 0 and x* initial feasible point 
#for the standard primal problem


def InitFeas(A: np.matrix, b: np.array, epsilon, sigma):
    
    print('\n \tCOMPUTATION OF ALGORITHM with parameters:\n \t Epsilon {}, Sigma {}\n\n'.format(epsilon,sigma))
    
    r_A, c_A = np.shape(A)
    
    # INPUT DATA OF NEW LP
    
    B_P = np.matrix(b)
    
    Q = (B_P - np.transpose(np.sum(A, axis = 1)))
    #                         ^ x_{o} = 1
    A_P = np.concatenate((A, Q.T), axis = 1)
    c_P = np.concatenate((np.zeros(c_A),[1]), axis = 0)    # New cost vector 
    
    r = (r_A*c_A)**(-0.5) # Method's parameter
    
    x = np.ones(c_A + 1) # initial feasible point:
    
    err = b - np.dot(A, x[:c_A]) #  Error vector of the initial point
    
    print('Starting initial vector x = [1...1]') 
    
    # If the || err ||_2 < toll  we Exit
    
    if np.linalg.norm(err,2) < epsilon: 
        
        print('The unit vector is optimal solution.\n')
        return(x[:c_A])
        
    # Iteration algorithm such that x[m] --> 0 and x[0:m] --> a IFS
    
    while np.abs(x[c_A]) > epsilon:
        
        # Construction of the vector Projection of q on the matrix projection P
        X = np.diag(x)
        A_P = np.dot(A_P, X)
        C = np.concatenate((A_P, -np.transpose(B_P)), axis = 1)
        
        C_ = np.linalg.pinv(C)
        P = np.identity(c_A + 2) - np.dot(C_,C)    # Projection matrix
        q = np.concatenate((np.dot(X,c_P), [np.dot(-c_P,x)]))
        y = np.dot(P,q)   # Projection of q
        
        w = sigma*(r/np.linalg.norm(y,2))  # Direction step-length
        z = np.ones(c_A + 2)/(c_A + 2) - w*y
        
        x = np.dot(z[0,:c_A + 1], X)*((z[0,c_A + 1])**-1) # Solution of the new LP
        x = np.array(x)[0]  # Clean the form of (x,epsilon)
        err = b - np.dot(A,x[:c_A])
        
        print('Improved value (x, lamda) = {}\nError (b - Ax) = {} with'.format(x, err))
        
        sol = np.dot(A,x[:c_A])
        
        print('Ax = {}\n'.format(sol))
            
    return(x[:c_A])


#if __name__ == "__main__":
    
#    A = np.array([[1, 1],[-1, -1]])
#    b = np.array([8, -11])
#    c = np.array([2, -3])
#    
#    (A, c) = stdForm(A, c)
#    
#    InitFeas(A, b, 0.3, 0.5)
#       Output x = [4.359, 4.359, 0.04, 0.372, 0.1228]
#            Ax = [8.76, -8...]
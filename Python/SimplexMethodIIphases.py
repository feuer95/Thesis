# -*- coding: utf-8 -*-
"""

Created on Tue Apr  9 11:31:52 2019

@author: Elena
"""
from print_boxed import print_boxed
from stdForm import stdForm
import numpy as np 

# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)

'''' SIMPLEX METHOX II PHASES '''
    
def SimplexMethod(A, b, c, max_iter, rule):
    
    """ Error checking """
        
    if not (isinstance(A, np.ndarray) or isinstance(b, np.ndarray) or isinstance(c, np.ndarray)):
        raise Exception('Inputs must be a numpy arrays')
                
    # Construction in a standard form [A | I]
    (A, c) = stdForm(A, c)   
    A = np.asmatrix(A)    
    r_A, c_A = np.shape(A)

    """ Check full rank matrix """
    
    # Remove ld rows:
    if not np.linalg.matrix_rank(A) == r_A:        
        A = A[[i for i in range(r_A) if not np.array_equal(np.linalg.qr(A)[1][i, :], np.zeros(c_A))], :]
        r_A = A.shape[0]  # Update no. of rows                                                       
    
    print('\n\tCOMPUTATION OF SIMPLEX ALGORITHM:\n\n')
    
    """ IIphases simplex method """

    # If b negative then I phase simplex fun with input data A_I, x_I, c_I
    # The solution x_I is the initial point for II phase.
     
    if sum(b < 0) > 0:
        print('Vector b < 0:\n\tStart phase I\n')
        A[[i for i in range(r_A) if b[i] < 0]] *= -1  # Change sign of constraints
        b = np.abs(b)
        # Phase I variables:
        A_I = np.matrix(np.concatenate((A, np.identity(r_A)), axis = 1))
        c_I = np.concatenate((np.zeros(c_A), np.ones(r_A)))  
        x_I = np.concatenate((np.zeros(c_A),b))  
        B = set(range(c_A, c_A + r_A)) 
        
        # Compute the simplex core 
        infoI, xI, BI, zI, itI = fun(A_I, c_I, x_I, B, 1, max_iter, rule)     
        assert infoI == 0
        
        if zI > 0:
           print('The problem is not feasible.\n{} iterations in phase I.'.format(itI))
           print_boxed('Optimal cost in phase I: {}\n'.format(zI))
           
        else: # Get initial BFS for original problem (without artificial vars.)
            xI = xI[:c_A]
            print("Found initial BFS at x: {}.\n\tStart phase II\n".format(xI))        
            info, x, B, z, itII = fun(A, c, xI, BI, itI + 1, max_iter, rule)
            
    # If b is positive phase II with B = [n+1,..., n+m]
    
    else: 
        x = np.concatenate((np.zeros(c_A-r_A), b))
        B = set(range(c_A-r_A,c_A))
        it = 1
        info, x, B, z, itII = fun(A, c, x, B, it, max_iter, rule)
        
    # Print termination of phase II 
    if info == 0:
        print_boxed("Found optimal solution at x* =\n{}\n\n".format(x) +
                    "Basis indexes: {}\n".format(B) +
                    "Nonbasis indexes: {}\n".format(set(range(c_A)) - B) +
                    "Optimal cost: {}\n".format(z.round(decimals = 3))+
                    "Number of iteration: {}.".format(itII))
    elif info == 1:
        print("Unlimited problem.")
    

"""Algorithm"""
    
def fun(A, c, x, B, it, max_iter, rule) -> (float, np.array, set, float, np.array):
    
    r_A, c_A = np.shape(A)
    B, NB = list(B), set(range(c_A)) - B  # Basic /nonbasic index lists
    
    B_inv = np.linalg.inv(A[:, B])
    z = np.dot(c, x)  # Value of obj. function
    
    while it <= max_iter:  # Ensure procedure terminates (for the min reduced cost rule)
        print("\tIteration: {}\n".format(it), end='')
        lamda = c[B] * B_inv
        if rule == 0:  # Bland rule
            optimum = True
            for s in NB:  # Read in lexicographical index order
                m = np.asscalar(c[s] - lamda * A[:, s])
                if m < 0:
                    optimum = False
                    break
        elif rule == 1:
            m , s = min([(np.asscalar(c[q] - lamda * A[:, q]), q) for q in NB],key=(lambda tup: tup[0]))
#           ^ c_s and position s
            optimum = (m >= 0) 
#                true if the minimum of the cost vector is positive
        if optimum:
            info = 0
            return info, x, set(B), z, it
            
        """Feasible basic direction"""
        d = np.zeros(c_A) 
#       ^ vector that increments x_B and x_NB: here c_A is m + n because A is in standard form
        
        for i in range(r_A):
            d[B[i]] = np.asscalar(-B_inv[i, :] * A[:, s]) #solve B*d = A_s -> -B^-1*A_s
        d[s] = 1
        neg = [(-x[B[i]] / d[B[i]], i) for i in range(r_A) if d[B[i]] < 0]
        
        if len(neg) == 0:
            print("Unlimited problem")
            info = 3            
            return info, x, B, None, it  #unlimited return
        
        theta, r = min(neg, key=(lambda t: t[0]))
        x = x + theta * d
        z = z + theta * m       
        
        # Update inverse:
        for i in set(range(r_A)) - {r}:
            B_inv[i, :] -= d[B[i]]/d[B[r]] * B_inv[r, :]
        B_inv[r, :] /= -d[B[r]]
        NB = NB - {s} | {B[r]}  # Update nonbasic index set
        B[r] = s  # Update basic index list
        print("\nCurrent x: {} \nCurrent B: {}\n".format(x,B))
        it += 1      

    raise TimeoutError('The problem is not solved after {} iterations.'.format(max_iter))

#%%
"""Input data"""

# Input data of canonical LP:
if __name__ == "__main__":
    
    # Example in cycle, need Bland's rule
    c = np.array([-0.75, 150, -0.02, 6])
    b = np.array([0, 0, 1])
    A = np.array([[0.25, -60, -0.04, 9],[0.5, -90, -0.02, 3],[0, 0, 1, 0]])
    SimplexMethod(A, b, c, 3, 0) # With Bland's rule
#    SimplexMethod(A, b, c, 500, 1)
    
#    A = np.array([[1, -1],[-1, 1]])
#    c = np.array([-1, -1])
#    b = np.array([1, 1])
#    SimplexMethod(A, b, c, 500, 1) # Unlimited problem
#    
#    # Example with b negative
#    A = np.array([[-1, 1, -1, 1, 1], [-1, -4, 1, 3, 1]])
#    b = np.array([-10, -5])
#    c = np.array([9, 16, 7, -3, -1])
#    SimplexMethod(A, b, c, 500, 1)
    
#    A = np.matrix([[1/2, -11/2, -5/2, 9],[1/2, -3/2, -1/2, 1],[1, 0, 0, 0]])
#    c = np.array([-10, 57, 9, 24])
#    b = np.array([0, 0, 1])

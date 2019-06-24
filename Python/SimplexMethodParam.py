# -*- coding: utf-8 -*-
"""

Created on Tue Apr  9 11:31:52 2019

@author: Elena
"""
from print_boxed import print_boxed # Print pretty boxes
from stdForm import stdForm # Convert in standard form
import numpy as np # Create vectors
import matplotlib.pyplot as plt # Create graphics
import pandas as pd # Export to excel 
from input_data import input_data

# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)


''' _ PRAMETRIC SELF-DUAL SIMPLEX METHOD _ '''

"""
Input data: np.arrays: A, vector b, cost vector c of the model LP
            maximum no of iterations          (default 500)
            rule: if Bland's rule             (default 0)
            c_form: if canonical form         (default 0)
            
Output: vector x* optimal vector
        list u = [iterations, bases, vectors x, solutions c*x]
"""
    
def SimplexMethod(A, b, c, max_it = 500, rule = 0, c_form = 0):
    
    """ Error checking """
        
    if not (isinstance(A, np.ndarray) or isinstance(b, np.ndarray) or isinstance(c, np.ndarray)):
        raise Exception('Inputs must be a numpy arrays')
                
    # Construction in a standard form [A | I]
    if c_form == 0:
        (A, c) = stdForm(A, c)   
    A = np.asmatrix(A)    
    r_A, c_A = A.shape

    """ Check full rank matrix """
    
    # Remove ld rows:
    if not np.linalg.matrix_rank(A) == r_A:        
        A = A[[i for i in range(r_A) if not np.array_equal(np.linalg.qr(A)[1][i, :], np.zeros(c_A))], :]
        r_A = A.shape[0]  # Update no. of rows                                                       
    
    print('\n\tCOMPUTATION OF SIMPLEX ALGORITHM')
    if rule == 0:
        print('\twith the Bland\'s rule:\n')
    else:
        print('\t without the Bland\'s rule:\n')
    
    """ IIphases simplex method """

    # If b negative then I phase simplex fun with input data A_I, x_I, c_I
    # The solution x_I is the initial point for II phase.
     
    if sum(b < 0) > 0:# Phase I variables:
        print('Vector b < 0:\n\tStart phase I\n')
        
        # Change sign of constraints
        A[[i for i in range(r_A) if b[i] < 0]] *= -1  
        b = np.abs(b)                
        A_I = np.matrix(np.concatenate((A, np.identity(r_A)), axis = 1))
        c_I = np.concatenate((np.zeros(c_A), np.ones(r_A)))  
        x_I = np.concatenate((np.zeros(c_A), b))  
        B = set(range(c_A, c_A + r_A)) 
        
        # Compute the Algorithm of the extended LP 
                        
        infoI, xI, BI, zI, itI, uI = fun(A_I, c_I, x_I, B, 0, max_it, rule)     
        assert infoI == 0
        
#         Two cases of the phase I: zI > 0: STOP. zI = 0: Phase II
        
        if zI > 0:
           print('The problem is not feasible.\n{} iterations in phase I.'.format(itI))           
           print_boxed('Optimal cost in phase I: {}\n'.format(zI))
           return xI, uI
       
        else: # Get initial BFS for original problem (without artificial vars.)
            xI = xI[:c_A]
            print("Found initial BFS at x: {}.\n\tStart phase II\n".format(xI))        
            info, x, B, z, itII, u = fun(A, c, xI, BI, itI + 1, max_it, rule)
            u = uI + u
    # If b is positive phase II with B = [n+1,..., n+m]
    
    else:
        x = np.concatenate((np.zeros(c_A-r_A), b))
        B = set(range(c_A-r_A,c_A))        
        info, x, B, z, itII, u = fun(A, c, x, B, 0, max_it, rule)
        
    # Print termination of phase II 
    
    if info == 0:
        print_boxed("Found optimal solution at x* =\n{}\n\n".format(x) +
#                    "Basis indexes: {}\n".format(B) +
#                    "Nonbasis indexes: {}\n".format(set(range(c_A)) - B) +
                    "Optimal cost: {}\n".format(z.round(decimals = 3))+
                    "Number of iterations: {}.".format(itII))
    elif info == 1:
        print("\nUnlimited problem.")
    elif info == 2:
        print('The problem is not solved after {} iterations.'.format(max_it))
    return x, u
    

"""Algorithm"""
    
def fun(A, c, x, B, it, max_it, rule) -> (float, np.array, set, float, np.array, list):
    
    r_A, c_A = np.shape(A)
    B, NB = list(B), set(range(c_A)) - B  # Basic /nonbasic index lists    
    B_inv = np.linalg.inv(A[:, B])
    An = A[:, list(NB)]
    z = np.dot(c, x)  # Value of obj. function
    u = []    
    Z = np.ones(c_A-r_A)
    z = -c[:c_A-r_A]
    X = np.zeros(c_A)
    X[B] = np.ones(r_A)
    
    while it <= max_it:  # Ensure procedure terminates (for the min reduced cost rule)
        
        print("\t\nIteration: {}\nCurrent x: {} \nCurrent B: {}\n".format(it, x, B), end = '')
        u.append([it, B.copy(), x, z.copy()]) # Update table
        neg1 = [(-x[B[i]] / X[B[i]]) for i in range(r_A) if X[B[i]] > 0]
        neg2 = [(-z[i] / Z[i]) for i in range(c_A - r_A) if Z[i] > 0]
        star = min(neg1 + neg2)
        z += star*Z
        x += star*X
        if star >= 0:
            info = 0
            return info, x, set(B), z, it, u
        else:
            xx = np.zeros(c_A)
            zz = np.zeros(len(z))
            for l1 in range(r_A):
                if x[B[l1]] == 0: 
                   # Find l1                
                   v = -An.T*B_inv[:,l1]
                   neg = [(z[i] / v[i], i) for i in list(NB) if v[i] > 0]
                   t, k = min(neg, key=(lambda t: t[0]))
                   
                   for i in range(r_A):    
                       xx[B[i]] = np.asscalar(B_inv[i,:]*An[:, l1])
                   break
               
            for k1 in range(len(z)):           
                if z[k1] == 0:
                   # Find k1 in NB
                   v = B_inv*An[:, k1]
                   neg = [(x[B[i]] / v[i], i) for i in B if v[i] > 0]
                   t, l = min(neg, key=(lambda t: t[0]))
                   
                   for i in range(c_A - r_A): 
                       zz[i] = np.asscalar(-An.T[i,:]*B_inv[:,l])
                   break
            
        # Update the current points and cost vectors
            
        
        
        """Feasible basic direction"""
        d = np.zeros(c_A) 
#       ^ vector that increments x_B and x_NB: here c_A is m + n because A is in standard form
        
        for i in range(r_A):
            d[B[i]] = np.asscalar(-B_inv[i, :] * A[:, s]) #solve B*d = A_s -> -B^-1*A_s
        d[s] = 1
       
        
        if len(neg) == 0: # If d > 0
            info = 1            
            return info, x, B, None, it, u  # info = 1 if Unlimited return
        
        theta, r = min(neg, key=(lambda t: t[0]))  # Find r
        
        x = x + theta * d
        z = z + theta * m       
        
        # Update inverse:
        
        for i in set(range(r_A)) - {r}:
            B_inv[i, :] -= d[B[i]]/d[B[r]] * B_inv[r, :]
        B_inv[r, :] /= -d[B[r]]
        
        NB = NB - {s} | {B[r]}  # Update non-basic index set
        B[r] = s                # Update basic index list       
        it += 1                 # Update iteration
    return 2, x, set(B), z, it, u # info = 2 if max_iteration


#%%
    
"""Input data"""

# Input data of canonical LP:
if __name__ == "__main__":
    
    A, b, c = input_data(3)

    # Run simplex method
    x, u = SimplexMethod(A, b, c, rule = 0) # With Bland's rule
    
    # Create a dataframe and convert to excel
#    dfu = pd.DataFrame(u, columns = ['it', 'Current Basis', 'Current x', 'Current cost value'])
#    dfu.to_excel("Simplex_Minty2.xlsx", index = False)
#    
#    # Plot the graphic with dataframe elements
#    plt.figure()
#    plt.plot(dfu['it'], dfu['Current cost value'], marker = 'o', ls = 'None', label = 'Cost value')
#    plt.title('Cost value')
#    plt.ylabel('cost value')
#    plt.xlabel('iterations')
#    locs, labels = plt.xticks(np.arange(0, len(u), step = 1))
#    
#    plt.grid(b = True, which = 'major')

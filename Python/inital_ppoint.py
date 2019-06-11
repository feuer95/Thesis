# -*- coding: utf-8 -*-
"""

Created on Sat May  4 12:32:39 2019

@author: Elena
"""
from SimplexMethodIIphases import SimplexMethod
from MehrotraMethod2 import mehrotra2
import numpy as np
from stdForm import stdForm
from input_data import input_data
from LPFMethod import longpath
from LPFMethod_PC import longpathPC
#Clean the form of printing vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)


'''YE AND LUSTIG VARIANT INTERIOR POINT METHOD'''

#                     NEW LINEAR PROGRAMMING:
# find {min(1*epsilon) | Ax + epsilon(b - Ax_{o}) = b, epsilon >= 0}
# with initial feasible point (1)
# solution (x*, epsilon*) with epsilon* = 0 and x* initial feasible point 
#for the standard primal problem


def InitFeas(A, b, c, c_form = 0):
    
    
    if c_form == 0:    
        (A, c) = stdForm(A, c)
    (AP, cP) = rep(A, b)
    x, s, u = longpathPC(AP, b, cP)
    r_A, c_A = np.shape(A)
    D = np.concatenate((A.T, np.identity(c_A)), axis = 1)
    (T, w) = rep(D, c)

    y, s, u = longpathPC(T, c, w)
    return(x[0:c_A], y)

def rep(A, b):
    r_A, c_A = np.shape(A)
    # INPUT DATA OF NEW LP   
    B_P = np.matrix(b)
    Q = (B_P - np.transpose(np.sum(A, axis = 1)))
    #                         ^ x_{o} = 1
    AP = np.concatenate((A, Q.T), axis = 1)
    AP = np.asarray(AP)
    cP = np.concatenate((np.zeros(c_A),[1]), axis = 0)    # New cost vector    
    return (AP, cP)

if __name__ == "__main__":
    (A, b, c) = input_data(8)
#    (AP, cP) = InitFeas(AI, b)
#    x, u  = SimplexMethod(AP, b, cP)
#    x = x[:c_A + r_A]
#    err = np.dot(AI, x[:c_A + r_A]) - b
#    
#    A = np.array([[0,0,3,2,-4,-1,0,0,0],[0,0,0,1,-3,2,0,0,0],[-3,-2,0,0,-1,2,-1,0,0],[0,-1,0,0,-1,2,0,-1,0],[4,3,1,1,0,-1,0,0,-1],[1,-2,-2,-2,1,0,0,0,0]])

#    print(err)
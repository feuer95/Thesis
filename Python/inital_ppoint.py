# -*- coding: utf-8 -*-
"""

Created on Sat May  4 12:32:39 2019

@author: Elena
"""
from SimplexMethodIIphases import SimplexMethod
from MehrotraMethod import mehrotra
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


def InitFeas(A: np.matrix):
    
    r_A, c_A = np.shape(A)
    
    # INPUT DATA OF NEW LP
    
    B_P = np.matrix(b)
    
    Q = (B_P - np.transpose(np.sum(A, axis = 1)))
    #                         ^ x_{o} = 1
    A_P = np.concatenate((A, Q.T), axis = 1)
    c_P = np.concatenate((np.zeros(c_A),[1]), axis = 0)    # New cost vector    
    A = np.asarray(A_P)     
    return(A, c_P)


if __name__ == "__main__":
    (A, b, c) = input_data(10)
    r_A, c_A = A.shape
    (AI, cI) = stdForm(A, c)
    (AP, cP) = InitFeas(AI)
    
    x, u  = SimplexMethod(AP, b, cP)
    x = x[:c_A + r_A]
    err = np.dot(AI, x[:c_A + r_A]) - b
    print(err)
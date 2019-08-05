# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:33:14 2019

@author: elena
"""
import numpy as np
from stdForm import stdForm
from AffineMethod import affine 

from MehrotraMethod import mehrotra
from MehrotraMethod2 import mehrotra2

from LPFMethod import longpath1          # longpath with fixed cp
from LPFMethod2 import longpath2         # longpath2 with cp iterative
from LPFMethod_PC import longpathPC      # longpathPredCorr with cp equal to Mehrotra's algorithm

from SimplexMethodIphase import SimplexMethodI # For swedish steel with known basis
from SimplexMethodIIphases import SimplexMethod
from scipy.optimize import linprog

import pandas as pd                     # Export to excel 
import matplotlib.pyplot as plt         # Print plot
from cent_meas import cent_meas
import time

# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)


'''                             =============
                                SWEDISH-STEEL
                                =============
                                  
Find the MAXIMUM total NPV: canonical form A x < b and compute:
    
    SimplexMethod(A, b, c, max_iter = 500, rule = 0, c_form = 0) 
    input data: A, b, c
                                  
Find the MAXIMUM total NPV: standard form A_{i} x     < b_{i}
                                          A_{j} x + 0 = b_{j} 

Optimal cost:   9953.672                                                                
solution in x^* = [75, 90.91, 672.28, 137.31, 13.59, 0, 10.91]
'''
print('\n\tTEST ON SWEDISH STEEL EXAMPLE MODEL\n')

""" import & construct input data for IPM """
def ssteel1():
    excel_file = 'Swedish_steel_IPM.xlsx'
    r = pd.read_excel('Swedish_steel_IPM.xlsx')
    q = r.as_matrix()
    q = np.asarray(q)
    
    # Input data in standard form
    A = q[:,:17]
    b = q[:,17]
    c2 = np.array([16, 10, 8, 9, 48, 60, 53])
    y = np.zeros(10)
    c = np.concatenate((c2,y))
    return(A, b, c)    
    
""" import & construct input data for Simplex Method """
'''With A2, b2, c2 the IPM find a non positive definite matrix, they work only with pc algorithms'''
def ssteel2():    
    excel_file = 'Swedish_steel_SM.xlsx'
    r = pd.read_excel('Swedish_steel_SM.xlsx')
    q = r.as_matrix()
    q = np.asarray(q)
    
    # Input data in canonical form
    A2 = q[:,:7]
    b2 = q[:,7]
    c2 = np.array([16, 10, 8, 9, 48, 60, 53])
    (A3, c3) = stdForm(A2, c2)
    return(A2, b2, c2)


#%%

""" run the methods """
if __name__ == "__main__":
    
    (A, b, c) = ssteel1()
    x, s , u = affine(A, b, c, c_form = 1, info = 1, ip = 1)
    dfu = cent_meas(x, u, label = 'Affine', plot = 0) # 17 it
    
    """                             LPF1                                    """
    #                        161 iterations
    x, s, u = longpath1(A, b, c, c_form = 1, info = 1, ip = 0)
    dful = cent_meas(x, u, label = 'LPF', plot = 0) 
    
    """                             LPF2                                    """
    #                             13 it
    x, s, u, sigma_l2 = longpath2(A, b, c, c_form = 1, info = 1, ip = 1)    
    dfc = cent_meas(x, u, label = 'LPF2', plot = 0)
    
    """                    LPF predictor corrector                          """
    #                          14 iterations
    x, s, u, sigma_pc = longpathPC(A, b, c, c_form = 1, info = 1, ip = 1)
    dfpc = cent_meas(x, u, label = 'LPF PC', plot = 0) 
    
    """                          Mehrotra                                   """
    #                            8 iterations
    x, s, u, sigma_m = mehrotra(A, b, c, c_form = 1, info = 1, ip = 0)
    dfm = cent_meas(x, u, label = 'Mehrotra', plot = 0) 



    """ Recall the simplex method """
    
    (A2, b2, c2) = ssteel2()
    P, u = SimplexMethod(A, b, c, rule = 1, c_form = 1) # 17!!!
    #        Start phase II
    #Iteration: 12
    #Current x: [ 75.  250.  568.   80.5 ...  24.    0.    0.    0. ] 
    
    #P2, u2 = SimplexMethodI(A, b, c, rule = 0, c_form = 1) # 0 it
     
#    P, u = SimplexMethod(A2, b2, c2, rule = 0) 
    #
    #u2 = np.array([75, 13, 717.503, 177.555, 16.084, 0.857, 0])
    #
    x = linprog(c, method = 'simplex', A_eq = A, b_eq = b)    # Exact solution
    x = linprog(c2, method = 'simplex', A_ub = A2, b_ub = b2) # Exact solution in 17 iterations

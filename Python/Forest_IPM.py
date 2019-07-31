# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:52:17 2019

@author: Elena
"""
import numpy as np

from SimplexMethodIIphases import SimplexMethod

from AffineMethod import affine 
from LPFMethod import longpath1
from LPFMethod2 import longpath2
from MehrotraMethod import mehrotra
from MehrotraMethod2 import mehrotra2
from LPFMethod_PC import longpathPC

import pandas as pd             # Export to excel 
import matplotlib.pyplot as plt # Print plot
from cent_meas import cent_meas


# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)


'''         =========================
            FOREST_SERVICE_ALLOCATION 
            =========================

Find the MAXIMUM total NPV: the constraint set in standard form A x = b using the IPM methods:
                input data: A, b, c, c_form = 1, w = 10^{-8} default
                expected output data: optimal solution -322515

import & construct input data in standard form
We construct the input data with zeros rows for the equailies
and identy for the submtrix respect to the inequalities
'''
excel_file = 'Forest.xlsx'
def forest():
    r = pd.read_excel('Forest.xlsx')
    c = np.array(r['p'])  # Cost vector of maximum problem
    T = np.array(-r['t'])
    G = np.array(-r['g'])
    W = np.array(-r['w'])/788
    
    # Construct A in a standard form
    B = np.zeros((7,21))
    for i in range(7):
        B[i,i*3:(i+1)*3] = np.ones(3)
    Y = np.vstack((T,G,W)) # inequality constraints
    r_Y, c_Y = Y.shape
    r_B, c_B = B.shape 
    AI = np.concatenate((B, np.zeros((r_B,r_Y))), axis = 1)  
    AII = np.concatenate((Y, np.identity(r_Y)), axis = 1)
    A = np.concatenate((AI, AII), axis = 0)
    
    # Concatenate c
    c = np.concatenate((c, np.zeros(r_Y)), axis = 0)
    
    # Construct b
    S = np.asarray(r['s'])
    S = S[np.logical_not(np.isnan(S))]
    b = np.array([-40000, -5, -70])
    b = np.concatenate((S, b))
    return (A, b, c)

#%%

""" RUN THE METHODS """
'''
For every method we obtain the optimal vector (x,s), a dataframe with all sequences
time of the algorithm 
'''

if __name__ == "__main__":
    
    (A, b, c) = forest() # already in standard form
    IP = 0
    """                           Affine                                    """
    #                              29 it
    x, s , u = affine(A, b, -c, c_form = 1, ip = IP)
    dfu = cent_meas(x, u, label = 'Affine', plot = 0) 
    
    """                               LPF1                                  """
    #                                183 it
    x, s, u = longpath1(A, b, -c, c_form = 1, info = 1, ip = IP)    
    dful = cent_meas(x, u, label = 'LPF', plot = 0) 
    
    """                               LPF2                                  """
    #                                15 it
    x_c, s_c, u_c, sigma_l2 = longpath2(A, b, -c, c_form = 1, info = 1, ip = IP)     
    dfc = cent_meas(x_c, u_c, label = 'LPF2', plot = 0)
    
    """                            LPF predictor corrector                  """
    #                               19 iterations
    x, s, u, sigma_pc = longpathPC(A, b, -c, c_form = 1, info = 1, ip = IP)
    dfpc = cent_meas(x, u, label = 'LPF PC', plot = 0) 
    
    """                                  Mehrotra                           """
    #                                 9 iterations
    x, s, u, sigma = mehrotra(A, b, -c, c_form = 1, info = 1, ip = IP)
    dfm = cent_meas(x, u, label = 'Mehrotra', plot = 0) 
    
    P1, u = SimplexMethod(A, b, -c, rule = 0, c_form = 1) # 45 iterations

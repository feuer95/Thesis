# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:58:29 2019

@author: elena
"""
import numpy as np
import pandas as pd             # Export to excel 
import matplotlib.pyplot as plt # Print plot
from cent_meas import cent_meas
from scipy.optimize import linprog
import time

'''         =========================
            FOREST_SERVICE_ALLOCATION 
            =========================

Find the MAXIMUM total NPV: the constraint set in standard form A x = b using the IPM methods:
                input data: A, b, c, c_form = 1, w = 10^{-8} default
                expected output data: optimal solution -322515
'''

print('\n\tfirst TEST ON FOREST SERVICE ALLOCATION\n')

""" import & construct input data in standard form"""
''' We construct the input data with zeros rows for the equailies
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
    
'''        =============================
           Blending model: SWEDISH STEEL
           =============================

optimal solution :
    x^* = [75, 90.91, 672.28, 137.31, 13.59, 0, 10.91]
    
'''
""" import & construct input data """

def ssteel():
    excel_file = 'Swedish_steel_IPM.xlsx'
    r = pd.read_excel('Swedish_steel_IPM.xlsx')
    q = r.as_matrix()
    q = np.asarray(q)
    
    # Input data in canonical form Ax <= b
    A = q[:,0:7]
    b = q[:,17]
    c = np.array([16, 10, 8, 9, 48, 60, 53])
    return(A, b, c)

#%%

'''                         ====================================
                            TUBULAR PRODUCTS OPERATIONS PLANNING
                            ==================================== 
                                           
Find the minimum of the TPC : the LP is in canonical form
'''
'''                                        ===
                            TUBULAR PRODUCTS OPERATIONS PLANNING
                                           === 
                                           
Find the minimum of the TPC : the LP is in canonical form

'''

""" import & construct input data for IPM """
def tubprod():
    excel_file = 'TubProd.xlsx'
    r = pd.read_excel('TubProd.xlsx')
    q = r.as_matrix()
    q = np.asarray(q)
    c = q[0,:64]
    A = q[1:21,:64]
    b = q[1:21,64]
    return(A, b, c)

#%%

def qa():    
    excel_file = 'QA.xlsx'
    r = pd.read_excel('QA.xlsx')
    q = r.as_matrix()
    q = np.asarray(q)
    
    A = q[:8,:15]
    b = q[:8,15]
    c = q[:15,16]
    
    # Standard form of input_dat:
    
    r_A, c_A = A.shape
    Z = np.zeros((r_A, 4))
    A = np.hstack((A, Z))
    c = np.concatenate((c, np.zeros(4)))
    
    # rows inequalities
    
    for i in range(4):
           A[i, c_A + i] = 1
    return(A, b, c)       
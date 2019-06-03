# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:52:17 2019

@author: elena
"""
import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # Print plot
import numpy as np

'''                 ===
               SWEDISH STEEL
                    ===
                    
Maximize the present net value:  max p_{ij} x_{ij}

s. t                            sum (x_{ij}) = s_{i} for every j
                                t_{ij}x_{ij} >= 40000
                                g_{ij}x_{ij} >= 5
                                w_{ij}x_{ij} >= 70*788
                                
optimal solution :
    [0,0,75,90,0,0,140,0,0,0,0,60,0,154,58,0,0,98,0,0,113]
    
'''
# Construct input data in canonical form Ax <= b:

""" import & construct input data """

excel_file = 'Forest.xlsx'
r = pd.read_excel('Forest.xlsx')

T = np.array(-r['t'])
G = np.array(-r['g'])
W = np.array(-r['w'])/788
    
# Construct A
B = np.zeros((7,21))
for i in range(7):
    B[i,i*3:(i+1)*3] = np.ones(3)
A = np.vstack((B,-B,T,G,W)) # inequality constraints

# Vector b
S = np.asarray(r['s'])
S = S[np.logical_not(np.isnan(S))]
b = np.array([-40000, -5, -70])
b = np.concatenate((S,-S, b))

# Vector c
c = np.array(r['p'])
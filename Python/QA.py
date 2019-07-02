# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:35:31 2019

@author: elena
"""
import numpy as np
from AffineMethod import affine 

from MehrotraMethod import mehrotra
from MehrotraMethod2 import mehrotra2

from LPFMethod import longpath1
from LPFMethod2 import longpath2

from LPFMethod_cp import longpathC
from LPFMethod_PC import longpathPC

import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # Print plot
from cent_meas import cent_meas

from SimplexMethodIIphases import SimplexMethod
from scipy.optimize import linprog
import time 

"""         ========= 
            QUICK Aid
            =========
"""

# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)

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
       

""" input data for Simplex Method """
 
D = q[:4,:15]
A1 = np.vstack((-D,q[:8,:15]))
S = q[:4,15]      
c1 = q[:15,16]
b1 = np.concatenate((-S, b))

#%%

""" run the methods """

#x_a, s_a , u_a = affine(A, b, c)
#dfu = cent_meas(x_a, u_a, label = 'Affine') # 29 it


"""                              LPF1                                       """
#                           174 iterations
start = time.time()
x_l, s_l, u_l = longpath1(A, b, c, c_form = 1, info = 1)
time_lpf1 = time.time()-start
print('Time of the algorithm is {} \n\n'.format("%2.2e"%time_lpf1))

dful = cent_meas(x_l, u_l, label = 'LPF', plot = 0) 

"""            LPF2             """

start = time.time()
x_c, s_c, u_c, sigma_l2 = longpath2(A, b, c, c_form = 1, info = 1) # 12 it
time_lpf2 = time.time()-start
print('Time of the algorithm is {} \n\n'.format("%2.2e"%time_lpf2))

dfc = cent_meas(x_c, u_c, label = 'LPF2', plot= 0)

"""                         LPF predictor corrector                         """
#                              12 iterations
start = time.time()
x_pc, s_pc, u_pc, sigma_pc = longpathPC(A, b, c, c_form = 1, info = 1)
time_lpfpc = time.time()-start
print('Time of the algorithm is {} \n\n'.format("%2.2e"%time_lpfpc))

dfpc = cent_meas(x_pc, u_pc, label = 'LPF PC', plot = 0) 
 
"""                              Mehrotra                                   """
#                              7 iterations
start = time.time()
x_m, s_m, u_m, sigma_m = mehrotra(A, b, c, c_form = 1, info = 1)
time_mer = time.time()-start
print('Time of the algorithm is {} \n\n'.format("%2.2e"%time_mer))

dfm = cent_meas(x_m, u_m, label = 'Mehrotra', plot = 0) 


" Recall the simplex method "

#P, u = SimplexMethod(A, b, c, rule = 1, c_form = 0) # BAD
#
x = linprog(c, A_eq =A, b_eq = b) # Exact solution
#
#plt.show()
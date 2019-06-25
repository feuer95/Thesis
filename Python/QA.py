# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:35:31 2019

@author: elena
"""
import numpy as np
from AffineMethod import affine 

from MehrotraMethod import mehrotra
from MehrotraMethod2 import mehrotra2

from LPFMethod import longpath
from LPFMethod2 import longpath2
from LPFMethod_cp import longpathC
from LPFMethod_PC import longpathPC

import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # Print plot
from cent_meas import cent_meas
from SimplexMethodIIphases import SimplexMethod
from scipy.optimize import linprog

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
A = np.hstack((A,Z))
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

# Recall the interior point methods

#x_a, s_a , u_a = affine(A, b, c, c_form = 1)
#dfu = cent_meas(x_a, u_a, label = 'Affine', plot = 0) # 16 it
#
#x_m, s_m, u_m = mehrotra(A, b, c, c_form = 1)
#dfm = cent_meas(x_m, u_m, label = 'Mehrotra', plot = 0) # it 7
#
#x_l, y_l, s_l, u_l = longpath(A, b, c, c_form = 1, max_it = 2)
#dful = cent_meas(x_l, u_l, label = 'LPF', plot = 0) # max iter 
#
#x_l, s_l, u_l = longpath2(A, b, c, c_form = 1)
#dful = cent_meas(x_l, u_l, label = 'LPF') # 27 it
#
#x_m, s_m, u_m = mehrotra2(A, b, c, c_form = 1)
#dfm = cent_meas(x_m, u_m, label = 'Mehrotra2', plot = 0) # 8 iterations
#
#cp = 0.0001  #0.5 it = 47 #0.2  22 # 0.8 = 142 
#x_c, s_c, u_c = longpathC(A, b, c, cp = cp) # 142
#dfc = cent_meas(x_c, u_c, label = 'LPF with cp {}'.format(cp))
#
#x_pc, s_pc, u_pc = longpathPC(A, b, c, c_form = 1)
#cfl = cent_meas(x_pc, u_pc, label = 'LPF PC', plot = 0) # 12 iterations
#
#" Recall the simplex method "
#
#P, u = SimplexMethod(A, b, c, rule = 0, c_form = 1) # BAD
#
#x = linprog(c1, A1, b1) # Exact solution

#plt.show()
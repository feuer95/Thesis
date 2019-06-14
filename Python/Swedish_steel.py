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

from LPFMethod import longpath
from longpath2 import longpath2
from LPFMethod_cp import longpathC
from LPFMethod_PC import longpathPC

from SimplexMethodIIphases import SimplexMethod
from scipy.optimize import linprog

import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # Print plot
from cent_meas import cent_meas



# Clean form of printed vectors
#np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)


'''                                  ===
                                SWEDISH-STEEL
                                     ===
                                  
Find the MAXIMUM total NPV: canonical form A x < b and compute:
    
    SimplexMethod(A, b, c, max_iter = 500, rule = 0, c_form = 0) 
    input data: A, b, c
                                  
Find the MAXIMUM total NPV: standard form A_{i} x < b ^ A_{j} x + 0 = b_{j} 
                            and compute:
    
    longpath and mehrotra (A, b, c, max_iter = 500, c_form = 1) 
    input data: A2, b2, c2

'''
print('\n\tsecond TEST ON SWEDISH STEEL EXAMPLE MODEL\n')

""" import & construct input data for IPM """

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

""" import & construct input data for Simplex Method """

excel_file = 'Swedish_steel_SM.xlsx'
r = pd.read_excel('Swedish_steel_SM.xlsx')
q = r.as_matrix()
q = np.asarray(q)

# Input data in canonical form
A2 = q[:,:7]
b2 = q[:,7]
(A3, c3) = stdForm(A2, c2)
# With A2, b2, c2 the IPM find a non positive definite matrix, they fork only with pc algorithms 

# La soluzione deve essere 
# u = np.array([75, 90.91, 672.28, 137.31, 13.59, 0, 10.91])

#%%

""" run the methods """

# Recall the interior point methods

#x_a, s_a , u_a = affine(A, b, c, c_form = 1)
#dfu = cent_meas(x_a, u_a, label = 'Affine', plot = 0) # 17 it
#
#x_m, s_m, u_m = mehrotra(A, b, c, c_form = 1)
#dfm = cent_meas(x_m, u_m, label = 'Mehrotra', plot = 0) # it 8
#
#x_l, s_l, u_l = longpath(A, b, c, c_form = 1)
#dful = cent_meas(x_l, u_l, label = 'LPF', plot = 0) # 27 it 
#
#x_l, s_l, u_l = longpath2(A, b, c, max_it = 1000, c_form = 1)
#dful = cent_meas(x_l, u_l, label = 'LPF') # BAD
#
#x_m, s_m, u_m = mehrotra2(A, b, c, c_form = 1)
#dfm = cent_meas(x_m, u_m, label = 'Mehrotra', plot = 0) # 33 iterations
#
#cp = 0.8
#x_c, s_c, u_c = longpathC(A, b, c, cp = cp) # BAD
#dfc = cent_meas(x_c, u_c, label = 'LPF with cp {}'.format(cp))
#
#x_pc, s_pc, u_pc = longpathPC(A, b, c, c_form = 1)
#cfl = cent_meas(x_pc, u_pc, label = 'LPF PC', plot = 0) # 15 iterations

#x_a, s_a , u_a = longpathPC(A, b, c)
#dfu = cent_meas(x_a, u_a, label = 'LPF PC') # E l'unico ipm che funzione con canonical form 

# Recall the simplex method

P, u = SimplexMethod(A, b, c, rule = 0, c_form = 1)
P, u = SimplexMethod(A2, b2, c2, rule = 0)

u2 = np.array([75, 13, 717.503, 177.555, 16.084, 0.857, 0])

x = linprog(c, method = 'simplex', A_eq = A, b_eq = b) # Exact solution
x = linprog(c2, method = 'simplex', A_ub = A2, b_ub = b2) # Exact solution

#plt.show()
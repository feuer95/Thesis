# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:33:14 2019

@author: elena
"""
import numpy as np
from MehrotraMethod import mehrotra
from MehrotraMethod2 import mehrotra2
from LPFMethod import longpath
from longpath2 import longpath2
from LPFMethod_cp import longpathC
from LPFMethod_PC import longpathPC
from AffineMethod import affine 
from scipy.optimize import linprog
import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # Print plot
from cent_meas import cent_meas
from SimplexMethodIIphases import SimplexMethod
from scipy.optimize import linprog
# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)


'''                                  ===
                                SWEDISH-STEEL
                                     ===
                                  
Find the MAXIMUM total NPV: the constraint set in canonical form A x < b using:
    
    4. SimplexMethod(A, b, c, max_iter = 500, rule = 0, c_form = 0) 
    
    input data: A, b, c

'''
print('\n\tsecond TEST ON SWEIDSH STEEL EXAMPLE MODEL\n')

""" import & construct input data """

excel_file = 'Swedish_steel_IPM.xlsx'
r = pd.read_excel('Swedish_steel_IPM.xlsx')
q = r.as_matrix()
q = np.asarray(q)

# Input data in canonical form
A = q[:,0:7]
b = q[:,7]
c = np.array([16, 10, 8, 9, 48, 60, 53])


""" import & construct input data for SM"""

excel_file = 'Swedish_steel_SM.xlsx'
r = pd.read_excel('Swedish_steel_SM.xlsx')
q = r.as_matrix()
q = np.asarray(q)

# Input data in standard form
A2 = q[:,0:17]
b2 = q[:,17]
y = np.zeros(10)
c2 = np.concatenate((c,y))

#%%

""" run the methods """

# Recall the interior point methods

#x_a, s_a , u_a = affine2(A, b, c)
#dfu = cent_meas(x_a, u_a, label = 'Affine') # BAD
#
#x_m, s_m, u_m = mehrotra(A, b, c)
#dfm = cent_meas(x_m, u_m, label = 'Mehrotra') # BAD
#x_l, s_l, u_l = longpath(A, b, c)
#dful = cent_meas(x_l, u_l, label = 'LPF') # BAD
#
#x_l, s_l, u_l = longpath2(A, b, c, max_it = 1000)
#dful = cent_meas(x_l, u_l, label = 'LPF') # BAD
#
#x_m, s_m, u_m = mehrotra2(A, b, c)
#dfm = cent_meas(x_m, u_m, label = 'Mehrotra', plot = 0) # 45 iterations
#
#cp = 0.8
#x_c, s_c, u_c = longpathC(A, b, c, cp = cp) # BAD
#dfc = cent_meas(x_c, u_c, label = 'LPF with cp {}'.format(cp))
#
#x_pc, s_pc, u_pc = longpathPC(A, b, c)
#cfl = cent_meas(x_pc, u_pc, label = 'LPF PC', plot = 0) # 15 iterations

# Recall for Simplex method
P, u = SimplexMethod(A, b, c, rule = 0)
P, u = SimplexMethod(A2, b2, c2, c_form = 1, rule = 0)
#x = linprog(c2, method = 'simplex', A_eq = A2, b_eq = b2) # Not exact solution
#x = linprog(c, method = 'simplex', A_ub = A, b_ub = b) # Exact solution

#plt.show()
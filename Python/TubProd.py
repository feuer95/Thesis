# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 09:37:46 2019

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

from SimplexMethodIIphases import SimplexMethod
from scipy.optimize import linprog

import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # Print plot
from cent_meas import cent_meas
import time

# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)

'''                                        ===
                            TUBULAR PRODUCTS OPERATIONS PLANNING
                                           === 
                                           
Find the minimum of the TPC : the LP is in canonical form

'''

""" import & construct input data for IPM """

excel_file = 'TubProd.xlsx'
r = pd.read_excel('TubProd.xlsx')
q = r.as_matrix()
q = np.asarray(q)
c = q[0,:64]
A = q[1:21,:64]
b = q[1:21,64]

#%%

""" run the methods """

# Recall the interior point methods (optimal cost 252607.143)
# Plot dual gap e centering measure
#
#x_a, s_a , u_a = affine(A, b, c)
#dfu = cent_meas(x_a, u_a, label = 'Affine') # 29 it
#

"""            LPF1             """

#start = time.time()
#x_l, s_l, u_l = longpath1(A, b, c, info = 1)
#time_lpf1 = time.time()-start
#print('Time of the algorithm is {} \n\n'.format("%2.2e"%time_lpf1))

dful = cent_meas(x_l, u_l, label = 'LPF', plot = 0) # 170 iterations

"""            LPF2             """

start = time.time()
x_c, s_c, u_c, sigma_l2 = longpath2(A, b, c, info = 1) # 13 it
time_lpf2 = time.time()-start
print('Time of the algorithm is {} \n\n'.format("%2.2e"%time_lpf2))

dfc = cent_meas(x_c, u_c, label = 'LPF2', plot= 0)

"""            LPF predictor corrector             """

start = time.time()
x_pc, s_pc, u_pc, sigma_pc = longpathPC(A, b, c, info = 1)
time_lpfpc = time.time()-start
print('Time of the algorithm is {} \n\n'.format("%2.2e"%time_lpfpc))

dfpc = cent_meas(x_pc, u_pc, label = 'LPF PC', plot = 0) # 13 iterations

"""            Mehrotra             """

start = time.time()
x_m, s_m, u_m, sigma_m = mehrotra(A, b, c, info = 1)
time_mer = time.time()-start
print('Time of the algorithm is {} \n\n'.format("%2.2e"%time_mer))
dfm = cent_meas(x_m, u_m, label = 'Mehrotra', plot = 0) # 10 iterations

" Recall the simplex method "

#P, u = SimplexMethod(A, b, c, rule = 1) # 51 it
# it doesn't work with rule = 0
#dfu = pd.DataFrame(u)


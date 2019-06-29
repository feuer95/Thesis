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

""" Interior point methods test """

# Recall the interior point methods (optimal cost 252607.143)

#x_a, s_a , u_a = affine(A, b, c)
#dfu = cent_meas(x_a, u_a, label = 'Affine', plot = 0) # 23 it
#
#x_m, s_m, u_m = mehrotra(A, b, c)
#dfm = cent_meas(x_m, u_m, label = 'Mehrotra', plot = 0) # it 10
#
#x_l, y_l, s_l, u_l = longpath(A, b, c)
#dful = cent_meas(x_l, u_l, label = 'LPF', plot = 0) # BAD
#
#x_l, s_l, u_l = longpath2(A, b, c)
#dful = cent_meas(x_l, u_l, label = 'LPF', plot = 0) # 41 it
#
#x_m, s_m, u_m = mehrotra2(A, b, c)
#dfm = cent_meas(x_m, u_m, label = 'Mehrotra2', plot = 0) # 10 it
#
#cp = 0.8
#x_c, s_c, u_c = longpathC(A, b, c, cp = cp) # BAD
#dfc = cent_meas(x_c, u_c, label = 'LPF with cp {}'.format(cp))
#
#x_pc, s_pc, u_pc = longpathPC(A, b, c) # 20 it
#cfl = cent_meas(x_pc, u_pc, label = 'LPF PC', plot = 0)
#

" Recall the simplex method "

P, u = SimplexMethod(A, b, c, rule = 1) # 51 it
# it doesn't work with rule = 0
dfu = pd.DataFrame(u)


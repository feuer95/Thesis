# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:58:29 2019

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
import time

'''                 ===
           Blending model: swedish steel
                    ===

optimal solution :
    x^* = [75, 90.91, 672.28, 137.31, 13.59, 0, 10.91]
    
'''
""" import & construct input data """

excel_file = 'Swedish_steel_IPM.xlsx'
r = pd.read_excel('Swedish_steel_IPM.xlsx')
q = r.as_matrix()
q = np.asarray(q)

# Input data in canonical form Ax <= b
A = q[:,0:7]
b = q[:,17]
c = np.array([16, 10, 8, 9, 48, 60, 53])


#%%

""" run the methods """

# Recall the interior point methods
# Plot dual gap e centering measure
#
#x_a, s_a , u_a = affine(A, b, -c, c_form = 1)
#dfu = cent_meas(x_a, u_a, label = 'Affine') # 29 it
#

"""            LPF1             """

start = time.time()
x_l, s_l, u_l = longpath1(A, b, c, info = 1)
time_lpf1 = time.time()-start
print('Time of the algorithm is {} \n\n'.format("%2.2e"%time_lpf1))

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

#P1, u = SimplexMethod(A, b, -c, rule = 0, c_form = 1) # 45 iterations

#plt.show()
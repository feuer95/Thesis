# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:13:57 2019

@author: elena
"""
import numpy as np

from SimplexMethodIIphases import SimplexMethod
from scipy.optimize import linprog
from AffineMethod import affine 

from MehrotraMethod import mehrotra
from MehrotraMethod2 import mehrotra2

from LPFMethod import longpath1
from LPFMethod2 import longpath2
from LPFMethod_PC import longpathPC

import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # Print plot
from cent_meas import cent_meas

# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)

"""                 ===   
                    ONB
                    ===
"""
def onb(): 
    excel_file = 'ONB.xlsx'
    r = pd.read_excel('ONB.xlsx')
    q = r.as_matrix()
    q = np.asarray(q)
    
    q[[i for i in range(15,26)]] *= -1
    c = q[0,:23]
    A = q[1:26,:23]
    b = q[1:26,23]
    b= np.concatenate((b, 20*np.ones((10))))
    r_A, c_A = A.shape
    A = np.vstack((A, np.zeros((10,c_A))))
    for i in range(1,11):
        A[r_A-1 + i,12 + i] = 1
    return(A, b, c)
    

#%%

""" run the methods """

(A, b, c) = onb() # input data in canonical form
IP = 0

"""                           Affine                                        """
# IP = 1 doesn't work
x, s , u = affine(A, b, c, ip = IP)
dfu = cent_meas(x, u, label = 'Affine', plot = 0) # 29 it


"""                          LPF1                                          """
#                         307 iterations
x, s, u = longpath1(A, b, c, c_form = 0, info = 1, ip = IP)
dfu = cent_meas(x, u, label = 'LPF', plot = 0)

"""                                 LPF2                                    """
#                                  16 it
x, s, u, sigma_l2 = longpath2(A, b, c, c_form = 0, info = 1, ip = IP) 

"""                        LPF predictor corrector                          """
#                            41 iterations / 17 iterations ip = 1
x_pc, s_pc, u_pc, sigma_pc = longpathPC(A, b, c, c_form = 0, info = 0)
x_pc, s_pc, u_pc, sigma_pc = longpathPC(A, b, c, c_form = 0, info = 0, ip = 1)

dfpc = cent_meas(x_pc, u_pc, label = 'LPF PC', plot = 0) 

"""                          Mehrotra                                       """
 #                          8 iterations

x_m, s_m, u_m, sigma_m = mehrotra2(A, b, c, c_form = 0, info = 1) # Mehrotra1 doesn't work
x_m, s_m, u_m, sigma_m1 = mehrotra2(A, b, c, c_form = 0, info = 1, ip = 1) # Mehrotra1 doesn't work

dfm = cent_meas(x_m, u_m, label = 'Mehrotra', plot = 0)

" Recall the simplex method "

#P, u = SimplexMethod(A, b, c, c_form = 0, rule = 0, max_it = 200) 

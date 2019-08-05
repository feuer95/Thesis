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

'''
Find the minimum of the TPC : the LP is in canonical form
The optimal cost is 252607.143
import & construct input data for IPM
'''
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

""" run the methods """
if __name__ == "__main__": 
    (A, b, c) = tubprod() # canonical form!
    """                              Affine                                 """
    #                                 29 it
    x, s, u = affine(A, b, c, ip = 1)
    dfu = cent_meas(x, u, label = 'Affine', plot = 0)
    
    
    """                            LPF1                                     """
    #                              IT DOESN'T WORK!!!
#    x_l, s_l, u_l = longpath1(A, b, c, info = 1, ip = 1)
    
    """                            LPF2                                     """
    #                              21 it
    x, s, u, sigma_l2 = longpath2(A, b, c, info = 1, ip = 0) 
    dfc = cent_meas(x, u, label = 'LPF2', plot = 0)
    
    """                 LPF predictor corrector                           """
    #                          20 iterations
    x, s, u, sigma_pc = longpathPC(A, b, c, info = 1, ip = 0)
    dfpc = cent_meas(x, u, label = 'LPF PC', plot = 0) 
    
    """                        Mehrotra                                     """
    #                             10 iterations
    start = time.time()
    x_m, s_m, u_m, sigma_m = mehrotra(A, b, c, info = 1, ip = 1) 
    dfm = cent_meas(x_m, u_m, label = 'Mehrotra', plot = 0)
    plt.plot(sigma_m)
    
    " Recall the simplex method "
    
    #P, u = SimplexMethod(A, b, c, rule = 1) # 51 it
    # it doesn't work with rule = 0
    #dfu = pd.DataFrame(u)
    x = linprog(c, method = 'simplex', A_ub = A, b_ub = b)    # Exact solution


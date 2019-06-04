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

import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # Print plot
from cent_meas import cent_meas

from scipy.optimize import linprog

# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)


''' 
                                 FOREST_SERVICE_ALLOCATION 


Find the MAXIMUM total NPV: the constraint set in canonical form A x < b using:
    
    4. SimplexMethod(A, b, c, max_iter = 500, rule = 0, c_form = 0) 
    
    input data: A, b, c

'''
print('\n\tsecond TEST ON SWEIDSH STEEL EXAMPLE MODEL\n')

""" import & construct input data """

excel_file = 'Swedish_steel.xlsx'
r = pd.read_excel('Swedish_steel.xlsx')
q = r.as_matrix()
q = np.asarray(q)

# Input data in canonical form
A = q[:,0:7]
b = q[:,7]
c = np.array([16, 10, 8, 9, 48, 60, 53])
#x, u = SimplexMethod(A, b, c) # With Bland's rule

x, s, u = mehrotra2(A, b, c)
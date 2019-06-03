# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:33:14 2019

@author: elena
"""
import numpy as np
import pandas as pd
from SimplexMethodIIphases import SimplexMethod
from stdForm import stdForm
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
A = q[:,0:7]
b = q[:,7]
c = np.array([16, 10, 8, 9, 48, 60, 53])
x, u = SimplexMethod(A, b, c) # With Bland's rule

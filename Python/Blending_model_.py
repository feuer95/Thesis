# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:58:29 2019

@author: elena
"""
import pandas as pd # Export to excel 
import numpy as np

'''                 ===
               Blending model: swedish steel
                    ===

optimal solution :
    x^* = [75, 90.91, 672.28, 137.31, 13.59, 0, 10.91]
    
'''
""" import & construct input data """

excel_file = 'Swedish_steel.xlsx'
r = pd.read_excel('Swedish_steel.xlsx')
q = r.as_matrix()
q = np.asarray(q)

# Input data in canonical form Ax <= b
A = q[:,0:7]
b = q[:,7]
c = np.array([16, 10, 8, 9, 48, 60, 53])

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:52:29 2019

@author: elena
"""
from SimplexMethodIIphases import SimplexMethod
from Simplex_dwn import simplex
from scipy.optimize import linprog
from input_data import input_data
from stdForm import stdForm

'''       === SIMPLEX METHOD ===

Recall all problem data: the 10 small-scale problems and the 3 applications
All problems are in canonical form: min cx | Ax <= b
In case we have both equalities and inequalities we study two cases A and B

The functions implemented are: 1. simplex(A: matrix, b: np.array, c: np.array, rule: int = 0)
                               2. SimplexMethod(A, b, c, max_iter = 500, rule = 0, c_form = 0)
                               3. linprog <- from linoptimize library, to check the solution
                     
'''
#           Recall input data
(A, b, c) = input_data(1)

x2 = SimplexMethod(A, b, c)
x3 = linprog(c, method = 'simplex', A_ub = A, b_ub = b)                           

# Same solution: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

# FOREST: IT IS INFEASIBLE but if use input data of IPM my simplex works!!

# SWEDISH STEEL: x = linprog(c, method = 'simplex', A_ub = A, b_ub = b) Exact solution
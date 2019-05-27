# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:54:55 2019

@author: elena
"""

import numpy as np
from SimplexMethodIIphases import SimplexMethod
import pandas as pd # Export to excel 

''' 
                 ===
            BEALE'S EXAMPLE 
                 ===
'''

# Computation of the Simplex algorithm to the Beale's example, a canonical LP
# and check the degenerate problem when we don't apply the Bland's rule (rule = 1).

print('\nOptimal solution of the Beale\'s problem:\n')

# Input data in canonical form.
c = np.array([-0.75, 150, -0.02, 6])
b = np.array([0, 0, 1])
A = np.array([[0.25, -60, -0.04, 9],[0.5, -90, -0.02, 3],[0, 0, 1, 0]])

# Run SM with Bland's rule
x0, u0 = SimplexMethod(A, b, c, max_iter = 15, rule = 0) 
dfu = pd.DataFrame(u0,columns = ["iteration", "Current Basis", "Current x", "Current cost value"])
dfu.to_excel("Beales_0.xlsx", index = False)

print('\n','-.-'*17,'\n')

# Run SM without Bland's rule
x1, u1 = SimplexMethod(A, b, c, max_iter = 15, rule = 1) 
dfu = pd.DataFrame(u1,columns = ["iteration", "Current Basis", "Current x", "Current cost value"])
dfu.to_excel("Beales_1.xlsx", index = False)
# Found optimal solution after 7 iterations



#c = np.array([-0.75, 150, -0.02, 6])
#b = np.array([0, 0, 1])
#A = np.array([[0.25, -60, -0.04, 9],[0.5, -90, -0.02, 3],[0, 0, 1, 0]])
#SimplexMethod(A, b, c, 15, 1, 0) # Without Bland's rule 
# TimeoutError: The problem is not solved after 15 iterations.

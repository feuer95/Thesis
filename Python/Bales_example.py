# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:54:55 2019

@author: elena
"""

import numpy as np
from SimplexMethodIIphases import SimplexMethod
import pandas as pd # Export to excel 

''' BEALE'S EXAMPLE '''

# Computation of the Simplex algorithm to the Beale's example, a canonical LP
# and check the degenerate problem whe we don't apply the Bland's rule.

print('\nOptimal solution of the Beale\'s problem:\n')


c = np.array([-0.75, 150, -0.02, 6])
b = np.array([0, 0, 1])
A = np.array([[0.25, -60, -0.04, 9],[0.5, -90, -0.02, 3],[0, 0, 1, 0]])
x, u = SimplexMethod(A, b, c, max_iter= 15, rule = 1) # With Bland's rule
dfu = pd.DataFrame(u,columns = ["it", "B", "x"])
dfu.to_excel("Simplex_.xlsx", index = False)

# Found optimal solution after 7 iterations

print('\n','-.-'*17,'\n')

#c = np.array([-0.75, 150, -0.02, 6])
#b = np.array([0, 0, 1])
#A = np.array([[0.25, -60, -0.04, 9],[0.5, -90, -0.02, 3],[0, 0, 1, 0]])
#SimplexMethod(A, b, c, 15, 1, 0) # Without Bland's rule 
# TimeoutError: The problem is not solved after 15 iterations.

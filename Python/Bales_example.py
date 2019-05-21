# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:54:55 2019

@author: elena
"""

import numpy as np
from SimplexMethodIIphases import SimplexMethod

''' BEALE'S EXAMPLE '''

# Computation of the Simplex algorithm to the Beale's example, a canonical LP
# and check the degenerate problem whe we don't apply the Bland's rule.

print('\nFind the optimal solution of the Beale\'s problem  with the simplex method:\n')


c = np.array([-0.75, 150, -0.02, 6])
b = np.array([0, 0, 1])
A = np.array([[0.25, -60, -0.04, 9],[0.5, -90, -0.02, 3],[0, 0, 1, 0]])
SimplexMethod(A, b, c, 100, 0, 0) # REcall the function With Bland's rule
# Found optimal solution after 7 iterations

print('\n','-.-'*17,'\n')

c = np.array([-0.75, 150, -0.02, 6])
b = np.array([0, 0, 1])
A = np.array([[0.25, -60, -0.04, 9],[0.5, -90, -0.02, 3],[0, 0, 1, 0]])
SimplexMethod(A, b, c, 15, 1, 0) # Without Bland's rule 
# TimeoutError: The problem is not solved after 15 iterations.

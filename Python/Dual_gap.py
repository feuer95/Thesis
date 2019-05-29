# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:20:16 2019

@author: elena
"""
from LPFMethod import longpath # Recall the long-path following function 
from MehrotraMethod import mehrotra # Recall mehrotra's method
import numpy as np # To create vectors
import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # Print plot
from input_data import input_data

'''
                                                  ===
                            Convergence of the dual gap of the LPF and Mehrotra method
                                                  ===

Input data in canonical form: matrix A, vector b and c as np.array

Implementation of the mehrotra(A, b, c, c_form = 0, w = 0.005) 
                      longpath(A, b, c, gamma = 0.001, s_min = 0.1, s_max = 0.9, c_form = 0, w = 0.005)

Output points x, s, table u

Plot the DataFrame in graphic with both convergences.

-----> PROBLEM IN EXAMPLE 5!!
'''

#%%

# Input data of canonical form
# 1
(A, b, c) = input_data(5)

#%%

# Recall the interior point methods
x_m, s_m, u_m = mehrotra(A, b, c)
x_l, s_l, u_l = longpath(A, b, c)

#Create a DataFrame for Mehrotra
dfm = pd.DataFrame(u_m, columns = ['it', 'gM', 'xM', 'sM'])

#Create a DataFrame for LPF
dfl = pd.DataFrame(u_l, columns = ['it', 'gl', 'xl', 'sl'])

plt.figure()
plt.plot(dfm['it'], dfm['gM'], color = 'b', marker = '.')
plt.plot(dfl['it'], dfl['gl'], color = 'g', marker = '.')
plt.title('Dual gap')
plt.xlabel('iterations')
plt.ylabel('current dual gap')
plt.legend()
plt.grid(b = True, which = 'major')
locs, labels = plt.xticks(np.arange(0, max(len(u_m),len(u_l)), step = 1))
plt.show()



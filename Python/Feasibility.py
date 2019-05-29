# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:18:31 2019

@author: elena
"""

from LPFMethod import longpath # Recall the long-path following function 
from MehrotraMethod import mehrotra
import numpy as np # To create vectors
import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # To crrreate graphics
from stdForm import stdForm
from input_data import input_data

'''
                                                  ===
                            Convergence of the feasibilty of the LPF and Mehrotra's method
                                                  ===

Input data in canonical form: matrix A, vector b and c as np.array

Implementation of the longpath(A, b, c, gamma = 0.001, s_min = 0.1, s_max = 0.9, c_form = 0, w = 0.005)
                      mehrotra(A, b, c, c_form = 0, w = 0.005) 
Output points x, s, table u

Check the Feasibility of the vectors computed with LPF method and Mehrotra method
    I. Input data
    II. Run the methods
    III. Construction of feasibility list for both met (A in std form)
    IV. Create Dataframe
    V. Plot both solutions
'''

#%%

# Input data of canonical form
(A, b, c) = input_data(5)

#%%

# Recall the interior point methods
x, s, u_l = longpath(A, b, c)
x, s, u_m = mehrotra(A, b, c)

# Construct list feas
feasl = []
feasm = []
y = A.shape[1] # Truncate the vector x solution of stdForm


# Cycle: t = b - A*x_k > 0

for i in range(len(u_l)):
    t = b - np.dot(A, u_l[i][2][:y]) 
    t = np.amin(t)
    feasl.append(t.copy())

for i in range(len(u_m)):
    t = b - np.dot(A, u_m[i][2][:y]) 
    t = np.amin(t)
    feasm.append(t.copy())
    
#Create a DataFrame for Mehrotra
dfm = pd.DataFrame(u_m, columns = ['it', 'Current g', 'Current x', 'Current s'])
dfm['feas'] = feasm

#Create a DataFrame for LPF
dfl = pd.DataFrame(u_l, columns = ['it', 'Current g', 'Current x', 'Current s'])
dfl['feas'] = feasl

# Plot
plt.figure()

# Plot feasibility P LPF
plt.subplot(2, 1, 1)
plt.title('Feasibility P LPF')
plt.plot(dfl['it'], dfl['feas'], color = 'g', marker = '.', label = '|b - Ax|')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Primal error')
plt.grid(b = True, which = 'both')

# Plot feasibility P Mehrotra
plt.subplot(2, 1, 2)
plt.title('Feasibility P Mehrotra')
plt.plot(dfm['it'], dfm['feas'], color = 'r', marker = '.', label = '|b - Ax|')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Primal error')
plt.grid(b = True, which = 'both')

plt.show()

#%%

# Plot
plt.figure()

# Plot feasibility D LPF
plt.subplot(2, 1, 1)
plt.title('Feasibility D LPF')
#plt.plot(dfl['it'], dfl['Current s'], color = 'g', marker = '.', label = '|b - Ax|')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Dual error')
plt.grid(b = True, which = 'both')

# Plot feasibility D Mehrotra
plt.subplot(2, 1, 2)
plt.title('Feasibility D Mehrotra')
#plt.plot(dfm['it'], dfm['Current s'], color = 'r', marker = '.', label = '|b - Ax|')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Dual error')
plt.grid(b = True, which = 'both')

plt.show()

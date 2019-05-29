# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:58:58 2019

@author: elena
"""

from LPFMethod import longpath # Recall the long-path following function 
from MehrotraMethod import mehrotra
import numpy as np # To create vectors
import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # To create graphics
from input_data import input_data

'''
                                                  ===
                            Convergence of the dual gap of the LPF and Mehrotra methods
                                                  ===

Input data in canonical form: matrix A, vector b and c as np.array

Implementation of the longpath(A, b, c, gamma = 0.001, s_min = 0.1, s_max = 0.9, c_form = 0, w = 0.005)

Output points x, s, table u

Check the behaviour of the duality measure mu for LPF method and Mehrotra method
    I. Input data
    II. long-path followinfg method:
        a. Dataframe
        b. Construct list of mu
        c. plot graphic for dual gap and duality measure 
    III. Merhotra's method
        a. Dataframe
        b. Construct list of mu
        c. plot graphic for dual gap and duality measure 
'''


# Input data
(A, b, c) = input_data(5)

#%%


x, s, u_l = longpath(A, b, c)

# Construct list mu
mu = []
for i in range(len(u_l)):
    mu.append(np.dot(u_l[i][2],u_l[i][3])/(sum(A.shape)))
    
# Dataframe and convert to excel           
dfl = pd.DataFrame(u_l, columns = ['it', 'Current g', 'Current x', 'Current s'])   
dfl['mu'] = mu

# Plot
plt.figure()

# Plot dual gap
plt.subplot(2, 1, 1)
plt.plot(dfl['it'], dfl['Current g'], color = 'g', marker = '.')
plt.legend()
plt.title('LPF method')
plt.xlabel('iterations')
plt.ylabel('current dual gap')
plt.grid(b = True, which = 'both')
locs, labels = plt.xticks(np.arange(0, len(u_l), step = 1))

# Plot dual measure
plt.subplot(2, 1, 2)
plt.plot(dfl['it'], dfl['mu'], color = 'r', marker = '.', label = 'Current mu')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('current mu')
plt.grid(b = True, which = 'both')
locs, labels = plt.xticks(np.arange(0, len(u_l), step = 1))
plt.show()

#%%

x, s, u_m = mehrotra(A, b, c)

# creo lista mu
mu = []
for i in range(len(u_m)):
    mu.append(np.dot(u_l[i][2],u_l[i][3])/sum(A.shape))
    
# Dataframe and convert to excel           
dfl = pd.DataFrame(u_m, columns = ['it', 'Current g', 'Current x', 'Current s'])   
dfl['mu'] = mu

# Plot
plt.figure()

# Plot dual gap
plt.subplot(2, 1, 1)
plt.title('Mehrotra\'s method')
plt.plot(dfl['it'], dfl['Current g'], color = 'g', marker = '.')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('current dual gap')
plt.grid(b = True, which = 'both')
locs, labels = plt.xticks(np.arange(0, len(u_m), step = 1))

# Plot dual measure
plt.subplot(2, 1, 2)
plt.plot(dfl['it'], dfl['mu'], color = 'r', marker = '.', label = 'Current mu')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('current mu')
plt.grid(b = True, which = 'both')
locs, labels = plt.xticks(np.arange(0, len(u_m), step = 1))
plt.show()


# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:58:58 2019

@author: elena
"""

from LPFMethod import longpath # Recall the long-path following function 
from LPFMethod_cp import longpathC
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
        
Target: study the behaviour at each iteration of the value s^{T}x and the dual gap (c^{T}x - b^{T}y)

'''


# Input data
(A, b, c) = input_data(3)

#%%

# Run CPF with centering parameter cp1 assigned
cp1 = 0.9  
x, s, u_l = longpathC(A, b, c, cp = cp1)

# Construct list mu
mu = []
for i in range(len(u_l)):
    mu.append(u_l[i][2]*u_l[i][3])
    
# Dataframe dfl of the list u_l          
dfl = pd.DataFrame(u_l, columns = ['it', 'Current g', 'Current x', 'Current s'])   
dfl['mu'] = mu
#
## Plot
#plt.figure()
#
## Plot dual gap g = (c^{T}x - b^{T}y)
#plt.plot(dfl['it'], dfl['Current g'], color = 'm', marker = 'o')
#plt.legend()
#plt.title('LPFc method with sigma {}'.format(cp1))
#plt.xlabel('iterations')
#plt.ylabel('current dual gap')
#plt.grid(b = True, which = 'both')
#locs, labels = plt.xticks(np.arange(0, len(u_l), step = 1))
#
## Plot duality measure s^{T}x
##plt.plot(dfl['it'], dfl['mu'], color = 'r', marker = 'x', label = 'Current mu')
##plt.legend()
##plt.xlabel('iterations')
##plt.ylabel('current mu')
##plt.grid(b = True, which = 'both')
##locs, labels = plt.xticks(np.arange(0, len(u_l), step = 1))
##plt.show()
#
##%%
#
## Run Mehrotra's method
#x, s, u_m = mehrotra(A, b, c)
#
## Construct lista mu
#mu = []
#for i in range(len(u_m)):
#    mu.append(np.dot(u_m[i][2],u_m[i][3]))
#    
## Convert u_m list in dataframe dfm          
#dfm = pd.DataFrame(u_m, columns = ['it', 'Current g', 'Current x', 'Current s'])   
#dfm['mu'] = mu
#
## Plot
#plt.figure()
#
## Plot dual gap
#plt.plot(dfm['it'], dfm['Current g'], color = 'g', marker = 'o')
#plt.title('Mehrotra\'s method')
#plt.legend()
#plt.xlabel('iterations')
#plt.ylabel('current dual gap')
#plt.grid(b = True, which = 'both')
#locs, labels = plt.xticks(np.arange(0, len(u_m), step = 1))
#
## Plot dual measure
#plt.plot(dfm['it'], dfm['mu'], color = 'r', marker = 'x', label = 'Current mu')
#plt.legend()
#plt.xlabel('iterations')
#plt.ylabel('current mu')
#plt.grid(b = True, which = 'both')
#locs, labels = plt.xticks(np.arange(0, len(u_m), step = 1))
#plt.show()
#

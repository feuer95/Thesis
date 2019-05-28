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
    V. Blot both solutions
'''

#%%

# Input data of canonical form
# 1
A = np.array([[3, 2], [0, 1]])
b = np.array([4, 3])
c = np.array([-1, -1])
#  2
A = np.array([[1, 0],[0, 1],[1, 1],[4, 2]])
c = np.array([-12, -9])
b = np.array([1000, 1500, 1750, 4800])
# 3
A = np.array([[1, 1, 2],[2, 0, 1],[2, 1, 3]])
c = np.array([-3, -2, -4])
b = np.array([4, 1, 7])
# 4
A = np.array([[2, 1],[2, 3]])
c = np.array([-4, -5])
b = np.array([32, 48])
# 5
A = np.array([[-1, 1, -1, 1, 1], [-1, -4, 1, 3, 1]])
b = np.array([-10, -5])
c = np.array([9, 16, 7, -3, -1])


#%%

# Recall the interior point methods
x, s, u_l = longpath(A, b, c)
x, s, u_m = mehrotra(A, b, c)

# Construct list feas
feasl = []
feasm = []
A, c = stdForm(A, c)

for i in range(len(u_l)):
    t = np.dot(A, u_l[i][2]) - b 
    feasl.append(np.linalg.norm(t, np.inf))

for i in range(len(u_m)):
    u = np.dot(A, u_m[i][2]) - b 
    feasm.append(np.linalg.norm(t, np.inf))  
    
#Create a DataFrame for Mehrotra
dfm = pd.DataFrame(u_m, columns = ['it', 'Current g', 'Current x', 'Current s'])
dfm['feas'] = feasm

#Create a DataFrame for LPF
dfl = pd.DataFrame(u_l, columns = ['it', 'Current g', 'Current x', 'Current s'])
dfl['feas'] = feasl

# Plot
plt.figure()

# Plot feasibility LPF
plt.subplot(2, 1, 1)
plt.title('Feasibility LPF')
plt.plot(dfl['it'], dfl['feas'], color = 'g', marker = '.', label = '||Ax - b||')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Primal error')
plt.grid(b = True, which = 'both')

# Plot feasibility Mehrotra
plt.subplot(2, 1, 2)
plt.title('Feasibility Mehrotra')
plt.plot(dfm['it'], dfm['feas'], color = 'r', marker = '.', label = '||Ax - b||')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Primal error')
plt.yscale('symlog')
plt.grid(b = True, which = 'both')

plt.show()


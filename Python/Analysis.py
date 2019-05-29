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
from input_data import input_data # Recall the examples 
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
(A, b, c) = input_data(1)

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
plt.title('Dual gap: example 1')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('current dual gap')
plt.grid()
plt.show()

#%%

# Input data
(A, b, c) = input_data(2)

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
plt.title('Dual gap: example 2')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('current dual gap')
plt.grid()
plt.show()

#%%

# Input data
A = np.array([[2, 1],[2, 3]])
c = np.array([-4, -5])
b = np.array([32, 48])


#Create a DataFrame for Mehrotra
dfm = pd.DataFrame(u_m, columns = ['it', 'g_M', 'x_M', 's_M'])

#Create a DataFrame for LPF
dfl = pd.DataFrame(u_l, columns = ['it', 'g_l', 'x_l', 's_l'])

plt.figure()
plt.plot(dfm['it'], dfm['gM'], color = 'b', marker = '.')
plt.plot(dfl['it'], dfl['gl'], color = 'g', marker = '.')
plt.title('Dual gap: example 3')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('current dual gap')
plt.grid()
plt.show()


#%%

# Input data
A = np.array([[-1, 1, -1, 1, 1], [-1, -4, 1, 3, 1]])
b = np.array([-10, -5])
c = np.array([9, 16, 7, -3, -1])

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
plt.title('Dual gap: example 4')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('current dual gap')
plt.grid()
plt.show()


#%%

# Input data
A = np.array([[1, 1, 2],[2, 0, 1],[2, 1, 3]])
c = np.array([-3, -2, -4])
b = np.array([4, 1, 7])

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
plt.title('Dual gap: example 5')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('current dual gap')
plt.grid()
plt.show()

#%%

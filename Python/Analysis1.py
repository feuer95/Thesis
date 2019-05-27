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
 
'''
                                     ===
                            Convergence of the dual gap
                                     ===
'''

#%%

# Input data of canonical form
A = np.array([[3, 2], [0, 1]])
b = np.array([4, 3])
c = np.array([-1, -1])

# Recall the interior point methods
x_m, s_m, u_m = mehrotra(A, b, c)
x_l, s_l, u_l = longpath(A, b, c)

#Create a DataFrame for Mehrotra
dfm = pd.DataFrame(u_m, columns = ['it_M', 'g_M', 'x_M'])
dfm.to_excel("M1.xlsx", index = False) 

#Create a DataFrame for LPF
dfl = pd.DataFrame(u_l, columns = ["it_l", "g_l", "x_l"])
dfl.to_excel("LPF.xlsx", index = False) 

dfm.plot(x = 'it_M', y = 'g_M', color = 'b', grid = True, title = 'Mehrotra algorithm 1')
dfl.plot(x = 'it_l', y = 'g_l', color = 'g', grid = True, title = 'LPF algorithm 1 ')

#%%

# Input data
A = np.array([[1, 0],[0, 1],[1, 1],[4, 2]])
c = np.array([-12, -9])
b = np.array([1000, 1500, 1750, 4800])

# Recall the interior point methods
x_m, s_m, u_m = mehrotra(A, b, c)
x_l, s_l, u_l = longpath(A, b, c)

#Create a DataFrame for Mehrotra
dfm = pd.DataFrame(u_m, columns = ['it_M', 'g_M', 'x_M'])
dfm.to_excel("M1.xlsx", index = False) 

#Create a DataFrame for LPF
dfl = pd.DataFrame(u_l, columns = ["it_l", "g_l", "x_l"])
dfl.to_excel("LPF.xlsx", index = False) 

dfm.plot(x = 'it_M', y = 'g_M', color = 'b', grid = True, title = 'Mehrotra algorithm 2')
dfl.plot(x = 'it_l', y = 'g_l', color = 'g', grid = True, title = 'LPF algorithm 2')

#%%

# Input data
A = np.array([[2, 1],[2, 3]])
c = np.array([-4, -5])
b = np.array([32, 48])

# Recall the interior point methods
x_m, s_m, u_m = mehrotra(A, b, c)
x_l, s_l, u_l = longpath(A, b, c)

#Create a DataFrame for Mehrotra
dfm = pd.DataFrame(u_m, columns = ['it_M', 'g_M', 'x_M'])
dfm.to_excel("M1.xlsx", index = False) 

#Create a DataFrame for LPF
dfl = pd.DataFrame(u_l, columns = ["it_l", "g_l", "x_l"])
dfl.to_excel("LPF.xlsx", index = False) 

dfm.plot(x = 'it_M', y = 'g_M', color = 'b', grid = True, title = 'Mehrotra algorithm 3')
dfl.plot(x = 'it_l', y = 'g_l', color = 'g', grid = True, title = 'LPF algorithm 3')

#%%

# Input data
A = np.array([[-1, 1, -1, 1, 1], [-1, -4, 1, 3, 1]])
b = np.array([-10, -5])
c = np.array([9, 16, 7, -3, -1])

# Recall the interior point methods
x_m, s_m, u_m = mehrotra(A, b, c)
x_l, s_l, u_l = longpath(A, b, c)

#Create a DataFrame for Mehrotra
dfm = pd.DataFrame(u_m, columns = ['it_M', 'g_M', 'x_M'])
dfm.to_excel("M1.xlsx", index = False) 

#Create a DataFrame for LPF
dfl = pd.DataFrame(u_l, columns = ["it_l", "g_l", "x_l"])
dfl.to_excel("LPF.xlsx", index = False) 

dfm.plot(x = 'it_M', y = 'g_M', color = 'b', grid = True, title = 'Mehrotra algorithm 4')
dfl.plot(x = 'it_l', y = 'g_l', color = 'g', grid = True, title = 'LPF algorithm 4')

#%%

# Input data
A = np.array([[1, 1, 2],[2, 0, 1],[2, 1, 3]])
c = np.array([-3, -2, -4])
b = np.array([4, 1, 7])

# Recall the interior point methods
x_m, s_m, u_m = mehrotra(A, b, c)
x_l, s_l, u_l = longpath(A, b, c)

#Create a DataFrame for Mehrotra
dfm = pd.DataFrame(u_m, columns = ['it_M', 'g_M', 'x_M'])
dfm.to_excel("M1.xlsx", index = False) 

#Create a DataFrame for LPF
dfl = pd.DataFrame(u_l, columns = ["it_l", "g_l", "x_l"])
dfl.to_excel("LPF.xlsx", index = False) 

dfm.plot(x = 'it_M', y = 'g_M', color = 'b', grid = True, title = 'Mehrotra algorithm 5')
dfl.plot(x = 'it_l', y = 'g_l', color = 'g', grid = True, title = 'LPF algorithm 5')

#%%

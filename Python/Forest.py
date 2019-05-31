# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:52:17 2019

@author: Elena
"""
import numpy as np
from SimplexMethodIIphases import SimplexMethod
from MehrotraMethod import mehrotra
from LPFMethod import longpath
from LPFMethod_PC import longpathPC
from AffineMethod import affine
import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # Print plot
# Clean form of printed vectors
#np.set_printoptions(precision=4, threshold=10, edgeitems=4, linewidth=120, suppress = True)


''' FOREST_SERVICE_ALLOCATION '''

"""
Find the MAXIMUM total NPV: the constraint set in standard form A x = b using the methods:
    
    1. affine(A, b, c, c_form = 0, w = 0.005):
    2. mehrotra(A, b, c, c_form = 0, w = 0.005)
    3. longpath(A, b, c, gamma = 0.001, s_min = 0.1, s_max = 0.9, c_form = 0, w = 0.005)
    
    input data: A, b, c, c_form = 1, w = 0.005 default
"""

print('\n\tfirst TEST ON FOREST SERVICE ALLOCATION\n')

""" import & construct input data """

excel_file = 'Forest.xlsx'
r = pd.read_excel('Forest.xlsx')
c = np.array(r['p'])
T = np.array(-r['t'])
G = np.array(-r['g'])
W = np.array(-r['w'])/788

# Construct A
B = np.zeros((7,21))
for i in range(7):
    B[i,i*3:(i+1)*3] = np.ones(3)
Y = np.vstack((T,G,W)) # inequality constraints
r_Y, c_Y = Y.shape
r_B, c_B = B.shape 
AI = np.concatenate((B, np.zeros((r_B,r_Y))), axis = 1)  
AII = np.concatenate((Y, np.identity(r_Y)), axis = 1)
A = np.concatenate((AI, AII), axis = 0)

# Concatenate c
c = np.concatenate((c, np.zeros(r_Y)), axis = 0)

# Construct b
S = np.asarray(r['s'])
S = S[np.logical_not(np.isnan(S))]
b = np.array([-40000, -5, -70])
b = np.concatenate((S, b))

""" run the methods """

# Recall the interior point methods
x_m, s_m, u_m = mehrotra(A, b, -c, c_form = 1)
x_l, s_l, u_l = longpath(A, b, -c, c_form = 1)
x_l, s_l, u_l = longpathPC(A, b, -c, c_form = 1, max_iter = 4)
# Create a DataFrame for Mehrotra
dfm = pd.DataFrame(u_m, columns = ['it', 'g_M', 'x_M', 's_M'])

#Create a DataFrame for LPF
dfl = pd.DataFrame(u_l, columns = ["it_l", "g_l", "x_l", 's_l'])

plt.figure()
plt.plot(dfm['it'], dfm['g_M'], label = 'Mehrotra')
plt.plot(dfl['it_l'],dfl['g_l'], label = 'LPF')
plt.legend()
plt.title('Dual gap')
plt.xlabel('iterations')
plt.ylabel('dual gap')

# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()

plt.grid(b = True, which = 'minor')
plt.grid(b = True, which = 'major')

plt.show()




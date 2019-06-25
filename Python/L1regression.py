# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:41:29 2019

@author: elena
"""
from input_data import input_data
from SimplexMethodIIphases import SimplexMethod
import numpy as np
from AffineMethod import affine 

from MehrotraMethod import mehrotra
from MehrotraMethod2 import mehrotra2

from LPFMethod import longpath
from LPFMethod2 import longpath2
from LPFMethod_cp import longpathC
from LPFMethod_PC import longpathPC
import matplotlib.pyplot as plt # Print plot

"""                 ===
       NUMBER OF ITERATIONS and L1 regression line
                    ===

"""         
B = []
a = []
Y = []
W = []
Z = []

q = np.log(2)
for i in range(25):
    if not i in {4, 20}:
        (A, b, c) = input_data(i)
        x, s, u = mehrotra2(A, b, c)
        x, v = SimplexMethod(A,b,c)
        x, s, z = longpathPC(A, b, c)        
        
        B.append(np.log(len(u)-1))
        W.append(np.log(len(v)-1))
        Z.append(np.log(len(z)-1))       
        r = sum(A.shape)        
        a.append([q, np.log(r)])
        Y.append(np.log(r))

A = np.asarray(a)
B = np.asarray(B)

""" """
# L1 regression of the function |Ax - b|

#r_A, c_A = np.shape(A)
#A1 = np.hstack((A, -np.identity(r_A)))
#A2 = np.hstack((-A, -np.identity(r_A)))
#
#A = np.vstack((A1, A2))
#b =  np.concatenate((-B, B))
#
#c = np.concatenate((np.zeros(c_A),np.ones(r_A)))

plt.plot(Y, B, 'o', color='black', label = 'Mehrotra')
plt.plot(Y, W, '<', color='red', label = 'Simplex')
plt.plot(Y, Z, '>', color='yellow', label = 'LPF predictorCorr')
plt.grid(b = True, which = 'major')
plt.ylabel('iterations')
plt.xlabel('size')
plt.legend()
plt.show()

#xm, sm, um = mehrotra(A, b, c)
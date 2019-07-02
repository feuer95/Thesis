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

from LPFMethod import longpath1
from LPFMethod2 import longpath2
from LPFMethod_cp import longpathC
from LPFMethod_PC import longpathPC
import matplotlib.pyplot as plt # Print plot

"""    ===========================================
       NUMBER OF ITERATIONS and L1 regression line
       ===========================================

""" 
it = []        
B = []
a = []
Y = []
W = []
Z = []

q = np.log(2)
for i in range(29):
    print(i)
    if not i in {4, 20, 25}:
        (A, b, c) = input_data(i)
        x, s, u, sigma_m = mehrotra2(A, b, c, info = 1)
        x, v = SimplexMethod(A,b,c, rule = 0)
        x, s, z, sigma_pc = longpathPC(A, b, c, info = 1)        
        B.append(np.log(len(u)-1))
        W.append(np.log(len(v)-1))
        Z.append(np.log(len(z)-1))       
        r = sum(A.shape)        
        a.append([q, np.log(r)])
        Y.append(np.log(r))
        it.append(i)
for i in range(29, 34):# we compute the models
    it.append(i)
    if i == 29:   
        B.append(np.log(8))
        W.append(np.log(42))
        Z.append(np.log(16))       
        r = sum(A.shape)        
        
    if i == 30: # ssteel
        (A, b, c) = input_data(i)
        B.append(np.log(7))
        W.append(np.log(16))
        Z.append(np.log(13))       
        r = A.shape[1]       
        
    if i == 31: #tubprod
        (A, b, c) = input_data(i)
        B.append(np.log(9))
        W.append(np.log(50))
        Z.append(np.log(19)) 
        r = sum(A.shape)         
    
    if i == 32: # Quick aid
        (A, b, c) = input_data(i)
        B.append(np.log(6))
        W.append(np.log(12))
        Z.append(np.log(11))       
        r = A.shape[1]         
        
    if i == 33: # onb
        (A, b, c) = input_data(i)
        B.append(np.log(9))
        W.append(np.log(59))
        Z.append(np.log(41))       
        r = sum(A.shape)        
    a.append([q, np.log(r)])
    Y.append(np.log(r)) 
    
A = np.asarray(a)
B = np.asarray(B)
W = np.asarray(W)
Z = np.asarray(Z)

#%%


plt.figure()

plt.plot(Y, B, 'o', color='black', label = 'Mehrotra')
plt.plot(Y, W, '<', color='red', label = 'Simplex')
plt.plot(Y, Z, '>', color='green', label = 'LPF predictorCorr')
plt.grid(b = True, which = 'major')
plt.ylabel('iterations')
plt.xlabel('size')
plt.legend()
plt.show()


# L2 regression of |Ax -b|_2 for Mehrotra
A = np.vstack([Y,np.ones(len(Y))])
A = A.T
q = np.dot(np.linalg.pinv(A), B)
x = np.arange(max(Y)+1)
ym = q[0]*x + q[1]
plt.plot(x,ym, c = 'black', linewidth = 2)
#T = 2**x[1]*(m + n)**x[2]

# L2 regression of |Ax -b|_2 for Simplex
#x = np.dot(np.linalg.pinv(A), W)
#x = [-np.Inf,  np.Inf]
#T = 2**x[1]*(m + n)**x[2]

# L2 regression of |Ax -b|_2 for LPF pc
t = np.dot(np.linalg.pinv(A), Z)
yl = t[0]*x + t[1]
plt.plot(x,yl, c = 'green', linewidth = 2)
#T = 2**x[1]*(m + n)**x[2]

# L1 regression of the function |Ax - b|
#r_A, c_A = np.shape(A)
#A1 = np.hstack((A, -np.identity(r_A)))
#A2 = np.hstack((-A, -np.identity(r_A)))
#
#A = np.vstack((A1, A2))
#b =  np.concatenate((-B, B))
#c = np.concatenate((np.zeros(c_A),np.ones(r_A)))
#
#xm, sm, um, sigmam = mehrotra(A, b, c)

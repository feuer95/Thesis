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
O = []
P = []

q = np.log(2)
for i in range(29):
    print(i)
    if not i in {4, 20, 23, 25}:
        (A, b, c) = input_data(i)
        x, s, u, sigma_m = mehrotra2(A, b, c, info = 1)
#        x, v = SimplexMethod(A, b, c, rule = 0)
        x, s, o = longpath1(A, b, c, c_form = 0,  info = 1, ip = 0)
        x, s, p, sigma_2 = longpath2(A, b, c, c_form = 0, info = 1)
        x, s, z, sigma_pc = longpathPC(A, b, c, info = 1)
        
        B.append(np.log(len(u)-1))
#        W.append(np.log(len(v)))
        Z.append(np.log(len(z)-1))   
        O.append(np.log(len(o)-1))
        P.append(np.log(len(p)-1))
        r = sum(A.shape)        
        a.append([q, np.log(r)])
        Y.append(np.log(r))
        it.append(i)
for i in range(29, 34):                               # we compute the models
    it.append(i)
    if i == 29:                                             # FOREST
        (A, b, c) = input_data(i)
        B.append(np.log(8))
#        W.append(np.log(42))
        Z.append(np.log(16))  
        O.append(np.log(224))
        P.append(np.log(15))
        r = sum(A.shape)        
        
    if i == 30:                                         # SWEDISH STEEL
        (A, b, c) = input_data(i)
        B.append(np.log(7))
#        W.append(np.log(16))
        Z.append(np.log(13)) 
        O.append(np.log(213))
        P.append(np.log(12))
        r = A.shape[1]       
        
    if i == 31:                                            #tubprod
        (A, b, c) = input_data(i)
        B.append(np.log(9))
#        W.append(np.log(50))
        Z.append(np.log(19)) 
        O.append(np.log(20))
        P.append(np.log(20))
        r = sum(A.shape)         
    
    if i == 32:                                           # Quick aid
        (A, b, c) = input_data(i)
        B.append(np.log(6))
#        W.append(np.log(12))
        Z.append(np.log(11)) 
        O.append(np.log(174))
        P.append(np.log(12))
        r = A.shape[1]         
        
    if i == 33:                                               # ONB
        (A, b, c) = input_data(i)
        B.append(np.log(9))
#        W.append(np.log(59))
        Z.append(np.log(41))
        O.append(np.log(306))
        P.append(np.log(16))
        r = sum(A.shape)   
        
    a.append([q, np.log(r)])
    Y.append(np.log(r)) 
    
A = np.asarray(a)
B = np.asarray(B)
#W = np.asarray(W)
Z = np.asarray(Z)

#%%


plt.figure()

plt.plot(Y, B, 'o', color='red', label = 'Mehrotra iterations')
#plt.plot(Y, W, '<', color='red', label = 'Simplex')
plt.plot(Y, Z, '>', color = 'cyan', label = 'LPF predictorCorr iterations')
plt.plot(Y, O, '>', color = 'green', label = 'LPF 1 iterations')
plt.plot(Y, P, '>', color = 'blue', label = 'LPF 2 iterations')
plt.grid(b = True, which = 'major')
plt.ylabel('log(T)')
plt.xlabel('log(m + n)')
plt.legend()
plt.show()


# L2 regression of |Ax -b|_2 for Mehrotra
A = np.vstack([Y,np.ones(len(Y))])
A = A.T
x = np.arange(max(Y)+1)

q = np.dot(np.linalg.pinv(A), B)
ym = q[0]*x + q[1]
plt.plot(x, ym, c = 'red', linewidth = 1)


# L2 regression of |Ax -b|_2 for LPF1
o = np.dot(np.linalg.pinv(A), O)
y1 = o[0]*x + o[1]
plt.plot(x, y1,c = 'green', linewidth = 2)

# L2 regression of |Ax -b|_2 for LPF2
p = np.dot(np.linalg.pinv(A), P)
y2 = p[0]*x + p[1]
plt.plot(x, y2, c = 'blue', linewidth = 2)


# L2 regression of |Ax -b|_2 for LPF pc
t = np.dot(np.linalg.pinv(A), Z)
yl = t[0]*x + t[1]
plt.plot(x,yl, c = 'cyan', linewidth = 2)

# L2 regression of |Ax -b|_2 for Simplex
#p = np.dot(np.linalg.pinv(A), W)
#ys = p[0]*x + p[1]
#plt.plot(x,ys, c = 'red', linewidth = 2)
#x = [-np.Inf,  np.Inf]

#%%

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

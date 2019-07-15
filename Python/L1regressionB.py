# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:17:18 2019

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
import pandas as pd             # Export to excel 

"""    ===========================================
       NUMBER OF ITERATIONS and L1 regression line
       ===========================================

""" 
it = []        
B = []
m = []
n = []
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
        
        B.append(len(u)-1)
#        W.append(np.log(len(v)))
        Z.append(len(z)-1)
        O.append(len(o)-1)
        P.append(len(p)-1)  
        m.append(A.shape[0])
        n.append(A.shape[1]) 
        it.append(i)
for i in range(29, 34):                               # we compute the models
    it.append(i)
    if i == 29:                                             # FOREST
        (A, b, c) = input_data(i)
        B.append(8)
#        W.append(np.log(42))
        Z.append(16)  
        O.append(224)
        P.append(15)       
        
    if i == 30:                                         # SWEDISH STEEL
        (A, b, c) = input_data(i)
        B.append(7)
#        W.append(np.log(16))
        Z.append(13) 
        O.append(213)
        P.append(12)
        
    if i == 31:                                            #tubprod
        (A, b, c) = input_data(i)
        B.append(9)
#        W.append(np.log(50))
        Z.append(19) 
        O.append(20)
        P.append(20)       
    
    if i == 32:                                           # Quick aid
        (A, b, c) = input_data(i)
        B.append(6)
#        W.append(np.log(12))
        Z.append(11) 
        O.append(174)
        P.append(12)
     #   r = A.shape[1]         
        
    if i == 33:                                               # ONB
        (A, b, c) = input_data(i)
        B.append(9)
#        W.append(np.log(59))
        Z.append(41)
        O.append(306)
        P.append(16)
        
    m.append(A.shape[0])
    n.append(A.shape[1])  
M = np.asarray(m)
N = np.asarray(n)
B = np.asarray(B)
#W = np.asarray(W)
Z = np.asarray(Z)

#%%
d = {'row': M, 'columns':N, 'Mehrotra':B, 'LPF PC':Z, 'LPF 1':O, 'LPF 2':P}
df = pd.DataFrame(d)
#df.to_excel("itertations_analysis.xlsx", index = False)  
'''
plt.figure()

plt.plot(np.log(Y), np.log(B), 'o', color='red', label = 'Mehrotra iterations')
#plt.plot(Y, W, '<', color='red', label = 'Simplex')
plt.plot(np.log(Y), np.log(Z), '>', color = 'cyan', label = 'LPF predictorCorr iterations')
plt.plot(np.log(Y), np.log(O), '>', color = 'green', label = 'LPF 1 iterations')
plt.plot(np.log(Y), np.log(P), '>', color = 'blue', label = 'LPF 2 iterations')
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
'''
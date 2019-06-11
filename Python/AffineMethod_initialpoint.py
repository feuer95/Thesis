# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:10:23 2019

@author: elena
"""
from input_data import input_data
from stdForm import stdForm
import numpy as np

(A, b, c) = input_data(0)
A, c = stdForm(A, c)
n, m = A.shape


# initial point

x0 = np.ones(m)
s0 = np.ones(m)
t0 = 1
k0 = 1
mi = sum(x0*s0 + t0*k0)/(m+1)
y0 = np.zeros(n)
B = b - np.dot(A, x0)
C = c - np.dot(A.T, y0) - s0

Z = sum(c) + 1

T = np.block([[np.zeros((n,n)), A, np.transpose([-b]), np.transpose([B])], 
              [np.zeros((m, n+m)),np.transpose([c]), np.transpose([-C])],
              [np.zeros((n + m + 1)), np.transpose([Z])],
              [np.zeros((n + m + 2))]
              ])

T = T - T.T 
U = np.block([[np.zeros((n, m + 1))], [-np.identity((m + 1))], [np.zeros((m + 1))]])
V = np.block([np.zeros((m, n)), np.diag((x0)), np.zeros((m,2)), np.diag((s0)), np.zeros((m,1))])
Z = np.block([np.zeros((m + n)), t0, np.zeros((m + 1)), k0])
A_h = np.block([[T, U], [V], [Z]])

A_h.shape
b_h = np.concatenate((np.zeros(n + m + 2), np.ones(m+1)*mi))

# Direction 

I = np.linalg.solve(A_h, b_h)
y0 += I[0:n] 
x0 += I[n:n+m] 
t0 += I[n + m] 
#d0 += I[n + m + 1]
s0 += I[n+m+2: n+m+m+2]
k0 += I[-1:]


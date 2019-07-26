# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:27:48 2019

@author: elena
"""
r = len(u_m)
print(r)

o = len(x_m)
q = np.dot(x_m,s_m)/o 
print("%2.6E"% q)


u = u_m[r-1]
t = np.concatenate((u[4],u[5]))
T = np.linalg.norm(t, 2)
q = T/W
print("%2.6E"% q)


# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:27:48 2019

@author: elena
"""
w = np.concatenate((b,c))
W = np.linalg.norm(w,2)

r = len(u)
print(r)

o = len(x)
q = np.dot(x,s)/o 
print("%2.6E"% q)


u = u[r-1]
t = np.concatenate((u[4],u[5]))
T = np.linalg.norm(t, 2)
q = T/W
print("%2.6E"% q)


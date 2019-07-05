# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 07:53:58 2019

@author: elena
"""
from starting_point import sp
from starting_point2 import sp2
from input_data import input_data
from stdForm import stdForm
import numpy as np
import pandas as pd # Export to excel 

x = []
y = []

u = []
q = []
p = []

v = []
w = []
k = []


for i in range(30):
    (A, b, c) = input_data(i)
    x.append(A.shape[0])
    y.append(A.shape[1])
        
    (A, c) = stdForm(A, c)
    (x1, y1, s1) = sp(A, c, b)
    (x2, y2, s2) = sp2(A, c, b)
    
    r1 = np.concatenate((np.dot(A, x1) - b, c - s1 - np.dot(A.T, y1)))
    r2 = np.concatenate((np.dot(A, x2) - b, c - s2 - np.dot(A.T, y2)))
    
    w1 = np.divide((np.dot(b,y1)- np.dot(c,x1)), A.shape[1])
    w2 = np.divide((np.dot(b,y2)- np.dot(c,x2)), A.shape[1])
    # Mehrotra's method
    u.append("%5.2e"%np.linalg.norm(r1))
    q.append("%5.2e"%w1)
    t = np.divide(np.linalg.norm(r1),w1)
    p.append(t)
    # STP1 method
    v.append("%5.2e"%np.linalg.norm(r2))
    w.append("%5.2e"%w2)
    m = np.divide(np.linalg.norm(r2),w2)
    k.append(m)
#%%

for i in range(30):
    print(u[i]<v[i])
    
d = {'n':x, 'm':y, 'infeas1':u, 'duality gap 1':q,'rate1':p, 'infeas2':v, 'duality gap 2':w, 'rate2':k}
dfu = pd.DataFrame(d)
dfu.to_excel("stp_analysis.xlsx", index = False)  
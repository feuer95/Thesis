# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:43:10 2019

@author: elena
"""
import pandas as pd                      # Export to excel 
from input_data import input_data        # Problem data
from LPFMethod import longpath1          # longpath with fixed cp
from LPFMethod2 import longpath2         # longpath2 with cp iterative
from LPFMethod_PC import longpathPC
import pandas as pd                      # Export to excel 
import matplotlib.pyplot as plt          # Print plot
import time

""" We construct lists with info about number of iterations and time to compute a LP problem """

it1 = []
time_longpath1 = []

it2 = []
time_longpath2 = []

it3 = []
time_longpathPC = []

# for to fill the lists

for i in range(28):
    if not i in {4,14,20,23,25}:
        print(i)
        A, b, c = input_data(i)
        
        start = time.time()
        x1, s1, u1 = longpath1(A, b, c, info = 1)
        time_longpath1.append(time.time() - start)
        it1.append(len(u1))
        
        start = time.time()
        x2, s2, u2, sig2 = longpath2(A, b, c, info = 1)
        time_longpath2.append(time.time() - start)
        it2.append(len(u2))
        
        start = time.time()
        x3, s3, u3, sig3 = longpathPC(A, b, c, info = 1)
        time_longpathPC.append(time.time() - start)
        it3.append(len(u3))


# Plot the times    
plt.figure()
plt.plot(time_longpath1, 'ro', label = 'time LPF 1')
plt.plot(time_longpath2,'b<', label = 'time LPF 2')
plt.plot(time_longpathPC, 'g>', label = 'time LPF PC')
ax = plt.axes()
ax.grid(True)
ax.set_xticks(range(22))
ax.set_yticks(time_longpath1+time_longpath2)
plt.yscale('log')
plt.legend()
plt.xlabel('Problem data')
plt.ylabel('Time (sec)')

# contruct dataframe using all 4 lists
d = {'col1': it1, 'col2': it2, 'col3': time_longpath1, 'col4': time_longpath2 }
dfu = pd.DataFrame(d)

# Plot the iterations number    
plt.figure()
plt.plot(it1, 'ro', label = 'it LPF 1')
plt.plot(it2,'b<', label = 'it LPF 2')
plt.plot(it3, 'g>', label = 'it LPF PC')
ax = plt.axes()
ax.set_xticks(range(22))
ax.set_yticks(it1+it2+it3)
plt.yscale('linear')
ax.grid(True)
plt.legend()

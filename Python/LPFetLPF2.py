# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:43:10 2019

@author: elena
"""
import pandas as pd                      # Export to excel 
from input_data import input_data        # Problem data
from LPFMethod import longpath           # longpath with fixed cp
from LPFMethod2 import longpath2         # longpath2 with cp iterative
import pandas as pd                      # Export to excel 
import matplotlib.pyplot as plt          # Print plot
import time
import pandas as pd                      # Export to excel 


""" We construct lists with info about number of iterations an time to compute a LP problem """

time_longpath = []
it = []
it1 = []
time_longpath2 = []

# for to fill the lists

for i in range(25):
    if not i in {4,14,20,23}:
        print(i)
        A, b, c = input_data(i)
        start = time.time()
        x, s, u = longpath(A, b, c)
        time_longpath.append(time.time() - start)
        it.append(len(u))
        start = time.time()
        x, s, v = longpath2(A, b, c)
        time_longpath2.append(time.time() - start)
        it1.append(len(v))
        
# Plot the times    
plt.figure()
plt.plot(time_longpath, 'ro', label = 'time LPF 1')
plt.plot(time_longpath2,'bo', label = 'time LPF 2')
ax = plt.axes()
ax.grid(True)
ax.set_xticks(range(24))
ax.set_yticks(time_longpath+time_longpath2)
plt.yscale('log')
plt.legend()
plt.xlabel('Problem data')
plt.ylabel('Time (sec)')
# contruct dataframe using all 4 lists

d = {'col1': it, 'col2': it1, 'col3': time_longpath, 'col4': time_longpath2 }
dfu = pd.DataFrame(d)
# , columns = ['iterations LPF1','iterations LPF2','time LPF1','time LPF2']
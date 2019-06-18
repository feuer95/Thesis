# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:44:04 2019

@author: elena
"""
from input_data import input_data

from MehrotraMethod import mehrotra
from MehrotraMethod2 import mehrotra2

from LPFMethod_PC import longpathPC
from matplotlib.ticker import NullFormatter  # useful for `logit` scale
import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # Create graphics
from mpl_toolkits.mplot3d import Axes3D
'''    _PLOT THE PATHs_   '''

# Recall lp istance with dimension 2
(A, b, c) = input_data(0)

xm, sm, um = mehrotra2(A, b, c)
#um = pd.DataFrame(um, columns = ['it', 'g', 'Current x', 'Current s', 'rb', 'rc'])

xl, sl, ul = longpathPC(A, b, c)
#ul = pd.DataFrame(ul, columns = ['it', 'g', 'Current x', 'Current s', 'rb', 'rc'])

# Create 3d point x*s for mehrotra
x1 = []
y1 = []
z1 = []
for i in range(len(um)):
    g = um[i][2]*um[i][3]
    x1.append(g[0].copy())
    y1.append(g[1].copy())
    z1.append(g[2].copy())
    
# Create 3d point x*s for Long-path predictor corrector
x2 = []
y2 = []
z2 = []    
c = []
for i in range(len(ul)):
    c = ul[i][2]*ul[i][3]
    x2.append(g[0].copy())
    y2.append(g[1].copy())
    z2.append(g[2].copy())
    
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x1, y1, z1, c = c, marker="o")
ax.scatter3D(x2, y2, z2, marker="o")
#ax.set(xscale='logit')
#ax.set(yscale='logit')
#ax.set(zscale='logit')
#plt.xscale('logit')
#plt.yscale('logit')
#plt.zscale('logit')

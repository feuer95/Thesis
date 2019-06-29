# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 16:01:36 2019

@author: elena
"""

'''
======================
Triangular 3D surfaces
======================

Plot a 3D surface with a triangular mesh.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from LPFMethod import longpath1
from MehrotraMethod2 import mehrotra2
from input_data import input_data

n_radii = 8
n_angles = 36

# Make radii and angles spaces (radius r=0 omitted to eliminate duplication).
radii = np.linspace(0.125, 10, n_radii)
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

# Repeat all angles for each radius.
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

# Convert polar (radii, angles) coords to cartesian (x, y) coords.
# (0, 0) is manually added at this stage,  so there will be no duplicate
# points in the (x, y) plane.
x = np.append(0, (radii*np.cos(angles)).flatten())
y = np.append(0, (radii*np.sin(angles)).flatten())

# Compute z to make the pringle surface.
z = x*x + y*y
for i in range(1,len(z)):
    z[i] = np.sqrt(z[i])
fig = plt.figure()

ax = fig.gca(projection='3d')
#plt.xscale('logit') 
#plt.yscale('logit') 
ax.plot_trisurf(x, y, z, linewidth=0.2,cmap='viridis')

ax.set_xlabel('x1s1')

ax.set_ylabel('x2s2')

ax.set_zlabel('x3s3')

w = []
(A, b, c) = input_data(0)

x, s, u = mehrotra2(A, b, c)
x1, s1, u1 = longpath1(A, b, c)

# Plot Mehrotra iterations
for i in range(1,len(u)):
    t = u[i][2]*u[i][3]
    w.append(t.copy())
    ax.scatter(t[0],t[1],t[2], linewidth= 2, zorder=1, lw=3, linestyle='dashed', c= "red")
    ax.text(t[0],t[1],t[2], "xs{}".format(len(u)-i))

# PlotLPF iterations
for i in range(1,len(u1)):
    t = u1[i][2]*u1[i][3]
    w.append(t.copy())
    ax.scatter(t[0],t[1],t[2], linewidth= 2, zorder=1, lw=3, linestyle='dashed', c= "g")
    ax.text(t[0],t[1],t[2], "xs{}".format(len(u1)-i))
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 08:59:04 2019

@author: elena
"""
$ pip install pyny3d

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import numpy as np
import pyny3d.geoms as pyny
#if using a Jupyter notebook, include:
#matplotlib inline

x = np.arange(-5,5,0.1)
y = np.arange(-5,5,0.1)
X,Y = np.meshgrid(x,y)
Z = X*np.exp(-X*2 - Y*2)


fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()

#import pyny3d.geoms as pyny


poly1 = pyny.Polygon(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]))
poly2 = pyny.Polygon(np.array([[0, 0, 3], [0.5, 0, 3], [0.5, 0.5, 3], [0, 0.5, 3]]))
polyhedron = pyny.Polyhedron.by_two_polygons(poly1, poly2)
polyhedron.plot('b')
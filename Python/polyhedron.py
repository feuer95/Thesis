# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:49:41 2019

@author: elena
"""
from SimplexMethodIIphases import SimplexMethod
from LPFMethod import longpath
from input_data import input_data
import pandas as pd # Export to excel 
from mpl_toolkits import mplot3d
from cent_meas import cent_meas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

'''             """
            Construct polyhedron
                """
'''

# Recall the input data of 3d data problem

A, b, c = input_data(0)

# Excel info of _Simplex method_

x, u = SimplexMethod(A, b, c) 

# Create a dataframe and convert to excel
dfu = pd.DataFrame(u, columns = ['it', 'Current Basis', 'Current x', 'Current cost value'])
dfu.to_excel("Simplex_polyhedron.xlsx", index = False)

# Excel info of _LPF method_

x, s, u = longpath(A, b, c)

# Create a dataframe and convert to excel
dfm = pd.DataFrame(u, columns = ['it', 'Current g', 'Current x', 'Current s', 'Primal Feasibility', 'Dual Feasibility'])   
#dfm.to_excel("LPF_polyhedron.xlsx", index = False)
Kx = [(u[i][2][0]) for i in range(len(u))]
Ky = [(u[i][2][1]) for i in range(len(u))]
Kz = [(u[i][2][2]) for i in range(len(u))]

fig = plt.figure()
ax = plt.axes(projection="3d")


z_line = np.linspace(-5, 5, 1000)
x_line = np.linspace(-5, 5, 1000)
y_line = np.linspace(-5, 5, 1000)
ax.plot3D(x_line, y_line, z_line, 'gray')
ax.scatter3D(Kx, Ky, Kz, c= 'r')
plt.show()

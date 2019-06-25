# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:13:57 2019

@author: elena
"""

import numpy as np
from AffineMethod import affine 

from MehrotraMethod import mehrotra
from MehrotraMethod2 import mehrotra2

from LPFMethod import longpath
from LPFMethod2 import longpath2
from LPFMethod_cp import longpathC
from LPFMethod_PC import longpathPC

import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # Print plot
from cent_meas import cent_meas
from SimplexMethodIIphases import SimplexMethod
# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)

excel_file = 'ONB.xlsx'
r = pd.read_excel('ONB.xlsx')
q = r.as_matrix()
q = np.asarray(q)

q[[i for i in range(15,25)]] *= -1
c = q[0,:23]
A = q[1:26,:23]
b = q[1:26,23]
b= np.concatenate((b, 20*np.ones((10))))
r_A, c_A = A.shape
A = np.vstack((A, np.zeros((10,c_A))))
for i in range(1,11):
    A[r_A-1+i,12+i] = 1
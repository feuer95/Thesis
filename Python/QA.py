# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:35:31 2019

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

excel_file = 'QA.xlsx'
r = pd.read_excel('QA.xlsx')
q = r.as_matrix()
q = np.asarray(q)
c = q[:,3]
A = q[0:57,4:103]
b = q[0:57,103]

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:42:45 2019

@author: elena
"""
from input_data import input_data
from LPFMethod import longpath
from MehrotraMethod import mehrotra
from cent_meas import cent_meas

'''             * ANALYSIS LONG-STEP PATH-FOLLOWING METHOD *


'''

# Input_data
(A, b, c) = input_data(10)

# Applicaton of longpath
x, s, u = longpath(A, b, c)

# Recall the dataframe

dfu = cent_meas(x, u, 'LPF')

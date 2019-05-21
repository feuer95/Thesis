# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:47:01 2019

@author: elena
"""

import numpy as np

''' stdForm: transform the canonical problem in standard form ''' 

def stdForm(A, c):
    A = np.concatenate((A, np.identity(A.shape[0])), axis = 1)
    c = np.concatenate((c, np.zeros(A.shape[0])), axis = 0)
    return A, c
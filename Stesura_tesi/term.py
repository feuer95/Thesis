# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:22:55 2019

@author: elena
"""
import numpy as np # To create vectors

def term(it, b = None, c = None, rb = None, rc = None, z = None, g = None, d = None):
    if it == 0: 
        return np.inf
    else:
        m = np.linalg.norm(rb)/(1 + np.linalg.norm(b))
        n = np.linalg.norm(rc)/(1 + np.linalg.norm(c))
        q = np.abs(g)/(1 + np.linalg.norm(z))
    return m, n, q  
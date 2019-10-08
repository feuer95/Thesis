# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:59:53 2019

@author: elena
"""

import matplotlib.pyplot as plt 
import numpy as np 

x = np.linspace(0,10) 

plt.plot(x, x, c = 'black') 
plt.plot(x, 5 * x,c = 'black') 
plt.plot(x, 0.30 * x,c = 'black') 
plt.show() 
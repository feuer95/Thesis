# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 14:48:56 2019

@author: elena
"""
import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # Print plot
import numpy as np

def cent_meas(x, u, label):
    # Create a dataframe and convert to excel        
    dfu = pd.DataFrame(u, columns = ['it', 'Current g', 'Current x', 'Current s'])   

    # Construct list mu
    mu = []
    for i in range(len(u)):
        mu.append(u[i][2]*u[i][3])
    
    # Dataframe dfu of the list mu            
    dfu['mu'] = mu

    cm = []
    for i in range(len(u)):
        r = sum(mu[i])/len(mu[i])
        s = mu[i] - r*np.ones(len(x))
        cm.append(np.linalg.norm(s))

    # Dataframe with centering deviation
    dfu['cd'] = cm
    
    # Plot the graphic with dataframe elements    
    plt.figure()
    plt.plot(dfu['it'], dfu['Current g'], label = 'Cost value', marker = '.')
    plt.plot(dfu['it'], dfu['cd'], label = 'Centering measure', marker = '.')
    plt.grid(b = True, which = 'major')
    locs, labels = plt.xticks(np.arange(0, len(u), step = 1))
    
    plt.title('Dual gap & Cenetring measure'+label)
    plt.ylabel('dual gap')
    plt.xlabel('iterations')
    plt.legend()

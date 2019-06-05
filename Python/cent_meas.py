# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 14:48:56 2019

@author: elena
"""
import pandas as pd # Export to excel 
import matplotlib.pyplot as plt # Print plot
import numpy as np

'''
                        ===
      Plot centering measure and dual gap
                        ===
'''


def cent_meas(x, u, label):
    # Create a dataframe and convert to excel        
    dfu = pd.DataFrame(u, columns = ['it', 'Current g', 'Current x', 'Current s', 'Primal Feasibility', 'Dual Feasibility'])   

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
        
    pf = []
    for i in range(len(u)):
        r = max(u[i][4])
        pf.append('%.5f'% r)

    # Dataframe with centering deviation
    dfu['pf'] = pf
    
    # Plot the graphic with dataframe elements    
    plt.figure()
    plt.plot(dfu['it'], dfu['Current g'], label = 'Dual gap', marker = '.')
    plt.plot(dfu['it'], dfu['cd'], label = 'Centering deviation', marker = '.')
    plt.grid(b = True, which = 'major')
    locs, labels = plt.xticks(np.arange(0, len(u), step = 1))
    
    plt.title('Dual gap & Cenetring measure '+label)
    plt.xlabel('iterations')
    plt.legend()
    return dfu
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 14:48:56 2019

@author: elena
"""
import pandas as pd             # Export to excel 
import matplotlib.pyplot as plt # Print plot
import numpy as np

'''
                        ===
      PLOT CENTERING MEASURE AND DUAL GAP
      
Input data: x -> optimal solution x 
            u -> list created in the algorithm u = [it, g, x, s, rb, rc]
            label -> Name of the tested algorithm
            plot -> by default 1 (NO PLOT)
                        ===
'''


def cent_meas(x, u, label, plot = 1):
    
    """ Dataframe """

    # Create a dataframe     
    dfu = pd.DataFrame(u, columns = ['it', 'Current g', 'Current x', 'Current s', 'Primal Feasibility', 'Dual Feasibility'])   

    # Construct list: at each iteration a vector [x1*s1, xi * si, ...]
    mu = []
    for i in range(len(u)):
        mu.append(u[i][2]*u[i][3])
    
    dfu['mu'] = mu # Dataframe with mu   

    # Construct list cm: at each iteration the norm || [x1*s1, xi * si, ...] || (centering deviation)
    cm = []
    sm = []
    for i in range(len(u)):
        r = sum(mu[i])/len(mu[i])
        s = mu[i] - r*np.ones(len(x))
        sm.append(r)
        cm.append(np.linalg.norm(s, 2))
        
    dfu['cd'] = cm # Dataframe with centering deviation
    dfu['sm'] = sm # Duality measure   
    pf = np.zeros(len(u))
    for i in range(len(u)):
        r = np.linalg.norm(u[i][4], 2)
        pf[i] = r

    dfu['pf'] = pf # Dataframe with feasibility
    
#     Convergence rate    
    pr = []
    for i in range(len(u)):
        r = abs(max(u[i][4]))
        pr.append(r)
    
    """ Plot the graphic with dataframe elements """   
    
    if plot == 0:
        plt.figure()
        plt.plot(dfu['it'], dfu['sm'], label = 'Duality measure', c = 'r', marker = '.')
        plt.plot(dfu['it'], dfu['Current g'], label = 'Current g', c = 'C1', marker = '.')
        plt.plot(dfu['it'], dfu['cd'], label = 'Centering deviation', c = 'b', marker = '.')
        plt.grid(b = True, which = 'major')
#        locs, labels = plt.xticks(np.arange(0, len(u), step = 1))
        
        plt.xlabel('iterations')
        plt.yscale('log')
        plt.legend()
       
        plt.figure()
        plt.yscale('log')   
        plt.xlabel('iterations')
        plt.ylabel('error')
        plt.plot(dfu['pf'], c = 'b')
#        locs, labels = plt.xticks(np.arange(0, len(u), step = 1))    
    return dfu
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
            label -> Name of th tested algorithm
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
        cm.append(np.linalg.norm(s, 1))
        
    dfu['cd'] = cm # Dataframe with centering deviation
    dfu['sm'] = sm # mi   
    pf = []
    for i in range(len(u)):
        r = max(u[i][4])
        pf.append('%.5f'% r)

    dfu['pf'] = pf # Dataframe with feasibility
    
    # Convergence rate    
#    pr = []
#    for i in range(len(u)-1):
#        r = abs(max(u[i][4]))
#        s = abs(max(u[i+1][4]))
#        pr.append('%.5f'% np.divide(s,r))
    
    """ Plot the graphic with dataframe elements """   
    
    if plot == 0:
        plt.figure()
        plt.plot(dfu['it'], dfu['sm'], label = 'Current g', marker = '.')
        plt.plot(dfu['it'], dfu['cd'], label = 'Centering deviation', marker = '.')
        plt.grid(b = True, which = 'major')
        locs, labels = plt.xticks(np.arange(0, len(u), step = 1))
        
        plt.title('Dual gap & Centering deviation '+ label)
        plt.xlabel('iterations')
        plt.yscale('log')
        plt.legend()
    return dfu
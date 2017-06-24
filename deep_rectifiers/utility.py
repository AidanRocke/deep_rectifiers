#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 20:50:13 2017

@author: aidanrocke
"""

from scipy.stats import mode
import numpy as np

def ranking_stats(rankings):
    
    #get the mean, mode and variance
    
    stats = np.zeros((3,len(rankings)))
    
    for i in range(len(rankings)):
        stats[:,i][0] = mode(rankings[:,i])
        stats[:,i][1] = np.mean(rankings[:,i])
        stats[:,i][2] = np.var(rankings[:,i])
        
    return stats
        
        
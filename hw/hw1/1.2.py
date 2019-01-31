# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:32:57 2018

@author: Xiruo Li
"""

import numpy as np
import matplotlib.pyplot as plt

#Q1
x=[34, 32, 53, 33, 43, 2, 43, 38, 41, 42, 49, 25, 41, 36, 42, 52, 32, 23, 43, 91]
print(np.mean(x))
print(np.median(x))
print(np.std(x))
print(np.percentile(x,25))
print(np.percentile(x,75))


#Q2
#(a)
plt.figure()
plt.boxplot(x)
A=[34, 32, 53, 33, 43,  43, 38, 41, 42, 49, 25, 41, 36, 42, 52, 32, 23, 43]
#(b)
x.sort()
B=[22.8,22.8,22.8,22.8,22.8,36.4,36.4,36.4,36.4,36.4,42.2,42.2,42.2,42.2,42.2,57.6,57.6,57.6,57.6,57.6]
#(c)
C=[2,32,32,32,32,33,33,33,41,41,41,41,41,43,43,43,43,43,43,91]

#Q3
print(np.mean(C))
print(np.median(C))
print(np.std(C))
print(np.percentile(C,25))
print(np.percentile(C,75))


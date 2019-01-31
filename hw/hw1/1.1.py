# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:34:25 2018

@author: Xiruo Li
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

#Q1
data = np.loadtxt('C:/Users/Xiruo Li/Desktop/CS412/HW/hw1/HW1-data/Q1-data.txt')
divide=(min(data[3,:])+max(data[3,:]))/2
n=0
for i in range(40):
    if data[3,i]>divide:
        n=n+1

Group1=np.zeros((4,17))
Group2=np.zeros((4,23))
n1=0
n2=0
for i in range(40):
    if data[3,i]>divide:
        Group1[:,n1]=data[:,i]
        n1=n1+1
    else:
        Group2[:,n2]=data[:,i]
        n2=n2+1

#Q2
for i in range(17):
    Group1[1,i]=(Group1[1,i]-min(Group1[1,:]))/(max(Group1[1,:])-min(Group1[1,:]))
    Group1[2,i]=(Group1[2,i]-min(Group1[2,:]))/(max(Group1[2,:])-min(Group1[2,:]))

for i in range(23):
    Group2[1,i]=(Group2[1,i]-min(Group2[1,:]))/(max(Group2[1,:])-min(Group2[1,:]))
    Group2[2,i]=(Group2[2,i]-min(Group2[2,:]))/(max(Group2[2,:])-min(Group2[2,:]))

#Q3
plt.figure()
plt.scatter(Group1[1,:],Group1[2,:])
plt.xlabel("substance A")
plt.ylabel("substance B")
plt.title("Group1: substance A vs substance B")

plt.figure()
plt.scatter(Group2[1,:],Group2[2,:])
plt.xlabel("substance A")
plt.ylabel("substance B")
plt.title("Group2: substance A vs substance B")


#Q4
print(pearsonr(Group1[1,:], Group1[2,:]))
print(pearsonr(Group2[1,:], Group2[2,:]))

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 11:25:09 2018

@author: eokte2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import norm
df = pd.read_csv("testing.csv")
#del df['Product_Info_1']
del df['Id']
import pickle
import seaborn as sns

#Load the raining data
f = open('storekaggle.pckl', 'rb')
obj = pickle.load(f)
f.close()

#load the mutual information data
f = open('storeinf.pckl', 'rb')
obj2 = pickle.load(f)
f.close()

#define variables with type
nominalvar=['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5',' InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9', 'Medical_History_11', 'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 'Medical_History_40', 'Medical_History_41']
contvar=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']
discretevar=['Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32']
dummyvar=[]
for i in range(48):
    strr='Medical_Keyword_' + str(i+1)
    dummyvar.append(strr)

#####Required Functions

#This code replaced product info 2 entries with numbers    
def product2replace(df):
    d={}
    dff=df
    count=1
    letters=['A','B','C','D','E']
    numbers=list(range(1,9))
    #First build the lookup table
    for i in range(len(letters)):
        for j in range(len(numbers)):
            strr=letters[i] + str(numbers[j])
            d[strr] = count
            count=count+1
            #Replace those values
            dff.Product_Info_2[dff.Product_Info_2==strr]=d[strr]          
    return dff

df=product2replace(df) #Replace product info 2 column with numbers

import math

#Normal distribution
def normpdf(x, mean, sd):
    var = float(sd)**2
    pi = 3.1415926
    denom = (2*pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

#Prediction algorithm
def PredictBayes(test,dd,contvar,classes,classcounts,testdata):
    #test is testing data, dd contains the conditional probabilities.
    probclass=[]
    ptemp=1 #Initial probability
    pred=[]
    suml=sum(classcounts) #For prior probabilities
    for i,row in test.iterrows(): #Loop through testing data
        probclass=[]
        for j in range(8): #Loop through clasees
            ptemp=1
            for key in dd: #loop through attributes in Naive bayes
                if pd.isnull(test[key][i]): #check if the entry is nan
                    if key in contvar:#if continous variable
                        attvalue=dd[key][1] #replace attribute with mean
                        mean=dd[key][0][1][j][0]
                        std=dd[key][0][1][j][1]
                        prob=normpdf(attvalue,mean,std) #use pdf if continious variable
                    else: #if not continious but still nan
                        attvaluet=round(dd[key][1]) #replace attribute with mean if nominal
                        attvalue = min(list(dd[key][0].keys()), key=lambda x:abs(x-attvaluet))
                        prob=dd[key][0][attvalue][j] #Prob of that class
                else:
                    if key in contvar: #Entry is not nan
                        attvalue=test[key][i]
                        mean=dd[key][0][1][j][0]
                        std=dd[key][0][1][j][1]
                        prob=normpdf(attvalue,mean,std) #pdf if countrinous
                    else:
                        try:
                            attvalue=test[key][i]
                            prob=dd[key][0][attvalue][j] #Prob of that class
                        except: #if not found skip
                            prob=1
                ptemp=ptemp*prob
                
            probclass.append(ptemp*classcounts[j]/suml) #Find class probabilities
        pred.append(np.argmax(probclass)+1) #Select the maximum
    if testdata==1: #If Kaggle data, since we do not know labels, there is no actual column    
        d = {'pred': pred, 'actual': test.Response}
    else:
        d = {'pred': pred,}
    predm = pd.DataFrame(data=d)    
    return predm  

#############################
    


#Code starts here!
dd=obj[0] #This variable stores conditional probabilities for all classes for all type of variables
contvar=obj[3] #Continious variable list
classes=obj[4] #Classes
classcounts=obj[5] #Class counts
dd.pop('Response', None) #We do not need the response column for prediction
#ddd = {k: dd.get(k, None) for k in nominalvar+contvar+dummyvar}  

res=list(obj2[0]) #This stores the mutual information of each variable wth respect to response in decreasng order
allvar=list(obj2[1]) #Stores the variables with maximum mutual information with decreasing order
res=res[::-1] 
allvar=allvar[::-1]
ddd = {k: dd.get(k, None) for k in allvar[0:12]} #For prediction, we need only the first few variables to avoid overfitting

pred=PredictBayes(df,ddd,contvar,classes,classcounts,0) #predict the results

#Write to excel
writer = pd.ExcelWriter('output.xlsx')
pred.to_excel(writer,'Sheet1')
writer.save()
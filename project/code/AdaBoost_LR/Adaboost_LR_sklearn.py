# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:29:29 2018

@author: qingwen2
"""

import pandas as pd
import numpy as np
import random as ran
import math

df = pd.read_csv("training.csv")
df2= pd.read_csv("testing.csv")
del df['Id']
del df2['Id']
nominalvar=['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5','InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9', 'Medical_History_11', 'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 'Medical_History_40', 'Medical_History_41']
contvar=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']
discretevar=['Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32']
dummyvar=[]
for i in range(48):
    strr='Medical_Keyword_' + str(i+1)
    dummyvar.append(strr)
    
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
df=product2replace(df)
df2=product2replace(df2)
    
for column in df.columns:
    if column in contvar:
        df[column]=df[column].fillna(df[column].mean())
    else:
        mode=df[column].mode()
        df[column]=df[column].fillna(mode[0])

for column in df2.columns:
    if column in contvar:
        df2[column]=df2[column].fillna(df2[column].mean())
    else:
        mode=df2[column].mode()
        df2[column]=df2[column].fillna(mode[0])

xt2=df2
xt2=xt2.reset_index()
del xt2['index']
xt2=xt2.values

x = df
x=x.reset_index()
del x['index']
y=x.iloc[:,-1].values
del x['Response']
x=x.values

def weighttuple(NPoints,Weights):   
    sumall=0
    WeightsCum=[]
    for j in range(NPoints):
        sumall=sumall+Weights[j]
        WeightsCum.append(sumall)
    WeightsCum=[0]+WeightsCum
    WeightsCum=np.asarray(WeightsCum)  
    
    idx=[]    
    for i in range(NPoints):
        randnum=ran.random()
        # the largest element of myArr less than myNumber
        temp=(WeightsCum[WeightsCum <= randnum].max())
        idx.append(list(WeightsCum).index(temp))   
        
    return idx

from sklearn.linear_model import LogisticRegression    
        
k=3
weight=[1/len(x) for number in range(len(x))]
classweight=[]
results=[]
for i in range(k):
    accuracy=0    
    while accuracy<0.5:
        ind=weighttuple(len(x),weight)
        D=x[ind]
        Dy=y[ind]
        clf=LogisticRegression()
        clf.fit(D, Dy)
        predictions=clf.predict(D)
        correct = [1 if a == b else 0 for (a, b) in zip(predictions, Dy)]  
        accuracy = (sum(map(int, correct)) / float(len(correct)))
        
    m=0
    A = np.squeeze(np.asarray(predictions))
    for t in range(len(Dy)):
        if A[m]==Dy[m]:
            weight[ind[m]]=weight[ind[m]]*(1-accuracy)/accuracy
        m=m+1
    weight=[float(i)/sum(weight) for i in weight]
    
    w=math.log10(accuracy/(1-accuracy))
    classweight.append(w)
    c=clf.predict(xt2)
    results.append(c.tolist())

predf=pd.DataFrame(np.array(results).reshape(k,10000))
final=[]
for i in range(predf.shape[1]):
    ww=[]
    data=predf.iloc[:,i]
    data.tolist()
    save=[]
    for j in range(len(data)):
        if data[j] in save:
            ww[save.index(data[j])]=ww[save.index(data[j])]+classweight[j]
        else:
            save.append(data[j])
            ww.append(classweight[j])
    final.append(save[ww.index(max(ww))])
final= np.array(final)
print(final)  

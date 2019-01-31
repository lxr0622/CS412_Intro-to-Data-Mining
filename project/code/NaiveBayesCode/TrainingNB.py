# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 10:06:22 2018

@author: Eggs910
"""
#Initialize and Pre Process
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
df = pd.read_csv("training.csv")
#del df['Product_Info_1']
del df['Id']
import warnings
warnings.filterwarnings('ignore')
#############################




####Required Functions and pre-processing



#Replace product Info 2 with numbers
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

#Main Function for Building Naive Bayes
def BuildNaiveBayes(dff,contvar,classes,classcount):
    d={}
    for column in dff: #For each attribute return conditional probabilities for each class
        d[column]=featuremat(dff,column,contvar,classes,classcounts)    
    return d

#This function returns the conditional probability for a given attribute
def featuremat(dff,col,contvar,classes,classcounts):

   #suml=sum(classcounts)
   dic={}
   cont=0
   if col in contvar: #If attribute is continious, it needs different treatment
       cont=1
       items=[1] 
   else:
       items=dff[col].dropna().unique() # equals to list(set(words)) if not continious

   for i in range(len(items)):
       if items[i]=='nan': #If item is nan Naive bayes skips
           continue
       dic[items[i]] = {}
       for j in range(len(classes)): #For each class find conditional probability
           if cont==0: #For non continious variables
               post=len(dff[dff[col]==items[i]][dff.Response==classes[j]])+1 #number of attributes
               lc=len(dff[dff.Response==classes[j]]) #Number of classes
               
               dic[items[i]][j]=post/lc #Conditional probability for given class
           else: #For contonious variables just return mean and standard deviation for given class and att
               dic[items[i]][j]=[dff[dff.Response==classes[j]][col].mean(), dff[dff.Response==classes[j]][col].std()]
   return [dic, dff[col].mean(),dff[col].std()]

###########################

#Code Starts here!

#Define the types of variables
nominalvar=['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5',' InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9', 'Medical_History_11', 'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 'Medical_History_40', 'Medical_History_41']
contvar=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']
discretevar=['Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32']
dummyvar=[]
for i in range(48):
    strr='Medical_Keyword_' + str(i+1)
    dummyvar.append(strr)

#Replace product info 2 values with numbers
df=product2replace(df)

#Test Train Split
msk = np.random.rand(len(df)) < 0.99 #Use All data for Kaggle
train = df[msk]
test = df[~msk]
test=test.reset_index()
train=train.reset_index()
del train['index']
del test['index']

#Histogram of class imbalance
plt.hist(train.Response)

#Count Classes and Class Numbers for prior probability
classnumbers=[]
classes=list(Counter(train.Response).keys()) # equals to list(set(words))
classcounts=list(Counter(train.Response).values()) # counts the elements' frequency
classes, classcounts = zip(*sorted(zip(classes, classcounts)))
for i in range(8):
    classnumbers.append(len(train.Response[train.Response==i+1]))

#Train the Naive Bayes Classifier    
dd=BuildNaiveBayes(train,contvar,classes,classcounts)

#Save the training data so we do not have to train each time we test
import pickle #To save the training data

f = open('storekaggle.pckl', 'wb')
pickle.dump([dd,test,train,contvar,classes,classcounts], f)
f.close()







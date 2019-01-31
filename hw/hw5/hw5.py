import numpy as np
import pandas as pd
import collections

n=14
name=['age', 'income', 'student', 'creditrating', 'buyscomputer?']
D=np.array([['l30', 'high', 'no', 'fair', 'no'],
 ['l30', 'high', 'no' ,'excellent', 'no'],
 ['31to40', 'high', 'no', 'fair', 'yes'],
 ['g40', 'medium', 'no', 'fair', 'yes'],
 ['g40', 'low', 'yes', 'fair', 'yes'],
 ['g40', 'low', 'yes', 'excellent', 'no'],
 ['31to40', 'low', 'yes', 'excellent' ,'yes'],
 ['l30', 'medium', 'no' ,'fair' ,'no'],
 ['l30', 'low', 'yes', 'fair', 'yes'],
 ['g40', 'medium', 'yes', 'fair', 'yes'],
 ['l30', 'medium', 'yes', 'excellent', 'yes'],
 ['31to40', 'medium', 'no', 'excellent', 'yes'],
 ['31to40', 'high', 'yes', 'fair', 'yes'],
 ['g40', 'medium', 'no', 'excellent', 'no'],])

D=pd.DataFrame(D,columns=name)


def Gain(n,D):
    #info
    freq=D[name[-1]].value_counts().to_dict()
    label=[]
    value=[]
    for k,v in freq.items():
        label.append(k)
        value.append(v)
    info_sub=[]
    for k in value:
        info_sub.append(-k/n*np.log2(k/n))
    info=sum(info_sub)
            
    #info_i
    info_i=[]
    split=[]
    for i in range(len(name)-1):
        attribute=name[i]
        freq_i=D[name[i]].value_counts().to_dict()
        
        name_i=[]
        value_i=[]
        for k,v in freq_i.items():
            name_i.append(k)
            value_i.append(v)
            
        split_sub=[]
        for l in value_i:
            split_sub.append(-l/n*np.log2(l/n))
        split.append(sum(split_sub))

        info_isub=[]
        for j in range(len(name_i)):
            sub=D.loc[(D[attribute] == name_i[j]), name[-1]].to_dict()
            sub_label=[]
            for k,v in sub.items():
                sub_label.append(v)
            sub_label_freq=collections.Counter(sub_label)
            for k,v in sub_label_freq.items():
                info_isub.append(-value_i[j]/n*v/value_i[j]*np.log2(v/value_i[j]))
        info_i.append(sum(info_isub))
    
    Gain=(np.ones((len(name)-1))*info-info_i).tolist()
    GainRatio=(np.divide(Gain,split)).tolist()
    return name[Gain.index(max(Gain))],name[GainRatio.index(max(GainRatio))]



def Gini(n,D):
    #gini
    freq=D[name[-1]].value_counts().to_dict()
    label=[]
    value=[]
    for k,v in freq.items():
        label.append(k)
        value.append(v)
    gini_sub=[]
    for k in value:
        gini_sub.append((k/n)**2)
    gini=1-sum(gini_sub)

    #gini_i
    gini_i=[]
    for i in range(len(name)-1):
        attribute=name[i]
        freq_i=D[name[i]].value_counts().to_dict()
        
        name_i=[]
        value_i=[]
        for k,v in freq_i.items():
            name_i.append(k)
            value_i.append(v)
                  
        n_value_i=len(value_i)

        gini_isub=[]
        for j in range(n_value_i):
            sub_v=[]
            sub=D.loc[(D[attribute] == name_i[j]), name[-1]].to_dict()
            sub_label=[]
            for k,v in sub.items():
                sub_label.append(v)
            sub_label_freq=collections.Counter(sub_label)
            for k,v in sub_label_freq.items():
                sub_v.append((v/value_i[j])**2)
            gini_isub.append(value_i[j]/n*(1-sum(sub_v)))     
        gini_i.append(sum(gini_isub))    
        
                
    Gain=(np.ones((len(name)-1))*gini-gini_i).tolist()
    return name[Gain.index(max(Gain))]

    
GainInfo, GainRatio=Gain(n,D)
gini=Gini(n,D)
print(GainInfo)
print(GainRatio)     
print(gini)     

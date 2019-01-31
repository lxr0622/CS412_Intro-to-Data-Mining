import pandas as pd
import numpy as np
import time
import seaborn as sns

##LDA##

def LDA(Xtrain,Ytrain,Xtest):
    #training
    train_split=[]
    df=pd.concat([Xtrain, Ytrain], axis=1)
    grouped=df.groupby(list(df.columns.values)[-1])
    for group in grouped:
        train_split.append(group)
    nlabel=len(train_split)
    #sample mean of each class
    Uk_total=[]
    #covariance
    C=np.zeros((nfeat,nfeat))
    for i in range(nlabel):
        Xk=train_split[i][1].iloc[:,0:nfeat].values
        Uk=np.mean(Xk, axis=0).reshape((1,nfeat))
        Uk_total.append(Uk)
        Ck=(Xk-np.ones((np.shape(Xk)[0],1))@Uk).T@(Xk-np.ones((np.shape(Xk)[0],1))@Uk)/(np.shape(Xk)[0]-1)
        C=C+Ck
    C=C/nlabel
    C_inv=np.linalg.inv(C)
    
    #testing
    W=C_inv@np.vstack((Uk_total)).T
    b_total=[]
    for j in range(nlabel):
        b_total.append(-0.5*Uk_total[j]@C_inv@Uk_total[j].T+np.log(1/nlabel))
    b=np.ones((np.shape(Xtest)[0],1))@np.hstack((b_total))
    pred=np.argmax(Xtest@W+b,axis=1)+1
    return pred




##preprocessing##

#read data and each type of variable
df = pd.read_csv("training.csv")
df_test=pd.read_csv("testing.csv")
attr=pd.read_csv("Attribute_List.csv")
del df['Id']
del df_test['Id']
nominalvar=['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5','InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9', 'Medical_History_11', 'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 'Medical_History_40', 'Medical_History_41']
contvar=['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']
discretevar=['Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32']
dummyvar=[]
for i in range(48):
    strr='Medical_Keyword_' + str(i+1)
    dummyvar.append(strr)

#convert product info 2 into numeric var
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
df_test=product2replace(df_test)

#replace NA by mean for numerical var, by mode for other var.
for column in df.columns:
    if column in contvar:
        df[column]=df[column].fillna(df[column].mean())
    else:
        mode=df[column].mode()
        df[column]=df[column].fillna(mode[0])
        
for column in df_test.columns:
    if column in contvar:
        df_test[column]=df_test[column].fillna(df_test[column].mean())
    else:
        mode=df_test[column].mode()
        df_test[column]=df_test[column].fillna(mode[0])


#Sort the feature according to mutual infomation gain, and select part of them
def Feature_Select(df,df_test,nfeat):
    #training data
    df2=pd.DataFrame()
    for i in range(nfeat):
        df2[attr['Attribute'][i]]=df[attr['Attribute'][i]]
    df2['Response']=df['Response']
    
    
    #testing data
    df2_test=pd.DataFrame()
    for i in range(nfeat):
        df2_test[attr['Attribute'][i]]=df_test[attr['Attribute'][i]]
    
    return df2,df2_test






##prediction for kaggle##

nfeat=126#number of features selected
df2,df2_test=Feature_Select(df,df_test,nfeat)
Xtrain=df2.iloc[:,0:nfeat]
Ytrain=df2.iloc[:,-1] 
Xtest=df2_test

t0 = time.time()
pred_kaggle=LDA(Xtrain,Ytrain,Xtest)
t1 = time.time()
print("time cost for LDA is %.2f s" %(t1-t0))


##Split training data(x,y,xt,yt) to get accuracy##
msk = np.random.rand(len(df)) < 0.85
train = df[msk]
test = df[~msk]
test=test.reset_index()
train=train.reset_index()
del train['index']
del test['index']

x=train.copy()
y=train['Response']
del x['Response']
xt=test.copy()
yt=test['Response']
del xt['Response']

yt=yt.values
pred_split=LDA(x,y,xt)
#plot
g=sns.regplot(yt, pred_split, fit_reg=False, scatter_kws={"color":"darkred","alpha":0.01,"s":200} )


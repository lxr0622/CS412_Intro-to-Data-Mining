from itertools import permutations

data=['3\n', 'b and c d and e f\n', 'b c d e f\n', 'b and c d b e g f']
#data=['4\n', 'Clustering and classification are important problems in machine learning.\n', 'There are many machine learning algorithms for classification and clustering problems.\n', 'Classification problems require training data.\n', 'Most clustering problems require user-specified group number.\n', 'SVM, LogisticRegression and NaiveBayes are machine learning algorithms for classification problems.\n', 'k-means, AGNES and DBSCAN are clustering algorithms.\n', 'Dimension reduction methods such as PCA are also learning algorithms for clustering problems.']


min_sup=int(data[0])
stop=['a' ,'an', 'are', 'as', 'at', 'by', 'be', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its','of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with']

#split and lower case
data1=[]
for i in range(1,len(data)):
    if "." in data[i]:
        data[i]=data[i].replace('.','')
    if "," in data[i]:
        data[i]=data[i].replace(',','')  
    data1.append((data[i].lower()).split())

#remove stop words
data2=[]
data2_L1=[]
for i in range(len(data1)):
    data2_1=[]
    for j in range(len(data1[i])):
        if data1[i][j] not in stop:
            data2_1.append(data1[i][j])
            data2_L1.append(data1[i][j])
    data2.append(data2_1)


#L1

data2_L1.sort()   
C1_name=[]
C1_sup=[]
cnt = 1
for i in range(1, len(data2_L1)):
    if (data2_L1[i] == data2_L1[i - 1]):
        cnt += 1
    else:
        C1_name.append(data2_L1[i-1])
        C1_sup.append(cnt)
        cnt = 1
    C1_name.append(data2_L1[i])
    C1_sup.append(cnt)
C1=dict(zip(C1_name, C1_sup))

L1_name=[]
L1_sup=[]
for k,v in C1.items():
    if v>=min_sup:
        L1_name.append(k)
        L1_sup.append(v)


#L2
C2_name=L1_name
if "and" in C2_name:
    C2_name.remove("and")
C2=list(permutations(C2_name,2))

L2_name=[]
L2_sup=[]
for i in range(len(C2)):
    cnt=0
    for j in range(len(data2)):
        if (C2[i][0] in data2[j]) and (C2[i][1] in data2[j]):
            i1=data2[j].index(C2[i][0])
            i2=data2[j].index(C2[i][1])
            if i1<i2:
                if not(data2[j][i1+1]=="and" and data2[j][i2-1]=="and" and abs(i2-i1)==2):
                    cnt=cnt+1

    if cnt>=min_sup:
        L2_name.append(list(C2[i]))
        L2_sup.append(cnt)

L2_name_1=[]
for i in range(len(L2_name)):
    L2_name_1.append(" ".join([L2_name[i][0],L2_name[i][1]]))    
 

#L3
C3=[]
for i in range(len(L2_name)):
    cnt=0
    for j in range(len(L2_name)):
        if L2_name[i][1]==L2_name[j][0]:
            C3.append([L2_name[i][0],L2_name[i][1],L2_name[j][1]])
            
L3_name=[]
L3_sup=[]
for i in range(len(C3)):
    cnt=0
    for j in range(len(data2)):
        if (C3[i][0] in data2[j]) and (C3[i][1] in data2[j]) and (C3[i][2] in data2[j]):
            i1=data2[j].index(C3[i][0])
            i2=data2[j].index(C3[i][1])
            i3=data2[j].index(C3[i][2])
            if (i1<i2<i3):
                cnt=cnt+1
    if cnt>=min_sup:
        L3_name.append(list(C3[i]))
        L3_sup.append(cnt)

L3_name_1=[]
for i in range(len(L3_name)):
    L3_name_1.append(" ".join([L3_name[i][0],L3_name[i][1],L3_name[i][2]]))
         
   
        
#main
if L3_name!=[]:
    L3_sup, L3_name_1 = zip(*sorted(zip(L3_sup, L3_name_1))) 
    for i in range(len(L3_name_1)):
        print ('%d [%s]' % (L3_sup[i],L3_name_1[i]))
        
elif L2_name!=[]:
    L2_sup, L2_name_1 = zip(*sorted(zip(L2_sup, L2_name_1)))
    for i in range(len(L2_name_1)):
        print ('%d [%s]' % (L2_sup[i],L2_name_1[i]))
        
elif L1_name!=[]:
    L1_name.sort()
    L1_sup.sort()
    for i in range(len(L1_name)):
        print ('%d [%s]' % (L1_sup[i],L1_name[i]))

else:
    pass








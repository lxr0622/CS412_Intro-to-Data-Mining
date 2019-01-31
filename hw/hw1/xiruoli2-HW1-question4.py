import numpy as np
import scipy.spatial.distance
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

f=open('C:/Users/Xiruo Li/Desktop/CS412/HW/hw1/HW1-data/Q4-analysis-input.in','r')

D=int(f.readline())
N=int(f.readline())
f.readline()
f.readline()
P=list(map(int, f.readline().split()))

#Original data
Sample=np.zeros((N,D))
distance1=[]
distance2=[]
distance3=[]
distance4=[]

for i in range(N):
    Sample[i,:]=list(map(int, f.readline().split()))
    distance1.append(scipy.spatial.distance.cityblock(P,Sample[i,:]))
    distance2.append(scipy.spatial.distance.euclidean(P,Sample[i,:]))
    distance3.append(scipy.spatial.distance.chebyshev(P,Sample[i,:]))
    distance4.append(scipy.spatial.distance.cosine(P,Sample[i,:]))
        
sort1=np.argsort(distance1,kind='mergesort')
sort2=np.argsort(distance2,kind='mergesort')
sort3=np.argsort(distance3,kind='mergesort')
sort4=np.argsort(distance4,kind='mergesort')

print("Original dataset")
print("Manhattan")
for i in range(5):
    print(sort1[i]+1)

print("Euclidean")
for i in range(5):
    print(sort2[i]+1)
    
print("Supremum")
for i in range(5):
    print(sort3[i]+1)

print("Cosine")
for i in range(5):
    print(sort4[i]+1)
print('\n')
    

#With PCA 
var=[]
for x in [100,10,2,1]:
    
    Total=np.vstack((P,Sample))
    pca = PCA(n_components=x)
    Total_pca = pca.fit_transform(Total)
    var.append(pca.explained_variance_.cumsum())
    
    distance1=[]
    distance2=[]
    distance3=[]
    distance4=[]
    for i in range(1,N+1):
        distance1.append(scipy.spatial.distance.cityblock(Total_pca[0,:],Total_pca[i,:]))
        distance2.append(scipy.spatial.distance.euclidean(Total_pca[0,:],Total_pca[i,:]))
        distance3.append(scipy.spatial.distance.chebyshev(Total_pca[0,:],Total_pca[i,:]))
        distance4.append(scipy.spatial.distance.cosine(Total_pca[0,:],Total_pca[i,:]))
        
    sort1=np.argsort(distance1,kind='mergesort')
    sort2=np.argsort(distance2,kind='mergesort')
    sort3=np.argsort(distance3,kind='mergesort')
    sort4=np.argsort(distance4,kind='mergesort')

    print("X=%d" %x)
    print("Manhattan")
    for i in range(5):
        print(sort1[i]+1)
    
    print("Euclidean")
    for i in range(5):
        print(sort2[i]+1)
        
    print("Supremum")
    for i in range(5):
        print(sort3[i]+1)
    
    print("Cosine")
    for i in range(5):
        print(sort4[i]+1)
    print('\n')

plt.figure()
plt.plot(var[0],label="X=100")
plt.plot(var[1],label="X=10")
plt.plot(var[2],label="X=2")
plt.plot(var[3],label="X=1")
plt.legend()
plt.ylabel('cumulative explained variance')
plt.xlabel('number of features')


import numpy as np
from itertools import combinations

K=2

A=np.array([['a1' ,'b2' ,'c1', 'f2' ,'d1' ,'e1']
 ,['a1', 'b2', 'c1', 'f2', 'd2' ,'e1']
 ,['a1' ,'b2' ,'c1' ,'f2', 'd1', 'e2']
 ,['a2' ,'b1' ,'c1' ,'f2', 'd1', 'e2']
 ,['a2' ,'b1', 'c1' ,'f2', 'd1', 'e3']])

#Frequency function
def Freq(List):
    List.sort()
    cnt = 1
    for i in range(1, len(List)):
        if (List[i] == List[i - 1]):
            cnt += 1
        else:
            print(List[i - 1] + ' : ' + str(cnt))
            cnt = 1
    print(List[-1] + ' : ' + str(cnt))


#dimension d of each partition
dim = int(np.shape(A)[1]/K)
#row n of each partition
nrow = int(np.shape(A)[0])
d0 = 0
for i in range(K):
    #partition matrix B
    B = A[:,d0:d0 + dim]
    d0 = d0 + dim
    #1 dimension
    for j in range(dim):
        Freq((B[:,j]).tolist())
    #more than 1 dimension
    if(dim>1):
        for ncomb in range(2,dim+1):
            comb=[]
            for k in range(nrow):
                comb.append(list(combinations(B[k,:],ncomb)))
            for l in range(np.shape(comb)[1]):
                comb_list = []
                for m in range(nrow):
                    #join the string
                    comb_ele = comb[m][l][0]
                    for n in range(1,ncomb):
                        comb_ele = " ".join([comb_ele,comb[m][l][n]])
                    comb_list.append(comb_ele)
                Freq(comb_list)
    if (i<K-1):
        print('')


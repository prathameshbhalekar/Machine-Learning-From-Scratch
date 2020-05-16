#   K nearest neighbors is a classification algorithm. It determines k points closest to the data to be classified using eucledian 
#   distance. The data is classified as the type having most number of votes among the k votes.

#   data source:http://mlr.cs.umass.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
import pandas as pd
import numpy as np
from collections import Counter
import random

df=pd.read_csv('data.csv')
df.drop(['id'],1,inplace=True) 
df.replace('?',-99999,inplace=True)
data=df.astype(float).values.tolist()
random.shuffle(data)

test_size=0.2

train_set=data[:int(len(data)*test_size)]
test_set=data[int(len(data)*test_size):]

train={2:[],4:[]}
test={2:[],4:[]}

for i in train_set:
    train[i[-1]].append(i[:-1])
    
for i in test_set:
    test[i[-1]].append(i[:-1])
total=0
correct=0
for i in test:
    for j in test[i]:
        result,confidence=get_type(train,j)
        total+=1
        if(result!=i):
            print(confidence)
        else:
            correct+=1
print(correct/total)

def get_type(train,case,k=5):
    distance=[]
    for i in train:
        for j in train[i]:
            euclidean_distance=np.linalg.norm(np.array(j)-np.array(case))
            distance.append([euclidean_distance,i])
    distance=sorted(distance)
    votes=[i[1] for i in distance[:k]]
    vote_result=Counter(votes).most_common(1)
    result=vote_result[0][0]
    confidence=vote_result[0][1]/k
    return result,confidence

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np

from tempfile import TemporaryFile
import os 
import pickle 
import random
import operator

import math
import numpy as np

#had to install the libraries - did this with 'pip install ______ ie numpy'

#function to get the distance between feature vectors and find neighbours

def getNeighbors(trainingSet, instance, k):
    distances = []      #declare empty list for distance values 
    for i in range(len(trainingSet)): #i is the index of the array/list? trainingSets
        dist = math.dist(trainingSet[i][0], instance[0]) + math.dist(instance[0], trainingSet[i][0]) 

        distances.append(((trainingSet[i][2]),dist))
    distances.sort(key=operator.itemgetter(1))    #key defines the sorting 'criteria'
    neighbours = []
    for i in range(k): #the k value specifies how many neighbours to include
        neighbours.append(distances[i][0]) #adds the nearest neighbours
    return neighbours

def nearestClass(neighbours):
    classVote = {}  #dictionary object stores the type and the number of appearances

    for i in range(len(neighbours)):
        response = neighbours[i]
        if response in classVote:
            classVote[response]+=1
        else:
            classVote[response]=1
        #sorts the items (classVote.items()  based on the second value (itemgetter(1) refers to the 2nd iterable))
        sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse = True)

    return sorter[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][2] == predictions[i]:
            correct+=1
    return 1.0*correct/len(testSet)


directory = "C:/Users/micha/OneDrive - University of Waterloo/Projects/Music Genre Identification/genres/"
#not sure what to put as the directory 

# f = open("my.dat",'wb')     #creating a new file to write the data to wb statnds for write binary
# i=0


# for folder in os.listdir(directory):
#     i+=1
#     if i==11: #folder contains 10 subfolders
#         break
    
#     for file in os.listdir(os.path.join(directory,folder)):
#         (rate,sig) = wav.read(os.path.join(directory,folder,file))     #(samping rate, entire dataset in numpy array)
#         mfcc_feat = mfcc(sig,rate ,winlen=0.020, appendEnergy = False)
#         covariance = np.cov(np.matrix.transpose(mfcc_feat))
#         mean_matrix = mfcc_feat.mean(0)
#         feature = (mean_matrix, covariance, i)
#         pickle.dump(feature, f)

    
# f.close()

# f= open("my.dat" ,'wb')
# i=0

# for folder in os.listdir(directory):
#     i+=1
#     if i==11 :
#         break 	
#     for file in os.listdir(directory+folder):	
#         (rate,sig) = wav.read(directory+folder+"/"+file)
#         mfcc_feat = mfcc(sig,rate ,winlen=0.020, appendEnergy = False)
#         covariance = np.cov(np.matrix.transpose(mfcc_feat))
#         mean_matrix = mfcc_feat.mean(0)
#         feature = (mean_matrix , covariance , i)
#         pickle.dump(feature , f)

# f.close()


dataset = []
def loadDataset(filename):
    with open("my.dat",'rb') as f: #rb for read binary 
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break

split = 0.75
loadDataset("my.dat")
trainingSet = []
testSet = []

for i in range(len(dataset)):
    if random.random() < split :
        trainingSet.append(dataset[i])        
    else:
        testSet.append(dataset[i])

print(len(dataset))
print(len(testSet))
print(len(trainingSet))

#leng = len(testSet)
predictions = []
for i in range(len(testSet)):
    predictions.append(nearestClass(getNeighbors(trainingSet, testSet[i], 4)))


accuracy1 = getAccuracy(testSet, predictions)

print(accuracy1)



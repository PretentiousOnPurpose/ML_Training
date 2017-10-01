'''
Created on Sep 29 ,2017
@author: chawat
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadDataSet(fileName):     
    dataMat = []               
    fr = open(fileName)
    for line in fr.readlines():
        dataMat.append(line.strip().split('\t'))
    return np.array(dataMat).astype(float)

def geoDistance(PtA, PtB):
    return np.sqrt(sum(np.power(PtA - PtB, 2))) 

def init_cluster(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = np.mat(minJ + rangeJ * np.random.rand(k,1))
    return centroids
    
def kMeans(dataSet, k):
    
    m = shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2))) 
                                      
    centroids = init_cluster(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = geoDistance(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust, axis=0)  
    return centroids, clusterAssment


#Main Function()
DataMatrix = loadDataSet('data.txt')
cent , clust = kMeans(dataSet=mat(DataMatrix) , k=4)
DataMatrix = pd.DataFrame(DataMatrix)
DataMatrix = pd.concat([DataMatrix, pd.DataFrame(np.array([X[0] for X in clust]).reshape((80, 2)))], axis=1)

DataMatrix = np.array(DataMatrix)

plt.plot([X[0] for X in DataMatrix if X[2] == 0], [Y[1] for Y in DataMatrix if Y[2] == 0], "r*")
plt.plot([X[0] for X in DataMatrix if X[2] == 1], [Y[1] for Y in DataMatrix if Y[2] == 1], "b*")
plt.plot([X[0] for X in DataMatrix if X[2] == 2], [Y[1] for Y in DataMatrix if Y[2] == 2], "g*")
plt.plot([X[0] for X in DataMatrix if X[2] == 3], [Y[1] for Y in DataMatrix if Y[2] == 3], "y*")

plt.show()



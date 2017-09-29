"""
Created on Fri Sep 29 18:40:28 2017

@author: chawat
"""
import numpy as np
np.random.seed(101)

def loadDataSet(filename):
    DataMatrix = []
    DataPoints = open(filename)
    
    for points in DataPoints.readlines():
        DataMatrix.append(points.strip().split('\t'))
    return np.array(DataMatrix)

def geoDistance(PtA , PtB):
    return np.sqrt(sum(((PtA - PtB)**2)))

# Not a good thing to do. Use KMeans++ instead.
def init_Cluster_Centroid(DataMatrix , K):
    min_vals = []; max_vals = []
    cluster_points = []
    for i in range(DataMatrix.shape[1]):
        min_vals.append(min([PtX[i] for PtX in DataMatrix]))
        max_vals.append(max([PtX[i] for PtX in DataMatrix]))
    
    min_vals = np.array(min_vals).astype(float)
    max_vals = np.array(max_vals).astype(float)
            
    for i in range(K):
        temp = []
        for j in range(DataMatrix.shape[1]):
            np.random.seed(i**2)
            temp.append(min_vals[j] + (max_vals[j]-min_vals[j]) * np.random.randint(0 , max_vals[j] - min_vals[j]))
            
        cluster_points.append(temp)
    
    return np.array(cluster_points)


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets

def plotData(X, y):
    pca = PCA(2)  # project from 64 to 2 dimensions
    projected = pca.fit_transform(X)
    plt.scatter(projected[:, 0], projected[:, 1], edgecolor='none', alpha=0.5, c=y, cmap=plt.cm.get_cmap('spectral', 10))
    plt.legend(y, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.scatter(projected[:, 0], projected[:, 1], edgecolor='none', alpha=0.5)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.show()

def loadData(csvPath):
    data = pd.read_csv(csvPath)
    return data.as_matrix()

def reduceData(data, numOfRows):
    idx = np.random.randint(data.shape[0], size=numOfRows)
    d = data[idx,:]
    c = [row[len(data[0]) - 1] for row in data]
    d = np.delete(d, -1, axis=1)
    return d, c

# we'll play with this function, right now it just randomly picks 20 centers, SAD
def getCandidateCenters(k, data):
    idx = np.random.randint(data.shape[0], size=20)
    return idx # we only need to return the index, not entire row

def initialCenters(k, C):
    idx = np.random.choice(C, size=k) # choose k candidates from C, not data 
    shadowIdx = [x for x in range(0,data.shape[0])  if x not in idx]
    s = data[idx,:] # pick centers from candidate set
    c = data[shadowIdx,:] # remove s from candidate set

    return s, c

# Calculates the distortion for centers S on data
def calculateDistortion(S, data):
    distances = {}
    distortion = 0

    for s in S: #loop over all centers
        for i,pt in enumerate(data): # loop over all points in data
            dist = np.linalg.norm(pt-s) #compute the distance
            if i not in distances: # if the distance between this point and any center has not been computed then add it
                distances[i] = dist
            elif distances[i] > dist: # otherwise check if the distance is less than the previously computed center dist, if so overwrite the previous one
                distances[i] = dist
    for i in distances: # now that we know all the distances between the centers and their closes points, compute the distortion
        distortion += distances[i]
    return distortion

def getOptimalCenters(S, C, data):
    tmpS = S.copy()
    tmpC = C.copy()
    currentDist = calculateDistortion(S, data)
    print(currentDist)
    for i, s in enumerate(tmpS):
        best_s_idx = -1
        for j, c in enumerate(tmpC):
            tmpS[i] = c
            tmpDist = calculateDistortion(tmpS, data)
            if tmpDist < currentDist:
                currentDist = tmpDist
                best_s_idx = j
        if best_s_idx > -1:
            tmpRow = tmpS[i]
            tmpS[i] = tmpC[best_s_idx]
            tmpC[best_s_idx] = tmpRow
    print(currentDist)
    return tmpS, tmpC

k = 5 # the number of centers to pick
# data = loadData('data/data.csv') # load data from csv file
dataset = datasets.load_digits()
# clusterIDs = datasets.load_digits().target_names
X = dataset['data']
y = dataset['target']
# data, clusterIDs = reduceData(data, n) # sample N rows from the data
# C = getCandidateCenters(k, data) # pick candidate centers
# S, C = initialCenters(k, C) # pick k centers from the candidate centers
# O = getOptimalCenters(S, C, data)
plotData(X, y)

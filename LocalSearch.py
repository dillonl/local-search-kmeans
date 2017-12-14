import pandas as pd
import numpy as np

def loadData(csvPath):
    data = pd.read_csv(csvPath)
    return data.as_matrix()

def reduceData(data, numOfRows):
    idx = np.random.randint(data.shape[0], size=numOfRows)
    d = data[idx,:]
    return d

# we'll play with this function, right now it just randomly picks 20 centers, SAD
def getCandidateCenters(data):
    idx = np.random.randint(data.shape[0], size=20)
    return idx # we only need to return the index, not entire row

def initialCenters(k, C):
    idx = np.random.choice(C, size=k) // choose k candidates from C, not data 
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
data = loadData('data/data.csv') # load data from csv file
data = reduceData(data, 1000) # sample N rows from the data
C = getCandidateCenters(data) # pick candidate centers
S, C = initialCenters(k, C) # pick k centers from the candidate centers
O = getOptimalCenters(S, C, data)

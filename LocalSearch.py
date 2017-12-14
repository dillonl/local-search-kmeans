import pandas as pd
import numpy as np
import time

def loadData(csvPath):
    data = pd.read_csv(csvPath)
    return data.as_matrix()

def reduceData(data, numOfRows):
    idx = np.random.randint(data.shape[0], size=numOfRows)
    d = data[idx,:]
    return d

# we'll play with this function, right now it just randomly picks 20 centers, SAD
def getCandidateCenters(idx,j,C):
    #C = np.random.randint(data.shape[0], size=20)
    #print(np.shape(data))
    #print(np.median(data[:,0]))
    m = np.median(data[:,0])
    #print(m)
    #print(j)
    size = np.shape(data)
    #print(size)
    #print("idx",idx)
    less = [x for x in idx if data[x,0] < m]
    more = [x for x in idx if data[x,0] > m]
    #print(type(idx))
    if j < 0 :
        return np.random.randint(data.shape[0], size=1)
    else:
        return np.append(C,[getCandidateCenters(less,j-1,C),getCandidateCenters(more,j-1,C)])
    #print(idx)
    #print(data[idx,0])
    #return C # just need the index of the cadidate centers, now row

def initialCenters(k, C):
    idx = np.random.choice(C, size=k)
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

print "Start time ...",time.clock(),"s"
j = 20 # candidate set size
k = 5 # the number of centers to pick
C = [] # initial candidate centers set
data = loadData('data/data.csv') # load data from csv file
data = reduceData(data, 100) # sample N rows from the data
C = getCandidateCenters(range(data.shape[0]),np.log2(j),C) # pick candidate centers
C = map(int, C)
S, C = initialCenters(k, C) # pick k centers from the candidate centers
O = getOptimalCenters(S, C, data)
print "End time ...",time.clock(),"s"

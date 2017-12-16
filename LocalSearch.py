import sys, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets

def plotData(X, y, k, filename):
    # pca = PCA(2)  # project from 64 to 2 dimensions
    # projected = pca.fit_transform(X)
    colors = ["green", "blue", 'red', "black", 'yellow', 'brown', 'violet', 'orange', 'darkgreen', 'magenta']
    c = []
    for i in y:
        c.append(colors[i])
    # plt.scatter(projected[:, 0], projected[:, 1], edgecolor='none', alpha=0.7, c=c, cmap=plt.cm.get_cmap('spectral', k))
    plt.scatter(X[:, 0], X[:,1], edgecolor='none', alpha=0.7, c=c, cmap=plt.cm.get_cmap('spectral', k))
    # plt.legend(y, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.scatter(projected[:, 0], projected[:, 1], edgecolor='none', alpha=0.5)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plotsDir = 'plots'
    if not os.path.exists(plotsDir):
        os.makedirs(plotsDir)

    plt.savefig(plotsDir + '/' + filename + '.png')
    # plt.show()

def loadData(csvPath):
    data = pd.read_csv(csvPath)
    return data.as_matrix()

def reduceData(data, numOfRows):
    idx = np.random.randint(data.shape[0], size=numOfRows)
    d = data[idx,:]
    c = [row[len(data[0]) - 1] for row in data]
    d = np.delete(d, -1, axis=1)
    return d, c

def getCandidateCentersByLargestDistance(K, data):
    idx = np.random.randint(data.shape[0], size=1)
    centroidIdxs = [idx]
    for k in range(K - 1):
        maxMinDist = 0
        maxMinDistPointIdx = -1
        for i, pt in enumerate(data):
            minDist = 2147483647 #max int value
            for c in centroidIdxs:
                dist = np.linalg.norm(pt-data[c]) #compute the distance
                if dist < minDist:
                    minDist = dist
            if minDist > maxMinDist:
                maxMinDist = minDist
                maxMinDistPointIdx = i
        centroidIdxs.append(maxMinDistPointIdx)

    c = data[centroidIdxs,:]
    shadowIdx = [x for x in range(0,data.shape[0])  if x not in centroidIdxs]
    d = data[shadowIdx,:] # remove s from candidate set
    return d, c # return data and centroids

# we'll play with this function, right now it just randomly picks 20 centers, SAD
def getCandidateCentersByRandom(k, data):
    centroidIdxs = np.random.randint(data.shape[0], size=k)
    c = data[centroidIdxs,:]
    shadowIdx = [x for x in range(0,data.shape[0])  if x not in centroidIdxs]
    d = data[shadowIdx,:] # remove s from candidate set
    return d, c # return data and centroids

def initialCenters(k, C):
    idx = np.random.choice(C, size=k) # choose k candidates from C, not data 
    shadowIdx = [x for x in range(0,data.shape[0])  if x not in idx]
    s = data[idx,:] # pick centers from candidate set
    c = data[shadowIdx,:] # remove s from candidate set

    return s, c

# Calculates the distortion for centers S on data
def calculateDistortion(C, data):
    distances = {}
    distortion = 0

    for c in C: #loop over all centers
        for i,pt in enumerate(data): # loop over all points in data
            dist = np.linalg.norm(pt-c) #compute the distance
            if i not in distances: # if the distance between this point and any center has not been computed then add it
                distances[i] = dist
            elif distances[i] > dist: # otherwise check if the distance is less than the previously computed center dist, if so overwrite the previous one
                distances[i] = dist
    for i in distances: # now that we know all the distances between the centers and their closes points, compute the distortion
        distortion += distances[i]
    return distortion

def calculateDistortion1(C, X):
    y = closestCentroid(X, C)
    # y = assignPointsToCenters(X, C)

def moveCentroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

def closestCentroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def getOptimalCenters(X, C, data):
    # move_centroids(points, closest_centroid(points, c), c)
    print(C)
    tmpX = X.copy()
    tmpC = C.copy()
    currentDist = calculateDistortion(C, X)
    for i, s in enumerate(tmpX):
        best_s_idx = -1
        for j, c in enumerate(tmpC):
            tmpX[i] = c
            tmpDist = calculateDistortion(tmpC, tmpX)
            # print(tmpDist, currentDist)
            if tmpDist < currentDist:
                currentDist = tmpDist
                best_s_idx = j
        if best_s_idx > -1:
            # print('swapping')
            tmpRow = tmpX[i]
            tmpX[i] = tmpC[best_s_idx]
            tmpC[best_s_idx] = tmpRow
    currentDist = calculateDistortion(tmpC, tmpX)
    print(currentDist)
    print(tmpC)
    return tmpX, tmpC

def assignPointsToClusters(X, C):
    y = []
    for i, x in enumerate(X):
        smallestDist = 2147483647
        smallestDistIdx = 0
        for j, c in enumerate(C):
            dist = np.linalg.norm(x-c)
            if dist < smallestDist:
                smallestDist = dist
                smallestDistIdx = j
        y.append(smallestDistIdx)
    return y

def plotAllData(X, y, XLD, cLD, k, label):
    yLD = [1] * len(XLD) + [2] * len(cLD)
    tmpX = []
    for x in XLD: tmpX.append(x)
    for c in cLD: tmpX.append(c)
    XLD = np.array(tmpX)
    yLD = np.array(yLD)
    plotData(XLD, yLD, len(np.unique(yLD)), label + '_centroids_only')

    yLD = assignPointsToClusters(XLD, cLD)
    plotData(XLD, yLD, k, label + '_points_assigned_to_centroids')

    oXLD, oCLD = getOptimalCenters(XLD, cLD, X)
    yOLD = assignPointsToClusters(oXLD, oCLD)
    plotData(oXLD, yOLD, k, label + '_lloyd_opt_clusters')

# data = loadData('data/data.csv') # load data from csv file
# dataset = datasets.load_digits()
# X = dataset['data']
# y = dataset['target']
k = 5
X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=k)
plotData(X, y, k, 'truth_set')
XR, cR = getCandidateCentersByRandom(k, X)
plotAllData(X, y, XR, cR, k, 'random')
XLD, cLD = getCandidateCentersByLargestDistance(k, X)
plotAllData(X, y, XLD, cLD, k, 'largest_distance')

def plotByCentersLargestDist(X, y, k):
    XLD, cLD = getCandidateCentersByLargestDistance(k, X)
    yLD = [1] * len(XLD) + [2] * len(cLD)
    tmpX = []
    for x in XLD: tmpX.append(x)
    for c in cLD: tmpX.append(c)
    XLD = np.array(tmpX)
    yLD = np.array(yLD)
    plotData(XLD, yLD, len(np.unique(yLD)), 'largest_dist_centroids')

    yLD = assignPointsToClusters(XLD, cLD)
    plotData(XLD, yLD, k, 'largest_dist_points')

    oXLD, oCLD = getOptimalCenters(XLD, cLD, X)
    yOLD = assignPointsToClusters(oXLD, oCLD)
    plotData(oXLD, yOLD, k, 'largest_dist_opt_clusters')

# data, clusterIDs = reduceData(data, n) # sample N rows from the data
# C = getCandidateCenters(k, data) # pick candidate centers
# S, C = initialCenters(k, C) # pick k centers from the candidate centers
# O = getOptimalCenters(S, C, data)
# plotData(X, y)

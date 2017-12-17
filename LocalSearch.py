import sys, os
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
from scipy.spatial import distance
from random import randint
import random

def plotDistortionHistogram(truthDistortion, randomDistortion, badPickDistortion, minMaxDistortion, label):
    plotsDir = 'plots'
    if not os.path.exists(plotsDir):
        os.makedirs(plotsDir)

    years = ('truth', 'random', 'bad pick', 'min-max pick')
    visitors = (truthDistortion, randomDistortion, badPickDistortion, minMaxDistortion)
    index = np.arange(len(visitors))
    bar_width = 0.3
    plt.bar(index, visitors, bar_width,  color="blue")
    plt.xticks(index + bar_width / 2, years) # labels get centered

    plt.savefig(plotsDir + '/' + label + '.png')
    plt.clf()
    return 0

def plotData(X, y, k, filename, shouldPCA=False):
    if shouldPCA:
        pca = PCA(2)  # project from 64 to 2 dimensions
        projected = pca.fit_transform(X)
        X = projected
    plt.scatter(X[:, 0], X[:,1], edgecolor='none', alpha=1, c=y, cmap=plt.cm.get_cmap('prism', k))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plotsDir = 'plots'
    if not os.path.exists(plotsDir):
        os.makedirs(plotsDir)

    plt.savefig(plotsDir + '/' + filename + '.png')
    plt.clf()

def loadData(csvPath):
    data = pd.read_csv(csvPath)
    return data.as_matrix()

def reduceData(data, numOfRows):
    idx = np.random.randint(data.shape[0], size=numOfRows)
    d = data[idx,:]
    c = [row[len(data[0]) - 1] for row in data]
    d = np.delete(d, -1, axis=1)
    return d, c

def getDist(p1, p2):
    dist = [(a - b)**2 for a, b in zip(p1, p2)]
    dist = sum(dist)
    return dist

def getCandidateCentersSegment(idx,j,C):
    d = np.random.randint(dim, size=1) # stores the random dimension on which to segment for this call
    m = np.median(data[:,0])

    less = [x for x in idx if data[x,d] < m]
    more = [x for x in idx if data[x,d] > m]

    if j < 0 :
        return np.random.randint(data.shape[0], size=1) # j is the countdown to the base case
    else:
        return np.append(C,[getCandidateCenters(less,j-1,C),getCandidateCenters(more,j-1,C)]) #recursive call to both "sides" of median

def getCandidateCentersByBadPick(k, X):
    idx = randint(0, len(X) - 1)
    firstPt = X[idx]
    c = []
    x = []
    dists = []
    for i, p in enumerate(X):
        dists.append((getDist(p, firstPt), p))
    dists = sorted(dists, key=lambda x: x[0])

    for d in dists:
        if len(c) < k:
            c.append(d[1])
        else:
            x.append(d[1])
    random.shuffle(x)
    c = np.array(c)
    x = np.array(x)
    return x, c # return data and centroids

def getCandidateCentersByLargestDistance(K, data):
    idx = randint(0, len(data) - 1)
    centroidIdxs = [idx]
    tmpC = [0] * K
    tmpC[0] = data[idx]
    tmp = 1
    for k in range(K - 1):
        maxMinDist = 0
        maxMinDistPointIdx = -1
        for i, pt in enumerate(data):
            minDist = 2147483647 #max int value
            for c in centroidIdxs:
                dist = getDist(pt, data[c])
                if dist < minDist:
                    minDist = dist
            if minDist > maxMinDist:
                maxMinDist = minDist
                maxMinDistPointIdx = i
        centroidIdxs.append(maxMinDistPointIdx)
        tmpC[tmp] = data[maxMinDistPointIdx]
        tmp += 1

    tmpX = []
    for i,x in enumerate(data):
        if i not in centroidIdxs:
            tmpX.append(x.copy())
    tmpX = np.array(tmpX)
    tmpC = np.array(tmpC)
    return tmpX, tmpC# return data and centroids

# we'll play with this function, right now it just randomly picks k centers, SAD
def getCandidateCentersByRandom(k, data):
    centroidIdxs = np.random.randint(data.shape[0], size=k)
    c = data[centroidIdxs,:]
    shadowIdx = [x for x in range(0,data.shape[0])  if x not in centroidIdxs]
    d = data[shadowIdx,:] # remove s from candidate set
    return d, c # return data and centroids

# Calculates the distortion for centers S on data
def calculateDistortion(C, X):
    Y = closestCentroid(X, C)
    totalDist = 0
    for i,y in enumerate(Y):
        c = C[y]
        x = X[i]
        totalDist += getDist(x, c)
    return totalDist

def closestCentroid(X, C):
    distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def getOptimalCenters(X, C):
    currentDist = calculateDistortion(C, X)
    print('start',currentDist)
    for i in range(len(X)):
        x = X[i].copy()
        best = {}
        for j in range(len(C)):
            c = C[j].copy()
            X[i] = c.copy()
            C[j] = x.copy()
            tmpDist = calculateDistortion(C, X)
            if tmpDist < currentDist:
                currentDist = tmpDist
                best = {'x': X.copy(), 'c': C.copy()}
            X[i] = x.copy() # put them back, we'll change it back if it's a lower distortion
            C[j] = c.copy()
        if 'x' in best:
            X = best['x']
            C = best['c']
    currentDist = calculateDistortion(C, X)
    print('end',currentDist)
    return X, C

def plotAllData(X, y, k, label, shouldPCA=False):
    yIdxs = closestCentroid(X, y)
    plotData(X, yIdxs, k, label + '_points_assigned_to_centroids', shouldPCA)

    dist1 = calculateDistortion(y, X)

    oX, oC = getOptimalCenters(X, y)
    yO = closestCentroid(oX, oC)

    dist2 = calculateDistortion(oC, oX)

    plotData(oX, yO, k, label + '_lloyd_opt_clusters', shouldPCA)

    yIdxs = [1] * len(X) + [2] * len(y)
    tmpX = []
    for x in X: tmpX.append(x)
    for c in y: tmpX.append(c)
    X = np.array(tmpX)
    yIdxs = np.array(yIdxs)
    plotData(X, yIdxs, len(np.unique(yIdxs)), label + '_centroids_only', shouldPCA)

    oY = [1] * len(oX) + [2] * len(oC)
    tmpX = []
    for x in oX: tmpX.append(x)
    for c in oC: tmpX.append(c)
    oX = np.array(tmpX)
    oY = np.array(oY)
    plotData(oX, oY, len(np.unique(oY)), label + '_centroids_only_after_opt', shouldPCA)
    return dist1, dist2

n = 1000
centers = np.array([[-2, -2], [1,1], [5,5], [-2,5], [3, -4]])
k = len(centers)
X, y = datasets.make_blobs(n_samples=n, centers=centers, n_features=2)

yIdxs = closestCentroid(X.copy(), centers.copy())
plotData(X, yIdxs, k, 'truth_set')

truthDistortion = calculateDistortion(centers, X.copy())

XR, cR = getCandidateCentersByRandom(k, X.copy())
randomPickDistortions = plotAllData(XR.copy(), cR.copy(), k, 'random', False)

XLD, cLD = getCandidateCentersByLargestDistance(k, X.copy())
minMaxDistortions = plotAllData(XLD, cLD, k, 'largest_distance', False)

XBP, cBP = getCandidateCentersByBadPick(k, X.copy())
badPickDistortions = plotAllData(XBP.copy(), cBP.copy(), k, 'bad_pick', False)

plotDistortionHistogram(truthDistortion, randomPickDistortions[0], badPickDistortions[0], minMaxDistortions[0], 'before_lloyds')
plotDistortionHistogram(truthDistortion, randomPickDistortions[1], badPickDistortions[1], minMaxDistortions[1], 'after_lloyds')

dataset = datasets.load_digits()
X = dataset['data']
y = dataset['target']

plotData(X, y, k, 'digits_truth_set', True)

XLD, cLD = getCandidateCentersByBadPick(k, X)
badPickDistortions = plotAllData(XLD, cLD, k, 'digits_bad_pick', True)

XR, cR = getCandidateCentersByRandom(k, X)
randomPickDistortion = plotAllData(XR, cR, k, 'digits_random', True)

XLD, cLD = getCandidateCentersByLargestDistance(k, X)
minMaxDistortions = plotAllData(XLD, cLD, k, 'digits_largest_distance', True)

plotDistortionHistogram(truthDistortion, randomPickDistortion[0], badPickDistortions[0], minMaxDistortions[0], 'digits_before_lloyds')
plotDistortionHistogram(truthDistortion, randomPickDistortion[1], badPickDistortions[1], minMaxDistortions[1], 'digits_after_lloyds')

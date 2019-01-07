# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 17:39:23 2018

@author: Suryam Sharma
"""

from sklearn import metrics, svm, linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
import pandas as pd
import numpy as np
import math
import operator

trainingdata = 'GenomeTrainXY.txt'
testingdata = "GenomeTestX.txt"
data = pd.read_csv('GenomeTrainXY.txt', header=-1).values
testData = pd.read_csv("GenomeTestX.txt", header=-1).values
headerinfo = data[0]
#headerinfo = row1 = genome(column) header
classlabelinfo = list(set(headerinfo))
#unique genome labels
classlbl, classlblcnt = np.unique(headerinfo, return_counts=True)
#classlbl = class label , classlblcount = genome lbl count
classlblcntdict = dict(zip(classlbl, classlblcnt))
#class label, class lbl count
genome_size = len(headerinfo)
#number of columns
k_groupsize = len(classlbl)
#number of classes
df = pd.DataFrame(data)
dftranspose = df.transpose()
#dataframe format and its transpose
fscores = pd.DataFrame()
fscorenumval = None
fscoredenom = None
fscorenumdf = pd.DataFrame()
fscoredenomdf = pd.DataFrame()
#empty dataframes and variables

#calculate mean of all features for a specific class label
featuremeandata = df.transpose().groupby(dftranspose[:][0]).mean()
#row-wise mean of all genomes along same feateures
featuremeandata = featuremeandata.loc[:, 1:]
centroidData = featuremeandata.transpose().values
#ndarray [row, collbl]

#calculate variance of all features for a specific class label
featurevardata = df.transpose().groupby(dftranspose[:][0]).var()
featurevardata = featurevardata.loc[:, 1:]
#dataframe [genome lbl, variance along the same feature

#calculate average of each of the feature
featureavg = df.mean(axis=1) # y-axis
#average of values in each gene vector
featureavgdata = pd.DataFrame(featureavg).transpose()
print(featureavgdata)
featureavgdata = featureavgdata.loc[:, 1:]
#store the gene vector avg data wothout column labels

#calculate f-score numerator
def getfeaturemeandata(classlblval, val):
    meanrowdata = pd.DataFrame()
    meanrowdatabyvalue = pd.DataFrame()
    meannumdata = pd.DataFrame()
    #for number of features labels
    for i in range(k_groupsize):
        if featuremeandata.index[i] == classlblval:
            meanrowdata = pd.DataFrame(featuremeandata.loc[classlblval, :]).transpose()
            meannumdata = meanrowdata.values - featureavgdata.values
            meanrowdatabyvalue = val*(pd.DataFrame((meannumdata)**2))
    return meanrowdatabyvalue

#calculate f-score denominator
def getfeaturevardata(classlblval, val):
    varrowdata = pd.DataFrame()
    varrowdatabyvalue = pd.DataFrame()
    #for number of features labels
    for i in range(k_groupsize):
        if featurevardata.index[i] == classlblval:
            varrowdata = pd.DataFrame(featurevardata.loc[classlblval, :]).transpose()
            varrowdatabyvalue = pd.DataFrame(((val-1)*varrowdata))
    return varrowdatabyvalue

def pickGenome():
    for key, value in classlblcntdict.items():
        # constructing fscore numerator and denominator vector
        if list(classlblcntdict.keys()).index(key) == 0:
            fscorenumdf = getfeaturemeandata(key, value)
            fscoredenomdf = getfeaturevardata(key, value)
        else:
            testnumdf = getfeaturemeandata(key, value)
            testdenomdf = getfeaturevardata(key, value)
            fscorenumdf = pd.concat([fscorenumdf, testnumdf], axis=0, ignore_index=True)
            fscoredenomdf = pd.concat([fscoredenomdf, testdenomdf], axis=0, ignore_index=True)

    # calculating all the f-score numerator vector by summing mean data and dividing by k-1
    fscorenumdata = ((pd.DataFrame(fscorenumdf.sum(axis=0)).transpose())/(k_groupsize - 1))
    #print(fscorenumdata)

    # calculating all the f-score denominator vector by summing var data and dividing by n-k
    fscorevardata = ((pd.DataFrame(fscoredenomdf.sum(axis=0)).transpose())/(genome_size - k_groupsize))
    #print(fscorevardata)

    fscorenumdata.columns = range(fscorenumdata.shape[1])
    fscorevardata.columns = range(fscorevardata.shape[1])

    #f-score
    fscores =  (fscorenumdata / fscorevardata).transpose()
    fscores.columns = ['Genome_fscore']
    #print(fscores)

    fscoreSorted = fscores.sort_values(by='Genome_fscore', ascending=False)
    print("========== Sorted fscores below ==============\n")
    print(fscoreSorted)

    top100fscoreindices = fscoreSorted.head(100).index.tolist()
    top100fscoreindices = [(x + 1) for x in top100fscoreindices]# bcos of class labels
    print("\n========== Top 100 fscore indices below ==============\n")
    print(top100fscoreindices)
    writeGenomes(top100fscoreindices)
    writeTestGenomes(top100fscoreindices)

def writeGenomes(genomeList):
    file = open("GenomeTop100TrainData.txt", "w")
    r1, = data[0][:].shape
    rx,cx = data.shape
    for i in range(0, r1):
        file.write(str(int(data[0][:][i])))
        if (i < r1 - 1):
            file.write(',')
    file.write("\n")
    for a in genomeList:
        for b in range(0, cx):
            file.write(str(data[a][:][b]))
            if(b < cx - 1):
                file.write(',')
        file.write("\n")
    file.close()

def writeTestGenomes(genomeList):
    file = open("GenomeTop100TestData.txt", "w")
    rx,cx = testData.shape
    for a in genomeList:
        for b in range(0, cx):
            file.write(str(testData[a-1][:][b]))
            if(b < cx - 1):
                file.write(',')
        file.write("\n")
    file.close()

pickGenome()

def data_classifier(classifier):
    if (classifier == "KNN"):
        #storeData(Xtrain, ytrain, Xtest, ytest, classifier)
        file1 = pd.read_csv('GenomeTop100TrainData.txt', header=-1)
        Xtrain = file1.loc[1:,:].transpose().values
        ytrain = file1.loc[0,:].transpose().values
        file2 = pd.read_csv('GenomeTop100TestData.txt', header=-1)
        Xtest = file2.transpose().values
        knneighbors = KNeighborsClassifier(n_neighbors=5)
        knneighbors.fit(Xtrain, ytrain)
        # calculating prediction
        predictions = knneighbors.predict(Xtest)
        print('\n KNN Predictions: ', predictions)

    elif (classifier == "Centroid"):
        file1 = pd.read_csv('GenomeTop100TrainData.txt', header=-1)
        Xtrain = file1.loc[1:,:].transpose().values
        ytrain = file1.loc[0,:].transpose().values
        file2 = pd.read_csv('GenomeTop100TestData.txt', header=-1)
        Xtest = file2.transpose().values
        centroid = NearestCentroid()
        centroid.fit(Xtrain, ytrain)
        # calculating prediction
        predictions = centroid.predict(Xtest)
        print('\n Centroid predictions: ', predictions)

    elif (classifier == "SVM"):
        file1 = pd.read_csv('GenomeTop100TrainData.txt', header=-1)
        Xtrain = file1.loc[1:,:].transpose().values
        ytrain = file1.loc[0,:].transpose().values
        file2 = pd.read_csv('GenomeTop100TestData.txt', header=-1)
        Xtest = file2.transpose().values
        svmclassifier = svm.LinearSVC()
        svmclassifier.fit(Xtrain, ytrain)
        # calculating prediction
        predictions = svmclassifier.predict(Xtest)
        print('\n SVM Predictions: ',predictions)

    elif(classifier == "Linear Regression"):
        file1 = pd.read_csv('GenomeTop100TrainData.txt', header=-1)
        Xtrain = file1.loc[1:,:].transpose().values
        ytrain = file1.loc[0,:].transpose().values
        file2 = pd.read_csv('GenomeTop100TestData.txt', header=-1)
        Xtest = file2.transpose().values
        lm = linear_model.LinearRegression()
        lm.fit(Xtrain,ytrain)
        # calculating prediction
        predictions = lm.predict(Xtest)
        print('\n LR Predictions: ',predictions)

knn_score = data_classifier("KNN")
centroid_score = data_classifier("Centroid")
svm_score = data_classifier("SVM")
linear_score = data_classifier("Linear Regression")

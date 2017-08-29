from math import log
import numpy as np
import operator
# def createData():
#     return np.array([[1 , 1, 'yes'] , [1 , 1, 'yes'] , [1 , 0 , 'no'] , [0 , 1 , 'no'] , [0 , 1, 'no']]) , ['no surfacing' , 'flippers']

def shannonEntropy(Data):
    classOcurr = {}
    for X in Data:
        if X[-1] not in classOcurr.keys():
            classOcurr[X[-1]] = 0
            classOcurr[X[-1]]   += 1
        else:
            classOcurr[X[-1]] += 1
    shanEntropy = 0.0

    for X in classOcurr.keys():
        px = classOcurr[X]/len(Data)
        shanEntropy -= px * log(px , 2)
    return shanEntropy

def splitFeature(Data , axis , val):
    splitData = []
    for X in Data:
        if X[axis] == val:
            temp = list(X[ : axis])
            temp.extend(list(X[axis+1:]))
            splitData.append(temp)
    return splitData

def majorityList(classList):
    classes = {}
    for val in classList:
        if val not in classes.keys():
            classes[val] = 0
        classes[val] += 1
    return sorted(classes.items() , key=operator.itemgetter(1) , reverse=True)[0][0]

def bestFeatToSplit(Data):
    num = len(Data[0]) - 1
    bastEnt = shannonEntropy(Data)
    bastInfoGain = 0.0 ; BestFeat = -1
    for Xs in range(num):
        X = [Y[Xs] for Y in Data]
        uniX = set(np.array(X))
        newEnt = 0
        for val in uniX:
            ds = splitFeature(Data , Xs , val)
            px = len(ds)/len(Data)
            newEnt += px * shannonEntropy(ds)
            infoGain = bastEnt - newEnt
        if (infoGain > bastInfoGain):
            bastInfoGain = infoGain
            BestFeat = Xs
    return BestFeat

def createTree(Data , labels):
    classList = [labels[-1] for labels in Data]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(Data[0]) == 1:
        return majorityList(classList)
    bestFeature = bestFeatToSplit(Data)
    myTree = {labels[bestFeature]: {} }
    featValues = [values[bestFeature] for values in Data]
    uniFeatValues = set(featValues)
    for val in uniFeatValues:
        sub = np.array(list(labels[:bestFeature]) + list(labels[bestFeature + 1:]))
        myTree[labels[bestFeature]][val] = createTree(splitFeature(Data , bestFeature , val) , sub)
    print(myTree.keys())
    return myTree

def classify(InTree , featLabels , testVec):
    firstFeat = list(InTree.keys())[0]
    secDict = InTree[firstFeat]
    firstIndex = list(featLabels).index(firstFeat)
    classlabel = 0
    for keys in list(secDict.keys()):
        if keys == testVec[firstIndex]:
            if type(secDict[keys]).__name__ == "dict":
                classlabel = classify(secDict[keys] , (featLabels[:firstIndex] + featLabels[firstIndex+1:]) , (testVec[:firstIndex] + testVec[firstIndex+1:]))
            else:
                classlabel = secDict[keys]

    return classlabel

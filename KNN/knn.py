# kNN Implemenation
import numpy as np
from collections import Counter

min_val = []
max_val= []

def classifyPoint(InX , dataSet , labels , k):

        Distances = []
        for X in dataSet:
            Distances.append(distanceCal(InX , X))
        finLabelInd = np.array(Distances).argsort()
        kNN = []
        for a in range(k):
            kNN.append(labels[finLabelInd[a]])
        return Counter(kNN).most_common()[0][0]

def distanceCal(inX , currX):

    x =  (inX[0] - currX[0]) * (inX[0] - currX[0]) + (inX[1] - currX[1]) *  (inX[1] - currX[1]) + (inX[2] - currX[2]) * (inX[2] - currX[2])
    return np.sqrt(np.abs(x))

def normalizeData(Data):
    global min_val
    global max_val
    min_val = Data.min(0)
    max_val = Data.max(0)
    for a in Data:
        for b in range(len(a)):
            a[b] = (a[b] - min_val[b]) / (max_val[b] - min_val[b]);
    return Data

def Matrix(filename):
    file = open(filename, "r")
    numLines = len(file.readlines())
    label = []
    lines = []
    file.close()

    file = open(filename, "r")
    for a in range(numLines):
        temp = file.readline().strip().split("\t")
        lines.append(temp[:3])
        label.append(temp[3])
    lines = np.array(lines).astype(float)
    return normalizeData(lines), label
#


def train_test_split(X , y , test_size):
    return X[:int((1-test_size)*len(X)) , :] , y[:int((1-test_size)*len(y))] , X[int((1-test_size)*len(X)): , :] , y[int((1-test_size)*len(y)):]

def trainNtest(xtr , ytr , xts , yts):
    success = 0
    for pts, ind in zip(xts , range(len(xtr))):
        if classifyPoint(pts[:3], xtr[: , :3] , ytr , 10) == yts[ind]:
             success += 1
        if 1:
            print("Pre:", classifyPoint(pts[:3], xtr[:, :3], ytr, 10), "Ans:", yts[ind])
    print("Error: " , (100 - (success*100)/len(yts)) , "%")

#End

def normalInput(Input):
    for a , ind in zip(Input , range(len(Input))):
        Input[ind] = (a - min_val[ind])/(max_val[ind] - min_val[ind])
    return np.array(Input)
def isMyMate(Data , label):
    amt = float(input("His income per m: "))
    game = float(input("Percent of time on Gaming: "))
    ice = float(input("Ice cream consumed (in Litres): "))
    nmip =  normalInput(np.array([amt, game, ice]))
    print("----------Result----------")
    res = classifyPoint(nmip, Data, label, 5)
    print("You might" , res)
    return res


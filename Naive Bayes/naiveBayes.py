import numpy as np

def loadDataset():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList ,classVec
def createVocabList(postingList):
    vocabList = set([])
    for vect in postingList:
        vocabList = vocabList | set(vect)
    return list(vocabList)
def setofWords2Vec(msgList , vocabList):
    resVect = [0]*len(vocabList)
    for words in vocabList:
        if words in msgList:
            resVect[vocabList.index(words)] = 1
    return resVect

def trainNB(postList , classVec , myVocab):
    p0s = []; p1s = []

    for clsInd in range(len(classVec)):
        if classVec[clsInd] == 1:
            tem = setofWords2Vec(postList[clsInd],myVocab)
            p1s.append(tem)
        else:
            tem = setofWords2Vec(postList[clsInd],myVocab)
            p0s.append(tem)
    P0V = np.array([1] * len(p0s[0])); P0D = 0.0
    P1V = np.array([1] * len(p1s[0])); P1D = 0.0
    for a in range(len(p0s)):
        P0V += np.array(p0s[a]);	
        P0D += sum(np.array(p0s[a]))
    for a in range(len(p1s)):
        P1V += np.array(p1s[a]); 
        P1D += sum(np.array(p1s[a]))
    return np.log((P0V/P0D).reshape(1,len(P0V)))[0] , np.log((P1V/P1D).reshape(1,len(P1V)))[0]

def PAbusive(classLabels):
	return np.log(sum(classLabels)/len(classLabels))

def classifyNB(InV , P0V, P1V , PAb):
	P1 = sum(InV*P1V) + PAb
	P0 = sum(InV*P0V) + 1-PAb
	if P1 > P0:
		return 1
	return 0

def testNB(InV):
	postIist , clsLabels = loadDataset()
	myVocab = createVocabList(postIist)
	p0 , p1 = trainNB(postIist, clsLabels , myVocab)
	pab = PAbusive(clsLabels)
	InX = setofWords2Vec(InV, myVocab)
	if classifyNB(InX, p0 , p1, pab) == 1:
		return 'Abusive'
	return 'Clean'
	

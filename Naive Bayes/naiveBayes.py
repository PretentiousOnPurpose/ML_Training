#----------Start Ab
import numpy as np
import re

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

def bagofWords2Vec(msgList , vocabList):
    resVect = [0]*len(vocabList)
    for words in vocabList:
        if words in msgList:
            resVect[vocabList.index(words)] += 1
    return resVect


def trainNB(postList , classVec , myVocab):
    p0s = []; p1s = []

    for clsInd in range(len(classVec)):
        if classVec[clsInd] == 1:
            p1s.append(setofWords2Vec(postList[clsInd],myVocab))
        else:
            p0s.append(setofWords2Vec(postList[clsInd],myVocab))

    P0V = np.array([1] * len(p0s[0])); P0D = 2.0
    P1V = np.array([1] * len(p1s[0])); P1D = 2.0
    for a in range(len(p0s)):
        P0V += np.array(p0s[a]);	
        P0D += sum(np.array(p0s[a]))
    for a in range(len(p1s)):
        P1V += np.array(p1s[a]); 
        P1D += sum(np.array(p1s[a]))
    return np.log((P0V/P0D).reshape(1,len(P0V)))[0] , np.log((P1V/P1D).reshape(1,len(P1V)))[0],P1V

#Synonymous for PSpam()
def PAbusive(classLabels):
	return float(sum(classLabels))/float(len(classLabels))

def classifyNB(InV , P0V, P1V , PAb):	
	P1 = sum(InV*P1V) + PAb
	P0 = sum(InV*P0V) + 1-PAb
	if P1 > P0:
		return 1
	return 0

def testNB(InV):
	postIist , clsLabels = loadDataset()
	myVocab = createVocabList(postIist)
	p0 , p1 ,ten= trainNB(postIist, clsLabels , myVocab)
	print(ten)
	pab = PAbusive(clsLabels)
	InX = setofWords2Vec(InV, myVocab)
	if classifyNB(InX, p0 , p1, pab) == 1:
		return 'Abusive'
	return 'Clean'

def emailTextParser(email):
	listOfWords = re.split(r'\w*' ,email)
	return [words for words in listOfWords if len(words) > 2]

def SpamOrHam(InV):
	InFile = emailTextParser(open(InV).read())	
	Docs = []
	cls = []
	for i in range(1, 26):
		File = emailTextParser(open('email/ham/%d.txt' % i).read())
		Docs.append(File)
		cls.append(0)
		File = emailTextParser(open('email/spam/%d.txt' % i).read())
		Docs.append(File)
		cls.append(1)

	myVocab = createVocabList(Docs)
	P0V = np.array([1]*len(myVocab));	P0D = 1
	P1V = np.array([1]*len(myVocab)); P1D = 1
	InV_Vect = setofWords2Vec(InV , myVocab)
	InV_Vect += np.array([1]*len(myVocab))
	for ind in cls:
		if ind == 1:
			P1V += setofWords2Vec(Docs[cls.index(ind)] , myVocab)
			P1D += sum(P1V)	
		else:
			P0V += setofWords2Vec(Docs[cls.index(ind)] , myVocab)
			P0D += sum(P0V)		
	PAb = PAbusive(cls)
	return InVProb(InV_Vect , P0V , P1V , PAb)

def InVProb(InV_Vect , P0V , P1V , PAb):	
	p0 = sum(InV_Vect*(np.log(P0V) + np.log(1 - PAb)))
	p1 = sum(InV_Vect*(np.log(P1V) + np.log(PAb)))
	if p1 > p0:
		return 'SPAM'
	return 'HAM'

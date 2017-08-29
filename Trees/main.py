import trees
Data =  []
labels = ['Outlook' , 'Temperatures' , 'Humidity' , 'Wind' , 'Play']
file = open("data.txt" , "r")
numLines = len(file.readlines())
file.seek(0)

for i in range(numLines):
    Data.append(file.readline().strip().split("\t"))
file.close()

inV = ['sunny', 'mild' , 'normal' ,'strong']

myTree = trees.createTree(Data , labels)
print(trees.classify(myTree , labels , inV))

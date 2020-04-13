"""
  Author: Yan Zichu
  StudentID: 300476924
  Victoria University of Wellington
"""

import numpy as np
import pandas as pd


def calEnt(dataSet):
    """
    function：Calculate Shannon entropy ( The higher the entropy, the more information can be transmitted.)
    :parameter：
        dataSet：original dataset
    :return：
        ent: Shannon entropy
    """
    n = dataSet.shape[0]                              # Total number of rows in the dataset
    iset = dataSet.iloc[:,-1].value_counts()          # All categories of labels
    p = iset/n                                        # The percentage of each type of label
    ent = (-p * np.log2(p)).sum()                     # Calculate information entropy ( Formula : '- sum(P * log(P)')
    return ent


# Select the best column to shard
def bestSplit(dataSet):
    """
    function：Select the best data set segmentation column according to information gain
    :parameter：
        dataSet：original dataset
    :return：
        axis:The index of the best cut column of a dataset
    """
    baseEnt = calEnt(dataSet)                                # Calculate the original entropy
    bestGain = 0                                             # Initialize the information gain
    axis = -1                                                # Initialize the best split column, label column

    for i in range(dataSet.shape[1]-1):                      # Loop through each column of features
        levels = dataSet.iloc[:,i].value_counts().index      # Extract all values of the current column
        ents = 0                                             # Initialize the information entropy of child nodes
        for j in levels:                                     # Loop through each value in the current column
            childSet = dataSet[dataSet.iloc[:,i]==j]         # A dataframe of a certain child node
            ent = calEnt(childSet)                           # Calculate the information entropy of a certain child node
            ents += (childSet.shape[0]/dataSet.shape[0])*ent # Calculate the information entropy of the current column
        # print(f'the information entropy of row{i} if {ents}')
        infoGain = baseEnt-ents                              # Calculate the information gain of the current column
        # print(f'the infomation gain of row{i} is {infoGain}')
        if (infoGain > bestGain):
            bestGain = infoGain                              # Select maximum information gain
            axis = i                                         # The index of the column where the maximum information gain is
    return axis


def mySplit(dataSet,axis,value):
    """
    function：Divide the data set according to the given column
    :parameter：
        dataSet：original dataset
        axis：The specified column index
        value：The specified attribute value
    :return：
        redataSet：The data set divided according to the specified column index and attribute value
    """
    col = dataSet.columns[axis]
    redataSet = dataSet.loc[dataSet[col]==value,:].drop(col,axis=1)

    return redataSet


def createTree(dataSet):
    """
    function：Split the data set based on the maximum information gain and construct the decision tree recursively
    :parameter：
        dataSet：original dataset（I make the last column is the label）
    :return：
        myTree：tree of Dictionary form
    """
    featlist = list(dataSet.columns)                          # Extract all the columns of the data set
    classlist = dataSet.iloc[:,-1].value_counts()             # Get the last column of class labels
    # Determine whether the maximum number of labels is equal to the number of rows in the data set, or whether the data set has only one column
    if classlist[0]==dataSet.shape[0] or dataSet.shape[1] == 1:
        return classlist.index[0]                             # If yes, return class label
    # print(dataSet)
    axis = bestSplit(dataSet)                                 # Determine the index of the current best split column
    bestfeat = featlist[axis]                                 # Get the features corresponding to the index
    myTree = {bestfeat:{}}                                    # Store tree information in a dictionary nesting manner
    del featlist[axis]                                        # Delete current feature
    valuelist = set(dataSet.iloc[:,axis])                     # Extract all attribute values of the best split column
    for value in valuelist:                                   # Recursively build tre for each attribute value
        myTree[bestfeat][value] = createTree(mySplit(dataSet,axis,value))
    return myTree


def classify(inputTree,labels, testVec):
    """
    function：Classify a test case
    :parameter：
        inputTree：The decision tree that has been generated
        labels：Store the selected optimal feature label
        testVec：Test data list, the sequence corresponds to the original data set
    :return：
        classLabel：Classification results
    """
    firstStr = next(iter(inputTree))                   # get the first node of decision tree
    secondDict = inputTree[firstStr]                   # Next dictionary
    featIndex = labels.index(firstStr)                 # The index of the column where the first node is
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict :
                classLabel = classify(secondDict[key], labels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def acc_classify(train,test):
    """
    function：Make predictions on the test set and return the predicted results
    :parameter：
        train：trainingSet
        test：testSet
    返回：
        test：Predict a well-classified test set
    """
    inputTree = createTree(train)                       # Generate a tree based on the test set
    labels = list(train.columns)                        # All column names in the data set
    result = []
    for i in range(test.shape[0]):                      # Loop through each piece of data in the test set
        testVec = test.iloc[i,:-1]                      # An instance in the test set
        # print(testVec)
        classLabel = classify(inputTree,labels,testVec) # Predict the classification of this instance
        result.append(classLabel)                       # Append the classification results to the result list
    test['predict']=result                              # Appends the prediction to the last column of the test set

    acc = (test.iloc[:,-1]==test.iloc[:,-2]).mean()     # Calculate accuracy
    # print(test.iloc[:,-1])
    # print(test.iloc[:,-2])
    print(f'The model prediction accuracy: {acc}')
    return test


def tenFoldstest():

    trainingSet = pd.read_csv("/Users/Patrick/Desktop/trainSet/hepatitis-training-run-1.csv")
    testSet = pd.read_csv("/Users/Patrick/Desktop/testSet/hepatitis-test-run-1.csv")
    baselineClassifier("/Users/Patrick/Desktop/testSet/hepatitis-test-run-1.csv")
    num1 = acc_classify(trainingSet,testSet)

    trainingSet = pd.read_csv("/Users/Patrick/Desktop/trainSet/hepatitis-training-run-2.csv")
    testSet = pd.read_csv("/Users/Patrick/Desktop/testSet/hepatitis-test-run-2.csv")
    baselineClassifier("/Users/Patrick/Desktop/testSet/hepatitis-test-run-2.csv")
    num2 =acc_classify(trainingSet, testSet)

    trainingSet = pd.read_csv("/Users/Patrick/Desktop/trainSet/hepatitis-training-run-3.csv")
    testSet = pd.read_csv("/Users/Patrick/Desktop/testSet/hepatitis-test-run-3.csv")
    baselineClassifier("/Users/Patrick/Desktop/testSet/hepatitis-test-run-3.csv")
    num3 = acc_classify(trainingSet, testSet)

    trainingSet = pd.read_csv("/Users/Patrick/Desktop/trainSet/hepatitis-training-run-4.csv")
    testSet = pd.read_csv("/Users/Patrick/Desktop/testSet/hepatitis-test-run-4.csv")
    baselineClassifier("/Users/Patrick/Desktop/testSet/hepatitis-test-run-4.csv")
    num4 = acc_classify(trainingSet, testSet)

    trainingSet = pd.read_csv("/Users/Patrick/Desktop/trainSet/hepatitis-training-run-5.csv")
    testSet = pd.read_csv("/Users/Patrick/Desktop/testSet/hepatitis-test-run-5.csv")
    baselineClassifier("/Users/Patrick/Desktop/testSet/hepatitis-test-run-5.csv")
    num5 = acc_classify(trainingSet, testSet)

    trainingSet = pd.read_csv("/Users/Patrick/Desktop/trainSet/hepatitis-training-run-6.csv")
    testSet = pd.read_csv("/Users/Patrick/Desktop/testSet/hepatitis-test-run-6.csv")
    baselineClassifier("/Users/Patrick/Desktop/testSet/hepatitis-test-run-6.csv")
    num6 = acc_classify(trainingSet, testSet)

    trainingSet = pd.read_csv("/Users/Patrick/Desktop/trainSet/hepatitis-training-run-7.csv")
    testSet = pd.read_csv("/Users/Patrick/Desktop/testSet/hepatitis-test-run-7.csv")
    baselineClassifier("/Users/Patrick/Desktop/testSet/hepatitis-test-run-7.csv")
    num7 = acc_classify(trainingSet, testSet)

    trainingSet = pd.read_csv("/Users/Patrick/Desktop/trainSet/hepatitis-training-run-8.csv")
    testSet = pd.read_csv("/Users/Patrick/Desktop/testSet/hepatitis-test-run-8.csv")
    baselineClassifier("/Users/Patrick/Desktop/testSet/hepatitis-test-run-8.csv")
    num8 = acc_classify(trainingSet, testSet)

    trainingSet = pd.read_csv("/Users/Patrick/Desktop/trainSet/hepatitis-training-run-9.csv")
    testSet = pd.read_csv("/Users/Patrick/Desktop/testSet/hepatitis-test-run-9.csv")
    baselineClassifier("/Users/Patrick/Desktop/testSet/hepatitis-test-run-9.csv")
    num9 = acc_classify(trainingSet, testSet)


def baselineClassifier(test_file_path):
    testSet = pd.read_csv(test_file_path)
    outcomes = testSet.iloc[:,-1]
    if len(outcomes) < 1:
        raise Exception('training data does not have any instances')
    outcomes2 =[]
    for i in range(1,len(outcomes)):
        outcomes2.append(outcomes.iloc[i])
    # print(outcomes2)
    majority_class = max(set(outcomes2), key=outcomes2.count)
    # print(majority_class)
    prob = outcomes2.count(majority_class) / len(outcomes2)
    print('Baseline classifier accuracy: ' + str(prob))
    return prob


#导入相应的包
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz

def drawTree():
    dataSet = pd.read_csv("/Users/Patrick/Desktop/hepatitis-training2.CSV")

    #特征
    Xtrain = dataSet.iloc[:,:-1]
    #标签
    Ytrain = dataSet.iloc[:,-1]
    labels = Ytrain.unique().tolist()
    Ytrain = Ytrain.apply(lambda x: labels.index(x))  #将本文转换为数字

    #绘制树模型
    clf = DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(Xtrain, Ytrain)

    tree.export_graphviz(clf)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graphviz.Source(dot_data)

    #给图形增加标签和颜色
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=['FEMALE','STEROID','ANTIVIRALS','FATIGUE','MALAISE','ANOREXIA','BIGLIVER','FIRMLIVER','SPLEENPALPABLE','SPIDERS','ASCITES','VARICES','BILIRUBIN','SGOT','HISTOLOGY','Class'],
                                    class_names=['live', 'die'],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graphviz.Source(dot_data)
    #利用render方法生成图形
    graph = graphviz.Source(dot_data)
    graph.render("hepatitis2")

if __name__ == '__main__':

    trainingSet = pd.read_csv("/Users/Patrick/Desktop/hepatitis-training2.CSV")
    testSet = pd.read_csv("/Users/Patrick/Desktop/hepatitis-test2.CSV")

    # Compare baseline and DT program
    baselineClassifier("/Users/Patrick/Desktop/hepatitis-test2.CSV")
    acc_classify(trainingSet,testSet)

    # print("--------------------10 train & test---------------------")
    # # ten training sets & test sets
    # tenFoldstest()

    myTree = createTree(trainingSet)
    print(myTree)

    # when call drawTree method , the picture will generate in the root folder
    # drawTree()


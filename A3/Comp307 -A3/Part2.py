
import pandas as pd
import numpy as np

class NaiveBayes(object):
    #
    def getTrainSet(self):
        ########################### pls change path here ############################
        dataSet = pd.read_csv('/Users/Patrick/Desktop/spamLabelled.csv')
        dataSetNP = np.array(dataSet)

        trainData = dataSetNP[:,0:dataSetNP.shape[1]-1]
        labels = dataSetNP[:,dataSetNP.shape[1]-1]


        return trainData, labels

    def getTestSet(self):
        ########################### pls change path here ############################
        dataSet = pd.read_csv('/Users/Patrick/Desktop/spamUnlabelled.csv')
        dataSetNP = np.array(dataSet)
        trainData = dataSetNP[:,0:dataSetNP.shape[1]]

        return trainData

    def classify(self, trainData, labels, features):
        #Find the prior probability of each label in labels
        labels = list(labels)    #Convert to list type
        labelset = set(labels)
        P_y = {}       #save the prob of label
        # print("---------------")
        for label in labelset:
            P_y[label] = labels.count(label)/float(len(labels))   # p = count(y) / count(Y)
            # print(label,P_y[label])
        # print("---------------")
        #Find the probability of simultaneous occurrence of label and feature
        P_xy = {}
        for y in P_y.keys():
            y_index = [i for i, label in enumerate(labels) if label == y]  # lSubscript index of all values with y value in abels
            # print(y_index)
            for j in range(len(features)):      # features[0] All subscript indexes of values appearing in trainData[:,0]
                x_index = [i for i, feature in enumerate(trainData[:,j]) if feature == features[j]]
                xy_count = len(set(x_index) & set(y_index))   # set(x_index)&set(y_index)List the same elements in both tables
                pkey = str(features[j]) + '*' + str(y)
                P_xy[pkey] = xy_count / float(len(labels))
                # print(pkey,P_xy[pkey])
        # print("---------------")
        #Find conditional probability
        P = {}
        l = []
        for y in P_y.keys():
            for x in features:

                pkey = str(x) + '|' + str(y)
                P[pkey] = P_xy[str(x)+'*'+str(y)] / float(P_y[y])    #P[X1|Y] = P[X1Y]/P[Y]
                if pkey in l:
                    pass
                else:
                    print(pkey,P[pkey])
                l.append(pkey)

        #Seeking [2,'S'] belongs to the category
        F = {}   #[2,'S']Probability of belonging to each category
        for y in P_y:
            F[y] = P_y[y]
            for x in features:
                F[y] = F[y] * P[str(x)+'|'+str(y)]     #P[y/X] = P[X/y]*P[y]/P[X]，The denominator is equal, just compare the numerator, so there isF=P[X/y]*P[y]=P[x1/Y]*P[x2/Y]*P[y]
                # print(str(x),str(y),F[y])
        # print("-------2--------")

        features_label = max(F, key=F.get)  #The category corresponding to the maximum probability
        print(F)
        return features_label


if __name__ == '__main__':
    nb = NaiveBayes()
    trainData, labels = nb.getTrainSet()
    testData = nb.getTestSet()

    # print(testData)
    features = [1,1,0,0,1,1,0,0,0,0,0,0]
    result = nb.classify(trainData, labels, features)
    print(features, 'belong to', result)
    print("---------------")
    for i in range(0,9):
        # print(testData[i])
        features = testData[i]
        result = nb.classify(trainData, labels, features)
        print(features,'belong to',result)
        print("---------------")
        # print(testData)

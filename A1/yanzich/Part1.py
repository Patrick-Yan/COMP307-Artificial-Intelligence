"""
  Author: Yan Zichu
  StudentID: 300476924
  Victoria University of Wellington
"""
import pandas as pd


# Classifier
def classify(inX, dataSet, labels, k):
    m, n = dataSet.shape   # shape（m, n）m rows have n features
    # Calculate the Euclidean distance from the test data to each point
    distances = []
    feature_range = []
    for i in range(13):
        sorted_trainingData = sorted(dataSet, key=lambda tup: tup[i])
        max = sorted_trainingData[-1][i]
        min = sorted_trainingData[0][i]
        feature_range.append(max - min)
        # print(feature_range)
    for i in range(m):
        sum = 0
        for j in range(n):
            sum += ((inX[j] - dataSet[i][j]) ** 2) / (feature_range[j] **2)
        distances.append(sum ** 0.5)

    sortDist = sorted(distances)

    # K categories of closest value
    classCount = {}
    for i in range(k):
        voteLabel = labels[ distances.index(sortDist[i])]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1 # 0:map default
    sortedClass = sorted(classCount.items(), key=lambda d:d[1], reverse=True)
    return sortedClass[0][0]


def loadTrainingset():
    trainingSet = pd.read_csv("/Users/Patrick/Desktop/wine-training.CSV") # Please change file path here
    features = trainingSet.iloc[:, :-1].values
    labels = trainingSet.iloc[:, -1].values
    return features, labels


def loadTestset():
    testSet = pd.read_csv("/Users/Patrick/Desktop/wine-test.CSV") # Please change file path here
    features = testSet.iloc[:, :-1].values
    labels = testSet.iloc[:, -1].values
    return features, labels


if __name__ == '__main__':
    dataSet, labels = loadTrainingset()
    dataSet2,labels2 = loadTestset()
    predict_result=[]
    prediction=[]
    #prediction
    for i,j in zip(dataSet2,labels2):
        predict_result.append([i,classify(i,dataSet2,labels2,3),j])  # k value tune here ^ ^
        prediction.append(classify(i,dataSet2,labels2,3))            # k value tune here ^ ^
        correct_count=0
    #compare prediction with coreect result，and caculate the acuuracy of prediction
    for i in predict_result:
        if(i[1]==i[2]):
            correct_count+=1
    ratio=float(correct_count)/len(predict_result)
    print("---------------------Prediction---------------------")
    print(prediction)
    print("---------------------Correct results---------------------")
    print(labels2)
    print ("Correct predicting ratio: ",ratio)

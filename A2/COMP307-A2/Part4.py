import random
import numpy as np
import pandas as pd
from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicClassifier
from sklearn.model_selection import train_test_split


random.seed(100)

# # # # # # # # # # # # # # Please change path here# # # # # # # # # # # # #
data = pd.read_csv("/Users/Patrick/Desktop/satellite.csv",header=0)


features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

for x in range(len(labels)):
    if labels[x] == "'Normal'":
        labels[x] = 1
    else:  # =="'Anomaly'"
        labels[x] = 0
labels = labels.astype('int')
# print(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)


def _accuracy(y, y_pred, w):
    """Calculate the accuracy."""
    if y_pred < 0:
        y_pred = 0
    else : y_pred = 1
    diffs = np.abs(y - y_pred) # calculate how many different values

    return 1 - (np.sum(diffs) / len(y_pred))

accuracy = make_fitness(_accuracy, greater_is_better = True)


est_gp = SymbolicClassifier(population_size=1000,
                           generations=200, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           feature_names=('V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                                          'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
                                          'V21','V22','V23','V24','V25','V26','V27','V28','V29','V30',
                                          'V31','V32','V33','V34','V35','V36'),function_set=('add','sub','mul','div'))
est_gp.fit(X_train,y_train)
print('The best individual is : ')
print(est_gp)
print('Training set accuracy is %0.2f%%' % (100 * est_gp.score(X_train,y_train)))

Predict_value = est_gp.predict(X_test)
count = 0
for i in range(len(Predict_value)):
    if Predict_value[i] == y_test[i]:
        count += 1
print('Test set accuracy is %0.2f%%' % (100 * count / len(Predict_value)))


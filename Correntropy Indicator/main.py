import math

import numpy as np
from scipy.spatial import distance
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# load Database
features, target = load_wine(return_X_y=True)

# Make a train/test split using 30% test size
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.30,
                                                    random_state=RANDOM_STATE)


# find center of data
def Find_Center_Normalize(data, gamma):
    Mu = np.mean(data,axis=0)
    Iter = 50
    for i in range(Iter):
        sum_x = 0
        sum_y = np.zeros((data.shape[1],))
        for sample in data:
            emp_x = math.exp(-1 * (distance.sqeuclidean(sample, Mu)) / gamma)
            emp_y = sample * emp_x
            sum_x = sum_x + emp_x
            sum_y = sum_y + emp_y
        Mu = sum_y / sum_x
    return Mu


# our definition of distance
# 1-emp(-1*distance(x-u)/gamma)
def distance_normal(sample, center, gamma):
    return 1 - math.exp(-1 * (distance.sqeuclidean(sample, center) / gamma))


# spite data to groups
dic = {label: X_train[y_train == label] for label in np.unique(y_train)}

for gamma in [1, 5, 25, 50, 100, 200, 500, 1000, 5000, 10000,50000,100000,1000000]:
    C1 = Find_Center_Normalize(dic[0], gamma)
    C2 = Find_Center_Normalize(dic[1], gamma)
    C3 = Find_Center_Normalize(dic[2], gamma)
    wrong = 0
    correct = 0
    i = 0
    while i < len(X_test):
        F1 = distance_normal(X_test[i], C1, gamma)
        F2 = distance_normal(X_test[i], C2, gamma)
        F3 = distance_normal(X_test[i], C3, gamma)
        if F1 == min(F1, F2, F3):
            ans = 0
        elif F2 == min(F1, F2, F3):
            ans = 1
        else:
            ans = 2
        if y_test[i] == ans:
            correct += 1
        else:
            wrong += 1
        i += 1
    print(gamma, ' : ', correct / (correct + wrong))

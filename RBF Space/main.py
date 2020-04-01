from math import exp

import numpy as np
from scipy.spatial import distance
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

dataset = load_wine()
train_data = dataset.data
train_label = dataset.target
x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.25)

dic = {label: x_train[y_train == label] for label in np.unique(y_train)}
C1 = np.mean(dic[0], axis=0)
C2 = np.mean(dic[1], axis=0)
C3 = np.mean(dic[2], axis=0)

y_kmeans = []
i = 0
while i < len(x_test):
    F1 = distance.sqeuclidean(x_test[i], C1)
    F2 = distance.sqeuclidean(x_test[i], C2)
    F3 = distance.sqeuclidean(x_test[i], C3)
    if F1 == min(F1, F2, F3):
        y_kmeans.append(0)
    elif F2 == min(F1, F2, F3):
        y_kmeans.append(1)
    else:
        y_kmeans.append(2)
    i += 1

results = confusion_matrix(y_test, y_kmeans)
acc = accuracy_score(y_test, y_kmeans)
print("results:")
print(results)
print("accuracy = ", acc * 100)

X = train_data
Y = train_label
gammaSample = [1, 3, 5, 10, 25, 50, 100, 1000, 10000, 50000, 100000, 500000, 1000000, 10000000, 100000000]
for gamma in gammaSample:
    print("gamma :", gamma)
    new_X = []
    centers = np.vstack((C1, C2, C3))

    for i, data in enumerate(X):
        new_data = []
        for j, center in enumerate(centers):
            d = exp(-(sum(pow((data - center), 2)) / gamma))
            new_data.append(d)
        new_X.append(np.array(new_data))
    new_X = np.array(new_X)

    new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(new_X, Y, test_size=0.25)

    dic = {label: new_x_train[new_y_train == label] for label in np.unique(new_y_train)}

    new_C1 = np.mean(dic[0], axis=0)
    new_C2 = np.mean(dic[1], axis=0)
    new_C3 = np.mean(dic[2], axis=0)

    y_kmeans = []
    i = 0
    while i < len(new_x_test):
        F1 = distance.sqeuclidean(new_x_test[i], new_C1)
        F2 = distance.sqeuclidean(new_x_test[i], new_C2)
        F3 = distance.sqeuclidean(new_x_test[i], new_C3)
        if F1 == min(F1, F2, F3):
            y_kmeans.append(0)
        elif F2 == min(F1, F2, F3):
            y_kmeans.append(1)
        else:
            y_kmeans.append(2)
        i += 1

    results = confusion_matrix(new_y_test, y_kmeans)
    acc = accuracy_score(new_y_test, y_kmeans)
    print("results:")
    print(results)
    print("accuracy sqeuclidean = ", acc * 100)
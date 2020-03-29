import numpy as np
from scipy.spatial import distance
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def calculateCovariance(X):
    meanX = np.mean(X, axis=0)
    lenX = X.shape[0]
    X = X - meanX
    covariance = X.T.dot(X) / lenX
    return covariance


RANDOM_STATE = 42
FIG_SIZE = (10, 7)

# load Database
features, target = load_wine(return_X_y=True)

# Make a train/test split using 30% test size
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.30,
                                                    random_state=RANDOM_STATE)

dic = {label: X_train[y_train == label] for label in np.unique(y_train)}
C1 = np.mean(dic[0], axis=0)
C2 = np.mean(dic[1], axis=0)
C3 = np.mean(dic[2], axis=0)


def Test(Fun, label):
    worng = 0
    correct = 0
    i = 0
    while i < len(X_test):
        F1 = Fun(X_test[i], C1)
        F2 = Fun(X_test[i], C2)
        F3 = Fun(X_test[i], C3)
        if F1 == min(F1, F2, F3):
            ans = 0
        elif F2 == min(F1, F2, F3):
            ans = 1
        else:
            ans = 2
        if y_test[i] == ans:
            correct += 1
        else:
            worng += 1
        i += 1
    print(label, ' : ', correct / (correct + worng))


Test(distance.sqeuclidean, 'sqeuclidean')
Test(distance.braycurtis, 'braycurtis')
Test(distance.canberra, 'Canberra ')
Test(distance.chebyshev, 'chebyshev')
Test(distance.cityblock, 'cityblock')
Test(distance.correlation, 'correlation')
Test(distance.cosine, 'cosine')
Test(distance.euclidean, 'euclidean')

# mahalanobis
worng = 0
correct = 0
i = 0
invdataCov1 = np.linalg.inv(calculateCovariance(dic[0]))
invdataCov2 = np.linalg.inv(calculateCovariance(dic[1]))
invdataCov3 = np.linalg.inv(calculateCovariance(dic[2]))
while i < len(X_test):
    F1 = distance.mahalanobis(X_test[i], C1, invdataCov1)
    F2 = distance.mahalanobis(X_test[i], C2, invdataCov2)
    F3 = distance.mahalanobis(X_test[i], C3, invdataCov3)
    if F1 == min(F1, F2, F3):
        ans = 0
    elif F2 == min(F1, F2, F3):
        ans = 1
    else:
        ans = 2
    if y_test[i] == ans:
        correct += 1
    else:
        worng += 1
    i += 1
print('mahalanobis :', correct / (correct + worng))

# seuclidean
worng = 0
correct = 0
i = 0
seuclideanMatrix1 = np.var(dic[0], axis=0)
seuclideanMatrix2 = np.var(dic[1], axis=0)
seuclideanMatrix3 = np.var(dic[2], axis=0)
while i < len(X_test):
    F1 = distance.seuclidean(X_test[i], C1, seuclideanMatrix1)
    F2 = distance.seuclidean(X_test[i], C2, seuclideanMatrix2)
    F3 = distance.seuclidean(X_test[i], C3, seuclideanMatrix3)
    if F1 == min(F1, F2, F3):
        ans = 0
    elif F2 == min(F1, F2, F3):
        ans = 1
    else:
        ans = 2
    if y_test[i] == ans:
        correct += 1
    else:
        worng += 1
    i += 1
print('seuclidean :', correct / (correct + worng))

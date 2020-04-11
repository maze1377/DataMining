import numpy as np
from scipy.spatial import distance
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

dataset = load_wine()
train_data = dataset.data
train_label = dataset.target
X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.2, random_state=np.random)


# find center of data
def Find_Center_Unsensitive(train_data, eps):
    Mu = np.mean(train_data, axis=0)
    zeros = np.zeros(train_data.shape)
    Iter = 50
    for i in range(Iter):
        sum_wx = 0
        sum_w = np.zeros((train_data.shape[1],))
        for data in train_data:
            mat = np.array(abs(data - Mu))
            mat = [x - eps if x - eps > 0 else 0 for x in mat]
            wi = [-1 / x if x > 0 else 0 for x in mat]
            sum_wx += wi * data
            sum_w += wi
        Mu = sum_wx / sum_w
        Mu = np.nan_to_num(Mu)
    return Mu


dic = {label: X_train[y_train == label] for label in np.unique(y_train)}


def distance_normal(sample, center, eps):
    d = abs(distance.sqeuclidean(sample, center)) - eps
    if d < 0:
        return 0
    else:
        return d


eps_vals = [0.0001, 000.1, 00.1, 0.1, 0.5, 0.75, 1.25, 1.5, 1.75, 2, 2.25, 3, 10, 20, 40, 100, 500, 1000, 2000, 10000,
            15000, 50000, 100000]
for eps in eps_vals:
    C1 = Find_Center_Unsensitive(dic[0], eps)
    C2 = Find_Center_Unsensitive(dic[1], eps)
    C3 = Find_Center_Unsensitive(dic[2], eps)
    wrong = 0
    correct = 0
    i = 0
    while i < len(X_test):
        F1 = distance_normal(X_test[i], C1, eps)
        F2 = distance_normal(X_test[i], C2, eps)
        F3 = distance_normal(X_test[i], C3, eps)
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
    print(eps, ' : ', correct / (correct + wrong))

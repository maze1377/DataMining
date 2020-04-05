import numpy as np
from numpy import linalg
from scipy.spatial import distance
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# load data from sklearn dataset:
dataset = load_wine()
train_data = dataset.data
train_label = dataset.target
x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.25, random_state=np.random)

# Calculate covariance of data:
cov = np.cov(x_train.T)
cov_inverse = cov.T
print(cov_inverse.shape)
print(cov_inverse)

# Decomposite covariace inverse of data, and calculate Q, landa, and Q.T so:
# Cov.T = Q * landa* Q’
landa, Q = linalg.eig(cov_inverse)
landa_mat = np.diag(landa)
Ql = np.dot(Q, landa_mat)
QlQT = np.dot(Ql, Q.T)
print(QlQT)


# Now let’s transform our data to  φ space using :
# Φ(x) = Q’*landa^(1/2)*x
def transform_data(Q, landa, x):
    u = np.dot(Q.T, landa_mat ** (1 / 2))
    return np.dot(x, u.T)


# convert data to new space
new_x_train = transform_data(Q, landa, x_train)
new_x_test = transform_data(Q, landa, x_test)
print(new_x_train[0])

dic = {label: new_x_train[y_train == label] for label in np.unique(y_train)}
m1 = np.mean(dic[0], axis=0)
m2 = np.mean(dic[1], axis=0)
m3 = np.mean(dic[2], axis=0)


# Now we can safely use Euclidean distance to classify out data:
def Euclidean(x):
    d1 = distance.euclidean(x, m1)
    d2 = distance.euclidean(x, m2)
    d3 = distance.euclidean(x, m3)

    min_dist = min(d1, d2, d3)

    if min_dist == d1:
        ret = 0
    elif min_dist == d2:
        ret = 1
    elif min_dist == d3:
        ret = 2

    return ret


# calculate result
i = 0
results = []
accs = []
while i < 100:
    predicted = []
    n, m = new_x_train.shape
    for j in range(n):
        dist = Euclidean(new_x_train[j])
        predicted.append(dist)

    result = confusion_matrix(y_train, predicted)
    acc = accuracy_score(y_train, predicted)
    results.append(result)
    accs.append(acc)
    i += 1

# print result
mean_result = sum(results) / 100
mean_accs = sum(accs) / 100
var = np.var(accs)
print(mean_result)
print("mean of accuracy: ", mean_accs)
print("variance: ", var)

# you can find the resualt of accuracy without change space in distance

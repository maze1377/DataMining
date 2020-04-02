from collections import Counter

import numpy as np
from scipy.spatial import distance as skidistance
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 47
K_Neighbors = 7
# load Database
features, target = load_wine(return_X_y=True)

# Make a train/test split using 25% test size
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.25,
                                                    random_state=RANDOM_STATE)


# the Euclidean distance
def distance(instance1, instance2):
    # just in case, if the instances are lists or tuples:
    instance1 = np.array(instance1)
    instance2 = np.array(instance2)
    return np.linalg.norm(instance1 - instance2)


variance_X = np.var(X_train, axis=0)


# the standardized Euclidean distance
def distance2(instance1, instance2):
    # just in case, if the instances are lists or tuples:
    instance1 = np.array(instance1)
    instance2 = np.array(instance2)
    return skidistance.seuclidean(instance1, instance2, variance_X)


def get_neighbors(training_set,
                  labels,
                  test_instance,
                  k,
                  distance=distance):
    """
    get_neighors calculates a list of the k nearest neighbors
    of an instance 'test_instance'.
    The list neighbors contains 3-tuples with
    (index, dist, label)
    where
    index    is the index from the training_set,
    dist     is the distance between the test_instance and the
             instance training_set[index]
    distance is a reference to a function used to calculate the
             distances
    """
    distances = []  # better implement by PriorityQueue
    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append(
            (training_set[index], dist, labels[index]))  # save data in tuple first data_x,distance,data_y(label)
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors


def vote(neighbors):
    class_counter = Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]] += 1
    return class_counter.most_common(1)[0][0]


# ∑ 1/(i+1)=1+1/2+1/3+...+1/k
def vote_distance_weights(neighbors, all_results=False):
    class_counter = Counter()
    number_of_neighbors = len(neighbors)
    for index in range(number_of_neighbors):
        dist = neighbors[index][1]
        label = neighbors[index][2]
        class_counter[label] += 1 / (dist ** 2 + 1)
    labels, votes = zip(*class_counter.most_common())

    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    if all_results:
        total = sum(class_counter.values(), 0.0)
        for key in class_counter:
            class_counter[key] /= total
        return winner, class_counter.most_common()
    else:
        return winner, votes4winner / sum(votes)


vote_ans = []
vote_distance_weights_ans = []
for sample in X_test:
    neighbors = get_neighbors(X_train,
                              y_train,
                              sample,
                              K_Neighbors,
                              distance=distance)
    vote_result = vote(neighbors)
    vote_distance_weights_result = vote_distance_weights(neighbors)
    vote_ans.append(vote_result)
    vote_distance_weights_ans.append(vote_distance_weights_result[0])

print("Euclidean distance vote", accuracy_score(y_test, vote_ans))
print("Euclidean distance weights vote", accuracy_score(y_test, vote_distance_weights_ans))

vote_ans = []
vote_distance_weights_ans = []
for sample in X_test:
    neighbors = get_neighbors(X_train,
                              y_train,
                              sample,
                              K_Neighbors,
                              distance=distance2)
    vote_result = vote(neighbors)
    vote_distance_weights_result = vote_distance_weights(neighbors)
    vote_ans.append(vote_result)
    vote_distance_weights_ans.append(vote_distance_weights_result[0])

print("standardized Euclidean distance vote", accuracy_score(y_test, vote_ans))
print("standardized Euclidean distance weights vote", accuracy_score(y_test, vote_distance_weights_ans))

for K_Neighbors in [1, 3, 7, 11, 17, 27, 41, 61]:
    vote_ans = []
    vote_distance_weights_ans = []
    for sample in X_test:
        neighbors = get_neighbors(X_train,
                                  y_train,
                                  sample,
                                  K_Neighbors,
                                  distance=distance2)
        vote_result = vote(neighbors)
        vote_distance_weights_result = vote_distance_weights(neighbors)
        vote_ans.append(vote_result)
        vote_distance_weights_ans.append(vote_distance_weights_result[0])
    print("k = ", K_Neighbors)
    print("standardized Euclidean distance vote", accuracy_score(y_test, vote_ans))
    print("standardized Euclidean distance weights vote", accuracy_score(y_test, vote_distance_weights_ans))

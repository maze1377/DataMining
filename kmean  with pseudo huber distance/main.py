from typing import List

import numpy as np
from PIL import Image

n_cluster = 30


def get_huber_weight(x, mu, gamma):
    subtraction = np.subtract(x, mu)
    return 1 / pow((1 + sum(pow((subtraction / gamma), 2))), (1 / 2))


def read_file(file_name: str):
    img = Image.open(file_name, 'r')
    pix_val = list(img.getdata())
    return img, np.asarray(pix_val, dtype=np.float32)


def get_label_dict(X_train, y_train) -> dict:
    return {label: X_train[y_train == label] for label in np.unique(y_train)}


def get_huber_index(x, mu):
    mu1 = mu
    nu = np.zeros(x.shape[1], int)
    den = 0
    for item in x:
        wi = get_huber_weight(item, mu1, gamma=10)
        nu = nu + np.dot(item, wi)
        den = den + wi
    return nu / den


def get_random_identifiers(x):
    indexes = np.random.randint(x.shape[0], size=n_cluster)
    return [{i: x[item]} for i, item in enumerate(indexes)]


def imag_filtering(y, img: Image, identifiers):
    new_image = []
    for x in range(img.size[1]):
        new_image_row = []
        for i in range(img.size[0]):
            new_image_row.append(identifiers[y[x * (img.size[0]) + i]])
        new_image.append(new_image_row)
    new_image = np.asarray(new_image, dtype=np.uint8)
    new_image = Image.fromarray(new_image, 'RGB')
    new_image.save('flower_out2.jpg')
    new_image.show()


def euclidean(x, m):
    x_minus_m = x - m
    euclid = np.dot(x_minus_m, x_minus_m.T)
    return euclid


def cluster_on_euclidean_distance(x, means: List[dict] = None):
    euclid = []
    for mean in means:
        euclid.append(euclidean(x, list(mean.values())[0]))
    min_dist = euclid.index(min(euclid))
    return list(means[min_dist].keys())[0]


def get_distance_euclidean_first_round(x=None, means=None):
    dist = []
    for i in range(0, len(x)):
        a = cluster_on_euclidean_distance(x[i], means=means)
        dist.append(a)
    return dist


def get_distance_euclidean(x, n, y=None, means=None):
    if n < 50:
        if means is None and y is None:
            means = get_random_identifiers(x)
            y = get_distance_euclidean_first_round(x, means)
        labeled_dictionary = get_label_dict(x, y)
        new_means = []
        for mean in means:
            new_means.append(
                {list(mean.keys())[0]: get_huber_index(labeled_dictionary[list(mean.keys())[0]],
                                                       list(mean.values())[0])})
        new_y = get_distance_euclidean_first_round(x, new_means)
        return get_distance_euclidean(x, n + 1, new_y, new_means)
    else:
        return means, y


img, image_data = read_file('photo_small.jpg')
identifiers, y = get_distance_euclidean(image_data, 0)
identifiers = {list(identifier.keys())[0]: list(identifier.values())[0] for identifier in identifiers}
imag_filtering(y, img, identifiers)

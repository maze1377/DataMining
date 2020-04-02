from PIL import Image
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def read_file(file_name: str) -> np.ndarray:
    img = Image.open(file_name, 'r')
    pix_val = list(img.getdata())
    return np.asarray(pix_val, dtype=np.float32)


def gmm(X: np.ndarray) -> GaussianMixture:
    gmm = GaussianMixture(n_components=5)
    gmm.fit(X)
    return gmm


def threshold_filtering(gmm: GaussianMixture, X: np.ndarray, img: Image):
    threshold = np.mean(gmm.means_)
    new_image = []
    for x in range(img.size[1]):
        new_image_row = []
        for i in range(img.size[0]):
            new_image_rgb = []
            for j in range(0, X.shape[1], 1):
                if X[x * (img.size[0]) + i][j] > threshold:
                    new_image_rgb.append(X[x * (img.size[0]) + i][j])
                else:
                    new_image_rgb.append(255)
            new_image_row.append(new_image_rgb)
        new_image.append(new_image_row)
    new_image = np.asarray(new_image, dtype=np.uint8)
    new_image = Image.fromarray(new_image, 'RGB')
    new_image.save('my.jpg')
    new_image.show()


def plotting(gmm, X):
    labels = gmm.predict(X)
    fig = plt.figure(1, figsize=(10, 10))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    ax.scatter(X[:, 2], X[:, 1], X[:, 0],
               c=labels, edgecolor="k", s=50)
    ax.set_xlabel("Petal width")
    ax.set_ylabel("Sepal length")
    ax.set_zlabel("Petal length")
    plt.title("Gaussian Mixture Model", fontsize=14)
    plt.show()


img = Image.open('photo.jpg', 'r')
X = read_file('photo.jpg')
gmm = gmm(X)
threshold_filtering(gmm, X, img)

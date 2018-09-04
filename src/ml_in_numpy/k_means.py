import numpy as np
import matplotlib.pyplot as plt
from utils import create_nd_dummy_data


def get_distances(X, centers, mode='sqdist'):
    X_broad = np.broadcast_to(X, (centers.shape[0], *X.shape))
    X_trans = np.transpose(X_broad, (1, 0, 2))

    if mode == 'sqdist':
        distances = np.sum((X_trans - centers)**2, axis=2)
    else:
        raise NotImplementedError("Mode {} not found. Supported: 'sqdist'.".format(mode))

    return distances


def centers_from_assignments(X, assignments):
    new_centers = []
    for i in np.unique(assignments):
        samples = X[assignments == i]
        new_centers.append(np.mean(samples, axis=0))
    return np.array(new_centers)


def k_means(X, num_clusters=3, mode='sqdist'):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    dim = X.shape[1]
    centers = np.random.normal(size=[num_clusters, dim])

    for i in range(num_iters):
        distances = get_distances(X, centers, mode=mode)
        assignments = np.argmin(distances, axis=1)
        centers = centers_from_assignments(X, assignments)

    return centers


num_iters = 10

data_params = [[[1, 1], [[2, 1], [1, 2]]], [[-1, -1], [[2, 1], [1, 2]]]]
X = create_nd_dummy_data(data_params=data_params, size=100)

centers = k_means(X, num_clusters=2)
assignments = np.argmin(get_distances(X, centers), axis=1)

plt.scatter(X[:, 0], X[:, 1], s=20, c=assignments)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=200, c=range(centers.shape[0]))
plt.show()

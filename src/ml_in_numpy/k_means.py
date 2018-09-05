import numpy as np
import matplotlib.pyplot as plt
from utils import create_nd_dummy_data


NUM_ITERS = 10
DUMMY_DATA_PARAMS = [[[1, 1], [[2, 1], [1, 2]]],
                     [[-1, -1], [[2, 1], [1, 2]]]]
SIZE = [100, 200]
MODE = 'gauss'
DIST = 'square'
NUM_CLUSTERS = 2


def get_distances(X, centers, dist='square'):
    X_broad = np.broadcast_to(X, (centers.shape[0], *X.shape))
    X_trans = np.transpose(X_broad, (1, 0, 2))

    if dist == 'square':
        distances = np.sum((X_trans - centers)**2, axis=2)
    else:
        raise NotImplementedError("Mode {} not found. "
                                  "Supported: 'square'.".format(mode))

    return distances


def centers_from_assignments(X, assignments):
    new_centers = []
    for i in np.unique(assignments):
        samples = X[assignments == i]
        new_centers.append(np.mean(samples, axis=0))
    return np.array(new_centers)


def k_means(X, num_iters=10, num_clusters=3, dist='square'):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    dim = X.shape[1]
    centers = np.random.normal(size=[num_clusters, dim])

    for i in range(num_iters):
        distances = get_distances(X, centers, dist=dist)
        assignments = np.argmin(distances, axis=1)
        centers = centers_from_assignments(X, assignments)

    return centers


X = create_nd_dummy_data(data_params=DUMMY_DATA_PARAMS, size=SIZE, mode=MODE)

centers = k_means(X, num_iters=NUM_ITERS, num_clusters=NUM_CLUSTERS, dist=DIST)
assignments = np.argmin(get_distances(X, centers), axis=1)

plt.scatter(X[:, 0], X[:, 1], s=20, c=assignments)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=200, c='k')
plt.show()

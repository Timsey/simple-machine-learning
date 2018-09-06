import numpy as np
import matplotlib.pyplot as plt
from utils import create_nd_dummy_data

np.random.seed(0)

NUM_ITERS = 10
DUMMY_DATA_PARAMS = [[[1, 1], [[2, 1], [1, 2]]],
                     [[-1, -1], [[2, 1], [1, 2]]]]
SIZE = [100, 200]
MODE = 'gauss'
DIST = 'square'
NUM_CLUSTERS = 2


def get_distances(X, centers, dist='square'):
    """
    Computes pairwise distances between each center and each data point.

    Arguments:
    - X: numpy array of shape (num_samples, num_features) representing data to
         cluster.
    - centers: numpy array of shape (num_clusters, num_ features), representing
               the centers found after num_iters iterations.
    - dist: str, distance metric to use. Default (and currently only supported)
            is square distance 'square'.

    Returns:
    - distances: numpy array of shape (num_samples, num_clusters), representing
                 the pairwise distances between all samples and all centers.
    """
    # Reshape X to (num_samples, num_centers, num_features)
    # I.e broadcast num_centers times, so distances between samples and centers
    # can be computed in a vectorised manner.
    X_broad = np.broadcast_to(X, (centers.shape[0], *X.shape))
    X_trans = np.transpose(X_broad, (1, 0, 2))

    if dist == 'square':
        distances = np.sum((X_trans - centers)**2, axis=2)
    else:
        raise NotImplementedError("Mode {} not found. "
                                  "Supported: 'square'.".format(mode))
    return distances


def centers_from_assignments(X, assignments):
    """
    Arguments:
    - X: numpy array of shape (num_samples, num_features) representing data to
         cluster.
    - assignments:

    Returns:
    - new_centers: numpy array of shape (num_clusters, num_ features),
                   representing the centers found in the maximisation (M) step.
    """
    new_centers = []
    for i in np.unique(assignments):
        # Get samples assigned to current cluster
        samples = X[assignments == i]
        # A new center is defined by the means of this cluster (M step)
        new_centers.append(np.mean(samples, axis=0))
    return np.array(new_centers)


def k_means(X, num_iters=10, num_clusters=3, dist='square'):
    """
    Performs k-means clustering. Default uses Lloyd's algorithm.

    Arguments:
    - X: numpy array of shape (num_samples, num_features) representing data to
         cluster.
    - num_iters: int, number of iterations to run the algorithm for.
    - num_clusters: int, number of clusters to find.
    - dist: str, distance metric to use. Default (and currently only supported)
            is square distance 'square'.

    Returns:
    - centers: numpy array of shape (num_clusters, num_ features), representing
               the centers found after num_iters iterations.
    """
    # Randomly initialise num_clusters centers
    centers = np.random.normal(size=[num_clusters, X.shape[-1]])

    # Get distances between centers and data, hard assign points to cluster,
    # and recompute new centers. Do this num_iters times.
    for i in range(num_iters):
        distances = get_distances(X, centers, dist=dist)
        # array of shape (num_samples,), where the entries are integers
        # representing the cluster the sample belongs to according to the
        # expectation step of the algorithm.
        assignments = np.argmin(distances, axis=1)
        centers = centers_from_assignments(X, assignments)

    return centers

if __name__ == '__main__':
    X = create_nd_dummy_data(data_params=DUMMY_DATA_PARAMS, size=SIZE, mode=MODE)

    centers = k_means(X, num_iters=NUM_ITERS, num_clusters=NUM_CLUSTERS, dist=DIST)
    assignments = np.argmin(get_distances(X, centers), axis=1)

    plt.scatter(X[:, 0], X[:, 1], s=20, c=assignments)
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=200, c='k')
    plt.show()

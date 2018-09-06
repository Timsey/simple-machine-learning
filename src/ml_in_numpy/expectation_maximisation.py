import numpy as np
from utils import create_1d_dummy_data

np.random.seed(0)

NUM_ITERS = 100
SIZE = [10000, 20000, 10000]  # interesting to play with data imbalance
MODE = 'gauss'
NUM_LATENT = 3
DUMMY_DATA_PARAMS = [[1, 1], [3, 2], [5, 1]]


def create_model_params(data, num_latent, mode):
    """
    Creates model parameters for num_latent models given some data.

    Arguments:
    - data: numpy array of shape (num_samples, num_features) representing data.
    - num_latent: int, number of latent variables (e.g. categories) to
                  consider.
    - mode: str, type of distribution to use for modelling. Default Gaussian
            'gauss'.
    Returns:
    - params: list of parameters of length num_latent, where each entry is a
              list of parameters specifying the distribution to use (e.g. a
              mean and standard deviation in case of Gaussian distribution).
    """
    if mode == 'gauss':
        # mean and variance for every cluster
        # Initial estimate: sort data on size, split data into num_latent
        # parts, and take as mean, var the mean, var of the parts.
        s_data = sorted(data)
        params = [[np.mean(s_data[int(i * data.size / num_latent):
                                  int((i + 1) * data.size / num_latent)]),
                   np.sqrt(np.var(s_data[int(i * data.size / num_latent):
                                         int((i + 1) * data.size /
                                         num_latent)]))]
                  for i in range(num_latent)]
    else:
        raise NotImplementedError("Only available mode is 'gaussian'.")
    return params


def gaussian_pdf(data, mu=0, sigma=1):
    """
    WARNING: Currently only implemented correctly for Univariate Gaussian.

    Gaussian probability density distribution. Supports univariate and
    multivariate Gaussian with diagonal correlation matrix.

    Arguments:
    - data: np.array, data array of shape (num_samples, num_features).
    - mu: number-like or array-like, means of Gaussian.
    - sigma: number-like or array-like, standard deviations of Gaussian.

    Returns:
    - np.array of size num_samples, with the probabilities under the Gaussian.
    """
    # Create a univariate or diagonal multivariate Gaussian
    mu, sigma = np.array(mu), np.array(sigma)
    if not mu.shape == sigma.shape:
        raise ValueError("mu and sigma not of same shape: {} and {}.".format(
            mu.shape, sigma.shape))

    # TODO: Check whether data has num features equal to mu.size and sigma.size
    # TODO: Make sure Gaussian returns a single value for every datapoint in
    #       diagonal multivariate case.
    numer = np.exp((data - mu)**2 / (-2 * sigma**2))
    denom = np.sqrt(2 * np.pi) * sigma
    return numer / denom


def expectation(data, params, mode):
    """
    Expectation step (E-step). Calculates likelihoods of the data given the
    current model parameters.

    Arguments:
    - data: numpy array of shape (num_samples, num_features) representing data.
    - params: list of parameters of length num_latent, where each entry is a
              list of parameters specifying the distribution to use (e.g. a
              mean and standard deviation in case of Gaussian distribution).
    - mode: str, type of distribution to use for modelling. Default Gaussian
            'gauss'.

    Returns:
    - likelihoods: numpy array of shape (num_latent, num_samples), representing
                   the likelihood for each sample under the num_latent
                   distributions.
    """
    if mode == 'gauss':
        likelihoods = np.array([gaussian_pdf(data, *param)
                                for param in params])
    else:
        raise NotImplementedError("Only available mode is 'gaussian'.")
    return likelihoods


def maximisation(data, likelihoods, mode):
    """
    Maximisation step (M-step). Uses the likelihoods found by the E-step to
    calculate new model parameters that maximise the likelihood under the
    current weighted data assignment.

    Arguments:
    - data: numpy array of shape (num_samples, num_features) representing data.
    - likelihoods: numpy array of shape (num_latent, num_samples), representing
                   the likelihood for each sample under the num_latent
                   distributions.
    - mode: str, type of distribution to use for modelling. Default Gaussian
            'gauss'.

    Returns:
    - params: list of parameters of length num_latent, where each entry is a
              list of parameters specifying the distribution to use (e.g. a
              mean and standard deviation in case of Gaussian distribution)
              found by the maximisation step.
    """

    # Calculate category weights (likelihoods normalised over categories)
    weights = likelihoods / np.sum(likelihoods, axis=0)

    if mode == 'gauss':
        # Calculate weighted means. Shape (num_latent,)
        means = np.sum(data * weights, axis=1) / np.sum(weights, axis=1)
        # Broadcast data to shape (num_latent, num_samples), so variance
        # can be calculated for each latent category in a vectorised manner
        sq_dist_to_means = (np.broadcast_to(data, (means.size, data.size)) -
                            means.reshape(-1, 1))**2
        # Calculate weighted variance from square distances to category means
        stddevs = (np.sqrt(np.sum(weights * sq_dist_to_means, axis=1) /
                   np.sum(weights, axis=1)))
        params = [[means[i], stddevs[i]] for i in range(means.size)]
    else:
        raise NotImplementedError("Only available mode is 'gauss'.")
    return params


if __name__ == '__main__':
    # Create data and initial model parameters
    X = create_1d_dummy_data(DUMMY_DATA_PARAMS, mode=MODE, size=SIZE)
    X = X.flatten()
    params = create_model_params(X, NUM_LATENT, MODE)

    # Iterate EM algorithm
    for t in range(NUM_ITERS):
        likelihoods = expectation(X, params, MODE)
        params = maximisation(X, likelihoods, MODE)

    print('Actual model parameters: {}'.format(DUMMY_DATA_PARAMS))
    print('Final model parameters: {}'.format(params))

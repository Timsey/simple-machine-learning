import numpy as np
from utils import create_1d_dummy_data

np.random.seed(0)

NUM_ITERS = 100
MODE = 'gauss'
NUM_LATENT = 3
DUMMY_DATA_PARAMS = [[1, 1], [3, 2], [5, 1]]


# def create_dummy_data(data_params, mode):
#     """
#     Creates dummy data given some model parameters.
#
#     Returns: numpy array of data.
#     """
#     if mode == 'uni_gauss':
#         data = np.array([np.random.normal(loc=data_params[i][0],
#                                           scale=data_params[i][1],
#                                           size=50000)
#                          for i in range(len(data_params))]).flatten()
#     else:
#         raise NotImplementedError("Only available mode is 'gaussian'.")
#     return data


def create_model_params(data, num_latent, mode):
    """
    Creates model parameters for num_latent models given some data.

    Returns: list of model parameters of length num_latent.
    """
    if mode == 'gauss':
        # mean and variance for every cluster
        # Initial estimate: sort data on size, split data into num_latent parts,
        # and take as mean, var the mean, var of the parts.
        s_data = sorted(data)
        params = [[np.mean(s_data[int(i * data.size / num_latent):
                                  int((i + 1) * data.size / num_latent)]),
                   np.sqrt(np.var(s_data[int(i * data.size / num_latent):
                                         int((i + 1) * data.size / num_latent)]))]
                  for i in range(num_latent)]
    else:
        raise NotImplementedError("Only available mode is 'gaussian'.")
    return params


def gaussian(x, mu=0, sigma=1):
    """
    WARNING: Currently only implemented correctly for Univariate Gaussian.

    Gaussian probability density distribution. Supports univariate and
    multivariate Gaussian with diagonal correlation matrix.

    Arguments:
    - x: np.array, data array of shape (num_samples, num_features).
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
    numer = np.exp((x - mu)**2 / (-2 * sigma**2))
    denom = np.sqrt(2 * np.pi) * sigma
    return numer / denom


def expectation(data, params, mode):
    """
    Expectation step (E-step). Calculates likelihoods of the data given the
    current model parameters.

    Returns: likelihoods.
    """
    if mode == 'gauss':
        likelihoods = np.array([gaussian(data, *param) for param in params])
    else:
        raise NotImplementedError("Only available mode is 'gaussian'.")
    return likelihoods


def maximisation(data, likelihoods, mode):
    """
    Maximisation step (M-step). Uses the likelihoods found by the E-step to
    calculate new model parameters that maximise the likelihood under the
    current weighted data assignment.

    Returns: model parameters.
    """
    if mode == 'gauss':
        weights = likelihoods / np.sum(likelihoods, axis=0)
        means = np.sum(data * weights, axis=1) / np.sum(weights, axis=1)
        unweighted_vars = (np.broadcast_to(data, (means.size, data.size)) -
                           means.reshape(-1, 1))**2
        stddevs = (np.sqrt(np.sum(weights * unweighted_vars, axis=1) /
                   np.sum(weights, axis=1)))
        params = [[means[i], stddevs[i]] for i in range(means.size)]
    else:
        raise NotImplementedError("Only available mode is 'gauss'.")
    return params


if __name__ == '__main__':
    # Create data and initial model parameters
    data = create_1d_dummy_data(DUMMY_DATA_PARAMS, mode=MODE, size=5000)
    data = data.flatten()
    params = create_model_params(data, NUM_LATENT, MODE)

    # Iterate
    for t in range(NUM_ITERS):
        likelihoods = expectation(data, params, MODE)
        params = maximisation(data, likelihoods, MODE)

    print('Actual model parameters: {}'.format(DUMMY_DATA_PARAMS))
    print('Final model parameters: {}'.format(params))

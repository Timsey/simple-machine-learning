import numpy as np

def create_1d_dummy_data(data_params, labeled=False, mode='gauss', size=1000):
    """
    Creates dummy data given some model parameters.

    Arguments:
    - data_params: list of parameters that define the data to be generated. If
                   list of lists, generates data using each parameter set in
                   the inner lists, and and concatenates these together.
    - labeled: bool, whether to return labels as well (0, 1, 2, etc.).
    - mode: str, type of data to generate. Currently only univariate Gaussian
            data is supported ('gauss').
    - size: number of data points to generate for every set of parameters. Can
            also be a list of length equal to data_params if data_params is a
            list of lists, to specify the number of data points for every set
            of paramaters separately.

    Returns: numpy array of data or tuple consisting of numpy array of data
             and labels if 'labeled' is set to True. Arrays will have shape
             (N, 1), where N is the total number of elements generated.
    """
    assert isinstance(data_params, list), "data_params must be a list."

    # Make data_params a list of lists for consistency
    if not isinstance(data_params[0], list):
        data_params = [data_params]
        if isinstance(size, list) and len(size) != 1:
            raise ValueError("size should be an int or list of length 1 if "
                             "data_params contains only one set of "
                             "parameters.")

    # Make size a list for consistency
    if isinstance(size, int):
        size = [size for _ in range(len(data_params))]

    # Check whether size and data_params are consistent
    if not isinstance(size, int):
        assert isinstance(size, list), "size must be an integer or list."
        assert len(size) == len(data_params), ("size and data_params must have "
                                               "the same length if size is a "
                                               "list.")

    # Create data
    if mode == 'gauss':
        data = np.array([np.random.normal(loc=data_params[i][0],
                                          scale=data_params[i][1],
                                          size=size[i])
                         for i in range(len(data_params))])
    else:
        raise NotImplementedError("Only available mode is 'gauss'.")

    # Reshape data to (N, 1) (either from (N,) or
    # from (len(data_params), sizes))
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    data = data.reshape(-1, data.shape[-1])

    # Create labels and return tuple of (data, labels) if 'labeled' is True
    if labeled:
        labels = np.array([np.ones(size=size[i]) * i
                           for i in range(len(data_params))])
        labels = labels.reshape(-1, 1)
        return (data, labels)

    return data


def create_nd_dummy_data(data_params, labeled=False, mode='gauss', size=1000):
    """
    Creates dummy data given some model parameters.

    Arguments:
    - data_params: list of lists representing parameters that define the data
                   to be generated. Every inner list should have form:
                   [[mean1, mean2], [covariance matrix]]. Data generated using
                   multiple inner lists is concatenated together.
    - labeled: bool, whether to return labels as well (0, 1, 2, etc.).
    - mode: str, type of data to generate. Currently only multivariate Gaussian
            data is supported ('gauss').
    - size: number of data points to generate for every set of parameters. Can
            also be a list of length equal to data_params if data_params is a
            list of lists, to specify the number of data points for every set
            of paramaters separately.

    Returns: numpy array of data or tuple consisting of numpy array of data
             and labels if 'labeled' is set to True. Arrays will have shape
             (N, 1), where N is the total number of elements generated.
    """
    assert isinstance(data_params, list), "data_params must be a list."

    # -------------
    # NOTE: Between lines is the only part that is different from 1d data
    # generator (besides mode). Generators should probably be combined.
    assert isinstance(data_params[0], list), ("data_params must be a list "
                                              "of lists.")
    # Make data_params a list of lists of lists for consistency
    if not isinstance(data_params[0][0], list):
    # -------------
        data_params = [data_params]
        if isinstance(size, list) and len(size) != 1:
            raise ValueError("size should be an int or list of length 1 if "
                             "data_params contains only one set of "
                             "parameters.")

    # Make size a list for consistency
    if isinstance(size, int):
        size = [size for _ in range(len(data_params))]

    # Check whether size and data_params are consistent
    if not isinstance(size, int):
        assert isinstance(size, list), "size must be an integer or list."
        assert len(size) == len(data_params), ("size and data_params must have "
                                               "the same length if size is a "
                                               "list.")

    if mode == 'gauss':
        data = np.array([np.random.multivariate_normal(mean=data_params[i][0],
                                                       cov=data_params[i][1],
                                                       size=size[i])
                         for i in range(len(data_params))])
    else:
        raise NotImplementedError("Only available mode is 'gauss'.")

    # Reshape data to (N, 1) (either from (N,) or
    # from (len(data_params), sizes))
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    data = data.reshape(-1, data.shape[-1])

    if labeled:
        labels = np.array([np.ones(size=size[i]) * i
                           for i in range(len(data_params))])
        labels = labels.reshape(-1, 1)
        return (data, labels)
    return data

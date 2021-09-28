import numpy as np

"""
A collection of methods that relate to importance sampling.

P.L.Green
"""

def normalise_weights(logw):
    """
    Description
    -----------
    Normalise importance weights. Note that we remove the mean here
        just to avoid numerical errors in evaluating the exponential.
        We have to be careful with -inf values in the log weights
        sometimes. This can happen if we are sampling from a pdf with
        zero probability regions, for example.

    Parameters
    ----------
    logw : array of logged importance weights

    Returns
    -------
    wn : array of normalised weights

    """

    # Identify elements of logw that are not -inf
    indices = np.invert(np.isneginf(logw))

    # Apply normalisation only to those elements of log that are not -inf
    logw[indices] = logw[indices] - np.max(logw[indices])

    # Find standard weights
    w = np.exp(logw)

    # Find normalised weights
    wn = w / np.sum(w)

    return wn


def resample(x, p_logpdf_x, wn, N):
    """
    Description
    -----------
    Resample given normalised weights.

    Parameters
    ----------
    x : array of current samples

    p_logpdf_x : array of current target evaluations.

    wn : array or normalised weights

    N : no. samples

    Returns
    -------
    x_new : resampled values of x

    p_logpdf_x_new : log pdfs associated with x_new

    wn_new : normalised weights associated with x_new

    """
    i = np.linspace(0, N-1, N, dtype=int)  # Sample positions
    i_new = np.random.choice(i, N, p=wn[:, 0])   # i is resampled
    wn_new = np.ones(N) / N           # wn is reset

    # New samples
    x_new = x[i_new]
    p_logpdf_x_new = p_logpdf_x[i_new]

    return x_new, p_logpdf_x_new, wn_new

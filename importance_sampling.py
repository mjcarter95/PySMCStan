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


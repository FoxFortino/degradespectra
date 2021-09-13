import numpy as np


def Gaussian(x, mu, sigma, amp=None):
    """
    Evaluates a Gaussian with mean mu and standard deviation sigma.

    Arguments
    ---------
    x : array-like
        The points to evaluate the Gaussian. For this work, these will be
        wavelength values.
    mu : float
        The mean of the Gaussian.
    sigma : float
       The standard deviation of the Gaussian.

    Keyword Arguments
    -----------------
    amp : float
        Set the amplitude of the Gaussian manually. Default argument will
        set the amplitude such that the Gaussian is normalized to 1.

    Returns
    -------
    gauss : array-like
    """

    if amp is not None:
        normalization = amp
    else:
        normalization = 1 / (np.sqrt(2 * np.pi) * sigma)

    gauss = np.exp((-1/2) * ((x - mu) / sigma)**2)
    gauss *= normalization
    return gauss

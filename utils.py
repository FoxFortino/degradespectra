import numpy as np


def calc_R_array(wvl):
    """
    Given an array of wavelength bins, find the array of spectral resolutions.

    Spectral resolution is defined as
        R = lambda / d_lambda
    and is a constant of a given spectrograph, no matter if d_lambda is a
    function of lambda (e.g., if the wavelength bins of the spectrgraph are
    evenly spaced in log-space).

    The spectral resolution of a spectrograph will not be exactly constant
    across the entire wavelength range. If you are interested in finding the
    array of spectral resoltuions at each wavelength bin, then this function
    can be used.

    This function assumes that the provided wavelengths denote the left edges
    of the spectrograph bins. This is because if the wavelength bins are not
    evenly spaced (for example if they are evenly spaced in log-space) then
    np.diff will not quite give the correct bin sizes to calculate R from. In
    practice, this is not a big issue though.

    Arguments
    ---------
    wvl : (N,) array-like
        Array of wavelength values that denote the bins of a spectrgraph.

    Returns
    -------
    R_arr : (N,) array_like
        Array of spectral resolutions for each wavelength bin.
    """
    wvl_plus1 = add_extra_point(wvl)
    R_arr = wvl / np.diff(wvl_plus1)
    return R_arr


def add_extra_point(arr):
    """
    Add an extra point to an array that is evenly spaced in log-space.

    Arguments
    ---------
    arr : (N,) array-like
        Array that is evenly spaced in log-space to add an extra point to.

    Returns
    -------
    arr_plus1 : (N+1,) array-like
        The same array as arr but with an additional point added such that the
        entire array is still evenly spaced in log-space.
    """
    log_arr = np.log(arr)
    avg_d_log_arr = np.mean(np.diff(log_arr))

    log_arr_plus1 = np.append(log_arr, log_arr[-1] + avg_d_log_arr)
    arr_plus1 = np.exp(log_arr_plus1)

    return arr_plus1


def calc_avg_R(wvl):
    """
    Given an array of wavelength bins, find the average spectral resolution.

    Arguments
    ---------
    wvl : array-like
        Array of wavelength values that denote the left bin edges of a
        spectrgraph.

    Returns
    -------
    R : float
        average R value of the spectrograph.
    """
    R_arr = calc_R_array(wvl)
    R = np.mean(R_arr)
    return R


def adjust_logbins(bins, current="center", new="leftedge"):
    """
    Redefines whether an array corresponds to bin centers or bin edges.

    Assuming the bins have a constant spacing in log-space (that is:
    ``np.diff(np.log(bins))`` is a constant array) then this function will
    shift the bins from bin-center-defined to bin-edge-defined or vice versa.

    Arguments
    ---------
    bins : array-like
        One dimensional array of bin positions.
    current : {"center", "leftedge",}, default: "center"
        Whether the bins array currently defines the bin centers or left edge.
    new : {"center", "leftedge"}, default: "leftedge"
        What the returned bins array should define: the bin centers or the left
        bin edge.

    Returns
    -------
    new_bins : array-like
        One dimensional array of new bin positions.
    """

    logbin = np.log(bins)
    d_logbin = np.mean(np.diff(logbin))

    if current == "center" and new == "leftedge":
        # diff_logbin / 2 will give the bin radius in log space. If our array
        # denotes bin-centers, then we need to subtract the bin radius from
        # the array so that now the array is denoting left-bin-edges.
        # Also note we need to add one more bin in log space before we take
        # np.diff. This is so that when we subtract arrays of the same shape
        # in the next line.
        bin_radii = np.diff(logbin, append=logbin[-1]+d_logbin)
        new_logbins = logbin - bin_radii * 0.5
        new_bins = np.exp(new_logbins)

    elif current == "leftedge" and new == "center":
        bin_widths = np.diff(logbin, append=logbin[-1]+d_logbin)
        new_logbins = logbin + bin_widths * 0.5
        new_bins = np.exp(new_logbins)

    return new_bins


def FWHM_to_STD(FWHM):
    """
    Given the FWHM of a Gaussian, convert to standard deviation.

    Arguments
    ---------
    FWHM : float
        Full width at half maximum of a Gaussian function.

    Returns
    -------
    stddev : float
        Standard deviation of the Gaussian function.
    """

    conversion = 2 * np.sqrt(2 * np.log(2))
    stddev = FWHM / conversion
    return stddev

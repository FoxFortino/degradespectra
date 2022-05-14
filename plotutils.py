import utils

import numpy as np
import matplotlib.pyplot as plt


@np.vectorize
def wavelen2rgb(nm):
    """
    Converts a wavelength between 380 and 780 nm to an RGB color tuple.

    Willow: This code taken from rsmith-nl/wavelength_to_rgb git repo.

    Arguments
    ---------
        nm : float
            Wavelength in nanometers.
    Returns
    -------
        rgb : 3-tuple
            tuple (red, green, blue) of integers in the range 0-255.
    """
    def adjust(color, factor):
        if color < 0.01:
            return 0
        max_intensity = 255
        gamma = 0.80
        rv = int(round(max_intensity * (color * factor)**gamma))
        if rv < 0:
            return 0
        if rv > max_intensity:
            return max_intensity
        return rv

    # if nm < 380 or nm > 780:
    #     raise ValueError('wavelength out of range')
    if nm < 380:
        nm = 380
    if nm > 780:
        nm = 780
    
    red = 0.0
    green = 0.0
    blue = 0.0
    # Calculate intensities in the different wavelength bands.
    if nm < 440:
        red = -(nm - 440.0) / (440.0 - 380.0)
        blue = 1.0
    elif nm < 490:
        green = (nm - 440.0) / (490.0 - 440.0)
        blue = 1.0
    elif nm < 510:
        green = 1.0
        blue = -(nm - 510.0) / (510.0 - 490.0)
    elif nm < 580:
        red = (nm - 510.0) / (580.0 - 510.0)
        green = 1.0
    elif nm < 645:
        red = 1.0
        green = -(nm - 645.0) / (645.0 - 580.0)
    else:
        red = 1.0
    # Let the intensity fall off near the vision limits.
    if nm < 420:
        factor = 0.3 + 0.7 * (nm - 380.0) / (420.0 - 380.0)
    elif nm < 701:
        factor = 1.0
    else:
        factor = 0.3 + 0.7 * (780.0 - nm) / (780.0 - 700.0)
    # Return the calculated values in an (R,G,B) tuple.
    return (adjust(red, factor), adjust(green, factor), adjust(blue, factor))


def plotSpec(wvl, flux, err=None, save=None):
    """
    Plot a spectrum with appropriate colors.

    Arguments
    ---------
    wvl : (N,) array-like
        Array defining the wavelength bin centers of the spectrograph.
    flux : (N,) array-like
        Array of flux values for each wavelength bin.

    Keyword Arguments
    -----------------
    err : (N,) array-like, Default: None
        Array of uncertainties in the flux measurement. If None, no errorbars
        are plotted.
    save : str, Default: None
        If save is None, then the resulting plot is not saved. If save is a
        string, then plt.savefig will be called with that string.
    """

    if not np.any(wvl > 7000):
        RGB = wavelen2rgb(wvl/10)
        RGBA = np.array(RGB).T / 255
    else:
        # If there are wavelength points above 7000 angstroms, make them an
        # RGB value corresponding to 7000 angstroms. This RGB code can't
        # handle wavelengths not between 4000 and 7000 angstroms.
        over7000 = np.where(wvl > 7000)[0]
        wvl_copy = wvl.copy()
        wvl_copy[over7000] = 7000
        RGB = wavelen2rgb(wvl_copy/10)
        RGBA = np.array(RGB).T / 255

    errmsg = ("flux and wvl arrays should be of same size but are "
              f"{flux.size} and {wvl.size} respectively. Each flux value "
              "should correspond to one wavelength bin. See docstring for "
              "more info.")
    assert wvl.size == flux.size, errmsg

    wvl_LE = utils.adjust_logbins(wvl)
    # wvl_LE_plus1 = add_extra_point(wvl_LE)  # Add rightmost bin edge plt.hist

    plt.figure(figsize=(20, 10))
    if err is not None:
        plt.errorbar(wvl[:-1], flux[:-1], yerr=err[:-1],
                     elinewidth=1, capsize=3,
                     ls="-", c="k", marker="*")
    else:
        plt.plot(wvl[:-1], flux[:-1],
                 ls="-", c="k", marker="*")
    _, _, patches = plt.hist(wvl_LE[:-1], bins=wvl_LE,
                             weights=flux[:-1], align="mid")

    # Each patch of the histogram (the rectangle underneath each point) gets
    # colored according to its central wavelength.
    for patch, color in zip(patches, RGBA):
        patch.set_facecolor(color)

    plt.xlabel(r"Wavelength [$\AA$]", fontsize=40)
    plt.ylabel(r"Normalized Flux [$F_{\lambda}$]", fontsize=40)

    bounds = flux.max() + flux.max()*0.05
    plt.ylim((-bounds, bounds))

    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.tight_layout()

    if save is not None:
        plt.savefig(save)

    plt.show()

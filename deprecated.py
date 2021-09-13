import sys
sys.path.insert(0, "/Users/admin/UDel/FASTLab/Summer2021_Research/SESNspectraPCA/code")

import os
import shutil
import imageio
import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas
import scipy
import astropy.units as u
import astropy.constants as c
from scipy.integrate import trapezoid
from astropy.convolution import convolve, Gaussian1DKernel

import SNIDsn
import SNIDdataset as snid
import SNePCA

from pysynphot import observation
from pysynphot import spectrum

from IPython import embed

imagedir = "/Users/admin/UDel/FASTLab/Summer2021_Research/imagedir"
# plt.style.use('~/GitHub/custom-matplotlib/custom.mplstyle')

# WWW This function DOES NOT create bins with constant spacing in log
# space. They are evenly spaced in normal space.
# Binspec implemented in python.
def binspec(wvl, flux, wstart, wend, wbin):
    """
    Rebins wavelengths of a spectrum and linearly interpolates the original fluxes
    to produce the new fluxes for the rebinned spectrum.
    Parameters
    ----------
    wvl : np.array
        wavelength values
    flux : np.array
        flux values
    wstart : float
        desired wavelength start for new binning
    wend : float
        desired wavelength end for new binning
    wbin : float
        desired bin size
    Returns
    -------
    answer/wvin : np.array
        interpolated fluxes
    outlam : np.array
        rebinned wavelength array
    """
    nlam = (wend - wstart) / wbin + 1
    nlam = int(np.ceil(nlam))
    outlam = np.arange(nlam) * wbin + wstart
    answer = np.zeros(nlam)
    interplam = np.unique(np.concatenate((wvl, outlam)))
    interpflux = np.interp(interplam, wvl, flux)

    for i in np.arange(0, nlam - 1):
        cond = np.logical_and(interplam >= outlam[i], interplam <= outlam[i+1])
        w = np.where(cond)
        if len(w) == 2:
            answer[i] = 0.5*(np.sum(interpflux[cond])*wbin)
        else:
            answer[i] = scipy.integrate.simps(interpflux[cond], interplam[cond])

    answer[nlam - 1] = answer[nlam - 2]
    cond = np.logical_or(outlam >= max(wvl), outlam < min(wvl))
    answer[cond] = 0
    return answer/wbin, outlam


"""Code from @mileslucas ono GitHub.
Found on spacetelescope/pysynphot repo issue #78."""

import pysynphot as S
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian1DKernel

def smear(sp, R, w_sample=1):
    '''
    Smears a model spectrum with a gaussian kernel to the given resolution, R.

    Parameters
    -----------

    sp: SourceSpectrum
        Pysynphot object that we willsmear

    R: int
        The resolution (dL/L) to smear to

    w_sample: int
        Oversampling factor for smoothing

    Returns
    -----------

    sp: PySynphot Source Spectrum
        The smeared spectrum
    '''

    # Save original wavelength grid and units
    w_grid = sp.wave
    w_units = sp.waveunits
    f_units = sp.fluxunits
    sp_name = sp.name

    # Generate logarithmic wavelength grid for smoothing
    w_logmin = np.log10(np.nanmin(w_grid))
    w_logmax = np.log10(np.nanmax(w_grid))
    n_w = np.size(w_grid)*w_sample
    w_log = np.logspace(w_logmin, w_logmax, num=n_w)

    # Find stddev of Gaussian kernel for smoothing
    R_grid = (w_log[1:-1]+w_log[0:-2])/(w_log[1:-1]-w_log[0:-2])/2
    sigma = np.median(R_grid)/R
    if sigma < 1:
        sigma = 1

    # Interpolate on logarithmic grid
    f_log = np.interp(w_log, w_grid, sp.flux)

    # Smooth convolving with Gaussian kernel
    gauss = Gaussian1DKernel(stddev=sigma)
    f_conv = convolve_fft(f_log, gauss)

    # Interpolate back on original wavelength grid
    f_sm = np.interp(w_grid, w_log, f_conv)

    # Write smoothed spectrum back into Spectrum object
    return S.ArraySpectrum(w_grid, f_sm, waveunits=w_units,
                            fluxunits=f_units, name=sp_name)

def rebin_spec(wave, specin, wavnew):
    spec = spectrum.ArraySourceSpectrum(
        wave=wave, keepneg=True,
        flux=specin, waveunits="angstrom", fluxunits="flam")
    f = np.ones(len(wave))
    filt = spectrum.ArraySpectralElement(wave, f, waveunits='angstrom')
    obs = observation.Observation(spec, filt, binset=wavnew, force='taper')

    return obs.binflux


def binspec_willow(spec, R):
    """
    Re-implementation of of binspec from SESNspectraPCA.

    This is a re-implementation of a function that rebins supernovae spectra
    into fewer bins. The original function was written in the repo
    nyusngroup/SESNspectraPCA/SNIDsn.py. I am rewriting this function to get a
    better understanding of it. Also, the original function creates a rebinned
    spectrum that has evenly spaced bins, NOT evenly spaced bins in log-space.

    Also I want to change how arguments are handled. I want it to only accept
    a SNIDsn object instance and an R value. This particular SNIDsn object is
    from Umer's repo: umerudel/SESNspectraPCA/SNIDsn.py. It is hopefully
    functionally equivalent to the SNIDsn class present in the nyusngroup
    SNIDsn.py file.

    This re-implementation also takes advantage of an algorithm to rebin
    spectra found on post by astrobetter.com by Jessica Lu from August 12,
    2013.


    Arguments
    ---------
    spec : SNIDsn object instance
        Contains information about a particular supernova's spectrum at a
        particular phase.
    R : float
        Spectral resolution, defined as lambda / (Delta lambda).

    Returns
    -------
    spec : SNIDsn object instance
        The same SNIDsn object instance that was provided as argument, except
        the spectrum has been rebinned according to the provided spectral
        resolution.
    """

    R_current = findR_fromSpec(spec)
    errmsg = (f"Current R: {R_current}. Chosen R: {R}. "
              f"You cannot upsample the spectrum. "
              "Choose a smaller spectral resolution, R.")
    assert R < R_current, errmsg

    # WWW Here I make an assumption about the units of these quantities. I am
    # confident that the wavelengths are in units of angstroms, but as far as
    # I can tell the units of flux are just "normalized flux" which obviously
    # isn't helpful. However, I have assumed that they have units of F_lambda
    # in cgs units.
    wvl = spec.wavelengths.astype(float)  # * u.angstrom
    flux = spec.data.astype(float)  # * u.erg / u.s / u.cm**2 / u.angstrom
    wvl = adjust_logbins(wvl, current="center", new="leftedge")

    # Renormalize the flux data such that there are no negative values. The
    # pysnphot code in rebin_spec expects non-negative fluxes. This is ok
    # because these fluxes are normalized in an unknown way (to me, Willow,
    # anyway) so it's ok if we normalize them again I think...
    flux -= flux.min()

    # First extract the first and last wavelength (bin) values. These are
    # necessary for creating the rebinned wavelength array. It is useful to
    # extract the log of these values too for later calculations
    logwvl = np.log(wvl)
    wvl0 = wvl[0]
    # wvlf = wvl[-1]
    logwvl0 = logwvl[0]
    logwvlf = logwvl[-1]

    # Calculate the change in lambda for the first bin based on the
    # relationship between R (sectroscopic resolving power) and wavelegth. See
    # docstring above.
    dlam0 = wvl0 / R

    # Calculating the second wavelength value (wvl1) is necessary in order to
    # calculate d_loglam, the difference between consecutive values of the log
    # wavelength arrray. This information is necessary to extract because
    # d_loglam is what is constant for the wavelength array, not dlam.
    # However, we can use dlam for the first pair of points to get a good
    # approximation of d_loglam.
    wvl1 = wvl0 + dlam0
    d_loglam = np.log(wvl1) - np.log(wvl0)
    # print(d_loglam, "This is what Umer's and the NYU code want I think")

    # Generate the new wavelength array in log space because in log space
    # d_loglam is the constant spacing between each element.
    logwvl_new = np.arange(logwvl0, logwvlf + d_loglam, d_loglam)
    wvl_new = np.exp(logwvl_new)

    wvl_new_center = adjust_logbins(wvl_new, current="leftedge", new="center")
    flux_new = rebin_spec(wvl, flux, wvl_new_center)

    return wvl_new, flux_new

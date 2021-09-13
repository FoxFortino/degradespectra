import sys
import os
import shutil

import imageio
import glob

import utils

import numpy as np
import matplotlib.pyplot as plt

codedir = "/Users/admin/UDel/FASTLab/Summer2021_Research/SESNspectraPCA/code"
sys.path.insert(0, codedir)
imagedir = "/Users/admin/UDel/FASTLab/Summer2021_Research/imagedir"
plt.style.use('~/GitHub/custom-matplotlib/custom.mplstyle')


def degrade(
    wvl,
    flux,
    err,
    R,
    PSF_width=None,
    spectral_PSF=None,
    PSF_args={},
    makegif=False
        ):
    """
    Re-binning spectra properly with convolution

    Using code from https://github.com/spacetelescope/pysynphot/issues/78 and
    with Veronique Petit's guidance, this code takes in a spectra with some
    spectral resolution and then rebins it, via convolution, to a new spectral
    resolution, R.

    Arguments
    ---------
    # spec : SNIDsn object instance
    #     Contains information about a particular supernova's spectrum at a
    #     particular phase.
    wvl : (N,) array-like
        Array containing center wavelength bins for the spectrograph
    flux : (N,) array-like
        Array containing fluxes for each wavelength bin
    err : (N,) array-like
        Array containing uncertainties on each flux value.
    R : float
        Spectral resolution, defined as lambda / (Delta lambda).

    Keyword Arguments
    -----------------
    PSF_width : array-like or float, Default: None
        If None, the degradation of the spectra is done by convolving the
        fluxes with a Gaussian with standard deviation proportional to the
        wavelength of the flux. Use this when the spectral PSF is a Gaussian
        and is not constant in wavelength but is constant in log-wavelength.
        If a float is supplied, then we assume the spectral PSF is a
        Gaussian again, but it is constant in wavelength. Therefore, the
        standard deviation of the Gaussian is whatever you set it to be for the
        entire convolution
        If an array is supplied, it must be the same length as the spectrum
        data. This array would specify the standard deviation at each point of
        the convolution. That is, the spectral PSF is still a Gaussian but its
        standard deviation has some complicated dependence on wavelength.
        Therefore, you can enter the exact array of standard deviations you
        want to use.
    spectral_PSF : callable, Default: None
        If None, we use a Gaussian function as the functional form of the
        convolutional kernel when degrading the spetra. See the kwarg PSF_width
        for information on how the standard deviation of the Gaussian is
        handled.
        If a callable is supplied then you must also supply PSF_args as a dict
        so that the spectral PSF can be called as
            spectral_PSF(wvl, wvl[i], stddev[i], **PSF_args)
    PSF_args : dict, Default: {}
        The arguments to be unpacked when keyword argument spectral_PSF is
        used.
    makegif : bool, Default: False
        If True, a gif will be made of the convolution which will be saved at
        the image directory (imagedir defined at the top of the script).

    Returns
    -------
    # spec : SNIDsn object instance
    #     Contains updated flux, wavelength, and uncertainties corresponding to
    #     degrading the original spectrum to resolving power R.

    """
    # data_dtype = spec.data.dtype

    # # WWW Here I make an assumption about the units of these quantities. I am
    # # confident that the wavelengths are in units of angstroms, but as far as
    # # I can tell the units of flux are just "normalized flux" which obviously
    # # isn't helpful. However, I have assumed that they have units of F_lambda
    # # in cgs units.
    # wvl = spec.wavelengths.astype(float)  # * u.angstrom
    # flux = spec.data.astype(float)  # * u.erg / u.s / u.cm**2 / u.angstrom
    # flux -= flux.min()

    # # Unpack the uncertainties on the spectrum. Assume it has the same units
    # # as the flux.
    # phase_key = list(spec.smooth_uncertainty.keys())[0]
    # err = spec.smooth_uncertainty[phase_key].astype(float)

    R_current = utils.calc_avg_R(wvl)
    errmsg = (f"Current R: {R_current}. Chosen R: {R}. "
              f"You cannot upsample the spectrum. "
              "Choose a smaller spectral resolution, R.")
    assert R < R_current, errmsg

    # Calculate the left edges of the bins for later on.
    wvl_LE = utils.adjust_logbins(wvl, current="center", new="leftedge")
    new_wvl_LE0 = wvl_LE[0]  # Set leftmost bin edge of new bins
    new_dlam0 = new_wvl_LE0 / R  # Find width of first new bin based on R
    new_wvl_LE1 = new_wvl_LE0 + new_dlam0  # Find left edge of second bin
    new_d_loglam = np.log(new_wvl_LE1) - np.log(new_wvl_LE0)  # Find log width

    # Generate the array of new bin left edges based on d_loglam.
    logwvl_LE0 = np.log(new_wvl_LE0)
    approx_end = np.log(wvl_LE[-1]) + new_d_loglam
    new_logwvl_LE = np.arange(logwvl_LE0, approx_end, new_d_loglam)
    new_wvl_LE = np.exp(new_logwvl_LE)
    new_wvl = utils.adjust_logbins(new_wvl_LE,
                                   current="leftedge",
                                   new="center")

    r = R_current / R
    if PSF_width is None:
        # In order to degrade the spectra we convolve it with a Gaussian with a
        # FWHM proprtional to d_lambda. This necessitates writing our own code
        # for this convolution because there is no existing code (that we are
        # aware of) for handling convolutions where the kernel changes at each
        # step of the integration.
        # Specifically we assume the FWHM of the appropriate Gaussian kernel is
        #     FWHM = d_lambda * R_old / R_new

        # Start by generating the array of FWHM for each kernel, and then
        # converting to an array of standard deviations. I could have written a
        # Gaussian function that accepts the FWHM as an argument, but I didn't.
        wvl_LE_plus1 = utils.add_extra_point(wvl_LE)
        FWHM = np.diff(wvl_LE_plus1) * r
        stddev = utils.FWHM_to_STD(FWHM)

    elif isinstance(PSF_width, (float, int)):
        # Let the standard deviation of the Gaussian PSF be constant.
        stddev = np.ones(wvl_LE.size + 1) * PSF_width

    elif PSF_width.shape == flux.shape:
        # Let the user define the standard deviation of the Gaussian PSF at
        # each wavelength
        stddev = PSF_width

    N = flux.size
    flux_conv = np.zeros(N)
    err_conv = np.zeros(N)
    for i in range(N):

        if spectral_PSF is None:
            G = utils.Gaussian(wvl, wvl[i], stddev[i])
        else:
            G = spectral_PSF(wvl, wvl[i], stddev[i], **PSF_args)

        # Renormalize the PSF so that the total integral is 1. This is
        # necessary to handle edge effects.
        renorm = np.trapz(G, x=wvl)
        G /= renorm

        # Perform one step of the integration of t
        flux_conv[i] = np.trapz(flux * G, x=wvl)
        err_conv[i] = np.trapz(err * G, x=wvl)

        if makegif:
            if i == 0:
                gifdir = os.path.join(imagedir, "gif")
                if os.path.isdir(gifdir):
                    shutil.rmtree(gifdir)
                os.mkdir(gifdir)
                G_max = np.zeros(N)

            G_max[i] = G.max()

            if (i % 5 == 0) or (i in np.arange(N - 5, N)):
                fig = plt.figure(figsize=(20, 10))
                plt.plot(wvl, flux, c="k",
                         label="Original Spectrum")
                plt.plot(wvl, G*r, c="tab:blue",
                         label=f"Kernel * {r:.2f}")
                plt.plot(wvl, G_max*r, c="k", ls=":",
                         label=f"Kernel Max * {r:.2f}", marker="o")
                plt.plot(wvl, flux_conv, c="tab:orange",
                         label="Convolution")
                plt.axhline(y=G.max()*r, c="k", ls=":")
                plt.legend(loc="upper right", fontsize=20)

                plt.title(f"Degrading to R = {R}", fontsize=30)
                plt.xlabel(r"Wavelength [$\AA$]")
                plt.ylabel(r"Normalized Flux [$F_{\lambda}$]")

                plt.ylim((0, None))

                file = os.path.join(gifdir, f"{i:03}.png")
                plt.savefig(file)
                fig.clear()
                plt.close(fig)

            if i == N - 1:
                filenames = os.path.join(gifdir, "*.png")
                filenames = sorted(glob.glob(filenames))

                images = []
                for filename in filenames:
                    images.append(imageio.imread(filename))
                file = os.path.join(imagedir, f"degrading_to_{R}.gif")
                imageio.mimsave(file, images)
                shutil.rmtree(gifdir)

    # Interpolate the convolved spectra at the new wavelength centers to get
    # the degraded spectra.
    new_flux = np.interp(new_wvl, wvl, flux_conv)
    new_err = np.interp(new_wvl, wvl, err_conv)

    # # Repack the SNIDsn object with the rebinned wavelength array and the
    # # degraded spectra and errors.
    # new_err_dict = {}
    # new_err_dict[phase_key] = np.array(new_err)

    # spec.wavelengths = np.array(new_wvl)
    # spec.data = np.array(new_flux).astype(data_dtype)
    # spec.smooth_uncertainty = new_err_dict

    return new_wvl, new_flux, new_err


if __name__ == "__main__":
    pass
    # Todo argparse -r (R) -f (csv filename)
    # assert R is a number (argparse might handle this)
    # assert os.isfile(args.filename)
    # write a help
    # degrade()
    # plotSpec(

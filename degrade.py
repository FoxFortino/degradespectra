import sys
import os
import shutil

import imageio
import glob

import utils
import kernels

import SNIDsn

import numpy as np


def degrade(
    R,
    SNIDobj=None,
    wvl=None,
    flux=None,
    uncer=None,
    PSF_width=None,
    spectral_PSF=None,
    PSF_args={},
    makegif=False,
    savepath=".",
    print_info=False
        ):
    """
    Re-binning spectra properly with convolution

    Using code from https://github.com/spacetelescope/pysynphot/issues/78 and
    with Veronique Petit's guidance, this code takes in a spectra with some
    spectral resolution and then rebins it, via convolution, to a new spectral
    resolution, R.
    
    I make an assumption about the units of the flux and wavelength arrays as
    provided by the SESNtemple and the astrodash GitHub repositories. I am
    confident that the wavelengths are in units of angstroms, but as far as I
    can tell the units of flux are just some normalized flux units (according
    to Williamson & Modjaz & Bianco (2019) about SESN PCA) which obviously
    isn't helpful. However, I have assumed that they have units of F_lambda in
    cgs units.

    Arguments
    ---------
    R : float
        Spectral resolution, defined as lambda / (Delta lambda).

    Keyword Arguments
    -----------------
    SNIDobj : SNIDsn object instance, Default: None
        Contains information about a particular supernova's spectrum at a
        particular phase. If None, then this program will expect you to supply
        wave, flux, and uncer kwargs instead.
    wvl : (N,) array-like, Default: None
        Array containing center wavelength bins for the spectrograph. If None,
        then this program will expect you to supply SNIDobj instead.
    flux : (N,) array-like, Default: None
        Array containing fluxes for each wavelength bin. If None, then this
        program will expect you to supply SNIDobj instead.
    uncer : (N,) array-like or None, Default: None
        Array containing uncertainties on each flux value. If None and wave and
        flux kwargs are supplied, then this program will assume that the given
        spectrum has no uncertainty information. If None and wave and flux
        aren't supplied then this program will expect SNIDobj to be supplied
        instead.
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
    savepath : str, Default: "."
        If makegif is True then you can use this kwarg to specify where the
        resulting gif is saved.

    Returns
    -------
    SNIDobj : SNIDsn object instance
        Contains updated flux, wavelength, and uncertainties corresponding to
        degrading the original spectrum to resolving power R.
    """
    if isinstance(SNIDobj, SNIDsn.SNIDsn):
        wvl = SNIDobj.wavelengths.astype(float)
        
        R_current = utils.calc_avg_R(wvl)
        errmsg = (f"Current R: {R_current}. Chosen R: {R}. "
                  f"You cannot upsample the spectrum. "
                  "Choose a smaller spectral resolution, R.")
        assert R < R_current, errmsg
        
        if print_info:
            print(f"{SNIDobj.header['SN']}", end=", ")
            print(f"{SNIDobj.header['TypeStr']}", end=", ")
            print(f"Current R: {R_current:.2f}", end=" -> ")
            print(f"New R: {R:.2f}", end="\n\n")

        # Calculate the left edges of the bins for later on.
        wvl_LE = utils.adjust_logbins(wvl,
                                      current="center",
                                      new="leftedge")
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
        
        # The flux data in the SNIDsn objects is in a structured array, so to
        # recreate that (with my limited understanding of numpy's structured
        # arrays) I need the resulting number of wavelength buns now.
        new_Nbins = len(new_wvl)
        
        # Generates a list of tuples which contain the column names and dtypes
        # for the structured array that will hold the new degraded flux values.
        phase_keys = list(SNIDobj.data.dtype.names)
        dtypes = [(phase_key, np.float32) for phase_key in phase_keys]
        new_data = np.zeros(new_Nbins, dtype=dtypes)

        
        if SNIDobj.smooth_uncertainty == {}:
            new_uncer_dict = {}
        else:
            new_uncer_dict = {phase: None for phase in SNIDobj.phases}

        # Need to loop through each phase because each phase represents a
        # different spectrum and we need to degrade each spectrum individually.
        for phase_key in phase_keys:
            # Access the structured array that this data is stored in with the
            # appropriate phase key, e.g.: "Ph-4.8" for data coresponding to a
            # phase of -4.8 days.
            flux = SNIDobj.data[phase_key].astype(float)
            
            # Unpack the uncertainties on the spectrum. Assume it has the same
            # units as the flux. The SNIDobj.smooth_uncertainty dictionary
            # might be empty, indicating no uncertainties associated withe
            # spectra.
            if SNIDobj.smooth_uncertainty == {}:
                uncer = None
            else:
                uncer = SNIDobj.smooth_uncertainty[phase_key].astype(float)

            r = R_current / R
            if PSF_width is None:
                # In order to degrade the spectra we convolve it with a
                # Gaussian with a FWHM proprtional to d_lambda. This
                # necessitates writing our own code for this convolution
                # because there is no existing code (that we are aware of) for
                # handling convolutions where the kernel changes at each step
                # of the integration. Specifically we assume the FWHM of the
                # appropriate Gaussian kernel is
                #     FWHM = d_lambda * R_old / R_new

                # Start by generating the array of FWHM for each kernel, and
                # then converting to an array of standard deviations. I could
                # have written a Gaussian function that accepts the FWHM as an
                # argument, but I didn't.
                wvl_LE_plus1 = utils.add_extra_point(wvl_LE)
                FWHM = np.diff(wvl_LE_plus1) * r
                stddev = utils.FWHM_to_STD(FWHM)

            elif isinstance(PSF_width, (float, int)):
                # Let the standard deviation of the Gaussian PSF be constant.
                stddev = np.ones(wvl_LE.size + 1) * PSF_width

            elif PSF_width.shape == flux.shape:
                # Let the user define the standard deviation of the Gaussian
                # PSF at each wavelength
                stddev = PSF_width

            N = flux.size
            flux_conv = np.zeros(N)
            uncer_conv = np.zeros(N)
            for i in range(N):

                if spectral_PSF is None:
                    G = kernels.Gaussian(wvl, wvl[i], stddev[i])
                else:
                    G = spectral_PSF(wvl, wvl[i], stddev[i], **PSF_args)

                # Renormalize the PSF so that the total integral is 1. This is
                # necessary to handle edge effects.
                renorm = np.trapz(G, x=wvl)
                G /= renorm

                # Perform one step of the integration of t
                flux_conv[i] = np.trapz(flux * G, x=wvl)
                
                # For Gaussian convolutions, the uncertainties should be
                # convolved with the square of the kernel.
                if (uncer is not None) and (spectral_PSF is None):
                    uncer_conv[i] = np.trapz(uncer * G**2, x=wvl)

                if makegif:
                    if i == 0:
                        gifdir = os.path.join(savepath, "tmp_gif")
                        if os.path.isdir(gifdir):
                            shutil.rmtree(gifdir)
                        os.mkdir(gifdir)
                        G_max = np.zeros(N)
                        
                    G_max[i] = G.max()
                        
                    if (i % 5 == 0) or (i in np.arange(N - 5, N)):
                        utils.make_frame(wvl, flux, flux_conv,
                                        G, G_max,
                                        R, r,
                                        i, gifdir)
                    if i == N - 1:
                        filenames = os.path.join(gifdir, "*.png")
                        filenames = sorted(glob.glob(filenames))

                        images = []
                        for filename in filenames:
                            images.append(imageio.imread(filename))
                        file = os.path.join(savepath, f"degrading_to_{R}.gif")
                        imageio.mimsave(file, images)
                        shutil.rmtree(gifdir)

            # Interpolate the convolved spectra at the new wavelength centers
            # to get the degraded spectra.
            new_flux = np.interp(new_wvl, wvl, flux_conv)
            new_data[phase_key] = new_flux
            
            if uncer is not None:
                new_uncer = np.interp(new_wvl, wvl, uncer_conv)
                new_uncer_dict[phase_key] = np.array(new_uncer)

        SNIDobj.wavelengths = np.array(new_wvl)
        SNIDobj.data = new_data
        if uncer is not None:
            SNIDobj.smooth_uncertainty = new_uncer_dict

        # Adjust the header of the SNIDsn object to reflect changes
        SNIDobj.header["Nbins"] = len(SNIDobj.wavelengths)

        # Continuum information is no longer valid so make it all some number.
        SNIDobj.continuum[...] = 0
    
    # elif isinstance(wvl, np.ndarray) and isinstance(flux, np.ndarray):
    #     assert wvl.ndim == 1, f"Wavelengths array must be 1D, is {wvl.ndim}"
    #     assert flux.ndim == 1, f"Flux array must be 1D, is {wvl.ndim}"
        
    #     errmsg = ("Wavelength and flux arrays should have the same shape but "
    #               "are {flux.shape} and {wvl.shape} respectively.")
    #     assert wvl.shape == flux.shape, errmsg
        
    #     if uncer is not None:
    #         errmsg = f"Uncertainties array must be 1D, is {wvl.ndim}"
    #         assert uncer.ndim == 1, errmsg
            
    #         errmsg = ("Uncertainties array must be same shape as "
    #                   f"wavelength and flux arrays but is {uncer.shape} "
    #                   "while the flux and wavelength arrays are "
    #                   f"shape {wvl.shape}.")
    #         assert uncer.shape == wvl.shape, errmsg

    
def degrade_lnw(lnw, R, savepath, print_info=True):
    SNIDobj = SNIDsn.SNIDsn()
    SNIDobj.loadSNIDlnw(lnw)

    if print_info:
        print(lnw)
    degrade(R, SNIDobj, print_info=print_info)
    utils.write_lnw(SNIDobj, overwrite=True)

    new_lnw = 'new_' + SNIDobj.header['SN'] + '.lnw'
    new_file = os.path.join(os.getcwd(), new_lnw)

    if "_bsnip" in os.path.basename(lnw):
        changed_lnw = SNIDobj.header['SN'] + "_bsnip" + ".lnw"
    else:
        changed_lnw = SNIDobj.header['SN'] + ".lnw"
        
    changed_file = os.path.join(savepath, changed_lnw)
    shutil.move(new_file, changed_file)

    return changed_file, SNIDobj


def degrade_all(R,
                lnw_files,
                savepath,
                overwrite_R_dir=True,
                print_info=True
                   ):
    errmsg = f"{savepath} does not exist. Please create it first."
    assert os.path.isdir(savepath), errmsg

    R_dir = os.path.join(savepath, str(R))
    if os.path.isdir(R_dir) and overwrite_R_dir:
        shutil.rmtree(R_dir)
        os.mkdir(R_dir)
    elif os.path.isdir(R_dir) and not overwrite_R_dir:
        pass
    else:
        os.mkdir(R_dir)
        
    # Create this templist file. In reality, astrodash does this in a more
    # complicated way, and removes some files in the process. If you want to
    # do that, then do that before running this function and just supply the
    # proper list of lnw files to this function.
    with open(os.path.join(R_dir, "templist.txt"), "w") as f:
        lnw_list = sorted([os.path.basename(file) for file in lnw_files])
        for lnw in lnw_list:
            f.write(os.path.basename(lnw) + "\n")

    for i, lnw in enumerate(lnw_files):
        if print_info:
            print(f"{int(i+1):05}", end=":  ")
        degrade_lnw(lnw, R, R_dir, print_info=print_info)


if __name__ == "__main__":
    pass
    # Todo argparse -r (R) -f (csv filename)
    # assert R is a number (argparse might handle this)
    # assert os.isfile(args.filename)
    # write a help
    # degrade()
    # plotSpec(

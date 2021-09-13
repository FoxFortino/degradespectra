# DegradeSpectra

This code can properly degrade spectra to any spectral resolution (R) that the user desires given fluxes, wavelength bins, and flux uncertainties. This code does not just re-bin the spectra into wider bins. This code convolves the spectrum with the spectral PSF (default Gaussian), and then linearly interpolates the convolved spectrum to new wavelength bins. <b>Additionally, this code allows for the width of the spectral PSF to vary throughout the convolution.</b>

# Required Packages
NumPy, Matplotlib, imageio, glob

# Usage:
<b>How to use `degrade()`:</b>
`degrade()` is the function that actually degrades spectra. The user must supply three arrays of the same length: the wavelength bin centers, the fluxes corresponding to those bins, and the uncertainties on each flux measurement. Additionally, the desired R value (the new spectroscopic resolution to degrade to) must also be supplied.

Recall that `R = $\frac{\lambda}{\Delta \lambda}$`. The new R value must be less than the current R of the spectra. The current R can be calculated with `calc_avg_R()`.

The default spectral PSF that the spectrum is convolved with is a Gaussian with a FWHM that is proportional to $\Delta \lambda$. That is, the standard deviation of the Gaussian changes throughout the convolution. The user can specify can instead specify one value for the standard deviation, such that it is constant throughout the convolution. The user can also specify an array of standard deviations so that the width of the spectral PSF can be whatever the user desires it to be.

The user can also supply their own spectral PSF function with the `spectral_PSF` keyword argument. It must be a callable and its first three arguments must be `wvl` (the array of wavelength bins), `mu` (the center of the spectral PSF), and `sigma` (the width of the spectral PSF). Any additional arguments must be passed to the function with the PSF_args keyword.

The `makegif` keyword argument can be used to create a gif of the convolution process. An example of a spectrum being degraded from R = 738 to R = 100 would look like this:

![makegif keyword argument example](img/degrading_to_100.gif)

Note how the peak of the kernel gets lower as the kernel moves to the right. This is because the width of the kernel is getting larger. Additionally notice at the edges the peak of the kernel is high. This is because the kernel is normalized to integrate to 1 on the domain of the bins, not from $[-\infinity,\infinity]$.

The `degrade()` function outputs three arrays: `new_wvl` (the new wavelength bin centers at the desired spectral resolution), `new_flux` (the properly degraded fluxes), and `new_err` (the new uncertainties).

<b>How to use `plotSpec`:</b>
`plotSpec()` is a handy function to make a pretty plot of spectra. The function needs the wavelength bin centers for the spectra, and the fluxes corresponding to those spectra. Optionally, the uncertainties on each flux measurement can be provided and they will be plotted as error bars on each flux measurement. An example plot of a spectrum at R = 100 looks like this:

![spectrum plot example](img/sn1998dt_phase0_degraded100.pdf)

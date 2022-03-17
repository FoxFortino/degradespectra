import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

import degrade

# I have not set this folder in my PYTHONPATH so I am doing this instead.
codedir = "/Users/admin/UDel/FASTLab/Summer2021_Research/SESNspectraPCA/code"
imagedir = "/Users/admin/UDel/FASTLab/Summer2021_Research/imagedir"
datadir = ("/Users/admin/Udel/FASTLab/Summer2021_Research/"
           "SESNspectraPCA/Data/DataProducts")
sys.path.insert(0, codedir)

import SNIDdataset as snid
import SNePCA

from IPython import embed


def degrade_snid_dataset(dataset, R):
    """
    Take in a dictionary of SNIDsn objects and degrade their spectra in place.

    Uses Willow Fox Fortino's spectra degradation code to optimially degrade
    the spectra to the desired spectral resolution, R.

    Arguments
    ---------
    dataset : SNIDsn.SNIDsn object
        Dictionary where the key is the supernova name and the value is the
        SNIDsn object corresponding to that supernova. This object should
        typically be read in from a pickle file which can be found at the
        SESNspectraPCA GitHub repo.
    R : float
        The new spectral resolution that you want to degrade the dataset to.
        Keep in mind that you must choose an R that is smaller than the current
        R. If you are unsure what the current R is for a given SNIDsn object,
        you can extract the wavelength array from one and use the calc_avg_R
        function in utils.py. Alternatively, you can just pass a large R value
        to this function and then degrade.degrade_inplace will raise an
        exception and tell you what R value the SNIDsn object is.

    Returns
    -------
    None
    """
    for sn, spec in dataset.items():
        degrade.degrade_inplace(spec, R)


def sn_pca_svc(snidPCA, dim5=False, pc1=None, pc2=None, nCV=50):
    """
    Perform Linear SVC using the PC decomposition of the SESN spectra.

    Given an SNePCA object (see SNePCA.py from SESNspectraPCA GitHub repo) that
    contains the principal component (PC) decomposition of a dataset of
    stripped envelope supernova spectra, where all spectra in a dataset are
    from a particular epoch of the supernova (0±5, 5±5, 10±5, or 15±5 days from
    peak brightness), perform linear support vector classification (Linear SVC)
    using scikit-learn's LinearSVC() function.

    There are four diffferent types of stripped envelope supernova to classify.
    The truth array is calculed using the SNePCA method
    SNePCA.getSNeTypeMasks().

    You can choose to either use just two PCs or the first five PCs for the
    SVC. In Williamson, Modjaz, and Bianco 2019 they figured out, for each
    epoch, which two PCs gave the best SVM score and used those. The reason for
    this is because they found that using more than two PCs did not
    significantly improve the SVM score, and it also denied them the ability to
    neatly graph the feature space.

    Arguments
    ---------
    snidPCA : SNePCA.SNePCA object
        The object which contains the PC decomposition for a dataset of
        supernova spectra from the same epoch.
    dim5 : bool
        Whether to the first five PCs in the SVC or not.
    pc1 : int, Default: None
        Which PC to use for the first feature in the SVC. Any number between 1
        and the number of spectra in the dataset is valid.
    pc2 : int, Default: None
        Which PC to use for the second feature in the SVC. Any number between 1
        and the number of spectra in the dataset is valid.
    nCV : int, Default: 50
        The number of cross validations to perform. Cross validations are
        performed with randomized training and test sets each time, rather than
        consistant sets between each cross validation as you would see in
        normal k-fold cross validation techniques. This is because the number
        of samples in our dataset is very low, and we would like to have a
        _stable_ estimate of the uncertainty. Performing only 3 cross
        validations with training and test sets that are consistent across each
        round would mean that the resulting calculated uncertainties are
        basically worthless since the dataset is so small.
    """
    rng = np.random.RandomState(193)

    IIbMask, IbMask, IcMask, IcBLMask = snidPCA.getSNeTypeMasks()
    truth = 1*IIbMask + 2*IbMask + 3*IcMask + 4*IcBLMask

    if dim5:
        data = snidPCA.pcaCoeffMatrix[:, :4]
    else:
        x = snidPCA.pcaCoeffMatrix[:, pc1-1]
        y = snidPCA.pcaCoeffMatrix[:, pc2-1]
        data = np.column_stack((x, y))

    # rng.shuffle(truth)
    # rng.shuffle(data)

    # WWW In the original work, dual was not set to false and the warnings
    # from sklearn were ignored. According to the documentation, dual should
    # be set to false when n_samples > n_features. We have 5 features for the
    # 5D problem, at 2 features for the 2D problem, and we always have more
    # samples than that. So this should always be set to false.
    linearsvc = LinearSVC(dual=False)

    CV_scores = []
    for i in range(nCV):
        split = train_test_split(data, truth, test_size=0.3, random_state=rng)
        trainX, testX, trainY, testY = split

        linearsvc.fit(trainX, trainY)
        score = linearsvc.score(testX, testY)
        CV_scores.append(score)

    SVM_score_mu = np.mean(np.array(CV_scores))
    SVM_score_sd = np.std(np.array(CV_scores))

    return SVM_score_mu, SVM_score_sd


def SVMscore_degraded(new_R, dataset_pickle, epoch, svc_args={}):
    # Each dataset is a dictionary containing about ~50 SNIDsn objects (see
    # SNIDsn.py in the SESNspectraPCA package). Each SNIDsn object contains
    # fluxes, wavelength bins, uncertainties, and metadata for a stripped
    # envelope supernova spectrum. Each dataset has supernova from only the
    # corresponding epoch (0, 5, 10, or 15). The epoch corresponds to the
    # number of days, ±5 days, since maximum brightness of the supernova.
    dataset = snid.loadPickle(dataset_pickle)

    # Degrade the dataset to the specified spectral resolution.
    if new_R is not None:
        degrade_snid_dataset(dataset, new_R)

    # Use the SNePCA package to create an SNePCA object which has lots of handy
    # function for performing the analysis from Williamson, Modjaz, and Bianco
    # 2019.
    snidPCA = SNePCA.SNePCA(dataset, epoch-5, epoch+5)

    # Calculate the eigenvectors ("eigenspectra"). Stores them in SNePCA.evecs
    snidPCA.snidPCA()

    # Calculates the PCA coefficients based on the eigenspectra and stores them
    # in SNePCA.pcaCoeffMatrix
    snidPCA.calcPCACoeffs()

    SVM_score_mu, SVM_score_sd = sn_pca_svc(snidPCA, **svc_args)

    return SVM_score_mu, SVM_score_sd


def plot_R_vs_SVMscore(R_vals, mu_vals, sd_vals, savedir, savename):
    fig, axes = plt.subplots(nrows=4, ncols=1,
                             sharex=True, sharey=True,
                             figsize=(10, 20))
    plt.subplots_adjust(hspace=0, wspace=0)

    plt.xscale("log")

    axes[0].errorbar(R_vals, mu_vals[0], yerr=sd_vals[0],
                     elinewidth=1, capsize=4, marker="o")
    axes[1].errorbar(R_vals, mu_vals[1], yerr=sd_vals[1],
                     elinewidth=1, capsize=4, marker="o")
    axes[2].errorbar(R_vals, mu_vals[2], yerr=sd_vals[2],
                     elinewidth=1, capsize=4, marker="o")
    axes[3].errorbar(R_vals, mu_vals[3], yerr=sd_vals[3],
                     elinewidth=1, capsize=4, marker="o")

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which="both",
                    top=False, bottom=False,
                    left=False, right=False)
    plt.xlabel("Spectral Resolution, R")
    plt.ylabel("SVM Score")

    axes[0].set_xlim((745, 4))
    axes[0].set_ylim((0.4, 0.9))

    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, savename))
    plt.show()


def plot_R_vs_SVMscore_2D5D(R_vals,
                            mu_vals2D, sd_vals2D,
                            mu_vals5D, sd_vals5D,
                            savedir, savename):
    fig, axes = plt.subplots(nrows=4, ncols=1,
                             sharex=True, sharey=True,
                             figsize=(15, 20))
    plt.subplots_adjust(hspace=0, wspace=0)

    plt.xscale("log")

    axes[0].errorbar(R_vals, mu_vals2D[0], yerr=sd_vals2D[0],
                     elinewidth=1, capsize=4, marker="o",
                     label="Best 2 PCs")
    axes[1].errorbar(R_vals, mu_vals2D[1], yerr=sd_vals2D[1],
                     elinewidth=1, capsize=4, marker="o")
    axes[2].errorbar(R_vals, mu_vals2D[2], yerr=sd_vals2D[2],
                     elinewidth=1, capsize=4, marker="o")
    axes[3].errorbar(R_vals, mu_vals2D[3], yerr=sd_vals2D[3],
                     elinewidth=1, capsize=4, marker="o")

    axes[0].errorbar(R_vals, mu_vals5D[0], yerr=sd_vals5D[0],
                     elinewidth=1, capsize=4, marker="o",
                     label="First 5 PCs")
    axes[1].errorbar(R_vals, mu_vals5D[1], yerr=sd_vals5D[1],
                     elinewidth=1, capsize=4, marker="o")
    axes[2].errorbar(R_vals, mu_vals5D[2], yerr=sd_vals5D[2],
                     elinewidth=1, capsize=4, marker="o")
    axes[3].errorbar(R_vals, mu_vals5D[3], yerr=sd_vals5D[3],
                     elinewidth=1, capsize=4, marker="o")

    axes[0].text(600, 0.4, r"Phase $0 \pm 5$ days")
    axes[1].text(600, 0.4, r"Phase $5 \pm 5$ days")
    axes[2].text(600, 0.4, r"Phase $10 \pm 5$ days")
    axes[3].text(600, 0.4, r"Phase $15 \pm 5$ days")

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which="both",
                    top=False, bottom=False,
                    left=False, right=False)
    plt.xlabel("Spectral Resolution, R")
    plt.ylabel("SVM Score")

    axes[0].set_xlim((745, 4))
    axes[0].set_ylim((0.35, 0.95))

    axes[0].legend(loc="upper right", fontsize=30)
    plt.grid(which="both", alpha=0.5)
    plt.savefig(os.path.join(savedir, savename))
    plt.show()


def calc_R_vs_SVMscore(R_vals, dim5, dataset_pkls):
    mu_vals = np.zeros((4, len(R_vals)))
    sd_vals = np.zeros((4, len(R_vals)))
    for i, new_R in enumerate(R_vals):
        mu0, sd0 = SVMscore_degraded(new_R, dataset_pkls[0], 0,
                                     {"dim5": dim5,
                                      "pc1": 1,
                                      "pc2": 5,
                                      "nCV": 50})
        mu5, sd5 = SVMscore_degraded(new_R, dataset_pkls[1], 0,
                                     {"dim5": dim5,
                                      "pc1": 1,
                                      "pc2": 3,
                                      "nCV": 50})
        mu10, sd10 = SVMscore_degraded(new_R, dataset_pkls[2], 0,
                                       {"dim5": dim5,
                                        "pc1": 1,
                                        "pc2": 3,
                                        "nCV": 50})
        mu15, sd15 = SVMscore_degraded(new_R, dataset_pkls[3], 0,
                                       {"dim5": dim5,
                                        "pc1": 1,
                                        "pc2": 3,
                                        "nCV": 50})

        mu_vals[0, i] = mu0
        mu_vals[1, i] = mu5
        mu_vals[2, i] = mu10
        mu_vals[3, i] = mu15

        sd_vals[0, i] = sd0
        sd_vals[1, i] = sd5
        sd_vals[2, i] = sd10
        sd_vals[3, i] = sd15

    return mu_vals, sd_vals

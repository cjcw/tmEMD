
import numpy as np
from scipy import stats
import emd
import multiprocessing 
import multiprocessing.pool
import time
import itertools

# fix - documentation: specify default values

### --- Multiprocessing --- ###
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class _pool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


### --- Power Spectral Density (PSD) --- ###
# fix - unsupervised way to cover suitable freqs?

def get_psd(X, sample_rate, window='hann'):
    """
    This function may need to be modified so that the variable freqAx_psd covers a suitable frequency range 
    # fix - words
    """

    def get_psd_(X, sample_rate, maxFreq, pointsPerHz):
        from scipy.signal import welch
        psdIndMax = int(maxFreq*pointsPerHz)
        psd = welch(X, fs=sample_rate, window=window, nperseg=int(sample_rate)*pointsPerHz)
        freqAx_psd, psd = psd[0][0:psdIndMax], psd[1][0:psdIndMax]
        return freqAx_psd, psd

    freqAx_psd, psd = [], []
    maxFreqs = [1, 10, 100, 200, 500]
    pointsPerHzs = [20, 4, 1, 0.04, 0.02]
    for maxFreq, pointsPerHz in zip(maxFreqs, pointsPerHzs):
        freqAx_psd_, psd_ = get_psd_(X, sample_rate, maxFreq, pointsPerHz)
        if not len(freqAx_psd):
            i = 0
        else:
            i = np.where(freqAx_psd_ > freqAx_psd[-1][-1])[0][0]
        freqAx_psd.append(freqAx_psd_[i:])
        psd.append(psd_[i:])
    freqAx_psd = np.concatenate(freqAx_psd)
    psd = np.concatenate(psd)

    return freqAx_psd, psd

    
    
def get_psd2(X, sample_rate, window='hann'):

    def get_psd_(X, sample_rate, maxFreq, pointsPerHz):
        from scipy.signal import welch
        psdIndMax = int(maxFreq*pointsPerHz)
        psd = welch(X, fs=sample_rate, window=window, nperseg=int(sample_rate)*pointsPerHz)
        freqAx_psd, psd = psd[0][0:psdIndMax], psd[1][0:psdIndMax]
        return freqAx_psd, psd

    freqAx_psd, psd = [], []
    maxFreqs = [301]
    pointsPerHzs = [1]
    for maxFreq, pointsPerHz in zip(maxFreqs, pointsPerHzs):
        freqAx_psd_, psd_ = get_psd_(X, sample_rate, maxFreq, pointsPerHz)
        if not len(freqAx_psd):
            i = 0
        else:
            i = np.where(freqAx_psd_ > freqAx_psd[-1][-1])[0][0]
        freqAx_psd.append(freqAx_psd_[i:])
        psd.append(psd_[i:])
    freqAx_psd = np.concatenate(freqAx_psd)
    psd = np.concatenate(psd)

    return freqAx_psd, psd

# fix - add another function for exploring consistency plots to [figplot_tmEMD] for return_label docstring
### --- Mode mixing --- ###
def get_modeMixScore_corr(imfs, imfis_4_scoring, sample_rate=None, compute=True,
                          return_label=False, label='Mode mixing score (r)'):
    """
    Computes the mode mixing score for a set of IMFs by measuring their pairwise Pearson correlations.
    
    Parameters
    ----------
    imfs : ndarray
        2D [time x N_imfs] array. (Note: this is called 'imf' in the emd package.)
    imfis_4_scoring : ndarray | None
        1D array containing the indices of the IMFs to be used to compute mix scoring. If None, all indices will be used.
    sample_rate : float | None
        The sampling rate for all the data provided in Xs. This is not needed for this function, so default = None.
    compute : bool
        Should the score be computed. Set to False if only the label is needed.
    return_label : bool
        This will be set to True when this function is used for plotting functions [figplot_tmEMD], so that 
        the axis is labelled appropriately.
    label : string
        A label for an axis for mode mixing plots (see above).
    
    Returns
    -------
    mixScore : float
        The mode mixing score for the set of IMFs.
    label : string
        Only if return_label=True
    """
    
    if compute:
        corMat = np.abs(np.corrcoef(imfs[:, imfis_4_scoring].T))
        corMat[np.tril_indices(corMat.shape[0], k=0)] = np.nan
        mixScore = np.nanmean(corMat)
    else:
        mixScore = None
    
    if return_label:
        return mixScore, label
    return mixScore


def get_modeMixScore_imfPSDs(imfs, imfis_4_scoring, sample_rate, psd_func=get_psd, compute=True,  
                             return_label=False, label='Mode mixing score (IMF PSD corr.)'):
    """ 
    Computes the mode mixing score for a set of IMFs by measuring the pairwise Pearson correlations between IMF power spectra (PSDs).
    
    Parameters
    ----------
    imfs : ndarray
        2D [time x N_imfs] array. (Note: this is called 'imf' in the emd package.)
    imfis_4_scoring : ndarray | None
        1D array containing the indices of the IMFs to be used to compute mix scoring. If None, all indices will be used.
    sample_rate : float | None
        The sampling rate for all the data provided in Xs
    psd_func : function
        Function to compute the Power Spectral Density (PSD) of IMFs.
        The function should take two arguments: (X, sample_rate) and return the frequency axis and PSD. 
        The frequency axis returned should cover an appropriate range for frequencies of interest.
    compute : bool
        Should the score be computed. Set to False if only the label is needed.
    return_label : bool
        This will be set to True when this function is used for plotting functions [figplot_tmEMD], so that 
        the axis is labelled appropriately.
    label : string
        A label for an axis for mode mixing plots (see above).
    
    Returns
    -------
    mixScore : float
        The mode mixing score for the set of IMFs.
    label : string
        Only if return_label=True
    """
    
    if compute:
        freqAx_psd, imfPSDs = get_imfPSDs(imfs, sample_rate=sample_rate, psd_func=psd_func)
        corMat = np.abs(np.corrcoef(imfPSDs[imfis_4_scoring, :]))
        corMat[np.tril_indices(corMat.shape[0], k=0)] = np.nan
        mixScore = np.nanmean(corMat)
    else:
        mixScore = None
    
    if return_label:
        return mixScore, label
    return mixScore
    
def get_modeMixScore_4_imfPSDs(imfPSDs, imfis_4_scoring, sample_rate=None, compute=True, 
                               return_label=False, label='Mode mixing score (IMF PSD corr.)'):
    """ 
    Computes the mode mixing score for a set of IMF Power Spectra (PSDs) by measuring their pairwise Pearson correlations.
    
    Parameters
    ----------
    imfPSDs : ndarray
        2D [N_imfs x frequency] array; returned by get_imfPSDs().
    imfis_4_scoring : ndarray | None
        1D array containing the indices of the IMFs to be used to compute mix scoring. If None, all indices will be used.
    sample_rate : float | None
        The sampling rate for all the data provided in Xs. This is not needed for this function, so default = None.
    compute : bool
        Should the score be computed. Set to False if only the label is needed.
    return_label : bool
        This will be set to True when this function is used for plotting functions [figplot_tmEMD], so that 
        the axis is labelled appropriately.
    label : string
        A label for an axis for mode mixing plots (see above).
    
    Returns
    -------
    mixScore : float
        The mode mixing score for the set of IMFs.
    label : string
        Only if return_label=True
    """
    if compute:
        corMat = np.abs(np.corrcoef(imfPSDs[imfis_4_scoring, :]))
        corMat[np.tril_indices(corMat.shape[0], k=0)] = np.nan
        mixScore = np.nanmean(corMat)
    else:
        mixScore = None
        
    if return_label:
        return mixScore, label
    return mixScore
    
def PMSI(imfs, imfi, method='both'):
    """
    #########################
    Code author: Marco Fabus 
    https://gitlab.com/marcoFabus/fabus2021_itemd/-/blob/main/Tools/analysis.py
    
    Method reference:
    Wang Y-H,Hu K,Lo M-T. (2018)
    Uniform phase empirical mode decomposition: an optimal hybridization of masking signal and ensembleapproaches.
    ###########################

    
    Computes pseudo-mode mixing index of an intrinsic mode function.

    Parameters
    ----------
    imf : 2D array
        Set of IMFs.
    m : int
        Mode to calculate the PMSI of.
    method : string, optional
        Calculate PMSI as sum of PMSI between mode m and both above / below 
        modes, or only above / below mode. The default is 'both'.

    Returns
    -------
    float
        PMSI calculated.
    """
    
    if method == 'both':
        abs1 = (imfs[:, imfi].dot(imfs[:, imfi]) + imfs[:, imfi-1].dot(imfs[:, imfi-1]))
        pmsi1 = np.max([np.dot(imfs[:, imfi], imfs[:,imfi-1]) / abs1, 0])
        abs2 = (imfs[:, imfi].dot(imfs[:, imfi]) + imfs[:, imfi+1].dot(imfs[:, imfi+1]))
        pmsi2 = np.max([np.dot(imfs[:, imfi], imfs[:,imfi+1]) / abs2, 0])
        return pmsi1 + pmsi2

    if method == 'above':
        abs1 = (imfs[:, imfi].dot(imfs[:, imfi]) + imfs[:, imfi-1].dot(imfs[:, imfi-1]))
        pmsi1 = np.max([np.dot(imfs[:, imfi], imfs[:,imfi-1]) / abs1, 0])
        return pmsi1

    if method == 'below':
        abs2 = (imfs[:, imfi].dot(imfs[:, imfi]) + imfs[:, imfi+1].dot(imfs[:, imfi+1]))
        pmsi2 = np.max([np.dot(imfs[:, imfi], imfs[:,imfi+1]) / abs2, 0])
        return pmsi2

def get_modeMixScore_pmsi(imfs, imfis_4_scoring, sample_rate=None, compute=True, 
                          return_label=False, label='Mode mixing score (PMSI)'):
    """ 
    Computes the mode mixing score for a set of IMFs by measuring the pseudo mode-splitting index (PMSI) for each IMF of interest.
    
    Parameters
    ----------
    imfs : ndarray
        2D [time x N_imfs] array. (Note: this is called 'imf' in the emd package.)
    imfis_4_scoring : ndarray | None
        1D array containing the indices of the IMFs to be used to compute mix scoring. If None, all indices will be used.
    sample_rate : float | None
        The sampling rate for all the data provided in Xs. This is not needed for this function, so default = None.
    compute : bool
        Should the score be computed. Set to False if only the label is needed.
    return_label : bool
        This will be set to True when this function is used for plotting functions [figplot_tmEMD], so that 
        the axis is labelled appropriately.
    label : string
        A label for an axis for mode mixing plots (see above).
    
    Returns
    -------
    mixScore : float
        The mode mixing score for the set of IMFs.
    label : string
        Only if return_label=True
    """
    
    if compute:
        mixScore = np.array([PMSI(imfs, imfi) for imfi in imfis_4_scoring if imfi+1 < imfs.shape[1]]).mean()
    else:
        mixScore = None
    
    if return_label:
        return mixScore, label
    return mixScore


### --- Consistency --- ###
def get_consistencyScores(freqAx_psd, X_imfPSDs, imfis_4_scoring, f_ranges0=None, use_f_ranges0=False, 
                          compute=True, return_label=False, label='Consistency (IMF PSD corr.)'):
    """ 
    Computes the consistency scores for the IMF PSDs for each input signal X.
    
    Parameters
    ----------
    freqAx_psd : ndarray
        1D frequency axis array, corresponding to the last dimension of X_imfPSDs.
    X_imfPSDs : ndarray
        3D [N_X x N_IMFs x frequency] array, containing the IMF PSDs obtained from the IMFs of each X.
    imfis_4_scoring : ndarray | None
        1D array containing the indices of the IMFs to be used to compute mix scoring. If None, all indices will be used.
    f_ranges0 : ndarray | None
        if not None and use_f_ranges0 is True, the PSDs of each IMF will be trimmed according to the frequency ranges 
        specified by f_ranges0. If None (default), the entire PSD will be used.
    use_f_ranges0 : bool
        If True, the PSDs will be trimmed according to the frequency limits specified in f_ranges0 before measuring 
        consistency.
    compute : bool
        Should the score be computed. Set to False if only the label is needed.
    return_label : bool
        This will be set to True when this function is used for plotting functions [figplot_tmEMD], so that 
        the axis is labelled appropriately
    label : string
        A label for an axis for consistency plots (see above).
    
    Returns
    -------
    consistencyScores : ndarray
        1D ndarray of length X_imfPSDs.shape[0]; each element being the mean consistency score for the IMF PSDs of that 
        X to all other Xs.
    label : string
        Only if return_label=True
    """
    
    try:
        n_X, n_imfs, _ = X_imfPSDs.shape
    except AttributeError:
        n_X, n_imfs = None, None
        
    if compute and n_X > 1:
        if imfis_4_scoring is None:
            imfis_4_scoring = np.arange(n_imfs)
        if f_ranges0 is None or not use_f_ranges0:
            f_sts = [0]*n_imfs
            f_ens = [len(freqAx_psd)]*n_imfs
        else:
            f_sts = [np.abs(freqAx_psd-f).argmin() for f in f_ranges0[:, 0]]
            f_ens = [np.abs(freqAx_psd-f).argmin() for f in f_ranges0[:, 1]]

        obs = np.row_stack([np.concatenate([imfPSDs[imfi, st:en] for imfi, st, en in \
                                            zip(imfis_4_scoring, f_sts, f_ens)]) for imfPSDs in X_imfPSDs])
        corMat = np.corrcoef(obs)
        consistencyScores = np.array([corMat[np.setdiff1d(np.arange(n_X), [xi]), xi].mean() for xi in range(n_X)])
    elif compute:
        consistencyScores = np.array([np.nan])
    else:
        consistencyScores = None
    
    if return_label:
        return consistencyScores, label
    return consistencyScores
    
    
### --- Utilities --- ###
def get_inds4propPSD(psd, prop_psd):
    """
    Computes the consistency scores across IMF PSDs for each input signal X. # fix -
    
    Parameters
    ----------
    psd : ndarray
        1D Power Spectral Density vector
    prop_psd : float (0 < prop_psd < 1)
        The proportion of the PSD to cover. Larger values will give a wider range
    Returns
    -------
    st, en : int, int
        The indices corresponding to the start and end of the psd that cover prop_psd
    """

    a0 = psd.sum()
    maxi = psd.argmax()
    
    if psd[maxi] / a0 > prop_psd:
        if maxi == 0:
            st, en = [maxi, maxi+2]
        elif maxi == len(psd)-1:
            st, en = [maxi-2, maxi]
        else:
            st, en = [maxi-1, maxi+1]
        return st, en
    
    
    if maxi in [0, len(psd)-1]:
        if maxi == 0:
            st, en = 0, 1
            finished_start, finished_end = True, False
        else:
            st, en = maxi-1, maxi
            finished_start, finished_end = False, True
    else:
        st, en = maxi-1, maxi+1
        finished_start, finished_end = False, False
    
    for _ in psd:
        if st == 0:
            finished_start = True
        elif en == len(psd)-1:
            finished_end = True
        
        if finished_start:
            en += 1
        elif finished_end:
            st -= 1
        elif psd[st-1] > psd[en+1]:
            st -= 1
        else:
            en += 1
        
        if psd[st:en].sum()/a0 > prop_psd:
            break
    
    return st, en


def it_X_2array(it_mask_freqs, it_mix_scores, it_adj_mix_scores, it_consistency_scores):
    """
    Converts arguments (which exist in list format between iterations) into numpy arrays.
    """
    if isinstance(it_mask_freqs, list):
        it_mask_freqs = np.row_stack(it_mask_freqs)
    if isinstance(it_mix_scores, list):
        it_mix_scores = np.row_stack(it_mix_scores)
    if isinstance(it_adj_mix_scores, list):
        it_adj_mix_scores = np.concatenate(it_adj_mix_scores)
    if isinstance(it_consistency_scores, list):
        it_consistency_scores = np.row_stack(it_consistency_scores)
    
    return it_mask_freqs, it_mix_scores, it_adj_mix_scores, it_consistency_scores


def get_imfPSDs(imfs, sample_rate, psd_func=get_psd):
    """
    Get the power spectral density (PSD) estimates for each IMF.
    """
    imfPSDs = []
    for imf in imfs.T:
        freqAx_psd, psd = psd_func(imf, sample_rate)
        imfPSDs.append(psd)
    imfPSDs = np.row_stack(imfPSDs)
    
    return freqAx_psd, imfPSDs


def get_f_ranges_from_imfPSDs(freqAx_psd, imfPSDs, prop_psd=0.8, f_set=None):
    """
    Defines frequency ranges for each IMF according to its PSD.
    
    Parameters
    ----------
    freqAx_psd : ndarray
        1D frequency axis array, corresponding to the last dimension of imfPSDs.
    imfPSDs : ndarray
        2D [n_imfs X frequency] array, containing the PSDs of each IMF
    prop_psd : float (0 < prop_psd < 1)
        The proportion of the PSD used to define the frequency range. Larger values will yield a wider range
    f_set : None | ndarray
        1D array specifying whether a mask frequency is to be fixed or variable for the algorithm. If fixed, the entry 
        should be the desired frequency (in Hz). If variable, the entry should be None.
        
    Returns
    -------
    f_ranges : ndarray
        2D [N_imfs X 2] array containing the minimum and maximum frequency values for that IMF.
    """

    f_ranges = []
    for imfi, psd in enumerate(imfPSDs):
        st, en = get_inds4propPSD(psd, prop_psd)
        f_ranges.append([freqAx_psd[st], freqAx_psd[en]])
    f_ranges = np.row_stack(f_ranges)
    
    if f_set is not None:
        for i, f in enumerate(f_set):
            if f is not None:
                f_ranges[i] = [f, f]
    
    return f_ranges



def get_adj_fis(nImfs):
    adj_fis = np.column_stack([np.arange(nImfs)[:-1], np.arange(nImfs)[1:]])
    return adj_fis

def get_adj_prop(it_mix_scores, it_adj_mix_scores, top_n):
    it_mix_scores_M = it_mix_scores.mean(axis=1)
    inds = np.where(np.argsort(np.argsort(it_mix_scores_M)) < top_n)[0]
    adj_prop = it_adj_mix_scores[inds].mean(axis=-1).mean(axis=0)
    for i in np.where(adj_prop < 0)[0]:
        adj_prop[i] = 0
    adj_prop /= adj_prop.sum()
    return adj_prop


def get_f_adjis(nImfs):
    f_adjis = []
    for fi in range(nImfs):
        if fi == 0:
            f_adjis.append([fi])
        elif fi < nImfs-1:
            f_adjis.append([fi-1, fi])
        else:
            f_adjis.append([fi-1])
    return f_adjis


def get_f_freqs(freqs0, f_ranges):
    """
    Get the frequency values to be used to generate mask frequency combinations. 
    """
    f_freqs = [freqs0[np.where(np.logical_and(freqs0 >= fMin, freqs0 <= fMax))[0]] for fMin, fMax in f_ranges]
    return f_freqs

def get_f_ranges(it_mask_freqs, it_mix_scores, it_adj_mix_scores, it_consistency_scores, top_n, f_set=None):
    """
    Defines frequency ranges for each IMF to select mask frequency values for the next iteration.
        
    Parameters
    ----------
    it_mask_freqs : list | ndarray
        2D array [N_sub-iterations x N_maskFreqs] containing the mask frequencies used for each mEMD sub-iteration
    it_mix_scores : list | ndarray
        2D array [N_sub-iterations x N_X] containing the mode mixing scores yeilded for each sub-iteration for each X
    it_adj_mix_scores : list | ndarray
        3D array [N_sub-iterations x N_maskFreqs-1 x N_X] containing the mode mixing scores for adjacent IMFs 
        yeilded for each sub-iteration for each X. The index of the second dimension (N_adj) corresponds to the mixing 
        between that IMF index and the successive one. This also corresponds to get_adj_fis(). # fix - 
    it_consistency_scores : list | ndarray
        2D array [N_sub-iterations x N_X] containing the consistency scores yeilded for each mEMD sub-iteration
    top_n : int
        Frequency ranges will be narrowed according those appearing in top N least mixed sub-iterations.
    f_set : None | ndarray
        1D array specifying whether a mask frequency is to be fixed or variable for the algorithm. If fixed, the entry 
        should be the desired frequency (in Hz). If variable, the entry should be None.

    Returns
    -------
    f_ranges : ndarray
        2D [N_imfs X 2] array containing the minimum and maximum frequency values for that IMF.
    """

    it_mask_freqs, it_mix_scores, it_adj_mix_scores, it_consistency_scores = \
    it_X_2array(it_mask_freqs, it_mix_scores, it_adj_mix_scores, it_consistency_scores)
    
    it_mix_scores_M = it_mix_scores.mean(axis=1)
    it_adj_mix_scores_M = it_adj_mix_scores.mean(axis=-1)
    it_sortInds = np.argsort(np.argsort(it_mix_scores_M))

    adj_fis = get_adj_fis(it_mask_freqs.shape[-1])
    topi = np.where(it_sortInds == 0)[0][0]
    inds = np.where(it_sortInds < top_n)[0]

    f_ranges = np.column_stack([it_mask_freqs[topi]]*2)

    adj_mix_scores_top = it_adj_mix_scores_M[topi]
    for adji, mix_score_top in enumerate(adj_mix_scores_top):
        betterInds = inds[np.where(mix_score_top > it_adj_mix_scores_M[inds][:, adji])[0]]
        if not len(betterInds):
            continue
        for fi in adj_fis[adji]:
            fmin, fmax = [f(it_mask_freqs[betterInds][:, fi]) for f in [np.min, np.max]]
            if f_ranges[fi, 0] > fmin:
                f_ranges[fi, 0] = fmin
            if f_ranges[fi, 1] < fmax:
                f_ranges[fi, 1] = fmax
        
    if f_set is not None:
        for i, f in enumerate(f_set):
            if f is not None:
                f_ranges[i] = [f, f]

    return f_ranges

def reduce_f_freqs(f_freqs, it_mix_scores, it_mask_freqs, it_adj_mix_scores, top_n, dim_red_prop):
    """
    # fix - 
    Work in progress - this function will only be implemented in the algorithm if 
    max_iterations > max_iterations_b4_dim_red in optimise_mask_freqs__().
    """
    
    it_mask_freqs, it_mix_scores, it_adj_mix_scores = it_X_2array(it_mask_freqs, it_mix_scores, it_adj_mix_scores)
    
    nImfs = len(f_freqs)
    adj_fis = get_adj_fis(nImfs)
    adj_prop = get_adj_prop(it_mix_scores, it_adj_mix_scores, top_n)
    f_adjis = get_f_adjis(nImfs)
    f_prop = np.array([np.mean([adj_prop[adji] for adji in adjis]) for adjis in f_adjis])
    f_nFreqs = np.array([len(fs) for fs in f_freqs])
    n_freqDim0 = f_nFreqs.sum()
    n2lose = n_freqDim0 - int(n_freqDim0*dim_red_prop)
    
    f_weights = np.array([((ni-1)/n_freqDim0) * (1-propi) for ni, propi in zip(f_nFreqs, f_prop)])
    f_weights /= f_weights.sum()
    f_n2lose = np.array(n2lose*f_weights, dtype=int)
    
    f_freqs_ = []
    for fi, n2lose, nFreqs, freqs in zip(range(nImfs), f_n2lose, f_nFreqs, f_freqs):
        if n2lose:
            n_left = nFreqs - n2lose
            if n_left >= 2:
                if freqs[0]:
                    freqs_ = np.geomspace(freqs[0], freqs[-1], n_left) # fix - sample from freqs0 rather than introduce novel freqs.
                else:
                    freqs_ = np.concatenate([[0.], np.geomspace(freqs[1], freqs[-1], n_left-1)])
            else:
                # pick the mask freq from the current space which yielded the lowest mixing scores for adjacent IMFs
                adjis = np.where([fi in inds for inds in adj_fis])[0]
                i = np.argmin([it_adj_mix_scores[it_mask_freqs[:, fi] == f][:, adjis].mean() for f in freqs])
                freqs_ = np.array([freqs[i]])
        else:
            freqs_ = freqs
        f_freqs_.append(freqs_)
    
    return f_freqs_



### --- tmEMD algorithm --- ###
def run_subIteration(args):
    """ 
    Randomly generates mask frequencies to apply mask-EMD to Xs and returns the 
    mask freqs and the mixing scores
    """
    
    Xs, f_freqs, imfis_4_scoring, sample_rate, mask_args, mixScore_func, consistency_func, compute_consistency, f_ranges0, nprocesses = args
    
    nImfs = len(f_freqs)
    
    # use fRange to randomly generate mask freq within each range
    np.random.seed()
    # fix - make nTries more elegant
    nTries = 50
    invalid = False
    
    # fix - ugly and use when pre_emd_mode is None
    perms = itertools.combinations(np.arange((len(f_freqs))), 2)
    if all([np.array_equal(f_freqs[i], f_freqs[j]) for i, j in list(perms)]):
        mask_freqs = np.array(sorted([np.random.choice(freqs) for freqs in f_freqs]))[::-1]
    
    else:
        for tryi in range(nTries):
            mask_freqs = np.array([np.random.choice(freqs) for freqs in f_freqs])
            if all(np.diff(mask_freqs) <= 0): # if randomly selected freqs are in descending order
                break
            else:
                if tryi == nTries-1:
                    mask_freqs = np.repeat(np.nan, nImfs)
                    invalid = True
    if invalid:
        mix_scores_ = np.repeat(np.nan, len(Xs))
        adjMixScores_ = np.full([nImfs-1, len(Xs)], np.nan)
        consistencyScores_ = np.repeat(np.nan, len(Xs))
        return mask_freqs, mix_scores_, adjMixScores_, consistencyScores_
        
    
    # get sift args
    sift_config = emd.sift.get_config('mask_sift')
    sift_config['mask_freqs'] = mask_freqs/sample_rate
    sift_config['max_imfs'] = len(mask_freqs)
    for k in mask_args:
        if mask_args[k] is not None:
            sift_config[k] = mask_args[k]
            
    mix_scores_ = []
    adjMixScores_ = []
    if compute_consistency:
        X_imfPSDs = []
    
    for X in Xs:
        imfs = emd.sift.mask_sift(X, **sift_config)
        if imfs.shape[1] != nImfs:
            mix_scores_.append(np.nan)
            adjMixScores_.append(np.repeat(np.nan, nImfs-1))
            continue
        mix_scores_.append(mixScore_func(imfs, imfis_4_scoring, sample_rate))
        adj_fis = get_adj_fis(nImfs)
        corMat = np.corrcoef(imfs.T)
        adjMixScores_.append(np.array([corMat[x, y] for x, y in adj_fis]))
        
        if compute_consistency:
            freqAx_psd, imfPSDs = get_imfPSDs(imfs, sample_rate)
            X_imfPSDs.append(imfPSDs)
    #
    
    if compute_consistency:
        X_imfPSDs = np.array(X_imfPSDs)
        consistencyScores_ = consistency_func(freqAx_psd, X_imfPSDs, imfis_4_scoring, f_ranges0)
    else:
        consistencyScores_ = None
    
    mix_scores_ = np.array(mix_scores_)
    adjMixScores_ = np.column_stack(adjMixScores_)
    
    return mask_freqs, mix_scores_, adjMixScores_, consistencyScores_
    

def run_iteration(Xs, f_freqs, imfis_4_scoring, sample_rate, mask_args, mixScore_func, consistency_func, compute_consistency, 
                  f_ranges0, nprocesses, n_per_it):
    
    pool = _pool(nprocesses)
    args = (Xs, f_freqs, imfis_4_scoring, sample_rate, mask_args, mixScore_func, consistency_func, compute_consistency, 
            f_ranges0, nprocesses)
    it_outputs = pool.map(run_subIteration, [args for i in range(n_per_it)])
    pool.close()
    pool.join()

    it_mask_freqs = np.row_stack([it_outputsi[0] for it_outputsi in it_outputs])
    it_mix_scores = np.row_stack([it_outputsi[1] for it_outputsi in it_outputs])
    it_adj_mix_scores = np.array([it_outputsi[2] for it_outputsi in it_outputs])
    
    
    if compute_consistency:
        it_consistency_scores = np.row_stack([it_outputsi[3] for it_outputsi in it_outputs])
    else:
        it_consistency_scores = None
        
    return it_mask_freqs, it_mix_scores, it_adj_mix_scores, it_consistency_scores



def optimise_mask_freqs__(Xs, sample_rate, psd_func=get_psd, freqs0=None, pre_emd_mode='eEMD', prop_psd=0.8,
                          nensembles=4, X_paths2imfs=None, max_imfs=10, f_set=None, imfis_4_scoring=None, 
                          mixScore_func=get_modeMixScore_imfPSDs, compute_consistency=False, consistency_func=get_consistencyScores, 
                          n_per_it=200, top_n=10, max_iterations=12, max_iterations_b4_dim_red=20, dim_red_prop=0.5, 
                          nprocesses=1, mask_amp=1, mask_amp_mode='ratio_imf', sift_thresh=1e-08, nphases=4, 
                          imf_opts={}, envelope_opts={}, extrema_opts={}):
    """
    Find the set of mask frequencies which yeild IMFs with the lowest loss function output.  
    
    Parameters
    ----------
    Xs : list
        Each element is a 1D array containing a sample time-series used to tune the optimisation.
    sample_rate : float
        The sampling rate for all the data provided in Xs.
    psd_func : function
        Function to compute the Power Spectral Density (PSD) of IMFs.
        The function should take two arguments: (X, sample_rate) and return the frequency axis and PSD. 
        The frequency axis returned should cover an appropriate range for frequencies of interest.
    freqs0 : ndarray | None
        1D array containing all the frequency values (in Hz) for the algorithm to select mask frequencies from. If None, 
        frequency values from the freqAx_psd returned by psd_func will be used. 
    pre_emd_mode : str | None
        The EMD variant to be used to first estimate mask frequency ramges from IMF PSDs. Default is ensemble EMD. 
        If None, mask frequency ranges will not be narrowed.
    prop_psd : float
        If pre-EMD is used as per the above argument, this will specify the proportion of the PSD of each IMF from the 
        pre-EMD to specify the frequency range.
    nensembles : int
        If eEMD is used, the number of ensembles to run
    X_paths2imfs : list | None
        If a list is given, it should the same length as Xs; each element being a path a .npy file which corresponds to 
        the IMFs of that signal (X) which are used instead of running pre-EMD. If the path is None, pre-EMD will be run 
        for that X as above.
    max_imfs : int
        The maximum number of IMFs. Used for pre-EMD and to specify the number of mask frequencies to be used for the 
        mEMD sub-iterations.
    f_set : None | ndarray
        1D array specifying whether a mask frequency is to be fixed or variable for the algorithm. If fixed, the entry 
        should be the desired frequency (in Hz). If variable, the entry should be None.
    imfis_4_scoring : ndarray | None
        1D array containing the indices of the IMFs to be used to compute mix scoring. If None, all indices will be used
    mixScore_func : function
        The function used to compute the mode mixing between IMFs. It should take three arguments: 
        (imfs, imfis_4_scoring, sample_rate) and return a single number; lower meaning less mode mixing (desirable)
    compute_consistency : bool
        Measure the IMF consistency for each mEMD process
    consistency_func : function
        The function to be used to compute the IMF consistency. It should take (freqAx_psd, X_imfPSDs, imfis_4_scoring) 
        as key arguments and return a 1D ndarray of length N_X; each element being the mean consistency score for that X to all other Xs
    n_per_it : int
        Number of mEMD sub-iterations to run within an iteration.
    top_n : int
        After each iteration, the frequency ranges for each mask frequency will be retricted by the ranges seen in the  
        to the best (i.e. least-mixed) sub-iterations.
    max_iterations : int
        The maximum number of iterations to run
    max_iterations_b4_dim_red : int
        If lower than max_iterations, once max_iterations_b4_dim_red is reached, the algorithm will attempt to increase 
        convergence speed by reducing the number of frequencies to choose from. This reduction is guided accrding to where 
        mode mixing is stronger for adjacent IMFs, and how many frequcies currently can be selected for a given mask frequency.
        Note: this option is still work in progess! fix - 
    dim_red_prop : float
        Proportion of frequencies to loose as per above (a lower number will increase convergence rate).
    nprocesses : int
        Integer number of parallel processes to compute (Default value = 1)
    mask_amp : float or array_like
        Amplitude of mask signals as specified by mask_amp_mode. If float the same value is applied to all IMFs, 
        if an array is passed each value is applied to each IMF in turn (Default value = 1)
    mask_amp_mode : {'abs', 'ratio_imf', 'ratio_sig'}:
        Method for computing mask amplitude. Either in absolute units ('abs'), or as a ratio of the standard deviation 
        of the input signal ('ratio_sig') or previous imf ('ratio_imf') (Default value = 'ratio_imf')
    sift_thresh : float
        By default will be ignored inplace of max_imfs. The threshold at which the overall sifting process 
        will stop. (Default value = 1e-8)
    nphases : int > 0
        The number of separate sinusoidal masks to apply for each IMF, the phase of masks are uniformly spread 
        across a 0<=p<2pi range (Default = 4).
    imf_opts : dict
        Optional dictionary of keyword arguments to be passed to emd.get_next_imf
    envelope_opts : dict
        Optional dictionary of keyword options to be passed to emd.interp_envelope
    extrema_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_padded_extrema
        
    
    Returns
    -------
    it_mask_freqs, it_mix_scores, it_adj_mix_scores, it_consistency_scores, it_is, optimised_mask_freqs, converged
    
    it_mask_freqs : ndarray
        2D array [N_sub-iterations x N_maskFreqs] containing the mask frequencies used for each mEMD sub-iteration
    it_mix_scores : ndarray
        2D array [N_sub-iterations x N_X] containing the mode mixing scores yeilded for each sub-iteration for each X
    it_adj_mix_scores : ndarray
        3D array [N_sub-iterations x N_maskFreqs-1 x N_X] containing the mode mixing scores for adjacent IMFs 
        yeilded for each sub-iteration for each X. The index of the second dimension (N_adj) corresponds to the mixing 
        between that IMF index and the successive one. This also corresponds to get_adj_fis(). # fix - 
    it_consistency_scores : ndarray
        2D array [N_sub-iterations x N_X] containing the consistency scores yeilded for each mEMD sub-iteration
    it_is : ndarray
        1D array [N_sub-iterations] containing the iteration index corresponding to each mEMD sub-iteration
    optimised_mask_freqs : ndarray
        1D array containing the mask frequencies which yeilded the lowest average score from mixScore_func
    converged : bool
        True if the algorithm converged to an 'optimised' set of mask frequencies
    """
    
    # checks - make a function
    if f_set is not None:
        if len(f_set) != max_imfs:
            print('Warning max_imfs different from f_set')
            return
            
    if len(Xs) == 1 and compute_consistency:
        print('Warning: len(Xs) must be more than 1 in order to compute consistency')
        compute_consistency = False
    
    if pre_emd_mode is not None:
        X_f_ranges = []
        if X_paths2imfs is None:
            X_paths2imfs = [None]*len(Xs)
        for X, path2imfs in zip(Xs, X_paths2imfs):
            if path2imfs is None:
                if pre_emd_mode == 'eEMD':
                    imfs = emd.sift.ensemble_sift(X, nensembles=nensembles, max_imfs=max_imfs, nprocesses=nprocesses)
                elif pre_emd_mode == 'itEMD':
                    from ccw_it_emd import it_emd
                    imfs = it_emd(X, sample_rate, N_imf=max_imfs)[0]
            else:
                imfs = np.load(path2imfs)
            freqAx_psd, imfPSDs = get_imfPSDs(imfs, sample_rate, psd_func=psd_func)
            f_ranges = get_f_ranges_from_imfPSDs(freqAx_psd, imfPSDs, f_set=f_set)
            X_f_ranges.append(f_ranges)

        X_nImfs = [X_f_rangesi.shape[0] for X_f_rangesi in X_f_ranges]

        if len(np.unique(X_nImfs)) > 1:
            print('Warning: Inconsistent number of IMFs detected between samples - using median number of IMFs')
            X_f_ranges_ = np.array([X_f_ranges[i] for i in np.where(X_nImfs == np.median(X_nImfs))[0]])
        else:
            X_f_ranges_ = np.array(X_f_ranges)

        f_ranges = np.column_stack([X_f_ranges_[:, :, 0].min(axis=0), X_f_ranges_[:, :, 1].max(axis=0)])
    else:
        freqAx_psd, _ = psd_func(Xs[0], sample_rate)
        f_ranges = np.row_stack([[0, freqAx_psd[-1]] for _ in range(max_imfs)])
        
    f_ranges0 = f_ranges.copy() # can be used for consistency scores
    
    if imfis_4_scoring is None:
        imfis_4_scoring = np.arange(f_ranges.shape[0])
    if freqs0 is None:
        freqs0 = freqAx_psd
    
    if f_set is not None:
        fs = np.array([f for f in f_set if f is not None])
        freqs0 = np.append(freqs0, np.setdiff1d(fs, freqs0))
    
    
    # mask_freq optimisation
    mask_args = {'mask_amp' : mask_amp,
                 'mask_amp_mode' : mask_amp_mode,
                 'sift_thresh' : sift_thresh, 
                 'nphases' : nphases,
                 'imf_opts' : imf_opts,
                 'envelope_opts' : envelope_opts,
                 'extrema_opts' : extrema_opts
                }
    
    it_mix_scores = []
    it_mask_freqs = []
    it_adj_mix_scores = []
    it_consistency_scores = [None, []][compute_consistency]
    it_is = []
    
    for iti in range(max_iterations):
        if iti:
            f_ranges = get_f_ranges(it_mask_freqs, it_mix_scores, it_adj_mix_scores, it_consistency_scores, top_n, f_set=f_set)
        
        f_freqs = get_f_freqs(freqs0, f_ranges)
        if iti >= max_iterations_b4_dim_red:
            print('reducing for iti=', iti)
            #return f_freqs, it_mix_scores, it_mask_freqs, it_adj_mix_scores, top_n, dim_red_prop
            f_freqs = reduce_f_freqs(f_freqs, it_mix_scores, it_mask_freqs, it_adj_mix_scores, top_n, dim_red_prop)
            
        it_mask_freqs_, it_mix_scores_, it_adj_mix_scores_, it_consistency_scores_ = run_iteration(Xs, f_freqs, imfis_4_scoring, sample_rate, mask_args, mixScore_func, 
                                                                                                   consistency_func, compute_consistency, f_ranges0, nprocesses, n_per_it)
        
        it_mix_scores.append(it_mix_scores_)
        it_mask_freqs.append(it_mask_freqs_)
        it_adj_mix_scores.append(it_adj_mix_scores_)
        if compute_consistency:
            it_consistency_scores.append(it_consistency_scores_)
        
        it_is.append(np.repeat(iti, it_mix_scores_.shape[0]))
        if iti and np.sum(np.subtract(f_ranges[:,1], f_ranges[:,0])) == 0 : #(n_main_freqs*freq_int): # if all freqs optimised
            converged = True
            break
        elif iti == (max_iterations-1):
            print('Warning: Did not converge. Consider:')
            print(' - increasing max_iterations, n_per_it')
            print(' - making max_iterations_b4_dim_red lower than max_iterations ??????') # fix - 
            converged = False
    it_mask_freqs = np.row_stack(it_mask_freqs)
    it_mix_scores = np.row_stack(it_mix_scores)
    it_adj_mix_scores = np.concatenate(it_adj_mix_scores)
    if compute_consistency:
        it_consistency_scores = np.concatenate(it_consistency_scores)
    it_is = np.concatenate(it_is)
    
    optimised_mask_freqs = it_mask_freqs[np.nan_to_num(it_mix_scores.mean(axis=1), nan=1).argmin()]
    
    return it_mask_freqs, it_mix_scores, it_adj_mix_scores, it_consistency_scores, it_is, optimised_mask_freqs, converged





### --- Figures --- ###
def get_figure_1():

    sample_rate = 1250.
    seconds = 5
    timeAx0 = np.linspace(0, seconds, int(seconds*sample_rate))

    # Create an amplitude modulation
    am = np.sin(2*np.pi*timeAx0)
    am[am < 0] = 0

    # Create a 25Hz signal and introduce the amplitude modulation
    xx = am*np.sin(2*np.pi*25*timeAx0)

    # Create a non-modulated 6Hz signal
    yy = .5*np.sin(2*np.pi*6*timeAx0)

    # Sum the 25Hz and 6Hz components together
    xy = xx+yy

    signals = np.column_stack([xx, yy])
    X = signals.sum(axis=1)
    signal_colors = sb.color_palette('Set2', signals.shape[1])
    
    return X, signals, signal_colors, sample_rate


emd_variants = ['EMD', 'eEMD', 'ceEMD', 'itEMD', 'mEMD_zc', 'mEMD']
def run_emd(X, sample_rate, variant, max_imfs=9, args=None):
    # fix incorperate set maskfreqs - new arg(s)... **kwargs
    
    import emd
    
    if X is None:
        print(emd_variants)
        return
    if variant == 'EMD':
        imfs = emd.sift.sift(X, max_imfs=max_imfs)
    elif variant == 'eEMD':
        imfs = emd.sift.ensemble_sift(X, max_imfs=max_imfs)
    elif variant == 'ceEMD':
        print('ccw to sort!')
        # fix -
        # imfs, noise = emd.sift.complete_ensemble_sift(X, max_imfs=max_imfs)
        imfs = None
    elif variant == 'itEMD':
        from ccw_it_emd import it_emd
        imfs = it_emd(X, sample_rate, N_imf=max_imfs)[0]
    elif variant == 'mEMD_zc':
        imfs = emd.sift.mask_sift(X, max_imfs=max_imfs)
    elif variant == 'mEMD':
        mask_freqs = args[0]
        imfs = emd.sift.mask_sift(X, mask_freqs=mask_freqs, max_imfs=max_imfs)
    else:
        print('method not recognised')
        imfs = None

    return imfs

def get_modeMixScores_4_emd(Xs, sample_rate, variant, psd_func, max_imfs, args=None, 
                            mixScore_funcs=[get_modeMixScore_corr, get_modeMixScore_imfPSDs, get_modeMixScore_4_imfPSDs], 
                            consistency_func=get_consistencyScores):
    """
    # fix - add documentation
    
    RETURNS:
    labelScores
    """
    
    labelScores = {}
    for mixScore_func in mixScore_funcs:
        _, label = mixScore_func(None, None, None, compute=False, return_label=True)
        labelScores[label] = []
    
    X_imfPSDs = []
    for X in Xs:
        if variant == 'eEMD':
            imfs = run_emd(X, sample_rate, variant, max_imfs+1, args)
            imfis_4_scoring = np.arange(1, imfs.shape[1])
        else:
            imfs = run_emd(X, sample_rate, variant, max_imfs, args)
            imfis_4_scoring = np.arange(imfs.shape[1])
        
        for mixScore_func in mixScore_funcs:
            mixScore, label = mixScore_func(imfs, imfis_4_scoring, sample_rate, return_label=True)
            labelScores[label].append(mixScore)
        freqAx_psd, imfPSDs = get_imfPSDs(imfs, sample_rate, psd_func)
        X_imfPSDs.append(imfPSDs)
    
    X_imfPSDs = np.array(X_imfPSDs)
    
    for label in labelScores:
        labelScores[label] = np.array(labelScores[label])
    
    consistencyScores, label = consistency_func(freqAx_psd, X_imfPSDs, imfis_4_scoring, return_label=True)
    
    labelScores[label] = consistencyScores
    
    
    return labelScores


''' PLOTTING '''
import seaborn as sb
import matplotlib.pyplot as plt

def set_plotStyle(i=0):
    s = ['Solarize_Light2', 'dark_background'][i]
    print(s)
    plt.style.use(s)

def plot_mask_freq_scores(it_mask_freqs, it_mix_scores, xi=None, imfis=None, ms=5, alpha=0.5, cmap='Spectral', inds=[], color_='k', ms_=8):
    
    if xi is None:
        it_mix_scores_M = it_mix_scores.mean(axis=1)
    else:
        it_mix_scores_M = it_mix_scores[:, xi]
    
    if imfis is None:
        imfis = np.arange(it_mask_freqs.shape[1])
    fCols = sb.color_palette(cmap, len(imfis))
    if cmap in ['husl', 'Spectral']:
        fCols = fCols[::-1]
    for fi, col in enumerate(fCols):
        plt.plot(it_mask_freqs[:,imfis][:, fi], it_mix_scores_M, 's', color=col, ms=ms, alpha=alpha, lw=0)
    
    for ind in inds:
        plt.plot(it_mask_freqs[ind, :], np.repeat(it_mix_scores_M[ind], it_mask_freqs.shape[1]), 's', ms=ms_, color=color_)
        
    return fCols


def get_nearestInd(val, array):
    array = np.array(array)
    d = np.abs(array - val)
    np.nan_to_num(d, False, np.nanmax(d))
    ind = d.argmin()
    return ind

# fix - tidy
def plot_emd(imfs, sample_rate, amps=None, ses=None, window=None, timeAx=None, color_X='k', imfCols=None, ampCol='k', cmap='gray', 
             alpha=1, ls='-', lw_lfp=1, lw_imfs=1, lw_amps=2, spaceFactor=0.2, lfp_shift=0., imfs_shift=0., flipCols=False, 
             focusImfis=None, unFocusCol='gray', unFocusAlpha=0.3, unFocusLw=1, zorder=2, alpha_se=0.5, return_imfYs=False):
    
    """imfs is  [nImfs x t] array """
    
    if window is not None:
        st, en = window
    else:
        st, en = [0, imfs.shape[0]-1]
    if imfCols is None:
        try:
            imfCols = sb.color_palette(cmap, imfs.shape[1])
        except:
            imfCols = [cmap]*imfs.shape[1]
    #
    if flipCols:
        imfCols = imfCols[::-1]
    
    lfp4plot = imfs[st:en, :].sum(axis=1)
    
    if amps is not None:
        amps4plot = amps[st:en, :].T
    if timeAx is None:
        timeAx = np.linspace(0, len(lfp4plot)/sample_rate, len(lfp4plot))
    
    plt.plot(timeAx, lfp4plot+lfp_shift, color=color_X, lw=lw_lfp, zorder=zorder)

    lfpMin, lfpMax = [f(lfp4plot) for f in [np.min, np.max]]
    emdYSt = lfpMin - (lfpMax-lfpMin)*spaceFactor + imfs_shift
    imfSpace = (lfpMax-lfpMin)*spaceFactor
    
    imfs4plot = imfs[st:en, :].T
    
    lfpMin, lfpMax = [f(lfp4plot) for f in [np.min, np.max]]
    emdYSt = lfpMin - (lfpMax-lfpMin)*spaceFactor + imfs_shift
    imfSpace = (lfpMax-lfpMin)*spaceFactor
    imfYs = []
    for imfi, imfTrace in enumerate(imfs4plot):
        if focusImfis is None:
            col = imfCols[imfi]
            alpha = 1
            lw = lw_imfs
        else:
            if imfi in focusImfis:
                col = imfCols[imfi]
                alpha = alpha
                lw = lw_imfs
                zorder = 3
            else:
                col = unFocusCol
                alpha = unFocusAlpha
                lw = unFocusLw
                zorder = 2
        plt.plot(timeAx, imfTrace+emdYSt-(imfSpace*imfi), color=col, alpha=alpha, ls=ls, lw=lw, zorder=zorder)
        
        imfYs.append(emdYSt-(imfSpace*imfi))
        
        if ses is not None:
            #
            plt.fill_between(timeAx, imfTrace+emdYSt-(imfSpace*imfi)+ses[imfi, :], imfTrace+emdYSt-(imfSpace*imfi)-ses[imfi, :], color=col, alpha=alpha_se, zorder=zorder)
        
        if amps is not None:
            plt.plot(timeAx, amps4plot[imfi]+emdYSt-(imfSpace*imfi), color=ampCol, alpha=alpha, ls=ls, lw=lw_amps, zorder=zorder)
    plt.xlim(timeAx[0], timeAx[-1])
    if return_imfYs:
        return imfYs

def plot_imfPSDs(freqAx_psd, imfPSDs, space=0.7, imfCols=None):
    if imfCols is None:
        imfCols = sb.color_palette('husl', imfPSDs.shape[0])[::-1]
    for imfi, psd in enumerate(imfPSDs):
        y = psd/psd.max()-imfi*space 
        plt.plot(freqAx_psd, y, color=imfCols[imfi])
        plt.fill_between(freqAx_psd, np.zeros_like(y)-imfi*space, y, color=imfCols[imfi], zorder=imfi, alpha=0.5)
    plt.yticks([])
    plt.xscale('log')

    

# fix - input mixScoreStr to determine ylabel 
def figplot_tmEMD(Xs, xi, it_mask_freqs, it_X_scores, sample_rate, mixScore_func, show_variants=True, variants=['EMD', 'eEMD', 'itEMD'], 
                  psd_func=get_psd, lw_variant=2, show_egs=True, window=None, eg_percs=[80, 30, 0], cmap='Spectral', set_style=True,
                  spaceFactor=0.2, fontsize=16, ms=4, ms_=6, nSecs=30, title=None, opt2xi=False, large=True, pad_egs=False):
    
    if cmap in ['Spectral']:
        color_text, color_eg, color_X = ['w']*3
        if set_style:
            set_plotStyle(1)
    else:
        color_text, color_eg, color_X = ['k']*3
        if set_style:
            set_plotStyle(0)
    
    facecolor=None
    
    _, label = mixScore_func(None, None, None, compute=False, return_label=True)
    
    if opt2xi:
        it_X_scores_M = it_X_scores[:, xi] 
    else:
        it_X_scores_M = it_X_scores.mean(axis=1) 
    
    X = Xs[xi]
    
    
    if window is None:
        if nSecs > len(X)/sample_rate:
            window = [0, len(X)-1]
        else:
            nSamples = int(sample_rate*nSecs)
            start = np.random.choice(np.arange(len(X)-nSamples))
            end = start+nSamples
            window = start, end

    if show_egs:
        eg_inds = np.array([get_nearestInd(np.nanpercentile(np.unique(it_X_scores_M), p), it_X_scores_M) for p in eg_percs])
        if large:
            w_freqs = [8, 6][show_egs]
            w_imfs = 20
            w_psd = 6
            
            wTot = w_freqs + w_imfs + w_psd
            h_eg = 7
            hTot = h_eg*len(eg_percs)
        else:
            w_freqs = 6
            w_imfs = 12
            w_psd = 3
            wTot = w_freqs + w_imfs + w_psd
            h_eg = [4, 2][len(eg_percs) > 3]
            hTot = h_eg*len(eg_percs)
    else:
        eg_inds = np.array([])
        w_freqs = 8 #[8, 6][show_egs]
        wTot = w_freqs
        hTot = 6
    
    if large:
        hspace, wspace = 3, 3
    else:
        hspace, wspace = 0.1, 0.2
    if pad_egs:
        h_eg -= 1
        h_pad = 1
    else:
        h_pad = 0
    
    plt.figure(figsize=(wTot, hTot))

    grid = plt.GridSpec(hTot, wTot, hspace=hspace, wspace=wspace)
    
    currW = 0
    ### ---------  PLOT mask freq space --------- ###
    plt.subplot(grid[:, currW:(currW+w_freqs)], facecolor=facecolor)
    plt.title(title, loc='left', fontweight='bold', color=color_text)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    imfCols = plot_mask_freq_scores(it_mask_freqs, it_X_scores, cmap=cmap, inds=eg_inds, 
                                    ms=ms, color_=color_eg, ms_=ms_) # fix - use xi input???
    
    if show_variants:
        
        max_imfs = it_mask_freqs.shape[-1]
        fmin, fmax = 0, np.round(np.nanmax(it_mask_freqs), -1)
        
        variant_colors = sb.color_palette('Set1', len(variants))
        
        for variant, color in zip(variants, variant_colors):
            labelScores = get_modeMixScores_4_emd(Xs, sample_rate, variant, psd_func, max_imfs)
            score = labelScores[label].mean()
            plt.hlines(score, fmin, fmax, color=color, linestyles='--', lw=lw_variant, label=variant)
            
        l = plt.legend()
        for text in l.get_texts():
            text.set_color(color_text)
    
    plt.xlabel('Mask freq. (Hz)', fontsize=fontsize)
    plt.ylabel(label, fontsize=fontsize)
    plt.xscale('log')
    currW += w_freqs

    # plot e.g. mEMDs 
    if show_egs:
        currW0 = np.copy(currW)
        currH = 0
        for egi, ind in enumerate(eg_inds):
            mask_freqs = it_mask_freqs[ind]
            sift_config = emd.sift.get_config('mask_sift')
            sift_config['mask_freqs'] = mask_freqs/sample_rate
            sift_config['max_imfs'] = len(mask_freqs)

            currW = np.copy(currW0)
            imfs = emd.sift.mask_sift(X, **sift_config)
            freqAx_psd, imfPSDs = get_imfPSDs(imfs, sample_rate)

            plt.subplot(grid[currH:(currH+h_eg), currW:(currW+w_imfs)], facecolor=facecolor)
            plt.xticks(fontsize=fontsize-2)
            plt.yticks([])
            plot_emd(imfs, sample_rate, window=window, imfCols=imfCols, color_X=color_X, lw_imfs=2, spaceFactor=spaceFactor)
            if egi == len(eg_inds)-1:
                plt.xlabel('Time (s)', fontsize=fontsize)
            currW += w_imfs
            plt.subplot(grid[currH:(currH+h_eg), currW:(currW+w_psd)], facecolor=facecolor)
            plt.xticks(fontsize=fontsize-2)
            plot_imfPSDs(freqAx_psd, imfPSDs, imfCols=imfCols)
            if egi == len(eg_inds)-1:
                plt.xlabel('Freq. (Hz)', fontsize=fontsize)
            currW += w_psd

            currH += h_eg + h_pad


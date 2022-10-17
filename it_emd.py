#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Extraction of IMFs using iterated masking EMD (itEMD).

Routines:
    it_emd
    it_emd_seg
    plot_imf
    
@author: MSFabus

"""
import matplotlib.pyplot as plt
import numpy as np
import emd
import warnings

def it_emd(x, sample_rate, mask_0='zc', N_imf=6, 
           N_iter_max=15, iter_th=0.1, N_avg=1, exclude_edges=False, 
           verbose=False, w_method='power', 
           ignore_last=False):
    """
    Compute iterated masking EMD.

    Parameters
    ----------
    x : 1D numpy array
        Input time-series signal.
    sample_rate : float
        Sample rate of signal in Hz.
    mask_0 : string or array, optional
        Mask initialisation type. Options are 'nyquist', 'random', zero-
        crossings ('zc'), or custom mask (numpy array matching
        N_imf in length). The default is 'zc'.
    N_imf : int, optional
        Maximum number of IMFs. The default is 6.
    N_iter_max : int, optional
        Maximum number of iterations before convergence. The default is 15.
    iter_th : float, optional
        Iteration stopping threshold (all masks stable to iter_th).
        The default is 0.1.
    N_avg : int, optional
        Optional averaging of N_avg iterations after convergence.
        The default is 1.
    exclude_edges : bool, optional
        Optionally exclude first/last 2.5% of data. The default is False.
    verbose : bool, optional
        Optionally return IMFs from all iterations. The default is False.
    w_method : string, optional
        Weighting method for mask calculation. Options are 'avg' (simple 
        average), 'IA' (instantanous amplitude), 'power' (IA**2). 
        The default is 'power'.
    ignore_last : bool, optional
        Optionally exclude the last IMF (often artifactual) from 
        mask variability for convergence calculation. The default is False.

    Returns
    -------
    list
        [IMF, mask equilibrium, mask std, # of iterations, 
         maximum iteration flag].

    """

    samples = len(x)
    
    # Initialise mask
    if mask_0 == 'nyquist':
        mask = np.array([sample_rate/2**(n+1) for n in range(1, N_imf+1)])/sample_rate
    elif mask_0 =='zc':
        _, mask = emd.sift.mask_sift(x, max_imfs=N_imf, mask_freqs='zc',
                                       ret_mask_freq=True)
    elif mask_0 == 'random':
        mask = np.random.randint(0, sample_rate/4, size=N_imf) / sample_rate
    else:
        mask = mask_0
    
    # Initialise output variables and counters
    mask_all = np.zeros((N_iter_max+N_avg, N_imf))
    imf_all = np.zeros((N_iter_max+N_avg, N_imf, samples))
    niters = 0; niters_c = 0
    navg = 0
    maxiter_flag = 0
    continue_iter = True
    converged = False
    
    while continue_iter:
        if not converged:
            print(niters, end=' ')
        else:
            print('Converged, averaging... ' + str(niters_c) + ' / ' + str(N_avg))
        
        mask_all[niters+niters_c, :len(mask)] = mask
        
        # Compute mask sift
        imf = emd.sift.mask_sift(x, max_imfs=N_imf, mask_freqs=mask,
                                 mask_amp_mode='ratio_imf')
        
        # Compute IF and weighted IF mean for next iteration
        IP,IF,IA = emd.spectra.frequency_transform(imf, sample_rate, 'nht')
        mask_prev = mask
    
        if exclude_edges:
            ex = int(0.025*samples)
            samples_included = list(range(ex, samples-ex)) #Edge effects ignored
        else:
            samples_included = list(range(samples)) #All
        
        if w_method == 'IA':
            IF_weighted = np.average(IF[samples_included, :], 0, weights=IA[samples_included, :])
        if w_method == 'power':
            IF_weighted = np.average(IF[samples_included, :], 0, weights=IA[samples_included, :]**2)
        if w_method == 'avg':
            IF_weighted = np.mean(IF[samples_included, :], axis=0)
            
        mask = IF_weighted/sample_rate
        imf_all[niters+niters_c, :imf.shape[1], :] = imf.T

        # Check convergence 
        l = min(len(mask), len(mask_prev))  # l to exclude potential nans
        if ignore_last:
            l -= 1
        mask_variance = np.abs((mask[:l] - mask_prev[:l]) / mask_prev[:l]) 
        
        if np.all(mask_variance[~np.isnan(mask_variance)] < iter_th) or converged: 
            converged = True
            if navg < N_avg:
                navg += 1
            else:
                continue_iter = False
        
        if not converged:
            niters += 1
        else:
            niters_c += 1
        
        # Check maximum number of iterations
        if niters >= N_iter_max:
            warnings.warn('Maximum number of iterations reached')
            maxiter_flag = 1
            continue_iter = False
        
    print('N_iter = ', niters)
        
    # Compute final IMFs
    if maxiter_flag == 1:
        imf_final = imf_all[niters-1, :, :].T
        IF_final = mask_all[niters-1, :]*sample_rate
    else:
        imf_final = np.nanmean(imf_all[niters:niters+N_avg, :, :], axis=0).T
        IF_final = np.nanmean(mask_all[niters:niters+N_avg, :], axis=0)*sample_rate
    IF_std_final = np.nanstd(mask_all[niters:niters+N_avg, :], axis=0)*sample_rate
    
    # If no averaging, make mask variance as deviation from last iteration
    if N_avg < 2:
        IF_std_final = mask_variance
    
    # Only output non-nan IMFs
    N_imf_final = int(np.sum(~np.isnan(mask_all[niters-1, :])))
    imf_final = imf_final[:, :N_imf_final]
    IF_final = IF_final[:N_imf_final]
    IF_std_final = IF_std_final[:N_imf_final]
    
    if verbose:
        return niters, mask_all, imf_final, IF_final, IF_std_final, imf_all
    return imf_final, IF_final, IF_std_final, niters, maxiter_flag


def it_emd_seg(data, t, segments, sample_rate, N_imf=8, joint=True, **kwargs):
    """
    Compute iterated masking EMD on segmented data.

    Parameters
    ----------
    data : 1D array
        Time-series data.
    t : 1D array
        Time vector of time series data.
    segments : 1D array
        Array of start and end times of segments.
    sample_rate : int
        Sampling rate of data.
    N_imf : int, optional
        Maximum number of IMFs. The default is 8.
    joint : bool, optional
        Return concatenated segment IMFs or individual IMFs. 
        The default is True.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        [IMF, mask equilibrium, mask std, number of iteration]

    """
        
    print('\n Processing segment:')
    out = [[] for _ in range(len(segments)-1)]
    imf_all = np.zeros((len(data), N_imf))*np.nan
    mask_all = np.zeros((len(segments)-1, N_imf))*np.nan
    niters_all = np.zeros(len(segments)-1)*np.nan
    ctr = 0
    
    for s in range(len(segments)-1):
        
        print('\n %s / %s' %(s+1, len(segments)-1))
             
        #Select slice
        ROI = np.logical_and(t > segments[s], t < segments[s+1])
        x = data[ROI]
        time_vect = t[ROI]
           
        [imf, mask_eq, mask_var, niters, _] = it_emd(x, sample_rate=sample_rate, **kwargs)
        
        out[s] = [imf, mask_eq, mask_var, time_vect, niters]
        
        N = imf.shape[1]
        imf_all[ctr:ctr+len(x), :N] = imf
        mask_all[s, :N] = mask_eq
        niters_all[s] = niters
        
        ctr += len(x)
        
    if joint:
        mask_avg = np.nanmean(mask_all, axis=0)
        mask_std = np.nanstd(mask_all, axis=0)
        N_imf_nz = np.sum(~np.isnan(mask_eq))
        imf_all = imf_all[:, :N_imf_nz]
        keep = ~np.isnan(imf_all).all(axis=1)
        imf_all = imf_all[keep, :]

        return [imf_all, mask_avg, mask_std, niters_all]
        
    else:
        return out


def plot_imf(imf, secs=[0, 2], sample_rate=256, figsize=(12,8), scale_y=True,
             **kwargs):
    """
    Quick function to plot IMFs using emd.plotting.plot_imfs with 
    extra functionality including picking a time window to plot.


    """
    idx = list(range(int(secs[0]*sample_rate), int(secs[1]*sample_rate)))
    t = np.linspace(0, secs[1]-secs[0], len(idx))
    plt.rc('font', size=18) 
    fig = plt.figure(figsize=figsize)
    emd.plotting.plot_imfs(imf[idx, :], cmap=True, fig=fig, time_vect=t,
                           scale_y=scale_y,  **kwargs)
    return fig
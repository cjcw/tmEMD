# tmEMD
Tailored Masked Empirical Mode Decomposition

Author: Charlie Clarke-Williams 

# Requirements
All figures can be reproduced in the example usage notebook, tmEMD_example.ipynb
All data used to produce figures can be found in the data folder in the main repository.
Also included is the iterated EMD module (from https://gitlab.com/marcoFabus/fabus2021_itemd), it_emd, which is needed for comparative analysis
The code was built using Python 3.6.8

# Notes
Example execution of tmEMD can be found in tmEMD_example.ipynb, or simply call the following, following the documentation:

    import tmEMD as temd
    it_mask_freqs, it_mix_scores, it_adj_mix_scores, it_consistency_scores, it_is, optimised_mask_freqs, converged = temd.run_tmEMD(Xs, sample_rate)

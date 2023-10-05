---
title: 'Tailored Masked Empirical Mode Decomposition of oscillatory time series with automatic minimization of mode mixing and tuning of between-sample consistency'   

tags:
- python
- frequency analysis
- time-series
- dynamics

authors:
- name: Charlie J. Clarke-Williams
- orcid: 0000-0002-6393-3161
- equal-contrib: true
- affiliation: "1"
- name: Vítor Lopes-dos-Santos
- orcid: 0000-0002-1858-0125
- equal-contrib: false
- affiliation: "1"
- name: David Dupret
- orcid: 0000-0002-0040-1766
- equal-contrib: false
- affiliation: "1"

affiliations:
- name: Medical Research Council Brain Network Dynamics Unit, Nuffield Department of Clinical Neurosciences, University of Oxford, Oxford, OX1 3TH, UK
- index: 1

date: 31 October 2022  
bibliography: paper.bib
---

# Summary
The tailored masked EMD (tmEMD) algorithm allows the automatic, non-arbitrary identification of the mask frequencies for analysing non-linear and non-stationary oscillatory time series with the EMD package [@Quinn2021]. Using tmEMD means that the mask frequency selection can be guided in an unsupervised and pragmatic manner.

# Statement of need
Empirical Mode Decomposition (EMD) is a powerful tool for describing the frequency content of a time-series signal, while preserving its non-sinusoidal properties [@Huang1998]. A fallback of the data-driven elegance of EMD comes in the result of mode mixing between the IMFs. Masked EMD (mEMD) offers an attractive solution to this problem [@Deering2005], but requires arbitrary mask frequency decisions to be made, taking away from the data-driven elegance of EMD. The mask frequency tailoring algorithm offered by the tmEMD toolkit offers a pragmatic and flexible way of selecting mask frequencies in a data-driven manner. The tmEMD process guides mask frequency selection such to minimise mode mixing, and also allows the consideration of between-sample consistency; two criteria which as of yet have not been directly addressed by the field. 

# State of the field
To help resolve the unfavourable phenomenon of mode mixing, two key EMD variants have been proposed: ensemble EMD (eEMD) [@Colominas2012; @Wu2009] and masked EMD (mEMD) [@Deering2005]. While eEMD can indeed improve mode mixing while requiring no key, user-made decisions for additional parameters, performance can be poor for “real-life” intermittent signals that contain complex combinations of IMFs of varying amplitude and frequency. mEMD can resolve these issues by introducing a masking signal of user-specified frequency for each EMD sifting iteration, effectively placing a lower-bound on the frequency content which can be extracted for each IMF. If suitable mask frequencies are chosen, mode mixing can be significantly reduced compared to “vanilla” EMD or eEMD (**Figure 1**). However, suitable mask frequency choices cannot be known a priori, and become increasingly difficult to identify as the number of IMFs to extract increases. This means that “default” mask frequency choices are arbitrary and highly unlikely to be best suited to the data.
To address this, the iterated EMD (itEMD) algorithm was recently introduced to select a set of mask frequencies based on a simple iterative process that tunes each mask to the amplitude-weighted frequency of the IMF extracted in the previous iteration [@Fabus2021]. While itEMD indeed deals with the problem of arbitrary mask frequency assignment and reduces mode mixing compared to the abovementioned EMD variants, as it is not directly assessed, the selected mask frequencies may not represent an “optimal” solution in terms of mode mixing. Moreover, this would not take into consideration meta-factors such as the consistency of extracted IMFs between data samples. We here introduce the tmEMD algorithm to incorporate these ideas, which allows automatic convergence to a set of mask frequencies that yield minimal mode mixing between IMFs (**Figure 2**). As tmEMD can be run on multiple samples, users can further retroactively explore other meta-factors like between-sample IMF consistency, to further tailor IMF extraction to suit additional requirements (**Figure 3**).

# Figures
![figure_1](https://github.com/cjcw/tmEMD/assets/35930153/cf37b981-4de7-4a2a-8788-82f39f61cd56)
**Figure 1**: Choice of mask frequency is important for unmixed IMFs. 
tmEMD was run on a dummy signal (black traces) to converge to an optimal mask frequency choice. **A:** Mode mixing as a function of mask frequency (green points). Dotted lines: mode mixing scores yielded by EMD variants. **B-C:** Example mask sifts (**B**) and the Power Spectral density estimates (PSDs) (**C**) of their IMFs corresponding to the black points in **A**. **D:** Mode mixing scores as a function of the tmEMD sub-iteration.



![figure_2](https://github.com/cjcw/tmEMD/assets/35930153/03f1d84d-2a61-4fc2-806c-9a87f32ffcff)
**Figure 2:** tmEMD algorithm applied to real data
Extending the logic from **Figure 1**, optimal sets of mask frequencies can be found for more complex signals. **A-B:** IMFs (**A**) and their PSDs (**B**) yielded by EMD variants. **C-E:** tmEMD algorithm visualisation for real data. **C:** mode mixing scores (y-axis) yielded by sets of mask frequencies (coloured points) in each tmEMD operation. Dotted lines: mode mixing scores yieled by EMD variants, as in **A**. **D-E:** example IMFs (**D**) and their PSDs (**E**) extracted by mask frequency sets corresponding to the black dots in **C**. **F:** tmEMD convergence to mode-mixing-minimised solution.



![figure_3](https://github.com/cjcw/tmEMD/assets/35930153/8ac78287-739b-4fb0-9970-fc64e2ea5d8e)
**Figure 3:** Tuning the mode mixing minimised solution with between-sample consistency.
Mean mode mixing scores (x-axis) plotted against the mean between-sample consistency (y-axis) for each tmEMD sub-iteration (blue-yellow dots). The sub-iteration which yielded the best overall combined mixing and consistency scores is denoted by the cross marker. Coloured, larger dots show the mode mixing and consistency scores yeilded by EMD variants.



# Installation
The tmEMD toolkit is implemented in Python (>=3.5) and can be downloaded from https://github.com/cjcw/tmEMD. To use tmEMD simply import the module and then call the function, <code>run_tmEMD</code>.

# Example usage
Example usage can be found in the iPython notebook <code>tmEMD_example.ipynb</code>. This notebook reproduces all figures presented here, showing the use of the algorithm on dummy and real data, with performance evaluation compared to other EMD variants, and also retroactive tailoring options to account for between-subject consistency. 

# References
Colominas, M., Schlotthauer, G., Torres, M.E., and Flandrin, P. (2012). Noise-Assisted EMD Methods in Action. Adv Adapt Data Anal https://doi.org/10.1142/S1793536912500252. 

Deering, R., and Kaiser, J.F. (2005). The use of a masking signal to improve empirical mode decomposition. p. iv/485-iv/488 Vol. 4. 

Fabus, M.S., Quinn, A.J., Warnaby, C.E., and Woolrich, M.W. (2021). Automatic decomposition of electrophysiological data into distinct nonsinusoidal oscillatory modes. J. Neurophysiol. 126, 1670–1684. https://doi.org/10.1152/jn.00315.2021. 

Quinn, A.J., Lopes-dos-Santos, V., Dupret, D., Nobre, A.C., and Woolrich, M.W. (2021). EMD: Empirical Mode Decomposition and Hilbert-Huang Spectral Analyses in Python. J. Open Source Softw. 6, 2977. https://doi.org/10.21105/joss.02977. 

Wu, Z., and Huang, N. (2009). Ensemble Empirical Mode Decomposition: a Noise-Assisted Data Analysis Method. Adv. Adapt. Data Anal. 1, 1–41. https://doi.org/10.1142/S1793536909000047. 

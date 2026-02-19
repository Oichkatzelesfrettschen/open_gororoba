"""Ghost spectral audit -- Python statistical bridge (Sprint 50).

Advanced spectral methods for testing the phi^{-1/2} ghost frequency hypothesis:
- Lomb-Scargle with Baluev (2008) analytic FAP
- IAAFT surrogates (Theiler et al. 1992)
- Thomson multitaper spectral estimation
- Quantile/Laplace periodogram (Li 2012)
- Pairwise magnitude-squared coherence
- Meta-analysis p-value combination (Fisher, Stouffer)
"""

GHOST_FREQ = 0.786_151_377_757_423
ALIASED_GHOST_FREQ = 1.0 - GHOST_FREQ
FREQ_TOL = 0.02

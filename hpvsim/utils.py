"""HPVsim utility helpers.

Slimmed in M02 to prefer starsim-native equivalents. v2's broader utility
surface is quarantined to hpvsim/_v2_legacy/utils.py for porter reference.

Most M02-onward code should use starsim directly:
    - distributions:  ss.bernoulli, ss.lognorm_ex, ss.choice, ss.normal
    - random seed:    set via ss.Sim(rand_seed=...)
    - boolean masks:  BoolArr.uids, FloatArr.notnan

Active-code consumers as of M02: none. All three helpers previously imported
by hpvsim/network.py (binomial_filter, participation_filter, choose_w) were
replaced with linked ss.bernoulli / ss.choice dist instances in M02.
"""

import numpy as np  # noqa: F401 — kept for any future helpers added here


__all__ = []   # nothing currently re-exported; helpers added below as needed


# Reserved for HPV-specific helpers without a starsim equivalent.
# As of M02 start: empty. Disease-progression math (logf2, compute_severity,
# transform_prob) is colocated with hpv.py in Tasks 4-6.

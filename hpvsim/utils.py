"""HPVsim utility helpers.

Reserved for HPV-specific helpers without a starsim equivalent. Currently
empty — disease-progression math (logf2, compute_severity, transform_prob)
is colocated with ``hpv.py``. The legacy v2 utility surface is quarantined
in ``hpvsim/_v2_legacy/utils.py`` for porter reference.

Active code should prefer starsim directly:
    - distributions:  ss.bernoulli, ss.lognorm_ex, ss.choice, ss.normal
    - random seed:    set via ss.Sim(rand_seed=...)
    - boolean masks:  BoolArr.uids, FloatArr.notnan
"""

import numpy as np  # noqa: F401 — kept for any future helpers added here


__all__ = []

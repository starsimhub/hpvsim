"""
Parameters of HPVsim's sexual network (``hpv.SexualNetwork``): its defaults and age-mixing data

The natural-history parameters of the genotypes are in ``genotypes.py`` (``hpv.genotype_pars()``).
"""

import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['NetworkPars', 'NETWORK_LAYERS']

# Sexual-network layer keys (marital, casual).
NETWORK_LAYERS = ('m', 'c')


# --------------------------------------------------------------------------- #
# Network mixing & participation matrices (data — used by NetworkPars)        #
# --------------------------------------------------------------------------- #

# Age mixing: rows = MALE age band, columns 1..N = FEMALE age band; column 0
# holds each row's male age-band lower bound as a label. So ``M[male_bin,
# female_bin + 1]`` is the relative weight males in that band give females in
# that band -- the orientation network.py indexes as ``mixing[:, ab + 1]`` with
# ``ab`` the female bin, inherited from HPVsim v2's population.py.
_MIXING_M = np.array([
    # col 0 = male band label; header below = female band
    #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
    [ 0,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 5,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [10,    0,  0, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [15,    0,  0, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [20,    0,  0, .1, .1, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [25,    0,  0, .5, .1, .5, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [30,    0,  0,  1, .5, .5, .5, .5, .1,  0,  0,  0,  0,  0,  0,  0,  0],
    [35,    0,  0, .5,  1,  1, .5,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0],
    [40,    0,  0,  0, .5,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0],
    [45,    0,  0,  0,  0, .1,  1,  1,  2,  1,  1, .5,  0,  0,  0,  0,  0],
    [50,    0,  0,  0,  0,  0, .1,  1,  1,  1,  1,  2, .5,  0,  0,  0,  0],
    [55,    0,  0,  0,  0,  0,  0, .1,  1,  1,  1,  1,  2, .5,  0,  0,  0],
    [60,    0,  0,  0,  0,  0,  0,  0, .1, .5,  1,  1,  1,  2, .5,  0,  0],
    [65,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  2, .5,  0],
    [70,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1, .5],
    [75,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1],
], dtype=float)
_MIXING_C = np.array([
    # col 0 = male band label; header below = female band
    #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
    [ 0,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 5,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [10,    0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [15,    0,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [20,    0,  0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [25,    0,  0, .5,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0],
    [30,    0,  0,  0, .5,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0],
    [35,    0,  0,  0, .5,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0],
    [40,    0,  0,  0,  0, .5,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0],
    [45,    0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0],
    [50,    0,  0,  0,  0,  0, .5,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0],
    [55,    0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0],
    [60,    0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0],
    [65,    0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  2, .5,  0],
    [70,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5],
    [75,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1],
], dtype=float)

# Age-band participation probabilities: row 0 = age-bin lower bounds,
# row 1 = annual female prob, row 2 = annual male prob.
_LAYER_PROBS_M = np.array([
    [ 0,    5,      10,     15,    20,    25,    30,    35,     40,     45,     50,    55,    60,    65,    70,    75],
    [ 0,    0,  0.0394, 0.938, 0.938, 0.938, 0.938, 0.938, 0.938, 0.938, 0.938, 0.760, 0.590, 0.344, 0.185, 0.0394],
    [ 0,    0,  0.0394, 0.590, 0.760, 0.938, 0.938, 0.938, 0.938, 0.938, 0.938, 0.760, 0.590, 0.344, 0.185, 0.0394],
], dtype=float)
_LAYER_PROBS_C = np.array([
    [ 0,    5,      10,     15,    20,    25,    30,    35,     40,     45,     50,    55,    60,    65,    70,    75],
    [ 0,    0,  0.590, 0.974, 0.998, 0.974, 0.870, 0.870, 0.870, 0.344, 0.0776, 0.0776, 0.0776, 0.0776, 0.0776, 0.0776],
    [ 0,    0,  0.590, 0.870, 0.870, 0.870, 0.870, 0.974, 0.998, 0.974, 0.590, 0.344, 0.185, 0.0776, 0.0776, 0.0776],
], dtype=float)


def _nbinom_mean_k(mean, k):
    """``ss.nbinom`` parametrized as (mean, dispersion k).

    Scipy uses (n=successes, p=success-prob); modellers think in
    (mean, dispersion). Mapping: n=k, p=k/(k+mean).
    """
    return ss.nbinom(n=k, p=k/(k+mean))


class NetworkPars(sc.objdict):
    """Default parameters for ``hpv.SexualNetwork``.

    All values are location-agnostic (HPVsim ships demographic data per
    country but not network calibration). Analysis scripts supply
    per-country overrides.

    Naming convention: per-layer pars carry a ``_marital`` / ``_casual``
    suffix; per-sex pars carry a ``_f`` / ``_m`` suffix. All Dist-typed
    pars are ``ss.Dist`` instances directly (no v2 ``{'dist': ..., 'par1':
    ...}`` dicts).

    Per-agent, per-layer partner-count targets are **always shifted by +1**
    in the network's sampling code (``set_network_states``). A participating
    agent in a layer wants at least one partner by definition; the ``lam``
    values below govern only the tail of *additional* concurrent partners.

    Keys:
        ``m_cross_layer``, ``f_cross_layer``: scalar annual probability of
            male/female cross-layer concurrency.
        ``debut_f``, ``debut_m``: per-sex debut-age ``ss.Dist``.
        ``{sex}_partners_{layer}``: per-layer per-sex partner-count Dist
            (``ss.poisson``; sampled value has +1 added at use).
        ``acts_{layer}``: per-partnership annual act-count Dist.
        ``dur_pship_{layer}``: partnership duration Dist (years).
        ``age_act_pars_{layer}``: per-layer age-based act modulation
            (peak, retirement, debut_ratio, retirement_ratio) dict.
        ``mixing_{layer}``: age-mixing matrix; rows = male age band, columns
            1..N = female age band, and column 0 holds each row's male
            age-band label. ``M[male_bin, female_bin + 1]`` is the relative
            weight males in that band give females in that band.
        ``layer_probs_{layer}``: (3, N) array of age-band participation:
            row 0 = age-bin lower bounds, rows 1/2 = annual f/m prob.
    """
    def __init__(self, **overrides):
        super().__init__()

        self.m_cross_layer = 0.760
        self.f_cross_layer = 0.185

        self.debut_f = ss.normal(loc=15.0, scale=2.1)
        self.debut_m = ss.normal(loc=17.6, scale=1.8)

        # Concurrent-partner-count Poisson lambda; +1 shift applied in network.
        self.m_partners_marital = ss.poisson(lam=0.01)
        self.m_partners_casual  = ss.poisson(lam=0.5)
        self.f_partners_marital = ss.poisson(lam=0.01)
        self.f_partners_casual  = ss.poisson(lam=0.5)

        self.acts_marital = _nbinom_mean_k(mean=80, k=40)
        self.acts_casual  = _nbinom_mean_k(mean=50, k=5)

        self.dur_pship_marital = _nbinom_mean_k(mean=80, k=3)
        self.dur_pship_casual  = ss.lognorm_ex(mean=1, std=2)

        # Acts ramp linearly: debut_ratio -> 1.0 at peak -> retirement_ratio, then 0.
        self.age_act_pars_marital = ss.Pars(peak=30, retirement=100, debut_ratio=0.5, retirement_ratio=0.1)
        self.age_act_pars_casual  = ss.Pars(peak=25, retirement=100, debut_ratio=0.5, retirement_ratio=0.1)

        self.mixing_marital = _MIXING_M
        self.mixing_casual  = _MIXING_C

        self.layer_probs_marital = _LAYER_PROBS_M
        self.layer_probs_casual  = _LAYER_PROBS_C

        self.update(overrides)

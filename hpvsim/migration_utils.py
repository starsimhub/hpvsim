"""Bridging utilities for the v2 → v3 migration.

Contents here translate v2-shaped values (parameter dicts, distributions)
into Starsim-native equivalents. Expected to be retired once the migration
is complete and the v2 quarantine in ``hpvsim/_v2_legacy/`` is deleted.
"""

import starsim as ss


class Poisson1(ss.poisson):
    """v2-style ``poisson1`` distribution: ss.poisson + a constant shift.

    v2's ``poisson1`` is ``Poisson(rate) + 1`` — i.e. every agent has at
    least one partner. We override ``rvs`` to add the shift after sampling
    rather than passing ``loc`` through to scipy, because Starsim's
    ``sync_pars`` plumbing only propagates ``mu`` to the underlying
    ``rv_discrete_frozen`` and a ``loc`` override is silently dropped.
    """
    def __init__(self, lam=1.0, shift=1, **kwargs):
        self._shift = shift
        super().__init__(lam=lam, **kwargs)

    def rvs(self, *args, **kwargs):
        return super().rvs(*args, **kwargs) + self._shift


def _v2_dist_to_starsim(d):
    """Convert a v2-format distribution dict to a Starsim Dist instance.

    v2 stores distributions as ``{'dist': name, 'par1': p1, 'par2': p2}``;
    Starsim distributions take named parameters and are sampled via
    ``.rvs(uids)``. M01 only needs poisson, poisson1, lognormal, neg_binomial
    (the four used in v2's default Nigeria network pars).

    Time-unit wrapping: we deliberately do NOT pass ``ss.years`` or
    ``ss.freqperyear`` to dur/acts dists. v2 samples once per pair in
    annual units and multiplies by ``dt`` at use-time, preserving the
    per-pair sample variance. Starsim's ``predraw`` auto-scaling for
    ``poisson``/``nbinom`` would instead scale the rate parameter inside
    the dist, which inflates per-sample variance by ``1/dt`` and breaks
    the v2 equivalence gate. Probability pars (``layer_probs``,
    ``cross_layer``) ARE wrapped via ``ss.prob`` — see
    ``hpvsim.data.country._network_pars``.
    """
    dist = d['dist']
    par1 = d.get('par1')
    par2 = d.get('par2')
    if dist == 'poisson':
        return ss.poisson(lam=par1)
    if dist == 'poisson1':
        return Poisson1(lam=par1, shift=1)
    if dist in ('lognormal', 'lognorm'):
        return ss.lognorm_ex(mean=par1, std=par2)
    if dist == 'neg_binomial':
        # v2: par1=mean, par2=k (dispersion).
        # scipy.stats.nbinom: n=number-of-successes, p=success-probability.
        # Mapping: n = k, p = k / (k + mean).
        n_param = par2
        p_param = par2 / (par2 + par1) if (par2 + par1) > 0 else 0.5
        return ss.nbinom(n=n_param, p=p_param)
    if dist == 'normal':
        return ss.normal(loc=par1, scale=par2)
    if dist == 'uniform':
        return ss.uniform(low=par1, high=par2)
    raise ValueError(f'Unsupported v2 distribution: {dist!r}')
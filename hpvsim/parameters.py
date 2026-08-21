"""HPV simulation parameters.

``SimPars`` subclasses ``ss.SimPars`` with HPV-specific defaults.
``GenotypePars`` holds per-genotype natural-history defaults (durations,
severity functions, immunity, age risk) consumed by ``HPV(ss.Infection)``;
``get_genotype_pars(genotype)`` is the factory multi-genotype consumers
use to look up defaults by name. Supports the four canonical genotypes:
``hpv16``, ``hpv18``, ``hi5`` (high-risk-5 pool), and ``ohr`` (other
high-risk pool).
"""

import numpy as np
import sciris as sc
import starsim as ss


__all__ = ['SimPars', 'GenotypePars', 'NetworkPars', 'get_genotype_pars',
           'genotype_aliases', 'GENOTYPE_KEYS', 'route_pars']

# Sexual-network layer keys (marital, casual).
NETWORK_LAYERS = ('m', 'c')

# Canonical 4-genotype ordering. The Connector uses this order as the
# default when no genotype list is supplied.
GENOTYPE_KEYS = ('hpv16', 'hpv18', 'hi5', 'ohr')

# Genotype name aliases for user ergonomics.
genotype_aliases = {
    'hpv16': ['hpv16', '16'],
    'hpv18': ['hpv18', '18'],
    'hi5':   ['hi5', 'high-risk-5'],
    'ohr':   ['ohr', 'other-high-risk'],
}


class SimPars(ss.SimPars):
    """HPV-specific defaults on top of ss.SimPars."""

    def __init__(self, **kwargs):
        super().__init__()
        self.n_agents  = 10_000
        self.total_pop = None  # If set, pop_scale = total_pop / n_agents.
        self.pop_scale = None  # Computed at init if total_pop is set.

        self.start     = ss.years(1990)
        self.stop      = ss.years(2060)
        self.dt        = ss.years(0.5)
        self.rand_seed = 0

        self.location = 'nigeria'
        self.verbose = ss.options.verbose

        self.update(kwargs)
        return


def _beta_from_mean_var(mean, var):
    """Build a beta ``ss.Dist`` from (mean, variance).

    Converts to the beta distribution's shape parameters (a, b):
        a = ((1 - mean) / var - 1 / mean) * mean ** 2
        b = a * (1 / mean - 1)
    Valid for mean in (0, 1) and var < mean * (1 - mean).

    Uses the ``ss.beta_dist`` factory rather than a bare
    ``ss.Dist(distname='beta', ...)``: starsim >=3.5 no longer resolves a
    name-only Dist to its scipy object, leaving ``self.dist=None`` so ``.rvs()``
    raises ``AttributeError: 'NoneType' ... 'ppf'``. ``ss.beta_dist`` wires the
    scipy dist explicitly and exists on both 3.4 and 3.5.
    """
    a = ((1 - mean) / var - 1 / mean) * mean ** 2
    b = a * (1 / mean - 1)
    return ss.beta_dist(a=a, b=b)


def _imm_init_dist(mean=0.35, var=0.025):
    """Beta sample for per-clearance humoral-immunity boost.

    Defaults are the calibrated values. Callers wanting a different
    immunity-boost distribution can construct their own and assign it to
    ``GenotypePars.imm_init`` (or pass via ``pars=``) on the per-genotype HPV
    module.
    """
    return _beta_from_mean_var(mean, var)


def _cell_imm_dist(mean=0.25, var=0.025):
    """Beta sample for per-clearance severity-immunity boost (cell_imm_init).

    Defaults are the calibrated values. See ``_imm_init_dist`` for
    how to override per-genotype.
    """
    return _beta_from_mean_var(mean, var)


# Per-genotype natural-history defaults. Duration entries are
# ``(mean, std)`` tuples in years.
_GENOTYPE_DEFAULTS = {
    'hpv16': dict(
        beta=0.25,
        dur_precin_yr=(3.0, 9.0),
        dur_cin_yr=(5.0, 20.0),
        dur_cancer_yr=(8.0, 3.0),
        dur_inf_male_yr=(1.0, 1.0),
        cin_fn=dict(form='logf2', k=0.3, x_infl=0, ttc=50),
        cancer_fn=dict(method='cin_integral', transform_prob=2e-3,
                       form='logf2', k=0.3, x_infl=0, ttc=50),
        age_risk=dict(age=30, risk=2),
        rel_beta=1.0,
        sero_prob=0.75,
        transf2m=1.0,
        transm2f=2.0,
    ),
    'hpv18': dict(
        beta=0.25,
        dur_precin_yr=(2.5, 9.0),
        dur_cin_yr=(5.0, 20.0),
        dur_cancer_yr=(8.0, 3.0),
        dur_inf_male_yr=(1.0, 1.0),
        cin_fn=dict(form='logf2', k=0.25, x_infl=0, ttc=50),
        cancer_fn=dict(method='cin_integral', transform_prob=2e-3,
                       form='logf2', k=0.25, x_infl=0, ttc=50),
        age_risk=dict(age=30, risk=2),
        rel_beta=0.75,
        sero_prob=0.56,
        transf2m=1.0,
        transm2f=2.0,
    ),
    'hi5': dict(
        beta=0.25,
        dur_precin_yr=(2.5, 9.0),
        dur_cin_yr=(4.5, 20.0),
        dur_cancer_yr=(8.0, 3.0),
        dur_inf_male_yr=(1.0, 1.0),
        cin_fn=dict(form='logf2', k=0.2, x_infl=0, ttc=50),
        cancer_fn=dict(method='cin_integral', transform_prob=1.5e-3,
                       form='logf2', k=0.2, x_infl=0, ttc=50),
        age_risk=dict(age=30, risk=2),
        rel_beta=0.9,
        sero_prob=0.60,
        transf2m=1.0,
        transm2f=2.0,
    ),
    'ohr': dict(
        beta=0.25,
        dur_precin_yr=(2.5, 9.0),
        dur_cin_yr=(4.5, 20.0),
        dur_cancer_yr=(8.0, 3.0),
        dur_inf_male_yr=(1.0, 1.0),
        cin_fn=dict(form='logf2', k=0.2, x_infl=0, ttc=50),
        cancer_fn=dict(method='cin_integral', transform_prob=1.5e-3,
                       form='logf2', k=0.2, x_infl=0, ttc=50),
        age_risk=dict(age=30, risk=2),
        rel_beta=0.9,
        sero_prob=0.60,
        transf2m=1.0,
        transm2f=2.0,
    ),
}


def _lognorm_yr(spec):
    """Build an ``ss.lognorm_ex`` from a ``(mean, std)`` tuple in years."""
    mean, std = spec
    return ss.lognorm_ex(mean=ss.years(mean), std=ss.years(std))


class GenotypePars(ss.Pars):
    """Per-genotype natural-history defaults for HPV.

    Each call returns fresh distribution instances so per-genotype RNG
    state stays independent. Supported genotypes: ``hpv16``, ``hpv18``,
    ``hi5``, ``ohr``.

    Notes:
      - ``beta`` is per-sex-act; SexualNetwork applies per-act via
        ``1 - (1-p)**acts``.
      - Female natural-history durations are lognormal in years; males
        clear via ``dur_inf_male`` without entering CIN/cancer.
      - ``cancer_fn`` carries ``cin_fn``'s keys so the ``cin_integral``
        branch can call ``compute_severity_integral`` on the same logf2.
      - ``imm_init`` is sampled per-clearance and feeds ``nab_imm``
        (read by the CrossImmunity Connector).
      - ``age_risk['age']``-and-older women get ``dur_cin`` scaled by
        ``age_risk['risk']``, shifting cancer onset to older ages.
      - ``transf2m`` / ``transm2f`` are sex-directional per-act scalars
        — same across genotypes (act-level, not genotype-level).
    """

    def __init__(self, genotype='hpv16', **kwargs):
        super().__init__()
        if genotype not in _GENOTYPE_DEFAULTS:
            raise ValueError(
                f'GenotypePars supports {GENOTYPE_KEYS}; got {genotype!r}.'
            )
        d = _GENOTYPE_DEFAULTS[genotype]
        self.genotype = genotype
        self.beta = d['beta']
        self.dur_precin = _lognorm_yr(d['dur_precin_yr'])
        self.dur_cin = _lognorm_yr(d['dur_cin_yr'])
        self.dur_cancer = _lognorm_yr(d['dur_cancer_yr'])
        self.dur_inf_male = _lognorm_yr(d['dur_inf_male_yr'])
        self.cin_fn = dict(d['cin_fn'])
        self.cancer_fn = dict(d['cancer_fn'])
        self.imm_init = _imm_init_dist()
        self.cell_imm_init = _cell_imm_dist()
        self.age_risk = dict(d['age_risk'])
        self.rel_beta = d['rel_beta']
        self.sero_prob = d['sero_prob']
        self.transf2m = d['transf2m']
        self.transm2f = d['transm2f']
        self.update(kwargs)
        return


def get_genotype_pars(genotype='hpv16'):
    """Return per-genotype natural-history defaults."""
    return GenotypePars(genotype=genotype)


# Sim-level keys route_pars will pass through to sim.pars. Explicit allowlist
# so a typo raises rather than silently creating a new sim par.
_SIM_KEYS = frozenset({'n_agents', 'dt', 'start', 'stop', 'rand_seed',
                       'ms_agent_ratio', 'total_pop', 'verbose'})


# --------------------------------------------------------------------------- #
# Network mixing & participation matrices (data — used by NetworkPars)        #
# --------------------------------------------------------------------------- #

# Age-mixing (rows = female age band start, cols = male age band start; first
# column is the female age-band label).
_MIXING_M = np.array([
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
        ``mixing_{layer}``: age-mixing matrix (rows = female age band start,
            cols = male age band; first column is female age label).
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

        # Age-based act modulation per layer. Ramp linearly from
        # debut_ratio at debut age to 1.0 at peak, then to retirement_ratio
        # at retirement, then 0 beyond.
        self.age_act_pars_marital = dict(peak=30, retirement=100, debut_ratio=0.5, retirement_ratio=0.1)
        self.age_act_pars_casual  = dict(peak=25, retirement=100, debut_ratio=0.5, retirement_ratio=0.1)

        self.mixing_marital = _MIXING_M
        self.mixing_casual  = _MIXING_C

        self.layer_probs_marital = _LAYER_PROBS_M
        self.layer_probs_casual  = _LAYER_PROBS_C

        self.update(overrides)


def _par_registry():
    """Category -> par-name-set (built lazily; lightweight)."""
    from .hpv import HPV
    from .cross_genotype import CrossImmunity
    from .network import SexualNetwork
    hpv_keys = set()
    for g in GENOTYPE_KEYS:
        hpv_keys.update(HPV(g).pars.keys())
    return {'sim': set(_SIM_KEYS), 'hpv': hpv_keys,
            'connector': set(CrossImmunity().pars.keys()),
            'network': set(SexualNetwork().pars.keys())}


def route_pars(sim, pars=None, calib_pars=None, verbose=True, strict=True, **_):
    """Route a flat ``{key: value}`` dict to the correct modules on ``sim``.

    Rules:
      - ``<genotype>.<par>[.<sub>]``: scoped to one HPV disease.
      - ``cross_immunity.<par>[.<sub>]``: scoped to ``CrossImmunity``.
        Legacy form ``cross_immunity.<matrix>.<tgt>.<src>`` (4 parts)
        writes a single matrix cell.
      - ``network.<par>[.<sub>]`` / ``sexualnetwork.<par>[.<sub>]``:
        scoped to ``SexualNetwork``. Also accepts a nested dict value.
      - Bare key: registry lookup (see ``_par_registry``); broadcast to
        every matching category. ``beta`` scalar broadcasts to every HPV
        module, preserving each genotype's F/M ratio.
      - Optuna spec dicts ``{'low','high','guess','value'}`` are unwrapped
        to their ``value``.
    """
    from .hpv import HPV
    from .cross_genotype import CrossImmunity
    from .network import SexualNetwork

    pars = pars if pars is not None else calib_pars
    if not pars:
        return sim
    pars = {k: (v['value'] if isinstance(v, dict) and 'value' in v else v)
            for k, v in pars.items()}

    if hasattr(sim, 'diseases') and sim.diseases is not None:
        diseases = {d.name: d for d in sim.diseases.values() if isinstance(d, HPV)}
        connectors = [c for c in sim.connectors.values() if isinstance(c, CrossImmunity)]
        networks = [n for n in sim.networks.values() if isinstance(n, SexualNetwork)]
    else:
        diseases = {d.name: d for d in sc.tolist(sim.pars.get('diseases'))
                    if isinstance(d, HPV)}
        connectors = [c for c in sc.tolist(sim.pars.get('connectors'))
                      if isinstance(c, CrossImmunity)]
        networks = [n for n in sc.tolist(sim.pars.get('networks'))
                    if isinstance(n, SexualNetwork)]

    registry = _par_registry()

    def apply_nested(container, parts, value):
        """Walk into nested pars/Dist to set at the leaf.

        Terminal cases:
        - ``target`` is an ``ss.Dist``: ``target.set(**{leaf: value})`` —
          used e.g. for ``cross_immunity.rel_sev.scale`` (leaf is a Dist par).
        - ``target`` is an ``ss.Pars``: delegate to ``Pars.update`` so
          starsim's ``_update_dist`` handles dict-to-Dist merges (e.g.
          ``dur_cin={'mean': 5}`` calls ``lognorm_ex.set(mean=5)`` without
          clobbering ``std``).
        - Otherwise (plain dict): direct assignment.
        """
        target = container
        for p in parts[:-1]:
            if isinstance(target, ss.Dist):
                target = target.pars
            target = target[p]
        leaf = parts[-1]

        # Preserve TimePar wrapping on scalar overrides. Dist.set(pars) writes
        # via plain dict update, dropping any ss.TimePar wrapping on the
        # existing par — e.g. dur_cin=6 turns `mean=years(5)` into `mean=6`
        # (unitless), which starsim then reads as timesteps, not years.
        def _preserve_units(cur, val):
            if sc.isnumber(val) and isinstance(cur, ss.TimePar):
                return cur.__class__(val)
            if isinstance(cur, ss.Dist) and sc.isnumber(val):
                first_key = next(iter(cur.pars))
                first = cur.pars[first_key]
                if isinstance(first, ss.TimePar):
                    return first.__class__(val)
            if isinstance(val, dict) and isinstance(cur, ss.Dist):
                return {k: _preserve_units(cur.pars.get(k), v) for k, v in val.items()}
            return val

        if isinstance(target, ss.Dist):
            cur = target.pars.get(leaf)
            value = _preserve_units(cur, value)
            target.set(**{leaf: value})
        elif isinstance(target, ss.Pars):
            cur = target.get(leaf)
            value = _preserve_units(cur, value)
            target.update({leaf: value})
        else:
            target[leaf] = value

    def apply_beta_scalar(disease, value):
        """Broadcast a scalar beta onto pars.beta ({'sexualnetwork': [f2m, m2f]}),
        preserving the F/M ratio. Per-act probabilities are clipped to
        [0, 1] via ``HPV._clip_beta`` (matches the construction-time clip);
        transm2f=3.69 defaults mean any scalar > ~0.271 would push m2f > 1
        and silently NaN transmission."""
        from .hpv import _clip_beta
        old = disease.pars.beta
        if not (sc.isnumber(value) and isinstance(old, dict)):
            disease.pars.beta = value
            return
        first = next(iter(old.values()))
        ref = first[0] if isinstance(first, list) else first
        scale = 1.0 if ref == 0 else value / ref
        new_beta = {}
        for net_name, v in old.items():
            if isinstance(v, list):
                new_beta[net_name] = _clip_beta(
                    v[0] * scale, v[1] * scale,
                    getattr(disease, 'genotype', getattr(disease, 'name', '?')),
                )
            else:
                new_beta[net_name] = min(1.0, v * scale)
        disease.pars.beta = new_beta

    unmatched = {}

    def apply_scope(container_pars, key, value):
        """Apply a scoped key (either 'a.b.c' dotted or 'a' with dict value)
        onto ``container_pars``. Dict values are unpacked one level so each
        sub-key gets its own apply_nested; the caller has already stripped
        the module-scope prefix."""
        if isinstance(value, dict):
            for sub_k, sub_v in value.items():
                apply_nested(container_pars, [key, *sub_k.split('.')], sub_v)
        else:
            apply_nested(container_pars, key.split('.'), value)

    for key, value in pars.items():
        parts = key.split('.')
        head = parts[0]

        # Scoped: <module>.<par>[.<sub>] OR bare <module> with a dict of pars.
        if head in diseases and (len(parts) > 1 or isinstance(value, dict)):
            if len(parts) > 1:
                apply_nested(diseases[head].pars, parts[1:], value)
            else:
                for sub_k, sub_v in value.items():
                    apply_scope(diseases[head].pars, sub_k, sub_v)
            continue
        if head in ('cross_immunity', 'crossimmunity') and (len(parts) > 1 or isinstance(value, dict)):
            if not connectors:
                raise ValueError(f'route_pars: {key!r} needs a CrossImmunity connector.')
            # Legacy: cross_immunity.<matrix>.<tgt>.<src> -> matrix cell.
            if len(parts) == 4 and parts[1] in ('cross_imm_sus', 'cross_imm_sev'):
                conn = connectors[0]
                idx = {m.name: i for i, m in enumerate(conn.hpv_modules)}
                getattr(conn, parts[1])[idx[parts[2]], idx[parts[3]]] = value
            elif len(parts) > 1:
                for c in connectors:
                    apply_nested(c.pars, parts[1:], value)
            else:
                for c in connectors:
                    for sub_k, sub_v in value.items():
                        apply_scope(c.pars, sub_k, sub_v)
            continue
        if head in ('network', 'sexualnetwork') and (len(parts) > 1 or isinstance(value, dict)):
            if not networks:
                raise ValueError(f'route_pars: {key!r} needs a SexualNetwork on the sim.')
            if len(parts) > 1:
                for n in networks:
                    apply_nested(n.pars, parts[1:], value)
            else:
                for n in networks:
                    for sub_k, sub_v in value.items():
                        apply_scope(n.pars, sub_k, sub_v)
            continue

        # Flat: registry lookup. Broadcast if a key lives in more than one category.
        cats = [c for c, kk in registry.items() if head in kk]
        if not cats:
            unmatched[key] = value
            continue
        if len(cats) > 1 and verbose:
            print(f'route_pars: {key!r} broadcast to {cats}')
        for cat in cats:
            if cat == 'sim':
                apply_nested(sim.pars, parts, value)
            elif cat == 'hpv':
                for d in diseases.values():
                    if parts == ['beta']:
                        apply_beta_scalar(d, value)
                    else:
                        apply_nested(d.pars, parts, value)
            elif cat == 'connector':
                if not connectors:
                    raise ValueError(f'route_pars: {key!r} needs a CrossImmunity connector.')
                for c in connectors:
                    apply_nested(c.pars, parts, value)
            elif cat == 'network':
                if not networks:
                    raise ValueError(f'route_pars: {key!r} needs a SexualNetwork on the sim.')
                for n in networks:
                    apply_nested(n.pars, parts, value)

    if unmatched and strict:
        raise ValueError(f'route_pars: unrecognized par(s): {sorted(unmatched)}')

    return sim

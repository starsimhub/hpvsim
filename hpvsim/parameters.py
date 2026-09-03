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

from . import misc


__all__ = ['SimPars', 'GenotypePars', 'NetworkPars', 'get_genotype_pars',
           'genotype_aliases', 'GENOTYPE_KEYS', 'route_pars',
           'expanddict', 'par_registry']

# Sexual-network layer keys (marital, casual).
NETWORK_LAYERS = ('m', 'c')

# Canonical genotype ordering; the Connector's default when none is supplied.
GENOTYPE_KEYS = ('hpv16', 'hpv18', 'hi5', 'ohr')

# Genotype name aliases for user ergonomics.
genotype_aliases = {
    'hpv16': ['hpv16', '16'],
    'hpv18': ['hpv18', '18'],
    'hi5':   ['hi5', 'high-risk-5'],
    'ohr':   ['ohr', 'other-high-risk'],
}


class SimPars(ss.SimPars):
    """HPV-specific defaults on top of ss.SimPars.

    ``location`` defaults to None to match ``hpv.Sim``'s own default: a bare
    ``hpv.Sim()`` is a natural-history playground (uniform ages 0-60, no
    births/deaths/migration, ``pop_scale=1``). Pass a country name for a real
    population, or call ``hpv.demo()`` for the canonical Nigeria sim.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.n_agents  = 10_000
        self.total_pop = None  # If set, pop_scale = total_pop / n_agents.
        self.pop_scale = None  # Computed at init if total_pop is set.

        self.start     = ss.years(1990)
        self.stop      = ss.years(2060)
        self.dt        = ss.years(0.25)
        self.rand_seed = 0

        self.location = None
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


# Per-genotype natural-history defaults; durations are (mean, std) in years.
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
        self.cin_fn = ss.Pars(d['cin_fn'])
        self.cancer_fn = ss.Pars(d['cancer_fn'])
        self.imm_init = _imm_init_dist()
        self.cell_imm_init = _cell_imm_dist()
        self.age_risk = ss.Pars(d['age_risk'])
        self.rel_beta = d['rel_beta']
        self.sero_prob = d['sero_prob']
        self.transf2m = d['transf2m']
        self.transm2f = d['transm2f']
        self.update(kwargs)
        return


def get_genotype_pars(genotype='hpv16'):
    """Return per-genotype natural-history defaults."""
    return GenotypePars(genotype=genotype)


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


def expanddict(flat):
    """Convert a flat {'a.b.c': value} dict to nested {'a': {'b': {'c': value}}}.
    The inverse of sc.flattendict. A key with no '.' becomes a plain
    top-level entry."""
    nested = {}
    for key, value in flat.items():
        sc.setnested(nested, key.split('.'), value)
    return nested


def par_registry():
    """Category -> par-name-set (built lazily; lightweight).

    No 'hiv' category: HIV's pars overlap HPV's on 'beta'/'init_prev' (both
    inherited from stisim's BaseSTIPars), so a bare-key broadcast would
    silently also flip on HIV transmission (e.g. bare beta= turning on
    hiv.pars.beta from its 0 default). The scoped hiv=dict(...) form in
    route_pars does NOT consult this registry -- it dispatches directly by
    instance name -- so scoped/hiv_pars= routing is unaffected.
    """
    from .hpv import HPV
    from .cross_genotype import CrossImmunity
    from .network import SexualNetwork
    hpv_keys = set()
    for g in GENOTYPE_KEYS:
        hpv_keys.update(HPV(g).pars.keys())
    # 'location' excluded: it is a construction-time argument absent from
    # sim.pars, so broadcasting it would raise. hpv.Sim.__init__ intercepts
    # pars=dict(location=...) before routing, so users never hit this.
    return {'sim': set(SimPars().keys()) - {'location'}, 'hpv': hpv_keys,
            'connector': set(CrossImmunity().pars.keys()),
            'network': set(SexualNetwork().pars.keys())}


def route_pars(sim, pars=None, calib_pars=None, verbose=True, strict=True, **_):
    """Route a nested ``{key: value}`` dict of overrides onto sim's modules.

    A top-level key is either:
      - Scoped: matches a live HPV genotype's name, the HIV disease's name
        (``'hiv'``), or ``cross_immunity``/``network`` (aliases
        ``crossimmunity``/``sexualnetwork``). Routed to that instance's pars
        via ONE ``instance.pars.update(value)`` call -- starsim's own
        ``Pars.update()`` handles arbitrarily nested overrides natively
        (merging into ``ss.Pars``-typed sub-pars, calling ``.set()`` on
        ``Dist`` leaves, preserving ``TimePar`` units). Applied AFTER
        broadcast keys, so a scoped override always wins regardless of
        input dict ordering.
      - A bare registry key: broadcast to every matching category (sim,
        hpv, hiv, connector, network) via ``instance.pars.update({key: value})``.
        Applied FIRST.
      - ``calib_pars`` (flat dotted) is converted to nested via
        ``expanddict`` before anything else runs.
    """
    from .hpv import HPV
    from .cross_genotype import CrossImmunity
    from .network import SexualNetwork

    hiv_cls = misc.hiv_class()
    disease_types = (HPV,) if hiv_cls is None else (HPV, hiv_cls)

    raw = calib_pars if calib_pars is not None else (pars or {})
    if not raw:
        return sim
    # calib_pars trial dicts are Optuna specs; unwrap to the sampled value.
    raw = {k: (v['value'] if isinstance(v, dict) and 'value' in v else v)
           for k, v in raw.items()}
    nested = expanddict(raw)

    if hasattr(sim, 'diseases') and sim.diseases is not None:
        diseases = {d.name: d for d in sim.diseases.values() if isinstance(d, disease_types)}
        connectors = [c for c in sim.connectors.values() if isinstance(c, CrossImmunity)]
        networks = [n for n in sim.networks.values() if isinstance(n, SexualNetwork)]
    else:
        diseases = {d.name: d for d in sc.tolist(sim.pars.get('diseases'))
                    if isinstance(d, disease_types)}
        connectors = [c for c in sc.tolist(sim.pars.get('connectors'))
                      if isinstance(c, CrossImmunity)]
        networks = [n for n in sc.tolist(sim.pars.get('networks'))
                    if isinstance(n, SexualNetwork)]

    registry = par_registry()
    # Scope names that are meaningful in hpvsim even when the module they
    # address is absent from this particular sim: an HIV scope in a no-HIV
    # counterfactual, or a genotype scope in a sim running a subset of
    # genotypes. Overrides aimed at a missing module are skipped with a
    # warning rather than raising, so one calibrated parameter set can drive a
    # whole scenario sweep without the caller pre-filtering it per scenario.
    optional_scopes = set(GENOTYPE_KEYS) | {'hiv'}
    scoped, broadcast, absent = {}, {}, []
    for key, value in nested.items():
        if key in diseases or key in ('cross_immunity', 'crossimmunity', 'network', 'sexualnetwork'):
            scoped[key] = value
        elif key in optional_scopes:
            absent.append(key)
        else:
            broadcast[key] = value
    if absent:
        misc.warn(f'route_pars: no module for {sorted(absent)} in this sim; '
                  f'those overrides were skipped.')

    unmatched = {}
    for key, value in broadcast.items():
        cats = [c for c, kk in registry.items() if key in kk]
        if not cats:
            unmatched[key] = value
            continue
        if len(cats) > 1 and verbose:
            print(f'route_pars: {key!r} broadcast to {cats}')
        for cat in cats:
            if cat == 'sim':
                sim.pars.update({key: value})
            elif cat == 'hpv':
                for d in diseases.values():
                    if isinstance(d, HPV):
                        d.pars.update({key: value})
            elif cat == 'hiv':
                for d in diseases.values():
                    if hiv_cls is not None and isinstance(d, hiv_cls):
                        d.pars.update({key: value})
            elif cat == 'connector':
                if not connectors:
                    raise ValueError(f'route_pars: {key!r} needs a CrossImmunity connector.')
                for c in connectors:
                    c.pars.update({key: value})
            elif cat == 'network':
                if not networks:
                    raise ValueError(f'route_pars: {key!r} needs a SexualNetwork on the sim.')
                for n in networks:
                    n.pars.update({key: value})

    for key, value in scoped.items():
        if key in diseases:
            diseases[key].pars.update(value)
        elif key in ('cross_immunity', 'crossimmunity'):
            if not connectors:
                raise ValueError(f'route_pars: {key!r} needs a CrossImmunity connector.')
            for c in connectors:
                c.pars.update(value)
        elif key in ('network', 'sexualnetwork'):
            if not networks:
                raise ValueError(f'route_pars: {key!r} needs a SexualNetwork on the sim.')
            for n in networks:
                n.pars.update(value)

    if unmatched and strict:
        raise ValueError(f'route_pars: unrecognized par(s): {sorted(unmatched)}')

    return sim

"""Rwanda HPV–HIV calibration (M08).

The NETWORK / behavioural / HIV-input parameters are ported from the published
v2.3 Rwanda calibration (debut, partners, layer_probs, init_hpv_dist, beta=0.12,
rel_imm). The CANCER natural-history scalars (_TP_SCALE, _CIN_K_SCALE,
_DUR_CIN_SCALE) and the HIV cancer-effect strengths (RWANDA_HIV_EFFECTS rel_sus /
rel_sev) were RE-CALIBRATED for v3 directly against the 2017 registry cancer-
incidence targets, because v2's values were fit against v2's multiscale-biased
engine and overshoot ~10x on v3's unbiased engine. See the _TP_SCALE comment
block below and tests/regression/calibrate_rwanda.py.

Two upstream sources for the ported parts, read once at authoring time:

  1. ``hpvsim_rwanda/results/rwanda_pars.obj`` (the Optuna posterior point
     estimate): ``genotype_pars`` (per-genotype dur/cin_fn/cancer_fn/rel_beta/
     sero), overall HPV ``beta=0.12``, ``sev_dist=normal_pos(0.87, 0.2)``,
     ``hiv_pars`` (CD4-stratified ``rel_sus``/``rel_sev``/``rel_imm``),
     ``m/f_partners``, ``m/f_cross_layer``.
  2. ``hpvsim_rwanda/run_sim.py:make_sim`` (hard-coded, not in the pickle):
     ``debut`` (f/m lognormals), the marital/casual ``layer_probs`` arrays,
     ``init_hpv_dist``, ``start=1960``.

PER-TIMESTEP vs ANNUAL.  v2 calibrates the probability-like network params in
*per-timestep* space (dt=0.25) and converts ``layer_probs`` + ``cross_layer``
to *annual* before building the network
(``run_sim._to_annual_prob``: ``annual = 1 - (1 - p)**(1/dt)``).  v3's
``SexualNetwork`` wraps ``layer_probs``/``cross_layer`` in ``ss.prob(.., annual)``
(see ``hpvsim/data/country._shape_network_pars``), so this module supplies the ANNUAL
forms.  ``beta``, ``partners`` (counts), genotype durations, and ``sev_dist`` are
NOT rate-converted.

WHAT v3 ALREADY MATCHES.  v3's genotype defaults equal the Rwanda calibration
except ``hi5.cin_fn.k`` (0.2 vs 0.24) and ``ohr.cin_fn.k`` (0.2 vs 0.14); the
directional scalars (``transf2m=1.0``, ``transm2f=3.69``) and base ``beta=0.25``
also match v2's defaults.  So the genotype port is: set base ``beta=0.12`` and
fix the two ``cin_fn.k`` (and the duplicate ``cancer_fn.k``).  The HIV effect
strengths differ from the generic connector defaults and are supplied via
``hpv_hiv_connector(effects=RWANDA_HIV_EFFECTS)``; the severity-scaler location
via ``CrossImmunity(rel_sev_loc=0.87)``.
"""

import numpy as np
import starsim as ss

import hpvsim as hpv
from hpvsim.cross_genotype import CrossImmunity
from hpvsim.data.country import _default_network_pars, _shape_network_pars
from hpvsim.hiv import hpv_hiv_connector
from hpvsim.parameters import get_genotype_pars

GENOTYPES = [16, 18, 'hi5', 'ohr']
_GENO_KEYS = ['hpv16', 'hpv18', 'hi5', 'ohr']
_DT = 0.25  # calibration timestep used for the per-timestep -> annual conversion

# --- rwanda_pars.obj scalars ------------------------------------------------
BASE_BETA = 0.12                  # overall HPV per-act transmissibility (calib)
REL_SEV_LOC = 0.87                # sev_dist = normal_pos(0.87, 0.2)
# Directional per-act scalars (unchanged from v3/v2 defaults; documented here
# because the genotype beta dict is rebuilt from BASE_BETA).
_TRANSF2M, _TRANSM2F = 1.0, 3.69
# cin_fn slope overrides (the only genotype-par diffs vs v3 defaults).
_RWANDA_CIN_K = {'hi5': 0.24, 'ohr': 0.14}
# Base dur_cin (mean, std) in years, per genotype (v3/v2 defaults).
_DUR_CIN = {'hpv16': (5.0, 20.0), 'hpv18': (5.0, 20.0),
            'hi5': (4.5, 20.0), 'ohr': (4.5, 20.0)}

# Cancer natural-history re-calibration for v3's multiscale-UNBIASED engine.
#
# The published v2.3 Rwanda numbers were produced under v2's multiscale-BIASED
# engine (extra fine agents re-roll the precin->CIN gate, deflating cancer
# ~4-10x); v2's transform_prob was fit against that. v3's engine is multiscale-
# unbiased, so the v2 params overshoot cancer ~10x. These three GLOBAL scalars
# are the converged best fit (gof 1.84) from a 7-param Optuna calibration to the
# 2017 registry cancer-incidence targets (by-age + aggregate, HIV-stratified) --
# see tests/regression/calibrate_rwanda.py and results/rwanda_calib/. BASE_BETA
# stays 0.12 (v2's HPV-prevalence calibration, uncontaminated by the cancer-only
# bug). A per-genotype refinement was tried (calibrate_rwanda_genotype.py) but
# did not improve the cancer genotype mix (structural 'ohr' lumping limit), so
# these global scalars are the adopted calibration.
_TP_SCALE = 0.102       # x cancer_fn.transform_prob (all genotypes) -- cancer LEVEL
_CIN_K_SCALE = 0.380    # x cin_fn.k (all genotypes; cancer_fn.k synced) -- age SHAPE
_DUR_CIN_SCALE = 1.043  # x dur_cin (mean & std, all genotypes)

# CD4-stratified HIV->HPV effects. rel_sus / rel_sev drive the HIV+/HIV- cancer
# RR (replace v2's 4.75/2.75 and 2.5/3.5); rel_imm retains the v2 Rwanda value
# (rwanda_pars.obj hiv_pars).
#
# RE-FIT ON THE GROW ENGINE (2026-07-07). The original 7-param ledger fit was
# run at ms_agent_ratio=1, where HIV+ cancer is severely under-resolved (sparse
# events, fit partly to noise). On the v2-faithful GROW multiscale engine
# (branch m08-rwanda-on-grow) the HPV natural-history scalars above transferred
# cleanly (HIV- flat/on-target), but HIV+ cancer over-predicted ~2x. A scoped
# re-fit of ONLY the 4 rel_sus/rel_sev params at ms_agent_ratio=5 (where fine
# agents resolve HIV+/cancer rare events) -- calibrate_rwanda_hiv_grow.py,
# results/rwanda_calib_hiv_grow/ -- restored the fit (honest gof 1.42 @ 20 seeds
# vs the ledger's ~2.05). rel_sus was ~unidentified (cancer responds to severity,
# not acquisition, by the 2010-19 window) and barely moved; the fix was almost
# entirely rel_sev_gt200 (3.25 -> 1.98): the CD4 200-500 band (bulk of the
# on-ART HIV+ population) had too-aggressive severity on the grow engine.
RWANDA_HIV_EFFECTS = {
    'rel_sus': {'lt200': 3.60, 'gt200': 1.96},
    'rel_sev': {'lt200': 5.23, 'gt200': 1.98},
    'rel_imm': {'lt200': 0.36, 'gt200': 0.76},
}

# Initial genotype mix among seeded infections (run_sim.make_sim init_hpv_dist).
RWANDA_INIT_HPV_DIST = dict(hpv16=0.4, hpv18=0.25, hi5=0.25, ohr=0.1)

# --- Network: debut (rwanda run_sim) ---------------------------------------
_RWANDA_DEBUT = dict(
    f=dict(dist='lognormal', par1=20.96, par2=3.34),
    m=dict(dist='lognormal', par1=17.91, par2=2.83),
)

# --- Network: partners + cross_layer (rwanda_pars.obj) ----------------------
_RWANDA_M_PARTNERS = dict(m=dict(dist='poisson1', par1=0.01),
                          c=dict(dist='poisson1', par1=0.30))
_RWANDA_F_PARTNERS = dict(m=dict(dist='poisson1', par1=0.01),
                          c=dict(dist='poisson1', par1=0.35))
_RWANDA_CROSS_LAYER_PT = 0.5  # per-timestep; -> annual below

# --- Network: layer_probs (rwanda run_sim, PER-TIMESTEP) --------------------
# Rows: age-bin lower bounds / female participation / male participation.
_RWANDA_LAYER_PROBS_PT = dict(
    m=np.array([
        [0, 5,    10,     15,     20,     25,    30,     35,    40,     45,    50,   55,   60,   65,   70,   75],
        [0, 0, 0.025, 0.0115, 0.1555, 0.313, 0.3875, 0.408, 0.3825, 0.334, 0.275, 0.20, 0.20, 0.20, 0.20, 0.20],
        [0, 0,  0.01,  0.023,  0.311, 0.626,  0.775, 0.816,  0.765, 0.668, 0.70,  0.80, 0.70, 0.60, 0.50, 0.60],
    ], dtype=float),
    c=np.array([
        [0, 5,   10,  15,  20,  25,  30,  35,  40,   45,   50,   55,   60,   65,   70,   75],
        [0, 0, 0.1, 0.6, 0.3, 0.2, 0.2, 0.2, 0.2, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        [0, 0, 0.1, 0.3, 0.4, 0.3, 0.3, 0.4, 0.5, 0.50, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    ], dtype=float),
)


def _to_annual(p, dt=_DT):
    """Per-timestep probability -> annual: ``1 - (1 - p)**(1/dt)`` (v2 form)."""
    p = np.clip(p, 0.0, 1.0 - 1e-10)
    return 1.0 - (1.0 - p) ** (1.0 / dt)


def _layer_probs_annual():
    """Rwanda layer_probs with the f/m rows converted per-timestep -> annual."""
    out = {}
    for lkey, lp in _RWANDA_LAYER_PROBS_PT.items():
        a = lp.copy()
        a[1, :] = _to_annual(a[1, :])
        a[2, :] = _to_annual(a[2, :])
        out[lkey] = a
    return out


def rwanda_genotype_pars():
    """Per-genotype overrides applying the v3 calibration:
      - beta: base BASE_BETA=0.12 rebuilt into the directional (f2m/m2f) dict.
      - dur_cin: scaled by _DUR_CIN_SCALE.
      - cin_fn.k: (Rwanda override for hi5/ohr, else the genotype default) x
        _CIN_K_SCALE, applied to ALL genotypes; cancer_fn.k kept in sync.
      - cancer_fn.transform_prob: scaled by _TP_SCALE (all genotypes)."""
    out = {}
    for key in _GENO_KEYS:
        gp = get_genotype_pars(key)
        rb = float(gp.rel_beta)
        d = {
            'beta': {'sexualnetwork': [BASE_BETA * rb * _TRANSF2M,
                                       BASE_BETA * rb * _TRANSM2F]},
        }
        if _DUR_CIN_SCALE != 1.0:
            mean, std = _DUR_CIN[key]
            d['dur_cin'] = ss.lognorm_ex(mean=ss.years(mean * _DUR_CIN_SCALE),
                                         std=ss.years(std * _DUR_CIN_SCALE))
        # cin_fn.k: Rwanda override (hi5/ohr) or genotype default, x _CIN_K_SCALE.
        base_k = _RWANDA_CIN_K.get(key, float(gp.cin_fn['k']))
        k = base_k * _CIN_K_SCALE
        cin = dict(gp.cin_fn)
        cin['k'] = k
        d['cin_fn'] = cin
        # cancer_fn (cin_integral): scale transform_prob (level); sync the
        # duplicate k with cin_fn.k.
        cf = dict(gp.cancer_fn)
        cf['transform_prob'] = cf['transform_prob'] * _TP_SCALE
        cf['k'] = k
        d['cancer_fn'] = cf
        out[key] = d
    return out


def rwanda_network_overrides():
    """Raw network overrides in ``_default_network_pars`` form.

    All probability-like entries are ANNUAL (v3 re-wraps them in
    ``ss.prob(.., annual)``); partners are counts and debut is an age dist.
    """
    cl_annual = float(_to_annual(_RWANDA_CROSS_LAYER_PT))
    return dict(
        debut=_RWANDA_DEBUT,
        layer_probs=_layer_probs_annual(),
        m_partners=_RWANDA_M_PARTNERS,
        f_partners=_RWANDA_F_PARTNERS,
        m_cross_layer=cl_annual,
        f_cross_layer=cl_annual,
    )


def make_rwanda_network():
    """A v3 ``SexualNetwork`` carrying the Rwanda network calibration.

    Pass the raw Rwanda overrides through the standard ``pars=`` override on
    ``_default_network_pars``, then shape the merged dict into SexualNetwork
    inputs (reusing ``_shape_network_pars`` so the annual->ss.prob / dist
    wrapping isn't duplicated here).
    """
    raw = _default_network_pars('rwanda', pars=rwanda_network_overrides())
    return hpv.SexualNetwork(**_shape_network_pars(raw))


def build_rwanda_sim(seed=0, n_agents=10_000, start=1960, stop=2020, dt=_DT,
                     incidence_driven=True, beta_m2f=0.0, ms_agent_ratio=5):
    """Rwanda HPV–HIV co-infection sim with the published v2.3 calibration.

    Args:
        incidence_driven: if True (default, v2-faithful), HIV transmission is
            off and the epidemic is imposed by ``hpv.hiv_incidence_import``;
            otherwise HIV transmits at ``beta_m2f`` over the sexual network.
        ms_agent_ratio: grow-multiscale ratio. Defaults to 5 -- the ratio the
            HIV rel_sus/rel_sev effects were re-fit at, where fine agents
            resolve the sparse HIV+/cancer events (see RWANDA_HIV_EFFECTS).
            ratio=1 is unbiased in expectation but leaves HIV+ by-age cancer
            very noisy (needs many seeds).
    """
    connectors = [
        CrossImmunity(rel_sev_loc=REL_SEV_LOC),
        hpv_hiv_connector(effects=RWANDA_HIV_EFFECTS),
    ]
    if incidence_driven:
        hiv = hpv.HIV.from_location('rwanda', beta_m2f=0.0, init_prev_data=0.0)
        interventions = [
            hpv.hiv_incidence_import.from_location('rwanda'),
            hpv.hiv_art.from_location('rwanda'),
        ]
    else:
        hiv = hpv.HIV.from_location('rwanda', beta_m2f=beta_m2f)
        interventions = [hpv.hiv_art.from_location('rwanda')]

    return hpv.Sim(
        location='rwanda',
        rand_seed=seed,
        n_agents=n_agents,
        start=start,
        stop=stop,
        dt=dt,
        ms_agent_ratio=ms_agent_ratio,
        genotypes=GENOTYPES,
        genotype_pars=rwanda_genotype_pars(),
        init_hpv_dist=RWANDA_INIT_HPV_DIST,
        networks=[make_rwanda_network()],
        connectors=connectors,
        diseases=[hiv],
        interventions=interventions,
    )

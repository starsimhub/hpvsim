"""HPV genotype as a Starsim Infection.

M01: single-genotype, transmission-only with SIS clearance (no precin/CIN/cancer).
M02 will add natural-history states (precin, CIN, cancer) and override step_state.
M03 will instantiate one HPV per genotype and add a CrossImmunity connector.

Default values for ``beta`` and ``init_prev`` are taken from v2's
``parameters.py``: ``beta=0.25`` (per-sex-act probability before sex-direction
scaling, applied per-pair via Starsim's standard
``net_beta = 1 - (1-p)**acts``), and ``init_hpv_prev`` is age- and sex-
stratified (see :data:`_INIT_HPV_PREV_TABLE`).
"""

import numpy as np
import starsim as ss


# ---------------------------------------------------------------------------
# Math helpers ported verbatim from hpvsim._v2_legacy.utils (v2).
# Renamed with leading underscore to mark as private to this module.
# These implement the logistic-2 (logf2) family used by cin_fn / cancer_fn.
# ---------------------------------------------------------------------------

def _get_asymptotes(k, x_infl, s=1, y_max=1, ttc=25):
    '''
    Get upper asymptotes for logistic functions
    '''
    term1 = (1 + np.exp(k*(x_infl-ttc)))**s # Note, this is 1 for most parameter combinations
    term2 = (1 + np.exp(k*x_infl))**s
    u_asymp_num = y_max*term1*(1-term2)
    u_asymp_denom = term1 - term2
    u_asymp = u_asymp_num / u_asymp_denom
    l_asymp = y_max * term1 / (term1 - term2)
    return l_asymp, u_asymp


def _logf3(x, k, x_infl, s=1, y_max=1, ttc=25):
    '''
    Logistic function passing through (0,0) and (ttc,y_max).
    This version is derived from the 5-parameter version here: https://www.r-bloggers.com/2019/11/five-parameters-logistic-regression/
    However, since it's constrained to pass through 2 points, there are 3 free parameters remaining.
    Args:
         k: growth rate, equivalent to b in https://www.r-bloggers.com/2019/11/five-parameters-logistic-regression/
         x_infl: a location parameter, equivalent to C in https://www.r-bloggers.com/2019/11/five-parameters-logistic-regression/
         s: asymmetry parameter, equivalent to s in https://www.r-bloggers.com/2019/11/five-parameters-logistic-regression/
         ttc (time to cancer): x value for which the curve passes through 1. For x values beyond this, the function returns 1
    '''
    l_asymp, u_asymp = _get_asymptotes(k, x_infl, s=1, y_max=y_max, ttc=ttc)
    return np.minimum(1, l_asymp + (u_asymp-l_asymp)/(1+np.exp(k*(x_infl-x)))**s)


def _logf2(x, k, x_infl, y_max=1, ttc=25):
    '''
    Logistic function constrained to pass through (0,0) and (ttc,y_max).
    This version is derived from the 5-parameter version here: https://www.r-bloggers.com/2019/11/five-parameters-logistic-regression/
    Since it's constrained to pass through 2 points, there are 3 free parameters remaining, and this verison fixes s=1
    Args:
         k: growth rate, equivalent to b in https://www.r-bloggers.com/2019/11/five-parameters-logistic-regression/
         x_infl: point of inflection, equivalent to C in https://www.r-bloggers.com/2019/11/five-parameters-logistic-regression/
         ttc (time to cancer): x value for which the curve passes through 1. For x values beyond this, the function returns 1
    '''
    return _logf3(x, k, x_infl, s=1, y_max=y_max, ttc=ttc)


# M01 ships defaults (beta, init_prev) tuned to HPV16 only. Other genotypes
# (hpv18, hi5, ohr) require per-genotype natural-history params that land
# in M02 + M03 — accepting them here without those defaults would silently
# run with wrong values. The validation is intentionally narrow until M03.
_KNOWN_GENOTYPES = ('hpv16',)


# v2 default initial HPV prevalence table from
# hpvsim/_v2_legacy/parameters.py make_pars: pars['init_hpv_prev'].
# Age brackets are inclusive lower bounds; the last bracket extends to age 150.
# For M01 (single genotype) we apply the full table to HPV16.
_INIT_HPV_PREV_AGE_BRACKETS = np.array([12, 17, 24, 34, 44, 64, 80, 150])
_INIT_HPV_PREV_M = np.array([0.0, 0.25, 0.60, 0.25, 0.05, 0.01, 0.0005, 0.0])
_INIT_HPV_PREV_F = np.array([0.0, 0.35, 0.70, 0.25, 0.05, 0.01, 0.0005, 0.0])


def _age_stratified_init_prev(module, sim, uids):
    """Per-uid initial-infection probability based on v2's age/sex table."""
    age = np.asarray(sim.people.age[uids])
    is_female = np.asarray(sim.people.female[uids])
    # Bin agent ages into the v2 brackets.
    bin_idx = np.searchsorted(_INIT_HPV_PREV_AGE_BRACKETS, age, side='right') - 1
    bin_idx = np.clip(bin_idx, 0, len(_INIT_HPV_PREV_AGE_BRACKETS) - 1)
    out = np.zeros(len(uids))
    out[is_female] = _INIT_HPV_PREV_F[bin_idx[is_female]]
    out[~is_female] = _INIT_HPV_PREV_M[bin_idx[~is_female]]
    return out


class HPV(ss.Infection):
    """Single-genotype HPV disease module.

    The ``genotype`` attribute identifies which strain this instance models
    and is the duck-type marker M03's ``hpv.CrossImmunity`` connector will
    use to discover HPV diseases (mirrors rotasim's
    ``hasattr(disease, 'G')`` pattern).
    """

    def __init__(self, genotype='hpv16', pars=None, **kwargs):
        if genotype not in _KNOWN_GENOTYPES:
            raise ValueError(
                f'M01 supports genotype={list(_KNOWN_GENOTYPES)} only; '
                f'got {genotype!r}. Other genotypes (hpv18, hi5, ohr) '
                f'require per-genotype natural-history params that land in M03.'
            )
        self.genotype = genotype
        if 'name' not in kwargs:
            kwargs['name'] = genotype
        super().__init__()
        # Defaults sourced from v2 parameters.py:
        #  - beta = 0.25 per sex-act (a scalar, NOT a Rate; SexualNetwork's
        #    net_beta applies it per-act via 1 - (1-p)**acts).
        #  - init_prev = age- and sex-stratified per v2's init_hpv_prev table.
        #  - dur_inf placeholder; M02 replaces with v2's get_genotype_pars
        #    duration distribution for HPV16.
        self.define_pars(
            init_prev=ss.bernoulli(p=_age_stratified_init_prev),
            beta=0.25,
            dur_inf=ss.lognorm_ex(mean=ss.years(2.0)),
        )
        self.update_pars(pars=pars, **kwargs)
        # ss.Infection already provides: susceptible, infected, rel_sus,
        # rel_trans, ti_infected. We add:
        #  - ti_clearance: SIS clearance time
        #  - ti_first_infection: timestep of the first-ever infection per
        #    agent. Set once and never overwritten — matches v2's
        #    date_infectious semantics (in v2's natural-history model
        #    immunity prevents re-infection, so date_infectious is
        #    naturally first-only; M01's SIS would overwrite ti_infected
        #    on re-infection, hence this separate state).
        self.define_states(
            ss.FloatArr('ti_clearance', label='Time of natural clearance'),
            ss.FloatArr('ti_first_infection', label='Time of first infection'),
        )

    def set_prognoses(self, uids, sources=None):
        """Mark uids as infected; schedule clearance per dur_inf; record
        ti_first_infection for never-before-infected agents.

        Initial seeding via ``init_post → set_prognoses`` also flows through
        here, so init_prev-seeded agents get their ti_first_infection set
        at ti=0 — matching v2's ``date_infectious`` behavior for initial
        prevalence and keeping the v2-baseline comparison apples-to-apples.
        """
        # ss.Infection.set_prognoses just appends to the infection_log if it's
        # enabled in pars (off by default); does not touch states. We set the
        # infection states explicitly below.
        super().set_prognoses(uids, sources)
        ti = self.ti
        # Record first-ever infection time only for agents whose
        # ti_first_infection is still NaN.
        first_uids = uids[self.ti_first_infection.isnan[uids]]
        self.ti_first_infection[first_uids] = ti
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = ti
        self.ti_clearance[uids] = ti + self.pars.dur_inf.rvs(uids)

    def step_state(self):
        """SIS: agents past ti_clearance return to susceptible."""
        clearing = (self.infected & (self.ti_clearance <= self.ti)).uids
        if len(clearing):
            self.infected[clearing] = False
            self.susceptible[clearing] = True
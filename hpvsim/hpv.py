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
import sciris as sc
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


def _transform_prob(tp, dysp):
    '''
    Returns transformation probability given dysplasia
    Using formula for half an ellipsoid:
        V = 1/2 * 4/3 * pi * a*b*c
          = 2 * a*b*c
          = 2* dysp * (dysp/2)**2, assuming that b = c = 1/2 a
          = 1/2 * dysp**3
    '''
    # return 1-np.power(1-tp, ((dysp*100)**2))
    return 1-np.power(1-tp, 0.5*((dysp)**3)*100)


def _indef_int_logf2(x, k, x_infl, ttc=25, y_max=1):
    '''
    Indefinite integral of logf2; see definition there for arguments
    '''
    t1 = 1 + np.exp(k*(x_infl-ttc))
    t2 = 1 + np.exp(k*x_infl)
    integ = np.log(np.exp(k*(x_infl-x)) + 1) / k + x
    result = y_max/(t1-t2)*(1-t1*t2*integ)
    return result


def _intlogf2(upper, k, x_infl, ttc=25, y_max=1):
    '''
    Integral of logf2 between 0 and the limit given by upper
    '''
    # Find the upper limits not including the part past time to cancer
    exceeding_ttc_inds = (upper > ttc).nonzero()
    lims_to_find = np.minimum(ttc, upper)

    # Take the integral
    val_at_0 = _indef_int_logf2(0, k, x_infl, ttc)
    val_at_lim = _indef_int_logf2(lims_to_find, k, x_infl, ttc)
    integral = val_at_lim - val_at_0

    # Deal with those whose duration of infection exceeds the time to cancer
    # Note, another option would be to set their transformation probability to 1
    excess_integral = upper[exceeding_ttc_inds] - ttc
    integral[exceeding_ttc_inds] += excess_integral

    return integral


def _compute_severity_integral(t, rel_sev=None, pars=None):
    '''
    Process functional form and parameters into values:
    '''

    pars = sc.dcp(pars)
    form = pars.pop('form')
    choices = [
        'logf2',
        'logf3 with s=1',
    ]

    # Scale t
    if rel_sev is not None:
        t = rel_sev * t

    # Process inputs
    if form is None or form == 'logf2':
        output = _intlogf2(t, **pars)

    elif form == 'logf3':
        s = pars.pop('s')
        if s == 1:
            output = _intlogf2(t, **pars)
        else:
            raise NotImplementedError(
                'Analytic integral for logf3 only implemented for s=1. '
                'Select integral=numeric.'
            )

    else:
        errormsg = (
            f'Analytic integral for the selected functional form "{form}" is '
            f'not implemented; choices are: {sc.strjoin(choices)}, or select '
            f'integral=numeric.'
        )
        raise NotImplementedError(errormsg)

    return output


def _compute_severity(t, rel_sev=None, pars=None):
    '''
    This function is used for two types of calculation related to disease progression:
        1. to model the probability of progressing to further disease stages
        2. to model the 'severity' of dysplasia on a scale from 0-1, historically interpreted as
           the percentage of the epithelium affected by dysplasia.
    Args:
        t: array of durations that women have been in their current health state
        rel_sev: array of individual relative severity values
        pars: dict with required key 'form', which dictates which subfunction will be used.

    Notes:
         If the pars dict contains the key 'cin_integral', then this function will call
         _compute_severity_integral to determine the progression probabilities.
    '''

    pars = sc.dcp(pars)

    # Complete these next stages if cancer progression probabilities are being modeled
    # as the cumulative severity-time of dysplasia.
    if pars.get('method') == 'cin_integral':
        del pars['method']
        if pars.get('ld50'):
            ld50 = pars.pop('ld50')
            if pars.get('transform_prob'):
                _ = pars.pop('transform_prob')
            sev_at_ld50 = _compute_severity_integral(np.array([ld50]), rel_sev=None, pars=pars)[0]
            transform_prob = 1 - 0.5**(1/sev_at_ld50**2)
        elif pars.get('transform_prob'):
            transform_prob = pars.pop('transform_prob')
        else:
            errormsg = ('If using calculating cancer probabilities using the integral of the CIN function, '
                        'must provide an LD50 or transform prob.')
            raise ValueError(errormsg)

        sev = _compute_severity_integral(t, rel_sev=rel_sev, pars=pars)
        cancer_probs = 1 - np.power(1 - transform_prob, sev**2)
        return cancer_probs

    # Proceed with severity calculations
    form = pars.pop('form')
    choices = [
        'logf2',
        'logf3',
        'linear',
    ]

    # Scale t
    if rel_sev is not None:
        t = rel_sev * t

    # Process inputs
    if form is None or form == 'logf2':
        output = _logf2(t, **pars)

    elif form == 'logf3':
        output = _logf3(t, **pars)

    elif form == 'linear':
        raise NotImplementedError('linear form not used in M02')

    elif callable(form):
        output = form(t, **pars)

    else:
        errormsg = f'The selected functional form "{form}" is not implemented; choices are: {sc.strjoin(choices)}'
        raise NotImplementedError(errormsg)

    return output


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
            # M02 progression durations (sourced from GenotypePars('hpv16') —
            # v2 _v2_legacy/parameters.py:337,339 + 96):
            #   dur_precin: lognormal(par1=3, par2=9)   — mean=3y, std=9y
            #   dur_cin:    lognormal(par1=5, par2=20)  — mean=5y, std=20y
            #   dur_cancer: lognormal(par1=8, par2=3)   — mean=8y, std=3y
            # par1/par2 are mean/std of the lognormal itself per
            # _v2_legacy/utils.py:239 (sample('lognormal') docstring).
            dur_precin=ss.lognorm_ex(mean=ss.years(3.0), std=ss.years(9.0)),
            dur_cin=ss.lognorm_ex(mean=ss.years(5.0), std=ss.years(20.0)),
            dur_cancer=ss.lognorm_ex(mean=ss.years(8.0), std=ss.years(3.0)),
            # M02 severity functions (passed verbatim to _compute_severity).
            # cancer_fn flattens cin_fn's keys so _compute_severity's cin_integral
            # branch can run _compute_severity_integral internally (matching v2's
            # _v2_legacy/people.py:274 merge-on-call pattern).
            cin_fn=dict(form='logf2', k=0.3, x_infl=0, ttc=50),
            cancer_fn=dict(method='cin_integral', transform_prob=2e-3,
                           form='logf2', k=0.3, x_infl=0, ttc=50),
            # M02 Bernoulli distributions for CIN and cancer draws.
            # Placeholder p=0.5 is overwritten per-call via .set(p=per_agent_arr)
            # in set_prognoses; they live in pars so starsim initializes them
            # (links RNG, registers slots) before the first set_prognoses call.
            _cin_bern=ss.bernoulli(p=0.5),
            _cancer_bern=ss.bernoulli(p=0.5),
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
            # M01 states
            ss.FloatArr('ti_clearance', label='Time of natural clearance'),
            ss.FloatArr('ti_first_infection', label='Time of first infection'),
            # M02 progression compartments
            ss.BoolState('precin', label='Precancerous infection'),
            ss.BoolState('cin', label='Cervical intraepithelial neoplasia'),
            ss.BoolState('cancerous', label='Invasive cancer'),
            # M02 scheduled transition times
            ss.FloatArr('ti_cin', label='Time of CIN onset'),
            ss.FloatArr('ti_cancerous', label='Time of invasive cancer onset'),
            ss.FloatArr('ti_dead_cancer', label='Time of cancer-caused death'),
            # M02 sampled durations (kept for analyzers/diagnostics)
            ss.FloatArr('dur_precin', label='Sampled duration of precin'),
            ss.FloatArr('dur_cin', label='Sampled duration of CIN'),
        )

    def set_prognoses(self, uids, sources=None):
        """Sample full natural-history trajectory for newly-infected agents.

        Mirrors v2's _v2_legacy/people.py:set_prognoses algorithm:
          - precin sampled for everyone (males + females)
          - probability of CIN computed via _compute_severity(dur_precin, cin_fn);
            only females eligible. Non-CIN agents clear after dur_precin.
          - probability of cancer computed via _compute_severity(dur_cin,
            cancer_fn). Non-cancer CIN agents clear after dur_precin + dur_cin.
          - cancer agents get ti_cancerous and ti_dead_cancer scheduled;
            cancer-caused removal is handled in step_state via people.request_death.

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
        p = self.pars

        # Record first-ever infection time only for agents whose
        # ti_first_infection is still NaN.
        first_uids = uids[self.ti_first_infection.isnan[uids]]
        self.ti_first_infection[first_uids] = ti

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = ti

        # Mark precin compartment for all newly-infected agents.
        self.precin[uids] = True

        # Reset trajectory fields so re-infection always starts from a clean slate.
        # v2 resets these fields at clearance time (people.py:696-701); here we
        # do it at the start of set_prognoses so that any stale values from a
        # prior infection don't conflict with the newly sampled trajectory.
        # Task 9 (step_state) will perform the same reset at clearance.
        nan = np.nan
        self.ti_clearance[uids] = nan
        self.ti_cin[uids] = nan
        self.ti_cancerous[uids] = nan
        self.ti_dead_cancer[uids] = nan
        self.dur_cin[uids] = nan

        # 1. Sample precin durations for everyone (males + females).
        dur_precin = p.dur_precin.rvs(uids)
        self.dur_precin[uids] = dur_precin

        # 2. Probability of CIN — computed from dur_precin via cin_fn.
        #    Only females are eligible for CIN; males always clear after precin.
        female = np.asarray(self.sim.people.female[uids])
        p_cin = _compute_severity(np.asarray(dur_precin), pars=p.cin_fn)
        p._cin_bern.set(p=p_cin)
        cin_draw = p._cin_bern.rvs(uids)
        cin_mask = cin_draw & female       # boolean array, len == len(uids)
        cin_uids = uids[cin_mask]
        nocin_uids = uids[~cin_mask]

        # 3. Branch A: clearance from precin (males + non-CIN females).
        self.ti_clearance[nocin_uids] = ti + dur_precin[~cin_mask]

        if len(cin_uids) == 0:
            return

        # 4. Branch B: progression to CIN.
        self.ti_cin[cin_uids] = ti + dur_precin[cin_mask]
        dur_cin = p.dur_cin.rvs(cin_uids)
        self.dur_cin[cin_uids] = dur_cin

        # 5. Probability of cancer given dur_cin.
        p_cancer = _compute_severity(np.asarray(dur_cin), pars=p.cancer_fn)
        p._cancer_bern.set(p=p_cancer)
        cancer_draw = p._cancer_bern.rvs(cin_uids)
        cancer_uids = cin_uids[cancer_draw]
        nocancer_uids = cin_uids[~cancer_draw]

        # 5a. Sub-branch: clear after CIN (no cancer).
        self.ti_clearance[nocancer_uids] = (
            self.ti_cin[nocancer_uids] + dur_cin[~cancer_draw]
        )

        # 5b. Sub-branch: progression to cancer.
        if len(cancer_uids) == 0:
            return
        self.ti_cancerous[cancer_uids] = (
            self.ti_cin[cancer_uids] + dur_cin[cancer_draw]
        )
        dur_cancer = p.dur_cancer.rvs(cancer_uids)
        self.ti_dead_cancer[cancer_uids] = (
            self.ti_cancerous[cancer_uids] + dur_cancer
        )

    def step_state(self):
        """Advance agents through the natural-history compartment chain.

        Transitions are applied in this order (order matters — clear first so a
        just-cleared agent is not accidentally re-flipped by a forward transition
        at the same timestep):

          1. Clear from precin (SIS path) — precin→susceptible, no CIN scheduled
          2. Clear from CIN (regression path) — CIN→susceptible, no cancer scheduled
          3. Precin → CIN (ti_cin reached)
          4. CIN → cancerous (ti_cancerous reached; stops transmitting)
          5. Cancer death (ti_dead_cancer reached; routed through people death pipeline)

        ``request_death`` API: ``self.sim.people.request_death(uids)`` — see
        starsim/people.py line 412; takes only a uid array, no extra kwargs.
        """
        ti = self.ti

        # --- 1. Clear from precin (SIS path: precin with no CIN scheduled) ---
        cleared = (self.infected & self.precin & ~self.cin & ~self.cancerous
                   & (self.ti_clearance <= ti)).uids
        if len(cleared):
            self.infected[cleared] = False
            self.susceptible[cleared] = True
            self.precin[cleared] = False

        # --- 2. Clear from CIN (CIN regression: had CIN, no cancer scheduled) ---
        cleared_from_cin = (self.infected & self.cin & ~self.cancerous
                            & (self.ti_clearance <= ti)).uids
        if len(cleared_from_cin):
            self.infected[cleared_from_cin] = False
            self.susceptible[cleared_from_cin] = True
            self.cin[cleared_from_cin] = False

        # --- 3. Precin → CIN ---
        to_cin = (self.precin & ~self.cin & (self.ti_cin <= ti)).uids
        if len(to_cin):
            self.precin[to_cin] = False
            self.cin[to_cin] = True

        # --- 4. CIN → cancerous (no longer infectious, no longer re-infectable) ---
        to_cancerous = (self.cin & ~self.cancerous & (self.ti_cancerous <= ti)).uids
        if len(to_cancerous):
            self.cin[to_cancerous] = False
            self.cancerous[to_cancerous] = True
            self.infected[to_cancerous] = False
            self.susceptible[to_cancerous] = False
            self.rel_trans[to_cancerous] = 0.0

        # --- 5. Cancer death: route through starsim's people death pipeline ---
        to_dead = (self.cancerous & (self.ti_dead_cancer <= ti)).uids
        if len(to_dead):
            self.sim.people.request_death(to_dead)

    def step_die(self, uids):
        """Reset all custom BoolStates for dying agents.

        Required for any custom BoolState per starsim disease pattern. Without
        this, dead agents retain their disease-compartment flags, corrupting
        n_precin / n_cin / n_cancerous result counts.
        """
        super().step_die(uids)
        self.precin[uids] = False
        self.cin[uids] = False
        self.cancerous[uids] = False
        self.infected[uids] = False
        self.susceptible[uids] = False
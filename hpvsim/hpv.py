"""HPV genotype as a Starsim Infection.

M01: single-genotype, transmission-only with SIS clearance (no precin/CIN/cancer).
M02: natural-history states (precin, CIN, cancer) + **partial permanent same-genotype
     immunity**. Clearance returns agents to ``susceptible=True`` (SIS-like), but
     multiplies ``rel_sus`` by ``(1 - imm_init)`` to apply a permanent per-clearance
     reduction in susceptibility. Default ``imm_init=0.35`` is the beta-distribution
     mean from v2's ``_v2_legacy/parameters.py:102`` (``imm_init = dict(dist='beta_mean',
     par1=0.35, par2=0.025)``), meaning agents retain ~65% of their original per-act
     susceptibility after clearance. Re-infection is allowed at this reduced rate,
     matching v2's observed ~151k cumulative infections over 70 years (neither full
     immunity / SIR at ~9k, nor no immunity / SIS at ~196k).
     Cross-genotype immunity and waning remain M03 scope.
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
    """Per-uid initial-infection probability based on v2's age/sex table.

    Matches v2 (_v2_legacy/sim.py:700) which uses np.digitize(age, brackets):
    that returns ``i`` such that ``brackets[i-1] <= age < brackets[i]``,
    which is equivalent to ``np.searchsorted(brackets, age, side='right')``
    (NOT minus 1). An earlier off-by-one shifted the entire prev-by-age
    table down one bracket — peak 0.70 prev landed on ages 24-34 instead
    of v2's 17-24, mis-seeding the initial infectious pool away from peak
    partnership-formation ages and depressing onward transmission.
    """
    age = np.asarray(sim.people.age[uids])
    is_female = np.asarray(sim.people.female[uids])
    bin_idx = np.searchsorted(_INIT_HPV_PREV_AGE_BRACKETS, age, side='right')
    bin_idx = np.clip(bin_idx, 0, len(_INIT_HPV_PREV_F) - 1)
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
            # Male HPV clearance — v2 uses a separate, much shorter distribution
            # for males (_v2_legacy/parameters.py:97 — dur_infection_male
            # lognormal(par1=1, par2=1)). v2's infect() routes females through
            # the precin/CIN/cancer trajectory but males directly through
            # dur_infection_male. Without this, M02 males stayed infected
            # ~3x longer (heavy-tailed dur_precin) → much higher transmission
            # → more cumulative female re-infections → ~10x more cancer events.
            dur_inf_male=ss.lognorm_ex(mean=ss.years(1.0), std=ss.years(1.0)),
            # M02 severity functions (passed verbatim to _compute_severity).
            # cancer_fn flattens cin_fn's keys so _compute_severity's cin_integral
            # branch can run _compute_severity_integral internally (matching v2's
            # _v2_legacy/people.py:274 merge-on-call pattern).
            cin_fn=dict(form='logf2', k=0.3, x_infl=0, ttc=50),
            cancer_fn=dict(method='cin_integral', transform_prob=2e-3,
                           form='logf2', k=0.3, x_infl=0, ttc=50),
            # M02 same-genotype partial permanent immunity.
            # Source: v2 _v2_legacy/parameters.py:102-104 — imm_init / cell_imm_init
            # both `dict(dist='beta_mean', par1=..., par2=0.025)`. We use the
            # scalar means (0.35 / 0.25) directly; v2's variance (0.025) is small
            # and beta-distributed sampling is M03 scope. use_waning=False in v2
            # by default, so no decay applied.
            #
            # On clearance:
            #   rel_sus[uid] = min(rel_sus[uid], 1 - imm_init)        # ~0.65
            #   rel_sev[uid] = min(rel_sev[uid], 1 - cell_imm_init)   # ~0.75
            # imm_init reduces per-act susceptibility (transmission); cell_imm_init
            # reduces severity (rel_sev * dur in compute_severity), damping
            # cancer-progression probability for re-infections.
            imm_init=0.35,
            cell_imm_init=0.25,
            # Age-based dur_cin multiplier (v2 _v2_legacy/parameters.py:99
            # ``age_risk = dict(age=30, risk=2)`` applied in v2 set_prognoses
            # _v2_legacy/people.py:237-241): women aged >= ``age`` get their
            # dur_cin sample multiplied by ``risk``, lengthening time-in-CIN
            # and pushing more cancer progression toward older ages. Without
            # this v3's cancer onset skews ~4y younger than v2.
            age_risk=dict(age=30, risk=2),
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
            # Lifetime "did this agent ever transition to cancerous?" flag.
            # Set in step_state when the cin → cancerous transition fires;
            # NOT reset in step_die. This is the correct counter for total
            # cancer events because ti_cancerous in .raw is preserved even
            # for agents who died of background mortality BEFORE their
            # scheduled ti_cancerous fired (those would otherwise be counted
            # as phantom cancers since the cin → cancerous transition never
            # happened — their `cin` flag was cleared by step_die first).
            ss.BoolState('ever_cancerous', label='Lifetime cancer flag'),
            # Lifetime cause-of-death flag: True iff request_death was
            # called via the cancer pipeline (step_state's
            # to_dead = cancerous & ti_dead_cancer <= ti branch). NOT reset
            # in step_die. Mirrors v2's people.dead_cancer
            # (_v2_legacy/people.py:1069). Required because ti_dead_cancer
            # is scheduled eagerly at cancer onset and may be in the past
            # for agents who died of background mortality after onset but
            # before ti_dead_cancer fired — those would otherwise inflate
            # the cancer-death count.
            ss.BoolState('dead_cancer', label='Cancer cause-of-death flag'),
            # M02 scheduled transition times
            ss.FloatArr('ti_cin', label='Time of CIN onset'),
            ss.FloatArr('ti_cancerous', label='Time of invasive cancer onset'),
            ss.FloatArr('ti_dead_cancer', label='Time of cancer-caused death'),
            # M02 sampled durations (kept for analyzers/diagnostics)
            ss.FloatArr('dur_precin', label='Sampled duration of precin'),
            ss.FloatArr('dur_cin', label='Sampled duration of CIN'),
            # M02 per-agent BIOLOGICAL relative severity (v2 sev_dist;
            # _v2_legacy/parameters.py:98). Sampled once at agent creation
            # from normal_pos(mean=1, std=0.2) and never modified — captures
            # an agent's intrinsic susceptibility to severe disease,
            # independent of immunity state.
            ss.FloatArr('rel_sev', label='Relative severity (biological)', default=1.0),
            # Tracker so newly-added agents (births, immigrants) get their
            # rel_sev sampled on first step; without this they keep the
            # default 1.0 forever, missing v2's per-agent severity variance.
            ss.BoolState('rel_sev_sampled', default=False),
            # M02 per-agent severity immunity. Defaults to 0; set to
            # cell_imm_init (~0.25) on clearance from precin or CIN. Mirrors
            # v2's people.sev_imm[g, :] (per-genotype). v2 keeps rel_sev and
            # sev_imm SEPARATE and combines them as
            # ``t_eff = dur_precin * (1 - sev_imm) * rel_sev`` in
            # compute_severity (_v2_legacy/people.py:222-231 +
            # compute_severity:209-210). An earlier collapse into a single
            # capped rel_sev broke the multiplicative independence and
            # inflated cancer probability for low-baseline-rel_sev agents.
            ss.FloatArr('sev_imm', label='Severity immunity', default=0.0),
        )
        # Per-agent rel_sev baseline distribution (v2 _v2_legacy/parameters.py:98
        # sev_dist=normal_pos(par1=1, par2=0.2)). Used to override the
        # FloatArr default 1.0 once the people array is initialized.
        self._rel_sev_dist = ss.normal(loc=1.0, scale=0.2)
        return

    def init_post(self):
        super().init_post()
        self._sample_rel_sev_for_unset()
        return

    def _sample_rel_sev_for_unset(self):
        """Sample rel_sev from sev_dist for any alive agent whose rel_sev
        hasn't been sampled yet (initial pop on first call, then newborns /
        immigrants on subsequent calls). Mirrors v2's set_static sampling at
        agent creation. Use abs() to match v2's normal_pos truncation.
        """
        unset = (~self.rel_sev_sampled).uids
        if not len(unset):
            return
        sampled = np.abs(np.asarray(self._rel_sev_dist.rvs(unset)))
        self.rel_sev[unset] = sampled
        self.rel_sev_sampled[unset] = True
        return

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

        # Reset trajectory fields on (re-)infection so stale values from a
        # prior infection don't fire spuriously in step_state. Cancer-bound
        # agents have ti_clearance=NaN (not set in set_prognoses for
        # cancer_uids), so they don't clear from CIN — they always progress
        # to cancer. Therefore the "preserve prior schedule across
        # reinfection" path is never exercised, and removing these resets
        # has no observable effect on cancer outcomes (verified empirically).
        # We keep them as defensive zeroing.
        self.ti_clearance[uids] = np.nan
        self.ti_cin[uids] = np.nan
        self.ti_cancerous[uids] = np.nan
        self.ti_dead_cancer[uids] = np.nan
        self.dur_cin[uids] = np.nan

        # 1. Sample precin durations. v2 routes males through a separate,
        #    much shorter dur_infection_male distribution (people.py:1051-1058).
        #    We replicate that here: females use dur_precin (which feeds the
        #    precin/CIN/cancer trajectory); males use dur_inf_male and clear
        #    after that duration without any progression.
        #    v2 (people.py:222-224) applies sev_imm reduction directly to
        #    dur_precin BEFORE compute_severity: dur_precin = sample *
        #    (1 - sev_imm). For same-genotype, sev_imm = cell_imm. We
        #    encode rel_sev (= 1 - cell_imm post-clearance) per agent and
        #    multiply here. Effect: re-infected agents (rel_sev=0.75) have
        #    shorter precin durations → less time infected per re-infection
        #    → lower steady-state HPV prevalence on the population.
        female_all = np.asarray(self.sim.people.female[uids])
        rel_sev_uids = np.asarray(self.rel_sev[uids])
        sev_imm_uids = np.asarray(self.sev_imm[uids])
        # Females: dur_precin = sample * (1 - sev_imm). Matches v2
        # set_prognoses (_v2_legacy/people.py:222-224). The biological
        # rel_sev is NOT applied to dur — it's passed separately to
        # compute_severity below (matches v2's two-factor product).
        dur_precin = p.dur_precin.rvs(uids) * (1.0 - sev_imm_uids)
        # Males: dur = dur_inf_male sample, no immunity reductions
        # (_v2_legacy/people.py:1051-1058 — male check_clearance samples
        # dur_infection_male directly, no sev_imm or rel_sev factor).
        if (~female_all).any():
            male_uids = uids[~female_all]
            dur_inf_male = p.dur_inf_male.rvs(male_uids)
            dur_precin[~female_all] = np.asarray(dur_inf_male)
        self.dur_precin[uids] = dur_precin

        # 2. Probability of CIN — computed from dur_precin via cin_fn.
        #    Only females are eligible for CIN; males always clear after precin.
        #    Matches v2 (_v2_legacy/people.py:229-231): pass rel_sev as a
        #    separate argument so compute_severity computes
        #    t_eff = (sample * (1-sev_imm)) * rel_sev internally. The
        #    two-factor product is what v2 evaluates.
        #    UNITS: dur_precin from p.dur_precin.rvs() comes back in starsim
        #    timesteps. v2's compute_severity / cin_fn expect YEARS (ttc=50
        #    is years time-to-cancer). Convert via *dt before passing.
        dt_yr = float(self.t.dt)
        female = female_all   # alias to keep the existing variable name below
        p_cin = _compute_severity(np.asarray(dur_precin) * dt_yr,
                                   rel_sev=rel_sev_uids, pars=p.cin_fn)
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
        # Apply v2's age_risk multiplier: women aged >= age_risk['age'] get
        # their dur_cin scaled by age_risk['risk']. Lengthens time-in-CIN
        # for older women, shifting cancer onset to older ages and
        # increasing per-CIN cancer probability via compute_severity.
        # Mirrors _v2_legacy/people.py:237-241.
        ages_at_cin = np.asarray(self.sim.people.age[cin_uids])
        age_mod = np.ones(len(cin_uids))
        age_mod[ages_at_cin >= p.age_risk['age']] = p.age_risk['risk']
        dur_cin = np.asarray(dur_cin) * age_mod
        self.dur_cin[cin_uids] = dur_cin

        # 5. Probability of cancer given dur_cin. v2 does NOT apply sev_imm to
        #    dur_cin (only to dur_precin); see _v2_legacy/people.py:241. So
        #    dur_cin stays as sampled (no sev_imm pre-multiplication here).
        #    Pass rel_sev separately for compute_severity to apply (matches
        #    v2 _v2_legacy/people.py:277-278). Same units conversion as #2:
        #    dur_cin (timesteps) → years before compute_severity.
        rel_sev_cin = rel_sev_uids[cin_mask]
        p_cancer = _compute_severity(np.asarray(dur_cin) * dt_yr,
                                      rel_sev=rel_sev_cin, pars=p.cancer_fn)
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
        # Sample rel_sev for any newly-added alive agents (births, immigrants)
        # before they can be selected for infection this step.
        self._sample_rel_sev_for_unset()

        ti = self.ti

        # --- 1. Clear from precin (partial-immunity path) ---
        # M02: clearance returns agents to susceptible=True (SIS-like) and
        # caps rel_sus at (1 - imm_init) (transmission immunity), and SETS
        # sev_imm to cell_imm_init (severity immunity, applied as
        # (1 - sev_imm) factor on dur_precin in set_prognoses). v2 takes
        # np.maximum(prior, new) on repeat clearances
        # (_v2_legacy/immunity.py:172,175); since cell_imm_init is constant,
        # max(prior, cell_imm_init) just sets the value (any prior would be
        # 0 or already cell_imm_init). rel_sev is the BIOLOGICAL baseline
        # and is NOT modified on clearance.
        cleared = (self.infected & self.precin & ~self.cin & ~self.cancerous
                   & (self.ti_clearance <= ti)).uids
        if len(cleared):
            self.infected[cleared] = False
            self.susceptible[cleared] = True
            self.precin[cleared] = False
            self.rel_sus[cleared] = np.minimum(self.rel_sus[cleared],
                                               1.0 - self.pars.imm_init)
            self.sev_imm[cleared] = self.pars.cell_imm_init

        # --- 2. Clear from CIN (CIN regression; same partial-immunity logic) ---
        cleared_from_cin = (self.infected & self.cin & ~self.cancerous
                            & (self.ti_clearance <= ti)).uids
        if len(cleared_from_cin):
            self.infected[cleared_from_cin] = False
            self.susceptible[cleared_from_cin] = True
            self.cin[cleared_from_cin] = False
            self.rel_sus[cleared_from_cin] = np.minimum(self.rel_sus[cleared_from_cin],
                                                        1.0 - self.pars.imm_init)
            self.sev_imm[cleared_from_cin] = self.pars.cell_imm_init

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
            self.ever_cancerous[to_cancerous] = True
            self.infected[to_cancerous] = False
            self.susceptible[to_cancerous] = False
            self.rel_trans[to_cancerous] = 0.0

        # --- 5. Cancer death: route through starsim's people death pipeline ---
        to_dead = (self.cancerous & (self.ti_dead_cancer <= ti)).uids
        if len(to_dead):
            self.dead_cancer[to_dead] = True
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
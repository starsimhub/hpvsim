"""Single-genotype HPV disease module.

Scope note: HPVsim only models cervical cancer outcomes. HPV is also
associated with anal, oropharyngeal, penile, vaginal, and vulvar cancers
(the first three of which can occur in males); those are out of scope here.

Models the natural-history pipeline as a Starsim Infection:

    susceptible -> precin -> (clear | CIN -> (clear | cancerous -> death))

Females are eligible for the full progression; males clear from precin without
entering CIN/cancer. Clearance grants partial permanent same-genotype immunity:
``rel_sus`` is reduced by ``imm_init`` (transmission immunity), and ``sev_imm``
accumulates the running max of beta samples (severity immunity, shortens
future precin durations).

``beta``, ``init_prev``, and progression-pars defaults are HPV16-specific.
Multi-genotype support, cross-immunity, and waning are out of scope for this
module — a future multi-genotype build will instantiate one HPV per genotype
and add a CrossImmunity connector.
"""

import numpy as np
import sciris as sc
import starsim as ss


# Logistic-2 (logf2) family used by cin_fn / cancer_fn. Private to this module.

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
        raise NotImplementedError('linear severity form not implemented')

    elif callable(form):
        output = form(t, **pars)

    else:
        errormsg = f'The selected functional form "{form}" is not implemented; choices are: {sc.strjoin(choices)}'
        raise NotImplementedError(errormsg)

    return output


# Other genotypes (hpv18, hi5, ohr) need per-genotype natural-history pars
# that aren't wired yet; reject them rather than silently using HPV16 defaults.
_KNOWN_GENOTYPES = ('hpv16',)


# Initial HPV prevalence by age bracket and sex. Brackets are inclusive lower
# bounds; the last bracket extends to age 150.
_INIT_HPV_PREV_AGE_BRACKETS = np.array([12, 17, 24, 34, 44, 64, 80, 150])
_INIT_HPV_PREV_M = np.array([0.0, 0.25, 0.60, 0.25, 0.05, 0.01, 0.0005, 0.0])
_INIT_HPV_PREV_F = np.array([0.0, 0.35, 0.70, 0.25, 0.05, 0.01, 0.0005, 0.0])


def _age_stratified_init_prev(module, sim, uids):
    """Per-uid initial-infection probability from the age/sex prevalence table.

    ``side='right'`` so ``brackets[i-1] <= age < brackets[i]``.
    """
    age = sim.people.age[uids]
    is_female = sim.people.female[uids]
    bin_idx = np.searchsorted(_INIT_HPV_PREV_AGE_BRACKETS, age, side='right')
    bin_idx = np.clip(bin_idx, 0, len(_INIT_HPV_PREV_F) - 1)
    out = np.zeros(len(uids))
    out[is_female] = _INIT_HPV_PREV_F[bin_idx[is_female]]
    out[~is_female] = _INIT_HPV_PREV_M[bin_idx[~is_female]]
    return out


class HPV(ss.Infection):
    """Single-genotype HPV disease module.

    The ``genotype`` attribute identifies which strain this instance models;
    a future multi-genotype CrossImmunity connector can use it to discover
    HPV diseases (duck-type marker pattern; cf. rotasim's ``hasattr(disease,
    'G')``).
    """

    def __init__(self, genotype='hpv16', pars=None, **kwargs):
        if genotype not in _KNOWN_GENOTYPES:
            raise ValueError(
                f'genotype must be one of {list(_KNOWN_GENOTYPES)}; got {genotype!r}.'
            )
        self.genotype = genotype
        if 'name' not in kwargs:
            kwargs['name'] = genotype
        super().__init__()
        # Pull natural-history defaults from GenotypePars so there's a single
        # source of truth per genotype.
        from .parameters import get_genotype_pars
        gpars = get_genotype_pars(genotype)
        self.define_pars(
            init_prev=ss.bernoulli(p=_age_stratified_init_prev),
            beta=gpars.beta,
            dur_precin=gpars.dur_precin,
            dur_cin=gpars.dur_cin,
            dur_cancer=gpars.dur_cancer,
            dur_inf_male=gpars.dur_inf_male,
            cin_fn=gpars.cin_fn,
            cancer_fn=gpars.cancer_fn,
            imm_init=gpars.imm_init,
            cell_imm_init=gpars.cell_imm_init,
            age_risk=gpars.age_risk,
            # Per-call Bernoullis for CIN and cancer draws; ``p`` is overwritten
            # via .set(p=...) in set_prognoses. Held in pars (vs. as plain
            # attributes) so the per-Dist RNG-slot identifier follows the
            # ``module.pars._cin_bern`` path — moving them changes which CRN
            # slot is drawn and shifts regression numbers past the ±10% gates.
            _cin_bern=ss.bernoulli(p=0.5),
            _cancer_bern=ss.bernoulli(p=0.5),
        )
        self.update_pars(pars=pars, **kwargs)
        # ss.Infection provides: susceptible, infected, rel_sus, rel_trans,
        # ti_infected. We add the natural-history states below.
        self.define_states(
            ss.FloatArr('ti_clearance', label='Time of natural clearance'),
            # Set on first infection only; preserves first-infection age
            # across reinfection (ti_infected is overwritten each time).
            ss.FloatArr('ti_first_infection', label='Time of first infection'),
            ss.BoolState('precin', label='Precancerous infection'),
            ss.BoolState('cin', label='Cervical intraepithelial neoplasia'),
            ss.BoolState('cancerous', label='Invasive cancer'),
            ss.FloatArr('ti_cin', label='Time of CIN onset'),
            ss.FloatArr('ti_cancerous', label='Time of invasive cancer onset'),
            ss.FloatArr('ti_dead_cancer', label='Time of cancer-caused death'),
            # Per-agent biological severity baseline. Sampled once at agent
            # creation and never modified; passed separately to _compute_severity
            # so the severity model evaluates t_eff = dur * (1 - sev_imm) * rel_sev.
            ss.FloatArr('rel_sev', label='Relative severity (biological)', default=1.0),
            # Tracks whether rel_sev has been sampled for an agent yet.
            ss.BoolState('rel_sev_sampled', default=False),
            # Severity immunity, accumulated as max-of-beta-samples on each
            # clearance. Shortens future dur_precin via (1 - sev_imm) factor.
            ss.FloatArr('sev_imm', label='Severity immunity', default=0.0),
        )
        # Baseline rel_sev distribution; abs() in _sample_rel_sev_for_unset
        # implements positive truncation.
        self._rel_sev_dist = ss.normal(loc=1.0, scale=0.2)
        return

    def init_post(self):
        super().init_post()
        self._sample_rel_sev_for_unset()
        return

    def init_results(self):
        """Per-step Results emitted from ``step_state``.

        ``new_cancers`` / ``new_cancer_deaths`` are realized-event counters
        (the cin -> cancerous and cancerous -> dead transitions);
        ``cum_*`` are populated as cumulative sums in ``finalize_results``.
        ``sum_age_at_*`` are per-step accumulators; mean age = ``sum / count``.
        """
        super().init_results()
        self.define_results(
            ss.Result('new_cancers', dtype=int, scale=True,
                      label='New cancers'),
            ss.Result('new_cancer_deaths', dtype=int, scale=True,
                      label='New cancer deaths'),
            ss.Result('cum_cancers', dtype=int, scale=True,
                      label='Cumulative cancers'),
            ss.Result('cum_cancer_deaths', dtype=int, scale=True,
                      label='Cumulative cancer deaths'),
            ss.Result('sum_age_at_cancer', dtype=float, scale=True,
                      label='Sum of ages at cancer onset'),
            ss.Result('sum_age_at_cancer_death', dtype=float, scale=True,
                      label='Sum of ages at cancer death'),
        )
        return

    def finalize_results(self):
        super().finalize_results()
        res = self.results
        res.cum_cancers[:] = np.cumsum(res.new_cancers)
        res.cum_cancer_deaths[:] = np.cumsum(res.new_cancer_deaths)
        return

    def _sample_rel_sev_for_unset(self):
        """Sample rel_sev for any alive agent without a sample yet.

        Runs once at init for the starting population, then per-step for
        newborns and immigrants. ``abs()`` folds the negative tail of the
        normal back onto positives; with loc=1.0, scale=0.2 the affected
        mass is < 1e-6, so this matches v2's positive-only convention
        without changing the practical distribution.
        """
        unset = (~self.rel_sev_sampled).uids
        if not len(unset):
            return
        sampled = np.abs(self._rel_sev_dist.rvs(unset))
        self.rel_sev[unset] = sampled
        self.rel_sev_sampled[unset] = True
        return

    def set_prognoses(self, uids, sources=None):
        """Sample full natural-history trajectory for newly-infected agents.

          - dur_precin sampled for everyone; males then clear after
            dur_inf_male without entering CIN/cancer.
          - For females, P(CIN) = _compute_severity(dur_precin, cin_fn).
            Non-CIN agents clear after dur_precin.
          - For CIN agents, P(cancer) = _compute_severity(dur_cin, cancer_fn).
            Non-cancer agents clear after dur_precin + dur_cin.
          - Cancer agents get ti_cancerous and ti_dead_cancer scheduled; the
            CIN -> cancerous transition and request_death fire from step_state.

        Initial seeding via ``init_post -> set_prognoses`` flows through here
        so init_prev-seeded agents get ti_first_infection set at ti=0.
        """
        super().set_prognoses(uids, sources)
        ti = self.ti
        p = self.pars

        # Record ti_first_infection only for agents seeing their first infection.
        first_uids = uids[self.ti_first_infection.isnan[uids]]
        self.ti_first_infection[first_uids] = ti

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = ti
        self.precin[uids] = True

        # Defensive reset so stale schedules from a prior infection don't
        # fire spuriously in step_state.
        self.ti_clearance[uids] = np.nan
        self.ti_cin[uids] = np.nan
        self.ti_cancerous[uids] = np.nan
        self.ti_dead_cancer[uids] = np.nan

        female_all = self.sim.people.female[uids]
        rel_sev_uids = self.rel_sev[uids]
        sev_imm_uids = self.sev_imm[uids]

        # 1. Sample precin durations.
        #    Females: sample * (1 - sev_imm); rel_sev passes separately to
        #    _compute_severity below so the model forms the two-factor product.
        #    Males: dur_inf_male sample with no immunity reductions; they then
        #    clear from precin without progression.
        dur_precin = p.dur_precin.rvs(uids) * (1.0 - sev_imm_uids)
        if (~female_all).any():
            male_uids = uids[~female_all]
            dur_inf_male = p.dur_inf_male.rvs(male_uids)
            dur_precin[~female_all] = dur_inf_male

        # 2. P(CIN) per female. Distributions return durations in starsim
        #    timesteps; convert to years before passing (cin_fn's ttc=50 is years).
        dt_yr = float(self.t.dt)
        female = female_all
        p_cin = _compute_severity(dur_precin * dt_yr,
                                   rel_sev=rel_sev_uids, pars=p.cin_fn)
        p._cin_bern.set(p=p_cin)
        cin_draw = p._cin_bern.rvs(uids)
        cin_mask = cin_draw & female
        cin_uids = uids[cin_mask]
        nocin_uids = uids[~cin_mask]

        # 3. Branch A: clearance from precin (males + non-CIN females).
        self.ti_clearance[nocin_uids] = ti + dur_precin[~cin_mask]

        if len(cin_uids) == 0:
            return

        # 4. Branch B: progression to CIN.
        self.ti_cin[cin_uids] = ti + dur_precin[cin_mask]
        dur_cin = p.dur_cin.rvs(cin_uids)
        # age_risk multiplier: women aged >= age_risk['age'] get dur_cin
        # scaled by age_risk['risk'].
        ages_at_cin = self.sim.people.age[cin_uids]
        age_mod = np.ones(len(cin_uids))
        age_mod[ages_at_cin >= p.age_risk['age']] = p.age_risk['risk']
        dur_cin = dur_cin * age_mod

        # 5. P(cancer) given dur_cin. sev_imm is NOT applied to dur_cin —
        #    only rel_sev (passed separately to _compute_severity). Same
        #    timestep -> years conversion as step 2.
        rel_sev_cin = rel_sev_uids[cin_mask]
        p_cancer = _compute_severity(dur_cin * dt_yr,
                                      rel_sev=rel_sev_cin, pars=p.cancer_fn)
        p._cancer_bern.set(p=p_cancer)
        cancer_draw = p._cancer_bern.rvs(cin_uids)
        cancer_uids = cin_uids[cancer_draw]
        nocancer_uids = cin_uids[~cancer_draw]

        # 5a. Clear after CIN (no cancer).
        self.ti_clearance[nocancer_uids] = (
            self.ti_cin[nocancer_uids] + dur_cin[~cancer_draw]
        )

        # 5b. Progression to cancer.
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

        Order matters: clearance fires first so a just-cleared agent isn't
        re-flipped by a forward transition at the same timestep.

          1. Clearance from precin or CIN (partial-immunity path)
          2. precin -> CIN
          3. CIN -> cancerous (stops transmitting)
          4. Cancer death (via people.request_death)
        """
        # Sample rel_sev for any newly-added alive agents (births, immigrants)
        # before they can be selected for infection this step.
        self._sample_rel_sev_for_unset()

        ti = self.ti

        # --- 1. Clearance (from precin OR CIN) — partial-immunity path ---
        # Returns agent to susceptible=True. rel_sus is capped at (1 - imm_init)
        # (transmission immunity). sev_imm accumulates as max(prior, new beta
        # sample) — the running max gives multi-cleared agents higher sev_imm
        # than the distribution mean. rel_sev (biological baseline) is unchanged.
        cleared = (self.infected & (self.precin | self.cin) & ~self.cancerous
                   & (self.ti_clearance <= ti)).uids
        if len(cleared):
            self.infected[cleared] = False
            self.susceptible[cleared] = True
            self.precin[cleared] = False
            self.cin[cleared] = False
            self.rel_sus[cleared] = np.minimum(self.rel_sus[cleared],
                                               1.0 - self.pars.imm_init)
            new_imm = self.pars.cell_imm_init.rvs(cleared)
            self.sev_imm[cleared] = np.maximum(self.sev_imm[cleared], new_imm)

        # --- 2. precin -> CIN ---
        to_cin = (self.precin & ~self.cin & (self.ti_cin <= ti)).uids
        if len(to_cin):
            self.precin[to_cin] = False
            self.cin[to_cin] = True

        # --- 3. CIN -> cancerous (no longer infectious, no longer re-infectable) ---
        to_cancerous = (self.cin & ~self.cancerous & (self.ti_cancerous <= ti)).uids
        if len(to_cancerous):
            self.cin[to_cancerous] = False
            self.cancerous[to_cancerous] = True
            self.infected[to_cancerous] = False
            self.susceptible[to_cancerous] = False
            self.rel_trans[to_cancerous] = 0.0
            ages_at_cancer = self.sim.people.age[to_cancerous]
            self.results.new_cancers[ti] = len(to_cancerous)
            self.results.sum_age_at_cancer[ti] = float(ages_at_cancer.sum())

        # --- 4. Cancer death (routed through starsim's people death pipeline) ---
        to_dead = (self.cancerous & (self.ti_dead_cancer <= ti)).uids
        if len(to_dead):
            ages_at_death = self.sim.people.age[to_dead]
            self.results.new_cancer_deaths[ti] = len(to_dead)
            self.results.sum_age_at_cancer_death[ti] = float(ages_at_death.sum())
            self.sim.people.request_death(to_dead)

    def step_die(self, uids):
        """Reset transient compartment flags for dying agents.

        Without this, dead agents would keep their compartment flags and
        corrupt n_precin / n_cin / n_cancerous counts.
        """
        super().step_die(uids)
        self.precin[uids] = False
        self.cin[uids] = False
        self.cancerous[uids] = False
        self.infected[uids] = False
        self.susceptible[uids] = False
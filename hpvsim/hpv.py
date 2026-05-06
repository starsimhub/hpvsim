"""Per-genotype HPV disease module.

Scope note: HPVsim only models cervical cancer outcomes. HPV is also
associated with anal, oropharyngeal, penile, vaginal, and vulvar cancers
(the first three of which can occur in males); those are out of scope here.

Models the natural-history pipeline as a Starsim Infection:

    susceptible -> precin -> (clear | CIN -> (clear | cancerous -> death))

Females are eligible for the full progression; males clear from precin without
entering CIN/cancer. Clearance grants partial same-genotype immunity: per-agent
beta samples accumulate as a running max into ``nab_imm`` (humoral) and
``cell_imm`` (cell-mediated). The ``CrossImmunity`` connector reads those
source states each step and writes per-target ``rel_sus`` and ``sev_imm``.

Multi-genotype runs instantiate one HPV per genotype, all sharing the People
and going through the same Connector path; a 1-genotype run uses a 1×1
identity matrix on the Connector.
"""

import numpy as np
import starsim as ss

from .utils import compute_severity


# Per-genotype initial HPV prevalence by age bracket and sex. Brackets are
# inclusive lower bounds; the last bracket extends to age 150. HPV16 retains
# M02's curves verbatim; hpv18/hi5/ohr scale-down defaults from v2 reference
# tables (we use 0.6x for hpv18 and 0.4x for hi5/ohr as proxies for the lower
# observed prevalence of those clades; calibrated values come in M04).
_INIT_HPV_PREV_AGE_BRACKETS = np.array([12, 17, 24, 34, 44, 64, 80, 150])

_INIT_PREV = {
    'hpv16': {
        'm': np.array([0.0, 0.25, 0.60, 0.25, 0.05, 0.01, 0.0005, 0.0]),
        'f': np.array([0.0, 0.35, 0.70, 0.25, 0.05, 0.01, 0.0005, 0.0]),
    },
    'hpv18': {
        'm': np.array([0.0, 0.15, 0.36, 0.15, 0.03, 0.006, 0.0003, 0.0]),
        'f': np.array([0.0, 0.21, 0.42, 0.15, 0.03, 0.006, 0.0003, 0.0]),
    },
    'hi5': {
        'm': np.array([0.0, 0.10, 0.24, 0.10, 0.02, 0.004, 0.0002, 0.0]),
        'f': np.array([0.0, 0.14, 0.28, 0.10, 0.02, 0.004, 0.0002, 0.0]),
    },
    'ohr': {
        'm': np.array([0.0, 0.10, 0.24, 0.10, 0.02, 0.004, 0.0002, 0.0]),
        'f': np.array([0.0, 0.14, 0.28, 0.10, 0.02, 0.004, 0.0002, 0.0]),
    },
}

# Public re-export — kept module-level for backward-compat with M02 imports.
_INIT_HPV_PREV_M = _INIT_PREV['hpv16']['m']
_INIT_HPV_PREV_F = _INIT_PREV['hpv16']['f']


def _make_init_prev_fn(genotype):
    """Return the per-uid init-prev sampler for a given genotype."""
    curves = _INIT_PREV[genotype]
    f_curve = curves['f']
    m_curve = curves['m']

    def _age_stratified(module, sim, uids):
        age = np.asarray(sim.people.age[uids])
        is_female = np.asarray(sim.people.female[uids])
        bin_idx = np.searchsorted(_INIT_HPV_PREV_AGE_BRACKETS, age, side='right')
        bin_idx = np.clip(bin_idx, 0, len(f_curve) - 1)
        out = np.zeros(len(uids))
        out[is_female] = f_curve[bin_idx[is_female]]
        out[~is_female] = m_curve[bin_idx[~is_female]]
        return out
    return _age_stratified


_KNOWN_GENOTYPES = tuple(_INIT_PREV.keys())


class HPV(ss.Infection):
    """Per-genotype HPV disease module.

    The ``genotype`` attribute identifies which strain this instance models.
    The CrossImmunity connector reads each registered HPV's nab_imm/cell_imm
    each step and writes per-target rel_sus/sev_imm.
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
            init_prev=ss.bernoulli(p=_make_init_prev_fn(genotype)),
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
            # Per-genotype beta scaler and serology probability (multi-genotype).
            rel_beta=gpars.rel_beta,
            sero_prob=gpars.sero_prob,
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
            # creation and never modified; passed separately to compute_severity
            # so the severity model evaluates t_eff = dur * (1 - sev_imm) * rel_sev.
            ss.FloatArr('rel_sev', label='Relative severity (biological)', default=1.0),
            # Tracks whether rel_sev has been sampled for an agent yet.
            ss.BoolState('rel_sev_sampled', default=False),
            # Severity immunity, accumulated as max-of-beta-samples on each
            # clearance. Shortens future dur_precin via (1 - sev_imm) factor.
            ss.FloatArr('sev_imm', label='Severity immunity', default=0.0),
            # M03 raw source-genotype immunity. Bumped on clearance; read by
            # CrossImmunity Connector to derive per-target rel_sus and sev_imm.
            ss.FloatArr('nab_imm', label='Humoral immunity (source genotype)', default=0.0),
            ss.FloatArr('cell_imm', label='Cell-mediated immunity (source genotype)', default=0.0),
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
          - For females, P(CIN) = compute_severity(dur_precin, cin_fn).
            Non-CIN agents clear after dur_precin.
          - For CIN agents, P(cancer) = compute_severity(dur_cin, cancer_fn).
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
        #    compute_severity below so the model forms the two-factor product.
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
        p_cin = compute_severity(dur_precin * dt_yr,
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
        #    only rel_sev (passed separately to compute_severity). Same
        #    timestep -> years conversion as step 2.
        rel_sev_cin = rel_sev_uids[cin_mask]
        p_cancer = compute_severity(dur_cin * dt_yr,
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
        # Returns agent to susceptible=True. nab_imm and cell_imm accumulate
        # the running max of per-agent beta samples; the CrossImmunity connector
        # reads them next step to derive rel_sus and sev_imm.
        cleared = (self.infected & (self.precin | self.cin) & ~self.cancerous
                   & (self.ti_clearance <= ti)).uids
        if len(cleared):
            self.infected[cleared] = False
            self.susceptible[cleared] = True
            self.precin[cleared] = False
            self.cin[cleared] = False
            new_nab = np.asarray(self.pars.imm_init.rvs(cleared))
            self.nab_imm[cleared] = np.maximum(self.nab_imm[cleared], new_nab)
            new_cell = np.asarray(self.pars.cell_imm_init.rvs(cleared))
            self.cell_imm[cleared] = np.maximum(self.cell_imm[cleared], new_cell)

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
"""HIV–HPV co-infection.

Two ways to drive HIV, both sharing the same CD4-stratified HIV->HPV effects
and HIV-stratified results (all owned by the shared ``HIV`` base class):
  - ``HIV_transmit``: network-transmitted, via beta_m2f/rel_beta_f2m onto
    hpv.SexualNetwork.
  - ``HIV_incidence``: a per-(year, sex, age) incidence curve imposed
    directly, no network transmission.
"""

import numpy as np
import starsim as ss
import stisim as sti

from . import misc
from .hpv import HPV
from .network import SexualNetwork

__all__ = ['HIV_transmit', 'HIV_incidence']


class HIV(sti.HIV):
    """Shared base for hpvsim's HIV variants (HIV_transmit, HIV_incidence).

    Owns the CD4-stratified HIV->HPV effect pars (rel_sus/rel_sev/rel_imm/
    rel_reactivation, each {effect}_lo/{effect}_hi, plus cd4_threshold/
    cd4_upper) and the HIV-stratified results (cancers, cancer incidence,
    HPV prevalence by HIV status, and a cancer rate ratio) -- the single
    place to look for HIV's effect on HPV, regardless of which subclass
    drives infection.

    rel_sus is written directly onto every HPV module in step_state(), with
    no separate connector: HPV.step_state() resets rel_sus=1.0 before any
    disease's step_state() runs (inside hpv.Sim, hpv diseases are always
    registered before other diseases, so this is guaranteed to have already
    happened -- a fully-manual diseases=[...] construction outside hpv.Sim
    does not get this guarantee automatically), and
    CrossImmunity multiplies (not assigns) its own factor onto rel_sus too --
    the two compose correctly regardless of order. rel_sev/rel_imm/
    rel_reactivation are read lazily by HPV's own set_prognoses/step_state at
    their point of use, unchanged from before.

    A raw `sti.HIV()` instance, constructed by a user bypassing hpvsim's
    classes entirely, does NOT get these HPV-modulation effects -- only
    `HIV_transmit`/`HIV_incidence` do. This is an intentional tradeoff:
    the CD4-effect pars/states/results live directly on this class, not a
    separate connector that could discover any `sti.HIV`-shaped disease.
    """

    def __init__(self, init_prev_data=None, name='hiv'):
        super().__init__(init_prev_data=init_prev_data, name=name)
        self.define_pars(
            rel_sus_lo=2.2,   rel_sus_hi=2.2,    # increased HPV acquisition
            rel_sev_lo=1.5,   rel_sev_hi=1.2,    # faster/worse CIN->cancer progression
            rel_imm_lo=0.36,  rel_imm_hi=0.76,   # reduced post-infection/vaccine immunity
            rel_reactivation_lo=1.0, rel_reactivation_hi=1.0,  # latent reactivation hazard multiplier (neutral default; no prior calibrated value)
            cd4_threshold=200.0,  # lo/hi stratum boundary
            cd4_upper=500.0,      # CD4 >= this -> no effect at all (see class docstring)
        )
        self.define_states(
            ss.FloatArr('hiv_rel_sus', default=1.0),
            ss.FloatArr('hiv_rel_sev', default=1.0),
            ss.FloatArr('hiv_rel_imm', default=1.0),
            ss.FloatArr('hiv_rel_reactivation', default=1.0),
        )
        self.hpv_modules = None
        # No update_pars() call here -- deferred to whichever leaf class
        # (HIV_transmit/HIV_incidence) is actually instantiated, after it
        # layers its own define_pars, matching HPV.__init__'s pattern.

    def init_pre(self, sim):
        super().init_pre(sim)
        self.hpv_modules = [m for m in sim.diseases.values() if isinstance(m, HPV)]
        if not self.hpv_modules:
            raise ValueError('hpv.HIV requires HPV genotype module(s) in the sim.')

    def _cd4_stratum(self, cd4):
        """True for the hi stratum (CD4 >= cd4_threshold), False for lo."""
        return cd4 >= self.pars.cd4_threshold

    def _factor_array(self, effect, hiv_pos, is_hi, n):
        """Build a per-agent factor array (1.0 for HIV-, stratum value for HIV+)."""
        out = np.ones(n, dtype=float)
        lo = self.pars[f'{effect}_lo']
        hi = self.pars[f'{effect}_hi']
        vals = np.where(is_hi, hi, lo)
        out[hiv_pos] = vals[hiv_pos]
        return out

    def step_state(self):
        super().step_state()
        auids = self.sim.people.auids
        cd4 = np.asarray(self.cd4[auids])
        # Effects apply only to HIV+ agents with an initialized CD4 in
        # [0, cd4_upper). CD4 >= cd4_upper (newly infected at ~594, or
        # ART-reconstituted) falls outside the hi band and gets NO effect (factor 1.0).
        infected = self.infected[auids]
        hiv_pos = infected & ~np.isnan(cd4) & (cd4 < self.pars.cd4_upper)
        is_hi = self._cd4_stratum(np.nan_to_num(cd4, nan=1e4))
        n = len(auids)
        self.hiv_rel_sus[auids] = self._factor_array('rel_sus', hiv_pos, is_hi, n)
        self.hiv_rel_sev[auids] = self._factor_array('rel_sev', hiv_pos, is_hi, n)
        self.hiv_rel_imm[auids] = self._factor_array('rel_imm', hiv_pos, is_hi, n)
        self.hiv_rel_reactivation[auids] = self._factor_array('rel_reactivation', hiv_pos, is_hi, n)
        # rel_sus is written for all agents, but Starsim only samples it for
        # susceptibles during step_infect.
        for m in self.hpv_modules:
            m.rel_sus[auids] = m.rel_sus[auids] * self.hiv_rel_sus[auids]

        # No testing cascade: diagnose all living HIV+ agents each step,
        # matching the deleted hiv_art's old behavior. No-op for
        # HIV_incidence (already diagnosed at infection).
        undiagnosed = self.infected & ~self.diagnosed & self.sim.people.alive
        uids = undiagnosed.uids
        if len(uids):
            self.diagnosed[uids] = True
            self.ti_diagnosed[uids] = self.ti
            self.ti_art[uids] = self.ti + 1
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('cancers_with_hiv', dtype=float, label='New cancers (HIV+)'),
            ss.Result('cancers_no_hiv', dtype=float, label='New cancers (HIV-)'),
            ss.Result('cancer_incidence_with_hiv', dtype=float, scale=False, label='Cancer incidence per 100k (HIV+)'),
            ss.Result('cancer_incidence_no_hiv', dtype=float, scale=False, label='Cancer incidence per 100k (HIV-)'),
            ss.Result('hpv_prevalence_with_hiv', dtype=float, scale=False, label='HPV prevalence (HIV+)'),
            ss.Result('hpv_prevalence_no_hiv', dtype=float, scale=False, label='HPV prevalence (HIV-)'),
            ss.Result('cancer_rate_ratio', dtype=float, scale=False, label='Cancer incidence rate ratio (HIV+/HIV-)'),
        )

    def update_results(self):
        """HIV-stratified HPV/cancer outcomes (absorbed from the former
        HIVStratifiedResults analyzer).

        Cancer detection: each HPV module fires its cin->cancerous transition
        in step_state for agents with ti_cancerous <= ti. Since ti_cancerous
        is scheduled as ti_cin + _randround(...) (an integer step >= ti_cin)
        and the agent is already CIN by then, the transition fires at exactly
        ti == ti_cancerous, so detecting flips this step via
        cancerous & (ti_cancerous == ti) matches the agents that just turned
        cancerous. NaN ti_cancerous (no scheduled cancer) compares False, so
        it is safely excluded.
        """
        super().update_results()
        ti = self.sim.ti
        people = self.sim.people
        alive = people.alive.values
        hiv_pos = self.infected.values & alive
        hiv_neg = (~self.infected.values) & alive

        any_hpv = np.zeros(alive.shape, dtype=bool)
        for m in self.hpv_modules:
            any_hpv |= m.infected.values

        # Scale-weight all counts by people.scale so grow-multiscale fine agents
        # (scale = 1/ms_agent_ratio) count fractionally, consistent with every
        # other hpvsim result. Prevalence weights numerator AND denominator;
        # an unweighted version over-counts fine agents at ms_agent_ratio > 1
        # (cancers concentrate on fine agents).
        scale = people.scale.values
        n_pos = float((hiv_pos * scale).sum())
        n_neg = float((hiv_neg * scale).sum())
        self.results['hpv_prevalence_with_hiv'][ti] = (
            float(((any_hpv & hiv_pos) * scale).sum()) / n_pos if n_pos else 0.0)
        self.results['hpv_prevalence_no_hiv'][ti] = (
            float(((any_hpv & hiv_neg) * scale).sum()) / n_neg if n_neg else 0.0)

        # New cancers this step, attributed by current HIV status. NOTE: this
        # runs after step_die in the Starsim loop, so an agent who turns
        # cancerous in step_state AND dies from background demographics the
        # same step has both `cancerous` and the HIV `infected` flag cleared
        # by the time we read them here -- that cancer is counted by
        # HPVTotal.new_cancers (recorded in step_state) but missed here. The
        # bias is O(P_death x P_cancer_transition) per step, negligible at
        # typical scales.
        new_cancer = np.zeros(alive.shape, dtype=bool)
        for m in self.hpv_modules:
            fired = (m.cancerous.values & (m.ti_cancerous.values == ti))
            new_cancer |= fired
        cancers_with_hiv = float(((new_cancer & hiv_pos) * scale).sum())
        cancers_no_hiv = float(((new_cancer & hiv_neg) * scale).sum())
        self.results['cancers_with_hiv'][ti] = cancers_with_hiv
        self.results['cancers_no_hiv'][ti] = cancers_no_hiv

        # Per-100k incidence rate (not age-standardized -- see all_hpv's
        # asr_cancer_incidence for the WHO2000 age-standardized version).
        inc_with_hiv = cancers_with_hiv / n_pos * 1e5 if n_pos else 0.0
        inc_no_hiv = cancers_no_hiv / n_neg * 1e5 if n_neg else 0.0
        self.results['cancer_incidence_with_hiv'][ti] = inc_with_hiv
        self.results['cancer_incidence_no_hiv'][ti] = inc_no_hiv
        self.results['cancer_rate_ratio'][ti] = inc_with_hiv / inc_no_hiv if inc_no_hiv else 0.0


class HIV_transmit(HIV):
    """Network-transmitted HIV -- see HIV for the CD4-effect/results machinery
    this inherits. Adds beta_m2f/rel_beta_f2m + validate_beta() targeting
    hpv.SexualNetwork.

    Use this when HIV prevalence should emerge from network transmission
    dynamics; use HIV_incidence instead to impose a known incidence curve
    directly.
    """

    def __init__(self, init_prev_data=None, pars=None, name='hiv', **kwargs):
        super().__init__(init_prev_data=init_prev_data, name=name)
        self.define_pars(beta_m2f=0.0035, rel_beta_f2m=0.5)
        self.update_pars(pars, **kwargs)

    def init_pre(self, sim):
        super().init_pre(sim)
        if not any(isinstance(n, SexualNetwork) for n in sim.networks.values()):
            misc.warn('hpv.HIV_transmit: no SexualNetwork found; HIV will not transmit.')

    def validate_beta(self):
        """Route beta_m2f/rel_beta_f2m onto every SexualNetwork's betamap entry.

        hpv.SexualNetwork puts females in p1, males in p2, so betamap[net][0]
        = f2m (smaller), betamap[net][1] = m2f = beta_m2f (larger).
        """
        betamap = super().validate_beta()
        for net in self.sim.networks.values():
            if isinstance(net, SexualNetwork):
                key = ss.standardize_netkey(net.name)
                betamap[key] = [self.pars.beta_m2f * self.pars.rel_beta_f2m, self.pars.beta_m2f]
        return betamap


class HIV_incidence(HIV):
    """Incidence-driven HIV: imposes a per-(age,sex,year) incidence curve
    directly as a disease, with no network transmission (infect() is fully
    overridden -- see stisim's SimpleBV for the precedent this follows).

    Every newly-infected agent is immediately diagnosed and ART-scheduled
    (ti_art = ti + 1 -- see the note in infect() on why +1, not ti, is
    required), so plain sti.ART(coverage=...) works against this disease
    directly, with no separate testing-cascade intervention needed.

    The incidence DataFrame has columns [age, sex, year, incidence] (sex
    'f'/'m'; incidence = the per-year HIV acquisition rate among susceptibles).

    FOI -> per-step probability: a per-year rate r is converted to a
    per-timestep infection probability with the exponential survival form
    p = 1 - exp(-r * dt_years) (correct for non-small rates), then drawn via
    a CRN-safe ss.bernoulli. Years outside the data's [min, max] range are
    nearest-year clamped (incidence is 0 before the curve begins).
    """

    def __init__(self, incidence=None, pars=None, init_prev_data=None, name='hiv', **kwargs):
        super().__init__(init_prev_data=init_prev_data, name=name)
        if incidence is None:
            raise ValueError(
                'HIV_incidence requires an incidence DataFrame '
                '[age, sex, year, incidence]; pass incidence=.'
            )
        self.incidence = incidence
        # Per-agent infection draw; p is set per-step from the looked-up rates.
        self.infect_bern = ss.bernoulli(p=0.0)
        # Filled in init_pre:
        self._years = None          # sorted unique years (int)
        self._year_index = None     # {year: row index into the rate cube}
        self._ages = None           # sorted unique ages (int)
        self._age_min = None
        self._age_max = None
        # rate_cube[sex][year_idx, age_idx] -> incidence rate; sex in {0:'f',1:'m'}
        self._rate_cube = None
        self.update_pars(pars, **kwargs)

    def init_pre(self, sim):
        super().init_pre(sim)
        # Precompute a fast lookup cube indexed by [sex, year, age].
        df = self.incidence
        self._years = np.sort(df['year'].unique()).astype(int)
        self._year_index = {int(y): i for i, y in enumerate(self._years)}
        self._ages = np.sort(df['age'].unique()).astype(int)
        self._age_min = int(self._ages.min())
        self._age_max = int(self._ages.max())
        n_year, n_age = len(self._years), len(self._ages)
        age_index = {int(a): i for i, a in enumerate(self._ages)}
        cube = {0: np.zeros((n_year, n_age)), 1: np.zeros((n_year, n_age))}
        sex_code = {'f': 0, 'm': 1}
        for r in df.itertuples(index=False):
            s = sex_code.get(str(r.sex).lower()[0])
            if s is None:
                continue
            yi = self._year_index[int(r.year)]
            ai = age_index[int(r.age)]
            cube[s][yi, ai] = float(r.incidence)
        self._rate_cube = cube

    def _lookup_rates(self, year, ages, female):
        """Per-agent annual incidence rate for the given calendar year.

        ages is a float array of agent ages; female a bool mask. Years are
        nearest-year clamped to the data range; ages are clamped to the data
        age range; integer-floored age indexes the per-single-year curve.
        """
        y = int(np.clip(int(np.floor(year)), int(self._years[0]), int(self._years[-1])))
        yi = self._year_index[y]
        ai = np.clip(np.floor(ages).astype(int), self._age_min, self._age_max) - self._age_min
        rates = np.empty(len(ages), dtype=float)
        f_curve = self._rate_cube[0][yi]
        m_curve = self._rate_cube[1][yi]
        rates[female] = f_curve[ai[female]]
        rates[~female] = m_curve[ai[~female]]
        return rates

    def infect(self):
        """Full override -- no super().infect() call, no network transmission."""
        people = self.sim.people
        eligible = (self.susceptible & ~self.infected & people.alive).uids
        if not len(eligible):
            return
        ages = np.asarray(people.age[eligible], dtype=float)
        female = np.asarray(people.female[eligible], dtype=bool)
        year = float(self.t.now('year'))
        rates = self._lookup_rates(year, ages, female)
        dt_years = float(self.t.dt_year)
        p = 1.0 - np.exp(-rates * dt_years)
        self.infect_bern.set(p=p)
        selected = self.infect_bern.filter(eligible)
        if len(selected):
            self.set_prognoses(selected)
            self.diagnosed[selected] = True
            self.ti_diagnosed[selected] = self.ti
            # +1, not self.ti: interventions.step() (loop position 8, where sti.ART
            # runs) executes BEFORE diseases.step() (position 9, this method) in the
            # SAME tick. ART's schedule-based gate is an exact-match `ti_art==self.ti`
            # check with no <= fallback, so ti_art=self.ti would never be seen -- a
            # permanent miss, not a one-tick lag. +1 mirrors how sti.HIVTest schedules
            # a future ART start from its own step(), which correctly lines up with
            # next tick's ART pass.
            self.ti_art[selected] = self.ti + 1
        return

    def step(self):
        """Full override of BaseSTI.step(): infect() already calls
        set_prognoses() directly, so there's no 3-tuple/set_outcomes() flow."""
        self.infect()
        return

    def init_post(self):
        super().init_post()  # HIV.init_post(): CD4/care_seeking init, init_prev seeding
        # Make ALL initially-seeded cases diagnosed + ART-eligible immediately
        # (HIV.init_post() only does this for its own init_diagnosed-sampled
        # subset, default 0). ti_art=0 (not +1) is correct here: init_post()
        # runs entirely before the tick loop starts, matching stisim's own
        # convention for initial cases (schedule ART start at ti=0, no delay,
        # so an ART intervention picks them up immediately).
        initial_cases = self.infected.uids
        self.diagnosed[initial_cases] = True
        self.ti_diagnosed[initial_cases] = 0
        self.ti_art[initial_cases] = 0
        return

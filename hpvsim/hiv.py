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
    cd4_upper) and HPV prevalence by HIV status -- the place to look for HIV's
    effect on HPV, regardless of which subclass drives infection. Cancer
    counts/incidence/rate-ratio by HIV status live on HPVTotal
    (sim.results.all_hpv) instead, since that module always ticks at the
    sim's own cadence.

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
            rel_reactivation_lo=1.0, rel_reactivation_hi=1.0,  # latent reactivation hazard multiplier
            cd4_threshold=200.0,  # lo/hi stratum boundary
            cd4_upper=500.0,      # CD4 >= this -> no effect at all
            dt=ss.years(0.25),
        )
        self.define_states(
            ss.FloatArr('hiv_rel_sus', default=1.0),
            ss.FloatArr('hiv_rel_sev', default=1.0),
            ss.FloatArr('hiv_rel_imm', default=1.0),
            ss.FloatArr('hiv_rel_reactivation', default=1.0),
        )
        self.hpv_modules = None

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
        # Only HIV+ agents with CD4 in [0, cd4_upper) get an effect; above that, 1.0.
        infected = self.infected[auids]
        hiv_pos = infected & ~np.isnan(cd4) & (cd4 < self.pars.cd4_upper)
        is_hi = self._cd4_stratum(np.nan_to_num(cd4, nan=1e4))
        n = len(auids)
        self.hiv_rel_sus[auids] = self._factor_array('rel_sus', hiv_pos, is_hi, n)
        self.hiv_rel_sev[auids] = self._factor_array('rel_sev', hiv_pos, is_hi, n)
        self.hiv_rel_imm[auids] = self._factor_array('rel_imm', hiv_pos, is_hi, n)
        self.hiv_rel_reactivation[auids] = self._factor_array('rel_reactivation', hiv_pos, is_hi, n)
        # Written for all agents; starsim only samples it for susceptibles.
        for m in self.hpv_modules:
            m.rel_sus[auids] = m.rel_sus[auids] * self.hiv_rel_sus[auids]

        # No testing cascade: diagnose every living HIV+ agent each step.
        undiagnosed = self.infected & ~self.diagnosed & self.sim.people.alive
        uids = undiagnosed.uids
        if len(uids):
            self.diagnosed[uids] = True
            self.ti_diagnosed[uids] = self.ti
            # sim.ti, not self.ti: sti.ART's gate checks ti_art against the sim clock.
            self.ti_art[uids] = self.sim.ti + 1
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('hpv_prevalence_with_hiv', dtype=float, scale=False, label='HPV prevalence (HIV+)'),
            ss.Result('hpv_prevalence_no_hiv', dtype=float, scale=False, label='HPV prevalence (HIV-)'),
        )

    def _rescale_stisim_results(self):
        """Recompute the inherited HIV results with per-agent scale weighting.

        ``stisim.utils.count`` is ``np.count_nonzero``, so every count-type STI
        result is a raw agent tally. Under grow-multiscale a fine agent carries
        ``scale = 1/ms_agent_ratio`` but is counted as a whole person, and the
        result is then multiplied by ``pop_scale`` -- so HIV stocks over-report
        by roughly the fraction of fine agents. At ``ms_agent_ratio=100`` that
        was ~6x in hpvsim's Zambia setup: 8.0M infections against a true 1.1M.

        Only the all-age quantities are corrected here. stisim's sex-by-age
        strata (``n_infected_f_15_20`` and friends) are built the same way and
        remain raw counts; use ``hpv.by_age`` or a scale-weighted analyzer for
        age-stratified output rather than those. Ratios of two equally-biased
        stocks (``p_on_art``) were already close to right; they are recomputed
        anyway so numerator and denominator are consistently weighted.
        """
        ti = self.ti
        res = self.results
        people = self.sim.people
        w = people.scale.values
        alive = people.alive.values
        infected = self.infected.values & alive

        def wsum(mask):
            return float((w * mask).sum())

        n_inf = wsum(infected)
        n_alive = wsum(alive)
        if 'n_infected' in res:
            res['n_infected'][ti] = n_inf
        if 'n_susceptible' in res:
            res['n_susceptible'][ti] = wsum(self.susceptible.values & alive)
        if 'n_diagnosed' in res:
            res['n_diagnosed'][ti] = wsum(infected & self.diagnosed.values)
        if 'n_on_art' in res:
            n_art = wsum(infected & self.on_art.values)
            res['n_on_art'][ti] = n_art
            if 'p_on_art' in res:
                res['p_on_art'][ti] = n_art / n_inf if n_inf else 0.0
        if 'prevalence' in res:
            res['prevalence'][ti] = n_inf / n_alive if n_alive else 0.0
        if 'prevalence_15_49' in res:
            age = people.age.values
            adult = alive & (age >= 15) & (age < 50)
            denom = wsum(adult)
            res['prevalence_15_49'][ti] = (wsum(adult & infected) / denom
                                            if denom else 0.0)

        # Per-step flows, then their cumulative partners from the fixed series.
        flows = (('new_infections', self.ti_infected),
                 ('new_diagnoses', self.ti_diagnosed),
                 ('new_deaths', self.ti_dead))
        for key, ti_arr in flows:
            if key in res and ti_arr is not None:
                res[key][ti] = wsum(np.asarray(ti_arr.values) == ti)
        for new_key, cum_key in (('new_infections', 'cum_infections'),
                                 ('new_diagnoses', 'cum_diagnoses'),
                                 ('new_deaths', 'cum_deaths')):
            if new_key in res and cum_key in res:
                res[cum_key][ti] = float(np.sum(res[new_key][:ti + 1]))
        return

    def update_results(self):
        """HPV prevalence by HIV status, plus scale-correction of the inherited
        stisim HIV results.

        Cancer counts/incidence by HIV status live on HPVTotal
        (sim.results.all_hpv) instead -- that module always ticks at the sim's
        own cadence, so a single new-cancer event can't be recorded more than
        once even if HIV's own dt differs from the sim's.
        """
        super().update_results()
        self._rescale_stisim_results()
        ti = self.ti
        people = self.sim.people
        alive = people.alive.values
        hiv_pos = self.infected.values & alive
        hiv_neg = (~self.infected.values) & alive

        any_hpv = np.zeros(alive.shape, dtype=bool)
        for m in self.hpv_modules:
            any_hpv |= m.infected.values

        # Scale-weight numerator and denominator so fine agents count fractionally.
        scale = people.scale.values
        n_pos = float((hiv_pos * scale).sum())
        n_neg = float((hiv_neg * scale).sum())
        self.results['hpv_prevalence_with_hiv'][ti] = (
            float(((any_hpv & hiv_pos) * scale).sum()) / n_pos if n_pos else 0.0)
        self.results['hpv_prevalence_no_hiv'][ti] = (
            float(((any_hpv & hiv_neg) * scale).sum()) / n_neg if n_neg else 0.0)


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
    ``age`` is an age-band lower bound, and any band width works: 5-year bands
    (0, 5, 10, ... -- the usual UNAIDS/Spectrum shape) or single years of age.
    Bands need not be evenly spaced; the first and last extend to cover ages
    below and above the data range.

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
        self._ages = None           # sorted unique age-band lower bounds (int)
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
        nearest-year clamped to the data range.

        Ages are bucketed into the age bands the data actually supplies, via
        searchsorted on ``self._ages``: an agent falls in the band whose lower
        bound is the greatest one <= its age, with the first/last bands
        extending to cover ages below/above the data range. This handles any
        band width -- 5-year bands (UNAIDS/Spectrum-style) as well as single
        years of age, which are just the width-1 case.
        """
        yi = int(np.abs(self._years - np.floor(year)).argmin())
        ai = np.clip(np.searchsorted(self._ages, np.floor(ages), side='right') - 1,
                     0, len(self._ages) - 1)
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
            # sim.ti + 1, not self.ti + 1: sti.ART's gate is an exact ti_art==ti
            # match on the sim clock, and interventions run before diseases.
            self.ti_art[selected] = self.sim.ti + 1
        return

    def step(self):
        """Full override of BaseSTI.step(): infect() already calls
        set_prognoses() directly, so there's no 3-tuple/set_outcomes() flow."""
        self.infect()
        return

    def init_post(self):
        super().init_post()  # HIV.init_post(): CD4/care_seeking init, init_prev seeding
        # All initially-seeded cases are diagnosed and ART-eligible immediately.
        # ti_art=0, not +1: init_post runs entirely before the tick loop.
        initial_cases = self.infected.uids
        self.diagnosed[initial_cases] = True
        self.ti_diagnosed[initial_cases] = 0
        self.ti_art[initial_cases] = 0
        return

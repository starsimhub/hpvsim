"""HIV–HPV co-infection.

HIV is STIsim's (``sti.HIV``), used as it is. ``HIVEffects`` (a connector)
gives HIV's effects on HPV, from each person's HIV status and CD4 count, and
HPV prevalence by HIV status. Two ways to drive HIV:
  - ``HIV_transmit()``: STIsim's HIV with hpvsim's defaults, transmitted per
    act on hpv.SexualNetwork (as on any sexual network between men and women).
  - ``HIV_incidence``: STIsim's HIV infected at the rate of a per-(year, sex,
    age) incidence curve, with no transmission.
"""

import numpy as np
import sciris as sc
import starsim as ss
import stisim as sti

from .hpv import HPV

__all__ = ['HIVEffects', 'HIV_transmit', 'HIV_incidence']


def hiv_values(pars, kwargs, **defaults):
    """hpvsim's defaults for STIsim's HIV (a quarterly step, and no results by age), updated by those given"""
    return sc.mergedicts(dict(dt=ss.years(0.25), age_bins=None), defaults, pars, kwargs)


def HIV_transmit(init_prev_data=None, pars=None, name='hiv', **kwargs):
    """Network-transmitted HIV: STIsim's HIV, transmitting at beta_m2f per act
    (and beta_m2f*rel_beta_f2m from women to men) on hpv.SexualNetwork.

    Use this when HIV prevalence should emerge from network transmission
    dynamics; use HIV_incidence instead to impose a known incidence curve
    directly.
    """
    return sti.HIV(init_prev_data=init_prev_data, name=name, **hiv_values(pars, kwargs, beta_m2f=0.0035))


class HIV_incidence(sti.HIV):
    """HIV driven by an imposed incidence curve, with no network transmission.

    STIsim's HIV, with one more transition (``acquisition``): infection at the
    rate of the curve. A testing intervention (e.g. ``sti.HIVTest``) diagnoses
    them, as for any HIV; ``hpv.Sim(model_hiv=...)`` adds one that diagnoses
    everyone with HIV each step, so a plain ``sti.ART(coverage=...)`` treats them.

    Args:
        incidence: DataFrame ``[age, sex, year, incidence]``; sex 'f'/'m',
            incidence the annual acquisition rate among susceptibles. ``age``
            is a band lower bound, of any width (5-year UNAIDS/Spectrum bands
            or single years) and not necessarily even; the first and last bands
            extend past the data range, as do the first and last years.
    """

    def __init__(self, incidence=None, pars=None, init_prev_data=None, name='hiv', **kwargs):
        if incidence is None:
            raise ValueError(
                'HIV_incidence requires an incidence DataFrame '
                '[age, sex, year, incidence]; pass incidence=.'
            )
        super().__init__(init_prev_data=init_prev_data, name=name,
                         **hiv_values(pars, kwargs, beta_m2f=0, beta_m2c=0, beta_breastfeed=0))
        self.incidence = incidence
        self.define_transitions(acquisition=ss.Transition(self.susceptible, self.acute, rate=self.incidence_rate)) # Infection at the rate of the curve, in place of transmission
        # Filled in init_pre:
        self._years = None          # sorted unique years (int)
        self._ages = None           # sorted unique age-band lower bounds (int)
        # rate_cube[sex][year_idx, age_idx] -> incidence rate; sex in {0:'f',1:'m'}
        self._rate_cube = None

    def init_pre(self, sim):
        super().init_pre(sim)
        # Precompute a fast lookup cube indexed by [sex, year, age].
        df = self.incidence
        self._years = np.sort(df['year'].unique()).astype(int)
        year_index = {int(y): i for i, y in enumerate(self._years)}
        self._ages = np.sort(df['age'].unique()).astype(int)
        n_year, n_age = len(self._years), len(self._ages)
        age_index = {int(a): i for i, a in enumerate(self._ages)}
        cube = {0: np.zeros((n_year, n_age)), 1: np.zeros((n_year, n_age))}
        sex_code = {'f': 0, 'm': 1}
        for r in df.itertuples(index=False):
            s = sex_code.get(str(r.sex).lower()[0])
            if s is None:
                continue
            cube[s][year_index[int(r.year)], age_index[int(r.age)]] = float(r.incidence)
        self._rate_cube = cube

    def _lookup_rates(self, year, ages, female):
        """Per-agent annual incidence rate for a calendar year.

        Nearest year, and the age band whose lower bound is the greatest one
        <= the agent's age, so any band width works. ``ages`` is a float array,
        ``female`` a bool mask.
        """
        yi = int(np.abs(self._years - np.floor(year)).argmin())
        ai = np.clip(np.searchsorted(self._ages, np.floor(ages), side='right') - 1,
                     0, len(self._ages) - 1)
        rates = np.empty(len(ages), dtype=float)
        rates[female] = self._rate_cube[0][yi][ai[female]]
        rates[~female] = self._rate_cube[1][yi][ai[~female]]
        return rates

    def incidence_rate(self, sim, uids):
        """The rate of the transition acquisition: the curve's annual rate for each person's age and sex"""
        people = sim.people
        ages = np.asarray(people.age[uids], dtype=float)
        female = np.asarray(people.female[uids], dtype=bool)
        return ss.peryear(self._lookup_rates(float(self.t.now('year')), ages, female))


class HIVEffects(ss.Connector):
    """HIV's effects on HPV, by CD4 count, and HPV prevalence by HIV status.

    People with HIV and a CD4 count below ``cd4_upper`` have their HPV
    acquisition (rel_sus), progression (rel_sev), conferred immunity
    (rel_imm), and reactivation of latent infection (rel_reactivation)
    multiplied by a factor, ``_lo`` below ``cd4_threshold`` and ``_hi`` at or
    above it. It reads the HIV disease's ``infected`` and ``cd4``; HPV reads
    the factors (``hiv_rel_sev`` etc.) from it. Cancer by HIV status is on
    ``HPVTotal`` (``sim.results.all_hpv``), which ticks at the sim's cadence.
    ``hpv.Sim`` adds one whenever the sim has HIV.
    """

    def __init__(self, pars=None, name='hiv_effects', **kwargs):
        super().__init__(name=name)
        self.define_pars(
            rel_sus_lo=2.2,   rel_sus_hi=2.2,    # increased HPV acquisition
            rel_sev_lo=1.5,   rel_sev_hi=1.2,    # faster/worse CIN->cancer progression
            rel_imm_lo=0.36,  rel_imm_hi=0.76,   # reduced post-infection/vaccine immunity
            rel_reactivation_lo=1.0, rel_reactivation_hi=1.0,  # latent reactivation hazard multiplier
            cd4_threshold=200.0,  # lo/hi stratum boundary
            cd4_upper=500.0,      # CD4 >= this -> no effect at all
        )
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.FloatArr('hiv_rel_sus', default=1.0),
            ss.FloatArr('hiv_rel_sev', default=1.0),
            ss.FloatArr('hiv_rel_imm', default=1.0),
            ss.FloatArr('hiv_rel_reactivation', default=1.0),
        )
        self.hpv_modules = None
        self.hiv = None

    def init_pre(self, sim):
        self.hpv_modules = [m for m in sim.diseases.values() if isinstance(m, HPV)]
        if not self.hpv_modules:
            raise ValueError('hpv.HIVEffects requires HPV genotype module(s) in the sim.')
        self.hiv = next((d for d in sim.diseases.values() if isinstance(d, sti.HIV)), None)
        if self.hiv is None:
            raise ValueError('hpv.HIVEffects requires an HIV disease (sti.HIV) in the sim.')
        super().init_pre(sim)

    def factor(self, effect, hiv_pos, is_hi):
        """A per-agent factor (1.0 for HIV-, the stratum value for HIV+)"""
        lo = self.pars[f'{effect}_lo']
        hi = self.pars[f'{effect}_hi']
        return np.where(hiv_pos, np.where(is_hi, hi, lo), 1.0)

    def step(self):
        auids = self.sim.people.auids
        cd4 = np.asarray(self.hiv.cd4[auids])
        # Only HIV+ agents with CD4 in [0, cd4_upper) get an effect; above that, 1.0.
        hiv_pos = np.asarray(self.hiv.infected[auids], dtype=bool) & ~np.isnan(cd4) & (cd4 < self.pars.cd4_upper)
        is_hi = np.nan_to_num(cd4, nan=1e4) >= self.pars.cd4_threshold
        self.hiv_rel_sus[auids] = self.factor('rel_sus', hiv_pos, is_hi)
        self.hiv_rel_sev[auids] = self.factor('rel_sev', hiv_pos, is_hi)
        self.hiv_rel_imm[auids] = self.factor('rel_imm', hiv_pos, is_hi)
        self.hiv_rel_reactivation[auids] = self.factor('rel_reactivation', hiv_pos, is_hi)
        # Written for all agents; starsim only samples it for susceptibles.
        for m in self.hpv_modules:
            m.rel_sus[auids] = m.rel_sus[auids] * self.hiv_rel_sus[auids]
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('hpv_prevalence_with_hiv', dtype=float, scale=False, label='HPV prevalence (HIV+)'),
            ss.Result('hpv_prevalence_no_hiv', dtype=float, scale=False, label='HPV prevalence (HIV-)'),
        )

    def update_results(self):
        """HPV prevalence by HIV status, each agent weighted by its scale, so fine agents count fractionally"""
        super().update_results()
        ti = self.ti
        people = self.sim.people
        alive = people.alive.values
        scale = people.scale.values
        infected = np.asarray(self.hiv.infected.values, dtype=bool)
        any_hpv = np.zeros(alive.shape, dtype=bool)
        for m in self.hpv_modules:
            any_hpv |= m.infected.values
        for key, group in (('hpv_prevalence_with_hiv', infected & alive), ('hpv_prevalence_no_hiv', ~infected & alive)):
            n = float((group * scale).sum())
            self.results[key][ti] = float(((any_hpv & group) * scale).sum()) / n if n else 0.0

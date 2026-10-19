"""
Across genotypes: immunity from one genotype protecting against others, and results over all genotypes

  - ``CrossImmunity``: the cross-protection matrices, and the modifiers that apply them. Each
    genotype holds the immunity its clearances gave (``nab``, ``cell``, as attributes); protection
    against target genotype ``i`` is ``sum_j m[i, j] * immunity_j``, clipped to [0, 1]. Humoral
    immunity reduces susceptibility; cell-mediated immunity (severity immunity) shortens the
    precancerous infection of women and lowers reactivation from latency.

  - ``HPVTotal`` (``ss.Analyzer``): results over all genotypes, at ``sim.results.all_hpv``:
    people infected with any genotype, sums of the genotypes' flows, and cancer incidence and
    mortality, crude and age-standardized (WHO 2000).
"""

import numpy as np
import sciris as sc
import starsim as ss

from .hpv import HPV
from .parameters import GENOTYPE_KEYS


__all__ = ['CrossImmunity', 'HPVTotal']

# Genotypes whose own immunity is 1.0; the others use own_imm_hr
_FULL_OWN_IMM_KEYS = frozenset({'hpv16', 'hpv18'})

# Pairs with 'high' cross-protection (same clade); everything else is 'med'
_CLADE_HIGH_PAIRS = frozenset({('hpv16', 'hpv18'), ('hpv18', 'hpv16')})

female = ss.people.female


def _build_cross_matrix(keys, scalar_med, scalar_high, own_imm_hr):
    """ An (n, n) cross-protection matrix: rows are the target genotype, columns the source """
    n = len(keys)
    m = np.full((n, n), scalar_med, dtype=float)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                m[i, j] = 1.0 if ki in _FULL_OWN_IMM_KEYS else own_imm_hr
            elif (ki, kj) in _CLADE_HIGH_PAIRS:
                m[i, j] = scalar_high
    return m


class CrossImmunity(sc.prettyobj):
    """
    Immunity across genotypes, applied by modifiers

    Convention: row = target genotype, column = source genotype. With ``cross=False``, only
    each genotype's own immunity applies (the diagonal), which a compartmental run can hold.

    Args:
        cross_imm_sus (array): the susceptibility matrix; by default built from the scalars below
        cross_imm_sev (array): the severity matrix; likewise
        pars (dict): the scalars: ``cross_imm_sus_med``, ``cross_imm_sus_high``, ``cross_imm_sev_med``, ``cross_imm_sev_high``, ``own_imm_hr``
        cross (bool): whether immunity crosses genotypes (default True)
        kwargs (dict): the scalars by name

    Examples:
        ```python
        imm = hpv.CrossImmunity(cross_imm_sus_med=0.2)
        sim = hpv.Sim(genotypes=[16, 18], connectors=imm)
        ```
    """
    defaults = dict(
        cross_imm_sus_med  = 0.3,
        cross_imm_sus_high = 0.5,
        cross_imm_sev_med  = 0.5,
        cross_imm_sev_high = 0.7,
        own_imm_hr         = 0.9,
    )

    def __init__(self, cross_imm_sus=None, cross_imm_sev=None, pars=None, cross=True, **kwargs):
        pars = sc.mergedicts(pars, kwargs)
        unknown = set(pars) - set(self.defaults)
        if unknown:
            raise ValueError(f'CrossImmunity: unknown parameter(s) {sorted(unknown)}; parameters are {sorted(self.defaults)}')
        self.pars = sc.objdict(sc.mergedicts(self.defaults, pars))
        self.cross_imm_sus = cross_imm_sus
        self.cross_imm_sev = cross_imm_sev
        self.cross = cross
        return

    def matrices(self, keys=None):
        """ The (susceptibility, severity) matrices for these genotypes """
        keys = GENOTYPE_KEYS if keys is None else list(keys)
        p = self.pars
        n = len(keys)
        out = []
        for given, med, high in [(self.cross_imm_sus, p.cross_imm_sus_med, p.cross_imm_sus_high), (self.cross_imm_sev, p.cross_imm_sev_med, p.cross_imm_sev_high)]:
            m = _build_cross_matrix(keys, med, high, p.own_imm_hr) if given is None else np.array(given, dtype=float)
            if m.shape != (n, n):
                raise ValueError(f'CrossImmunity: a matrix of shape {m.shape} for {n} genotypes')
            diag = np.diag(m)
            if (diag < 0).any() or (diag > 1).any():
                raise ValueError(f'CrossImmunity: diagonal entries must be in [0, 1]; got {diag}')
            if not self.cross:
                m = np.diag(diag)
            out.append(m)
        return out

    def modifiers(self, hpvs):
        """ The modifiers that apply immunity to these genotype diseases """
        hpvs = list(hpvs)
        m_sus, m_sev = self.matrices([h.genotype for h in hpvs])
        out = []
        for i, h in enumerate(hpvs):
            sus = sum(m_sus[i, j] * hj.nab for j, hj in enumerate(hpvs) if m_sus[i, j])
            sev = sum(m_sev[i, j] * hj.cell for j, hj in enumerate(hpvs) if m_sev[i, j])
            sus = np.clip(sus, 0, 1)
            sev = np.clip(sev, 0, 0.999) # A severity immunity of 1 clears at once, as HPVsim 3's zero duration does
            out += [
                ss.modify(h.rel_sus, multiply=1 - sus, name=f'nab_{h.name}'),
                ss.modify(h.precin_exit, multiply=1 / (1 - sev), where=female, name=f'sev_{h.name}'), # A rate: HPVsim 3 multiplies dur_precin by (1 - sev_imm)
                ss.modify(h.reactivation, multiply=1 - sev, name=f'sev_react_{h.name}'),
            ]
        return out


class HPVTotal(ss.Analyzer):
    """
    Results over all genotypes, at ``sim.results.all_hpv``

    - People in each state of any genotype (``n_infected``, ``n_precin``, ``n_cin``,
      ``n_cancerous``, ``n_latent``), ``n_susceptible`` (infected with none) and ``prevalence``
    - Sums of the genotypes' flows (``new_infections``, ``new_cancers``, ``new_cancer_deaths``, ...);
      cancer counts are exact, since each woman's cancer is of one genotype
    - ``cancer_incidence`` (crude, per 100,000 women per year), ``asr_cancer_incidence`` and
      ``asr_cancer_mortality`` (WHO 2000 standard, per 100,000 person-years), counted when
      cancers are detected; with HIV, incidence among women with and without HIV
    """

    _UNION_STATES = dict(n_infected='infected', n_precin='precin', n_cin='cin', n_cancerous='cancerous', n_latent='latent')
    _SUMS = ('new_infections', 'cum_infections', 'new_reactivations', 'new_cancers', 'new_undetected_cancers', 'new_cancer_deaths',
             'cum_cancers', 'cum_cancer_deaths', 'sum_age_at_cancer')

    # WHO 2000 World Standard Population weights per 5-year band (0-4 to 100+)
    WHO2000_5YR_EDGES = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 150], dtype=float)
    WHO2000_5YR_WEIGHTS = np.array([8860, 8690, 8600, 8470, 8220, 7930, 7610, 7150, 6590, 6040, 5370, 4550, 3720, 2960, 2210, 1520, 910, 440, 150, 40, 5], dtype=float)

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'all_hpv')
        super().__init__(**kwargs)
        return

    @classmethod
    def who2000_weights_for_edges(cls, edges):
        """ WHO 2000 5-year weights summed into the bins given by edges """
        edges = np.asarray(edges, dtype=float)
        mid = (cls.WHO2000_5YR_EDGES[:-1] + cls.WHO2000_5YR_EDGES[1:]) / 2.0
        return np.array([cls.WHO2000_5YR_WEIGHTS[(mid >= lo) & (mid < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])

    @classmethod
    def compute_asr(cls, cancers_by_age, n_female_by_age, edges):
        """ Age-standardized rate per 100,000 (WHO 2000) """
        weights = cls.who2000_weights_for_edges(edges)
        with np.errstate(divide='ignore', invalid='ignore'):
            rates = np.where(n_female_by_age > 0, cancers_by_age / n_female_by_age * 1e5, 0.0)
        return np.sum(rates * weights) / weights.sum()

    def init_pre(self, sim):
        self.hpv_modules = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        from .hiv import HIV
        self.hiv_module = next((d for d in sim.diseases.values() if isinstance(d, HIV)), None)
        super().init_pre(sim)
        return

    def init_results(self):
        super().init_results()
        res = []
        for key in self._UNION_STATES:
            res.append(ss.Result(key, dtype=float, label=f'{key} (any genotype)'))
        res += [
            ss.Result('n_susceptible', dtype=float, label='Infected with no genotype'),
            ss.Result('prevalence', dtype=float, scale=False, label='Prevalence of any HPV genotype'),
            ss.Result('cancer_incidence', dtype=float, scale=False, summarize_by='mean', label='Crude cervical cancer incidence (per 100,000 women per year)'),
            ss.Result('asr_cancer_incidence', dtype=float, scale=False, summarize_by='mean', label='Age-standardized cervical cancer incidence (WHO 2000, per 100,000)'),
            ss.Result('asr_cancer_mortality', dtype=float, scale=False, summarize_by='mean', label='Age-standardized cervical cancer mortality (WHO 2000, per 100,000)'),
        ]
        if self.hiv_module is not None:
            res += [
                ss.Result('new_cancers_with_hiv', dtype=float, label='New cancers (HIV+)'),
                ss.Result('new_cancers_no_hiv', dtype=float, label='New cancers (HIV-)'),
                ss.Result('cancer_incidence_with_hiv', dtype=float, scale=False, summarize_by='mean', label='Cervical cancer incidence (per 100,000 women with HIV per year)'),
                ss.Result('cancer_incidence_no_hiv', dtype=float, scale=False, summarize_by='mean', label='Cervical cancer incidence (per 100,000 women without HIV per year)'),
            ]
        self.define_results(*res)
        return

    def step(self):
        ti = self.ti
        hpvs = self.hpv_modules
        ppl = self.sim.people
        alive = ppl.alive.values
        n_alive = alive.sum()
        if not hpvs or not n_alive:
            return
        for key, attr in self._UNION_STATES.items():
            union = np.zeros(len(alive), dtype=bool)
            for h in hpvs:
                union |= getattr(h, attr).values
            self.results[key][ti] = np.count_nonzero(union & alive)
        n_inf = self.results.n_infected[ti]
        self.results.n_susceptible[ti] = n_alive - n_inf
        self.results.prevalence[ti] = n_inf / n_alive

        # Cancers detected and cancer deaths this step, by age, among women
        age = ppl.age.values
        fem = ppl.female.values
        edges = self.WHO2000_5YR_EDGES
        dt = self.t.dt_year
        women = np.histogram(age[fem], bins=edges)[0]
        detected = np.zeros(len(alive), dtype=bool)
        for h in hpvs:
            if h.detection:
                detected |= (h.ti_cancer_detection.values == ti) & h.X.values
        cancers = np.histogram(age[detected], bins=edges)[0]
        died = ss.uids.concatenate([h.graph.transitions.cancer_death.uids for h in hpvs])
        deaths = np.histogram(ppl.age.raw[died], bins=edges)[0]
        self.results.asr_cancer_incidence[ti] = self.compute_asr(cancers / dt, women, edges)
        self.results.asr_cancer_mortality[ti] = self.compute_asr(deaths / dt, women, edges)
        self.results.cancer_incidence[ti] = cancers.sum() / dt / women.sum() * 1e5 if women.sum() else 0.0

        if self.hiv_module is not None:
            hiv = self.hiv_module.infected.values
            for key, mask in [('with_hiv', fem & hiv), ('no_hiv', fem & ~hiv)]:
                n_cancer = np.count_nonzero(detected & mask)
                n_women = np.count_nonzero(mask & alive)
                self.results[f'new_cancers_{key}'][ti] = n_cancer
                self.results[f'cancer_incidence_{key}'][ti] = n_cancer / dt / n_women * 1e5 if n_women else 0.0
        return

    def finalize_results(self):
        """ Sums of the genotypes' results """
        super().finalize_results()
        for key in self._SUMS:
            if all(key in h.results for h in self.hpv_modules):
                vals = sum(np.asarray(h.results[key].values, dtype=float) for h in self.hpv_modules)
                self.results += ss.Result(key, values=vals, scale=False, label=f'{self.hpv_modules[0].results[key].label} (all genotypes)',
                                          timevec=self.results.prevalence.timevec, module=self.name)
        return

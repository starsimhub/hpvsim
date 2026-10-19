"""
HIV as a coinfection of HPV: a small declared HIV, with CD4 as states, and its effects on HPV

HPVsim 3 used STIsim's HIV. This stand-in keeps HPVsim's HIV parameters and effects, and does not
need STIsim: CD4 is in bands, as states, so every effect on HPV is a modifier with a condition.

    S  HIV-negative
    A  CD4 >= 500 (no effect on HPV)
    B  CD4 200-500 (the "hi" effects)
    D  CD4 < 200 (the "lo" effects)
    T  on ART (no effect on HPV: CD4 restored)

Effects on each genotype, for people in B or D: susceptibility (``rel_sus_hi``/``_lo``),
the time axis of the severity curves (``rel_sev_hi``/``_lo``), the immunity drawn at clearance
(``rel_imm_hi``/``_lo``), and reactivation from latency (``rel_reactivation_hi``/``_lo``).
"""

import numpy as np
import sciris as sc
import starsim as ss


__all__ = ['HIV', 'HIV_transmit', 'HIV_incidence', 'ART']


def _yrs(mean, std):
    return ss.lognorm_ex(mean=ss.years(mean), std=ss.years(std))


class HIV(ss.Disease):
    """
    HIV with CD4 in bands, as states, for HIV-HPV coinfection

    Args:
        pars (dict): parameter values
        name (str): the name (default 'hiv')
        kwargs (dict): parameter values by name

    Parameters:
        beta_m2f (float): per-act probability of transmission, male to female (default 0.0035)
        rel_beta_f2m (float): female-to-male relative to male-to-female (default 0.5)
        init_prev (float): initial prevalence (default 0.05)
        dur_cd4_500 (dur): time with CD4 >= 500 (default lognormal, mean 4 y)
        dur_cd4_200 (dur): time with CD4 200-500 (default lognormal, mean 4 y)
        dur_aids (dur): time from CD4 < 200 to death, without ART (default lognormal, mean 2 y)
        art_rel_trans (float): infectiousness on ART (default 0.04)
        rel_sus_lo, rel_sus_hi, rel_sev_lo, rel_sev_hi, rel_imm_lo, rel_imm_hi, rel_reactivation_lo, rel_reactivation_hi:
            HPVsim's effects of HIV on HPV, below 200 (lo) and 200-500 (hi)

    Examples:
        ```python
        hiv = hpv.HIV(beta_m2f=0.005)
        sim = hpv.Sim(genotypes=[16, 18], diseases=hiv, interventions=hpv.ART(hiv, coverage={2000: 0, 2020: 0.8}))
        ```
    """
    defaults = dict(
        beta_m2f = 0.0035,
        rel_beta_f2m = 0.5,
        init_prev = 0.05,
        dur_cd4_500 = _yrs(4, 2),
        dur_cd4_200 = _yrs(4, 2),
        dur_aids = _yrs(2, 1),
        rel_sus_lo = 2.2, rel_sus_hi = 2.2,
        rel_sev_lo = 1.5, rel_sev_hi = 1.2,
        rel_imm_lo = 0.36, rel_imm_hi = 0.76,
        rel_reactivation_lo = 1.0, rel_reactivation_hi = 1.0,
    )

    def __init__(self, pars=None, name='hiv', art_rel_trans=0.04, incidence=None, **kwargs):
        given = sc.mergedicts(pars, {k: kwargs.pop(k) for k in list(kwargs) if k in self.defaults})
        unknown = set(given) - set(self.defaults)
        if unknown:
            raise ValueError(f'HIV: unknown parameter(s) {sorted(unknown)}; parameters are {sorted(self.defaults)}')
        v = sc.mergedicts(self.defaults, given)
        p = sc.objdict({k: ss.par(val, name=k) for k, val in v.items() if k not in ('beta_m2f', 'rel_beta_f2m')})
        S, A, B, D = ss.states('S A B D')
        T = ss.State('T', rel_trans=art_rel_trans)
        aids_death = D >> ss.null @ p.dur_aids.value
        if incidence is None: # By network, from beta_m2f and rel_beta_f2m: see init_pre()
            infection = S >> A @ ss.par(0.0, name='beta') * (A + B + D + T)
        else: # Imposed: an annual rate by year, age, and sex
            infection = S >> A @ ss.par(_IncidenceTable(incidence), name='incidence')
        members = dict(
            infection  = infection,
            decline    = A >> B @ p.dur_cd4_500.value,
            falling    = B >> D @ p.dur_cd4_200.value,
            aids_death = aids_death,
            start_art  = (A | B | D) >> T, # Moved only by ART (hpv.ART)
            hpv_rel_sus = ss.attr({B: p.rel_sus_hi, D: p.rel_sus_lo}),
            hpv_rel_sev = ss.attr({B: p.rel_sev_hi, D: p.rel_sev_lo}),
            hpv_rel_imm = ss.attr({B: p.rel_imm_hi, D: p.rel_imm_lo}),
            hpv_rel_reactivation = ss.attr({B: p.rel_reactivation_hi, D: p.rel_reactivation_lo}),
            n_infected = A + B + D + T,
            prevalence = (A + B + D + T) / (S + A + B + D + T),
            new_deaths = ss.outcome(aids_death),
        )
        super().__init__(S, A, B, D, T, name=name, init_prev=p.init_prev, **members, **kwargs)
        self.define_pars(beta_m2f=v['beta_m2f'], rel_beta_f2m=v['rel_beta_f2m'])
        return

    def init_pre(self, sim):
        """ beta per act on each sexual network: [female to male, male to female] (hpv.SexualNetwork puts women first) """
        if 'beta' in self.pars and not isinstance(self.pars.beta.value, dict):
            m2f, rel = self.pars.beta_m2f, self.pars.rel_beta_f2m
            self.pars.beta.value = {k: ([m2f * rel, m2f] if isinstance(n, ss.SexualNetwork) else [0, 0]) for k, n in sim.networks.items()}
        super().init_pre(sim)
        return

    @property
    def infected(self):
        return self.A | self.B | self.D | self.T

    @property
    def on_art(self):
        return self.T

    def effects(self, hpvs):
        """ The modifiers of HPV by HIV, for these genotype diseases """
        low = self.B | self.D
        out = []
        for h in hpvs:
            out += [
                ss.modify(h.rel_sus, multiply=self.hpv_rel_sus, where=low, name=f'hiv_sus_{h.name}'),
                ss.modify(h.rel_sev, multiply=self.hpv_rel_sev, where=low, name=f'hiv_sev_{h.name}'),
                ss.modify(h.rel_imm, multiply=self.hpv_rel_imm, where=low, name=f'hiv_imm_{h.name}'),
                ss.modify(h.reactivation, multiply=self.hpv_rel_reactivation, where=low, name=f'hiv_react_{h.name}'),
            ]
        return out


HIV_transmit = HIV # HPVsim 3's name for network-transmitted HIV


class HIV_incidence(HIV):
    """
    HIV acquired at an imposed rate, with no transmission

    Args:
        incidence (DataFrame): columns [age, sex, year, incidence]: the annual rate among the susceptible, by age band
            (lower bounds, any width), sex ('f'/'m'), and year (nearest)
        kwargs (dict): as for `hpv.HIV`
    """
    def __init__(self, incidence=None, **kwargs):
        if incidence is None:
            raise ValueError('HIV_incidence requires an incidence DataFrame [age, sex, year, incidence]; pass incidence=.')
        super().__init__(incidence=incidence, **kwargs)
        return


class _IncidenceTable:
    """ The annual HIV incidence of each agent, from a table by year, age band, and sex: a rate as a function of (sim, uids); agents only """
    def __init__(self, df):
        self.years = np.sort(df['year'].unique()).astype(float)
        self.ages = np.sort(df['age'].unique()).astype(float)
        yi = {y: i for i, y in enumerate(self.years)}
        ai = {a: i for i, a in enumerate(self.ages)}
        self.cube = np.zeros((2, len(self.years), len(self.ages)))
        for r in df.itertuples(index=False):
            s = {'f': 0, 'm': 1}.get(str(r.sex).lower()[0])
            if s is not None:
                self.cube[s, yi[float(r.year)], ai[float(r.age)]] = float(r.incidence)
        self.__name__ = 'hiv_incidence'
        return

    def __call__(self, sim, uids):
        year = float(sim.t.now('year'))
        yi = int(np.abs(self.years - np.floor(year)).argmin())
        ages = sim.people.age[uids]
        ai = np.clip(np.searchsorted(self.ages, np.floor(ages), side='right') - 1, 0, len(self.ages) - 1)
        sex = np.where(sim.people.female[uids], 0, 1)
        return ss.peryear(self.cube[sex, yi, ai])


def ART(hiv, coverage, name='art', **kwargs):
    """
    ART as a stock target: the share of people with HIV on ART follows ``coverage``

    Args:
        hiv (HIV): the HIV disease
        coverage (float/dict/ss.timeseries): the share on ART; a dict is {year: share}, linear between
        kwargs (dict): passed to `ss.treat`

    Examples:
        ```python
        art = hpv.ART(hiv, coverage={2000: 0, 2010: 0.5, 2020: 0.85})
        ```
    """
    if isinstance(coverage, dict):
        coverage = ss.timeseries(coverage, mode='linear')
    return ss.treat(ss.Treatment(moves=hiv.start_art, name=f'{name}_tx'), target=ss.share(coverage, among=hiv.A | hiv.B | hiv.D | hiv.T),
                    priority=[hiv.D, hiv.B, hiv.A], name=name, **kwargs)

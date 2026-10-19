"""
HPV natural history: one declared Starsim disease per genotype

Scope note: HPVsim only models cervical cancer outcomes. HPV is also associated with anal,
oropharyngeal, penile, vaginal, and vulvar cancers; those are out of scope here.

States (per genotype):

    S  susceptible
    P  precin: infection without a lesion (infectious)
    C  CIN: cervical intraepithelial neoplasia (infectious; women only)
    X  invasive cancer (not infectious)
    L  latent infection (not infectious)
    Z  another genotype's cancer: this genotype's course stopped

Transitions:

    infection    S -> P            per sex act, over the sexual network
    precin_exit  P -> C | S | L    after dur_precin (women) or dur_inf_male (men); P(CIN) = logf2 of the time in P
    cin_exit     C -> X | S | L    after dur_cin; P(cancer) from the integral of logf2 over the time in C
    reactivation L -> P            a hazard
    cancer_death X -> (dead)       dur_cancer after onset
    cancel       S|P|C|L -> Z      when any other genotype reaches cancer
    treat_*      P, C, L -> S      undriven: moved only by treatment products

Immunity: clearance (to S, naturally or by treatment) draws humoral (`nab`) and cell-mediated
(`cell`) immunity, kept as a running max. `hpv.CrossImmunity` turns them into protection
across genotypes. `rel_sev` is a trait of each woman that scales the time axis of the
severity curves; `rel_imm` scales the immunity drawn at clearance (1, unless HIV modifies it).
"""

import numpy as np
import sciris as sc
import starsim as ss

from .parameters import genotype_aliases, get_genotype_pars, GENOTYPE_KEYS
from .utils import logf2, intlogf2


__all__ = ['HPV', 'GENOTYPE_KEYS']

_KNOWN_GENOTYPES = GENOTYPE_KEYS

# Values by sex, as people's levels: [women, men]
female = ss.by(sex=[1.0, 0.0])

# Parameters given as dicts in HPVsim 3, kept flat here so each is a number: {dict par: {key: flat par}}
_FN_PARS = dict(
    cin_fn    = dict(k='cin_fn_k', x_infl='cin_fn_x_infl', ttc='cin_fn_ttc'),
    cancer_fn = dict(k='cancer_fn_k', x_infl='cancer_fn_x_infl', ttc='cancer_fn_ttc', transform_prob='transform_prob'),
)
_FN_IGNORED = {'form', 'method'} # Only logf2 and the CIN integral are ported


def _normalize_genotype(key):
    """Resolve aliases (16 -> 'hpv16', 'hi5' -> 'hi5') to canonical keys."""
    s = str(key).lower().strip()
    for canonical, aliases in genotype_aliases.items():
        if s == canonical or s in aliases:
            return canonical
    raise ValueError(f'Unknown genotype {key!r}; valid: {list(genotype_aliases)}')


def flatten_pars(pars):
    """ HPVsim 3 parameters in the flat form of the declared model: cin_fn/cancer_fn dicts become one parameter per key """
    out = {}
    for key, val in sc.mergedicts(pars).items():
        if key in _FN_PARS:
            for k, v in dict(val).items():
                if k in _FN_IGNORED:
                    if k == 'form' and v not in (None, 'logf2'):
                        raise NotImplementedError(f'{key}: only form="logf2" is ported, not {v!r}')
                    continue
                if k not in _FN_PARS[key]:
                    raise NotImplementedError(f'{key}: "{k}" is not ported (keys: {list(_FN_PARS[key])})')
                out[_FN_PARS[key][k]] = v
        else:
            out[key] = val
    return out


def _age_risk_ramp(ages, age_risk):
    """
    Multiplier on dur_cin by age: 1 below age_risk['age'], age_risk['risk'] from
    age_risk['age_end'], and linear in between
    """
    frac = np.clip((np.asarray(ages) - age_risk['age']) / (age_risk['age_end'] - age_risk['age']), 0.0, 1.0)
    return 1.0 + (age_risk['risk'] - 1.0) * frac


class HPV(ss.Disease):
    """
    One HPV genotype, as a declared Starsim disease

    Parameters keep HPVsim's names and per-genotype defaults (`hpv.get_genotype_pars()`);
    `cin_fn` and `cancer_fn` may be given as dicts, as in HPVsim 3, and are stored flat
    (`cin_fn_k`, `cancer_fn_k`, `transform_prob`, ...).

    Args:
        genotype (str): 'hpv16', 'hpv18', 'hi5', or 'ohr' (or an alias, e.g. 16)
        genotypes (list): all the genotypes in the sim; with more than one, reaching cancer in any other cancels this one's course (`cancel`)
        pars (dict): parameter values
        detection (bool): count cancers when detected (`new_cancers`), `dur_undetected` after onset (agents only)
        kwargs (dict): parameter values by name, or `name`

    Results (besides each state's count and each transition's flow): `n_susceptible`,
    `n_precin`, `n_cin`, `n_cancerous`, `n_latent`, `n_infected`, `new_reactivations`,
    `new_cancer_deaths`, `cum_cancer_deaths`, and as agents `new_undetected_cancers`
    (onsets), `new_cancers` (detections), `cum_cancers`, `sum_age_at_cancer`.

    Examples:
        ```python
        hpv16 = hpv.HPV('hpv16', genotypes=['hpv16', 'hpv18'])
        hpv18 = hpv.HPV(18, genotypes=['hpv16', 'hpv18'], cin_fn=dict(k=0.3))
        ```
    """

    def __init__(self, genotype='hpv16', genotypes=None, pars=None, detection=True, **kwargs):
        genotype = _normalize_genotype(genotype)
        name = kwargs.pop('name', genotype)
        label = kwargs.pop('label', None)
        others = [] if genotypes is None else [_normalize_genotype(g) for g in genotypes if _normalize_genotype(g) != genotype]
        defaults = flatten_pars(get_genotype_pars(genotype))
        names = set(defaults) | set(_FN_PARS)
        given = flatten_pars(sc.mergedicts(pars, {k: kwargs.pop(k) for k in list(kwargs) if k in names}))
        unknown = set(given) - set(defaults)
        if unknown:
            raise ValueError(f'HPV {genotype}: unknown parameter(s) {sorted(unknown)}; parameters are {sorted(defaults)}')
        if int(given.get('ms_agent_ratio', 1)) != 1:
            raise NotImplementedError('ms_agent_ratio > 1 (multiscale cancer agents) is not ported to Starsim 4: results are in people, with no per-agent scale')
        v = sc.mergedicts(defaults, given)
        _validate_duration_units(genotype, v)

        p = sc.objdict({key: ss.par(val, name=key) for key, val in v.items() if key not in _PLAIN_PARS})
        S, P, C, X, L, Z = ss.states('S P C X L Z')
        yr = ss.years(1)

        # Severity: the time axis of the curves is scaled by rel_sev (a trait, which HIV can modify)
        rel_sev = ss.attr(p.rel_sev_dist)
        sev = np.abs(rel_sev)
        p_cin = female * logf2(sev * P.time / yr, p.cin_fn_k, p.cin_fn_x_infl, ttc=p.cin_fn_ttc)
        dysp = intlogf2(sev * C.time / yr, p.cancer_fn_k, p.cancer_fn_x_infl, ttc=p.cancer_fn_ttc)
        p_cancer = 1 - (1 - p.transform_prob)**(dysp**2)
        ctrl = p.hpv_control_prob * female # Women only enter latency

        dur = lambda d: d.value if isinstance(d, ss.par) else d
        precin_exit = ss.Transition(P, [C, S, L], p=[p_cin, (1 - p_cin) * (1 - ctrl), None],
                                    dur=ss.by(sex=[dur(p.dur_precin), dur(p.dur_inf_male)]))
        cin_exit = ss.Transition(C, [X, S, L], p=[p_cancer, (1 - p_cancer) * (1 - ctrl), None], dur=dur(p.dur_cin))
        treat_precin = P >> S # Moved only by treatment products
        treat_cin = C >> S
        treat_latent = L >> S
        cleared = precin_exit.to(S) | cin_exit.to(S) | treat_precin.to(S) | treat_cin.to(S) | treat_latent.to(S)
        reactivation = L >> P @ p.hpv_reactivation
        cancer_death = X >> ss.null @ dur(p.dur_cancer)

        # Immunity at clearance, women only: humoral gated by seroconversion, both kept as a running max
        rel_imm = ss.attr(1.0)
        nab = ss.attr(p.imm_init * ss.bernoulli(p=float(v['sero_prob'])) * female * rel_imm, on=cleared, combine='max', init=0)
        cell = ss.attr(p.cell_imm_init * female * rel_imm, on=cleared, combine='max', init=0)

        members = dict(
            infection    = S >> P @ p.beta * (P + C),
            precin_exit  = precin_exit,
            cin_exit     = cin_exit,
            reactivation = reactivation,
            cancer_death = cancer_death,
            treat_precin = treat_precin,
            treat_cin    = treat_cin,
            treat_latent = treat_latent,
            rel_sev = rel_sev,
            rel_imm = rel_imm,
            nab     = nab,
            cell    = cell,
            # Results with HPVsim's names
            n_susceptible     = S + 0,
            n_precin          = P + 0,
            n_cin             = C + 0,
            n_cancerous       = X + 0,
            n_latent          = L + 0,
            n_infected        = P + C,
            new_reactivations = ss.outcome(reactivation),
            new_cancer_deaths = ss.outcome(cancer_death),
        )
        if others: # Cancer in another genotype stops this one's course
            other_cancer = ss.any(ss.state(f'{o}.X') for o in others)
            members['cancel'] = (S | P | C | L) >> Z @ ss.on(other_cancer)
        if detection: # TODO(v4-rs): ss.outcome(cin_exit.to(X), delay=dur_undetected) once outcomes take a branch
            members['onset'] = ss.func(self.cancer_onset, on=cin_exit.to(X))

        super().__init__(S, P, C, X, L, Z, name=name, label=label, init_prev=p.init_prev, **members, **kwargs)
        self.genotype = genotype
        self.genotypes = [genotype] + others
        self.detection = detection
        self.define_pars(**{k: v[k] for k in _PLAIN_PARS})
        if detection:
            self.define_states(ss.FloatArr('ti_cancer_detection', label='Time of cancer detection'))
        return

    def set_pars(self, pars=None, **kwargs):
        """ Set parameter values by name, accepting HPVsim 3's dict forms (`cin_fn=dict(k=0.3)`) """
        pars = flatten_pars(sc.mergedicts(pars, kwargs))
        plain = {k: pars.pop(k) for k in list(pars) if k in _PLAIN_PARS and k in self.pars}
        if 'sero_prob' in pars:
            raise NotImplementedError('sero_prob is set when the genotype is made: give it to hpv.HPV() (or genotype_pars= of hpv.Sim)')
        super().set_pars(pars)
        for k, val in plain.items():
            self.pars[k] = val
        return

    def init_pre(self, sim):
        """ Expand beta into the network's betas: per act, [female to male, male to female] on hpv.SexualNetwork """
        beta = self.pars.beta
        if sc.isnumber(beta.value):
            base = beta.value * self.value_plain('rel_beta')
            f2m, m2f = _clip_beta(base * self.value_plain('transf2m'), base * self.value_plain('transm2f'), self.genotype)
            nets = {k: n for k, n in sim.networks.items() if isinstance(n, ss.SexualNetwork)}
            if not nets:
                raise ValueError(f'HPV {self.genotype}: beta is per sex act, and the sim has no sexual network; give beta as a rate (e.g. ss.peryear(0.5)) for homogeneous mixing')
            beta.value = {k: ([f2m, m2f] if k in nets else [0, 0]) for k in sim.networks.keys()}
        super().init_pre(sim)
        return

    def value_plain(self, key):
        val = self.pars[key]
        return val.value if isinstance(val, ss.par) else val

    def init_results(self):
        super().init_results()
        if self.detection:
            self.define_results(
                ss.Result('new_undetected_cancers', dtype=float, scale=True, label='New cancer onsets (pre-detection)'),
                ss.Result('new_cancers', dtype=float, scale=True, label='New detected cancers'),
                ss.Result('sum_age_at_cancer', dtype=float, scale=True, label='Sum of ages at cancer onset'),
            )
        return

    def cancer_onset(self, sim, uids):
        """ Cancer onsets: counted, and their detection scheduled dur_undetected later (TODO(v4-rs): an outcome with a delay) """
        ti = self.ti
        self.results.new_undetected_cancers[ti] += len(uids)
        self.results.sum_age_at_cancer[ti] += sim.people.age[uids].sum()
        delay = self.pars.dur_undetected.rvs(uids) # In steps
        self.ti_cancer_detection[uids] = ti + np.round(np.asarray(delay, dtype=float)).astype(int)
        return

    def update_results(self):
        super().update_results()
        if self.detection:
            ti = self.ti
            self.results.new_cancers[ti] = np.count_nonzero((self.ti_cancer_detection.values == ti) & self.X.values)
        return

    def finalize_results(self):
        super().finalize_results()
        res = self.results
        res += ss.Result('cum_cancer_deaths', values=np.cumsum(res.new_cancer_deaths.values), scale=False, label='Cumulative cancer deaths', timevec=res.new_cancer_deaths.timevec, module=self.name)
        if self.detection:
            res += ss.Result('cum_cancers', values=np.cumsum(res.new_cancers.values), scale=False, label='Cumulative cancers', timevec=res.new_cancers.timevec, module=self.name)
        return

    def modifiers(self):
        """ The modifiers of this genotype's own natural history: dur_cin longer with age at CIN (age_risk) """
        age_risk = self.pars.age_risk
        if age_risk is None or age_risk['risk'] == 1:
            return []
        ramp = lambda sim: 1 / _age_risk_ramp(sim.people.age.raw, age_risk) # A rate, so the inverse of the multiplier on the duration
        return [ss.modify(self.cin_exit, multiply=ramp, where=self.C, name=f'age_risk_{self.name}')]

    # States as HPVsim 3 named them, as agents, for analyzers and custom code
    @property
    def susceptible(self): return self.S
    @property
    def precin(self):      return self.P
    @property
    def cin(self):         return self.C
    @property
    def cancerous(self):   return self.X
    @property
    def latent(self):      return self.L
    @property
    def infected(self):    return self.P | self.C
    @property
    def ti_cin(self):      return self.ti_C


# Parameters that are not read by the declared model, so are held as plain values
_PLAIN_PARS = ('rel_beta', 'transf2m', 'transm2f', 'age_risk', 'sero_prob', 'ms_agent_ratio', 'dur_undetected')


def _validate_duration_units(genotype, pars):
    """ A duration without time units is read in steps; refuse it, as HPVsim 3 did """
    for key in ('dur_precin', 'dur_cin', 'dur_cancer', 'dur_undetected', 'dur_inf_male'):
        dist = pars.get(key)
        if isinstance(dist, ss.Dist):
            vals = list(getattr(dist, 'pars', {}).values())
            bad = any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals) and not any(isinstance(v, ss.dur) for v in vals)
        else:
            bad = isinstance(dist, (int, float)) and not isinstance(dist, bool)
        if bad:
            raise ValueError(f'HPV genotype {genotype!r}: duration parameter {key!r} has no time units; wrap it in ss.years(), e.g. ss.lognorm_ex(mean=ss.years(3), std=ss.years(9))')
    return


def _clip_beta(f2m, m2f, genotype='?'):
    """ Per-act probabilities, clipped to [0, 1] with a warning """
    if f2m > 1.0 or m2f > 1.0:
        ss.warn(f'HPV[{genotype}] beta clipped to [0, 1]: f2m={f2m:.4f}, m2f={m2f:.4f}')
    return [min(1.0, f2m), min(1.0, m2f)]

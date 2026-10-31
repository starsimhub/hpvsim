"""Per-genotype HPV disease module.

Scope note: HPVsim only models cervical cancer outcomes. HPV is also
associated with anal, oropharyngeal, penile, vaginal, and vulvar cancers
(the first three of which can occur in males); those are out of scope here.

Models the natural history of one genotype as a declared Starsim disease:

    susceptible -> precin -> (susceptible | latent | cin)
    cin -> (susceptible | latent | cancerous -> death)
    latent -> precin

The genotypes of a sim are the strains of one disease (``make_family()``, which
``hpv.Sim`` calls), each with a course of its own (``ss.strains.Family``). Cancer is
shared: entering it in one genotype ends the others (in ``blocked``). Clearing a
genotype gives ``nab_imm``, which protects against each genotype through the
cross-immunity matrix (``ss.immunity``).

Females are eligible for the full progression; males clear from precin without
entering CIN/cancer. Clearance gives females partial immunity: beta samples
accumulate as a running max into ``nab_imm`` (humoral) and ``cell_imm``
(cell-mediated). Vaccine-conferred immunity is stored separately in ``vax_imm`` and
does NOT flow through the cross-immunity matrix: the CSV's per-genotype ``rel_imm``
is the complete vaccine cross-protection profile. The ``CrossImmunity`` connector
applies ``vax_imm`` and works out ``sev_imm`` each step.
"""

import numpy as np
import sciris as sc
import starsim as ss

from . import misc
from .parameters import genotype_aliases, get_genotype_pars
from .seeding import _make_init_prev_fn
from .utils import compute_severity


_KNOWN_GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')

# Per-agent Arrs that must not be cloned (identity, not biology).
_NO_CLONE_STATES = {'uid', 'slot'}

# Lifecycle states reset after cloning, mapped to fresh-agent defaults.
_RESET_ON_CLONE = {'ti_dead': np.nan, 'ti_removed': np.nan, 'alive': True}


def _clone_agents(sim, src_uids, new_uids):
    """Copy every per-agent Arr from src_uids to new_uids.

    starsim registers every module's per-agent states into ``people.states``
    under combined names, so one loop clones ALL per-agent state (People,
    network, demographics, disease, connector), except the identity keys in
    ``_NO_CLONE_STATES``. Cloning network/demographics state is harmless: fine
    agents are excluded from those subsystems, so those values are inert.

    The values are read raw, since a time reads as the step it happened in, which
    is not what is stored (see ``ss.TimeArr``), and set as usual, which keeps the
    engines' counts of who is in each state.

    Lifecycle states in ``_RESET_ON_CLONE`` (``ti_dead``/``ti_removed``/``alive``)
    are then reset to fresh-agent defaults so a clone never inherits its source's
    death/removal schedule. This is centralized here (rather than in the caller)
    so every caller of ``_clone_agents`` is correct without a separate reset step.

    src_uids and new_uids must align element-wise and be equal length.
    """
    ppl = sim.people
    for key, arr in ppl.states.items():
        if key not in _NO_CLONE_STATES:
            arr[new_uids] = arr.raw[src_uids]
    for name, val in _RESET_ON_CLONE.items():
        if name in ppl.states:
            ppl.states[name][new_uids] = val
    return


def _clip_beta(f2m, m2f, genotype='?'):
    """Clip HPV per-act transmission probabilities to [0, 1].

    Guards against ``gpars.beta * rel_beta * transm2f`` exceeding 1: the
    network's per-act calc ``1 - (1 - p) ** acts`` silently produces NaN
    for p > 1, so the sim would run to completion with garbage
    prevalence rather than erroring. Warn on clip so unusual parameter
    combinations that hit this cap are visible. With defaults
    (``transm2f=2.0``, ``beta=0.25``) no clip triggers; it fires when
    users pin transm2f higher, set ``rel_beta > 1``, or pass a very
    high per-genotype beta.
    """
    if f2m > 1.0 or m2f > 1.0:
        import warnings
        warnings.warn(
            f'HPV[{genotype}] beta clipped to [0, 1]: '
            f'raw f2m={f2m:.4f}, m2f={m2f:.4f} '
            f'-> clipped f2m={min(f2m, 1.0):.4f}, m2f={min(m2f, 1.0):.4f}. '
            f'(transm2f amplifies m2f; verify transm2f, rel_beta, and '
            f'top-level beta scalar are consistent with per-act probs ≤ 1.)',
            RuntimeWarning, stacklevel=2,
        )
    return [min(1.0, f2m), min(1.0, m2f)]


def _age_risk_ramp(ages, age_risk):
    """Age-dependent multiplier on dur_cin.

    Returns 1 below ``age_risk['age']``, ``age_risk['risk']`` at/above
    ``age_risk['age_end']``, and a linear interpolation in between. The
    ramp replaces the previous hard step at ``age_risk['age']``, which
    produced a visible artifact in the model's age-at-cancer distribution
    (two modes on either side of the step).
    """
    start = age_risk['age']
    end = age_risk['age_end']
    risk = age_risk['risk']
    frac = np.clip((np.asarray(ages) - start) / (end - start), 0.0, 1.0)
    return 1.0 + (risk - 1.0) * frac


def _normalize_genotype(key):
    """Resolve aliases (16 -> 'hpv16', 'hi5' -> 'hi5') to canonical keys."""
    s = str(key).lower().strip()
    for canonical, aliases in genotype_aliases.items():
        if s == canonical or s in aliases:
            return canonical
    raise ValueError(
        f'Unknown genotype {key!r}; valid: {list(genotype_aliases)}'
    )


def _in_years(v):
    """A duration's number of years, or a plain number"""
    return v.years if hasattr(v, 'years') else v




class HPV(ss.Disease):
    """The natural history of one genotype of HPV.

    Declared, so its results and its place in a family come from its states and
    transitions; ``make_family()`` ties the genotypes of a sim together. How long
    someone stays in ``precin`` and ``cin`` decides what comes next (the longer, the
    likelier progression): see ``p_cin()`` and ``p_cancer()``, read as people leave.

    Args:
        genotype (str): 'hpv16', 'hpv18', 'hi5', or 'ohr'
        pars (dict): parameters, as for the keywords
        init_prev (Dist): who starts infected; by default, this genotype's curve by age and sex; None if the family seeds it
        kwargs (dict): parameters (see below), or ``name``
    """

    def __init__(self, genotype='hpv16', pars=None, init_prev='curve', **kwargs):
        if genotype not in _KNOWN_GENOTYPES:
            raise ValueError(
                f'genotype must be one of {list(_KNOWN_GENOTYPES)}; got {genotype!r}.'
            )
        self.genotype = genotype
        super().__init__(name=kwargs.pop('name', genotype))
        S, P, C, Ca, L, X = ss.states('susceptible precin cin cancerous latent blocked')
        L.keep_time = True
        S.label, P.label, C.label, Ca.label, L.label = 'Susceptible', 'Precancerous infection', 'Cervical intraepithelial neoplasia', 'Invasive cancer', 'Latent infection'
        X.label = 'Cancer of another genotype' # Where the family moves them (see ss.strains.Family.step())
        self.define_states(S, P, C, Ca, L, X)
        # Natural-history defaults come from GenotypePars.
        gpars = get_genotype_pars(genotype)
        self.define_pars(
            # Scalar beta; validate_beta() expands it. A dict bypasses expansion.
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
            # Per-genotype beta scaler, transmission asymmetry, serology prob.
            rel_beta=gpars.rel_beta,
            transf2m=gpars.transf2m,
            transm2f=gpars.transm2f,
            sero_prob=gpars.sero_prob,
            # Agents each cancer-capable agent represents; >1 grows fine agents
            # at scale 1/ms_agent_ratio, 1 is a bit-identical no-op.
            ms_agent_ratio=1,
            # Probability a clearing female enters latency instead of clearing.
            hpv_control_prob=0.0,
            # Reactivation hazard for latents; only active when hpv_control_prob>0.
            hpv_reactivation=ss.probperyear(0.025),
            dur_undetected=ss.lognorm_ex(mean=ss.years(1), std=ss.years(1)),
        )
        self.update_pars(pars=pars, **kwargs)
        self.pars.ms_agent_ratio = int(self.pars.ms_agent_ratio)

        # A duration without ss.years() is read in timesteps, not years.
        self._validate_duration_units()

        # A parameter that is a function is read for the people concerned; see the methods
        par = lambda func: ss.par(func, name=func.__name__)
        beta = dict(sexualnetwork=ss.peract(m2f=self.beta_m2f, f2m=self.beta_f2m), default=0)
        infection = ss.Transition(S, P, rate=ss.par(beta, name='beta_by_net')*(P + C))
        progression = ss.Transition(P, [C, L, S], dur=ss.par(self.pars.dur_precin, name='dur_precin')*par(self.rel_dur_precin),
                                    p=[self.p_cin, self.p_latent_precin, None], where=ss.ppl.female)
        male_clearance = ss.Transition(P, S, dur=ss.par(self.pars.dur_inf_male, name='dur_inf_male'), where=~ss.ppl.female)
        transformation = ss.Transition(C, [Ca, L, S], dur=ss.par(self.pars.dur_cin, name='dur_cin')*par(self.rel_dur_cin),
                                       p=[self.p_cancer, self.p_latent_cin, None])
        reactivation = ss.Transition(L, P, rate=ss.par(self.pars.hpv_reactivation, name='hpv_reactivation')*par(self.rel_reactivation))
        cancer_death = ss.Transition(Ca, ss.null, dur=ss.par(self.pars.dur_cancer, name='dur_cancer'))
        cure = ss.Transition([P, C], S) # Only by treatment: see clear_infection()
        self.define_transitions(infection=infection, progression=progression, male_clearance=male_clearance, transformation=transformation,
                                reactivation=reactivation, cancer_death=cancer_death, cure=cure)
        cleared = progression.to(S) | transformation.to(S) | cure # Females only: men clear by male_clearance, with no immunity
        self.declare(kwargs=dict(
            infected=P + C,
            # Fixed at infection, for the course that follows
            rel_sev=ss.attr(par(self.rel_sev_now), on=P.enter, label='Relative severity at infection'),
            # Running max over clearances; an HIV-reduced draw can fail to boost immunity but never erases it. sero_prob
            # gates nab_imm until a woman has seroconverted, and never gates cell_imm
            sero=ss.attr(ss.bernoulli(p=self.p_sero), on=cleared, combine='max', init=0, label='Whether seroconverted'),
            nab_imm=ss.attr(ss.par(self.pars.imm_init, name='imm_init')*par(self.rel_nab), on=cleared, combine='max', init=0, label='Humoral immunity (clearance-conferred, source genotype)'),
            cell_imm=ss.attr(ss.par(self.pars.cell_imm_init, name='cell_imm_init')*par(self.rel_imm), on=cleared, combine='max', init=0, label='Cell-mediated immunity (source genotype)'),
        ))
        init_prev = _make_init_prev_fn(genotype) if isinstance(init_prev, str) else init_prev.pars.p if isinstance(init_prev, ss.bernoulli) else init_prev
        if init_prev is not None:
            self.declare(kwargs=dict(init_prev=init_prev))

        self.define_states(
            # A subset of cancerous: onset has happened and detection has not
            ss.BoolArr('undetected_cancerous', label='Undetected invasive cancer'),
            ss.TimeArr('ti_cancer_detection', label='Time of cancer detection'),
            ss.TimeArr('ti_cancer_death', label='Time of cancer-caused death'),
            ss.FloatArr('ti_reactivation', label='Time of reactivation from latency'),
            # Per-target: protection against this genotype, applied directly in
            # CrossImmunity.step() and never through the cross-immunity matrix.
            ss.FloatArr('vax_imm', label='Vaccine-conferred immunity (against this/target genotype)', default=0.0),
            ss.FloatArr('txvx_sev_imm', label='Therapeutic-conferred severity immunity (against this/target genotype)', default=0.0),
            ss.FloatArr('sev_imm', label='Severity immunity', default=0.0), # From CrossImmunity: shortens dur_precin at infection
            ss.FloatArr('ca_fate', label='Whether cancer follows CIN, if decided at its onset'), # See _grow_fine_agents()
        )
        self._round_detection_bern = ss.bernoulli(p=0.5)
        # Separate RNG streams so growing fine agents doesn't perturb real agents' draws.
        self._fate_unif = ss.random()
        self._extra_dur_cin = ss.lognorm_ex(mean=_in_years(self.pars.dur_cin.pars['mean']), std=_in_years(self.pars.dur_cin.pars['std']))
        self._extra_cancer_unif = ss.random()
        return

    def _validate_duration_units(self):
        """Raise if a duration par lost its time units.

        Duration distributions must carry ``ss.years`` (an ``ss.dur``): starsim
        converts duration TimePars to timesteps by dividing by ``dt``, so a
        unit-less value (e.g. ``ss.lognorm_ex(mean=3, std=9)``) is treated as
        timesteps and, at ``dt<1``, makes the infectious period ~``1/dt`` too
        short — the epidemic silently collapses. Only the public module duration
        pars are checked (the private ``_extra_dur_*`` dists are intentionally
        unit-less for the grow multiscale path).
        """
        for key in ('dur_precin', 'dur_cin', 'dur_cancer', 'dur_undetected',
                    'dur_inf_male'):
            dist = self.pars.get(key, None)
            if dist is None or isinstance(dist, ss.dur):
                continue  # a bare ss.dur constant is fine
            if isinstance(dist, ss.Dist):
                vals = list(getattr(dist, 'pars', {}).values())
                has_dur = any(isinstance(v, ss.dur) for v in vals)
                has_plain = any(isinstance(v, (int, float)) and not isinstance(v, bool)
                                for v in vals)
                bad = has_plain and not has_dur
            else:
                # A plain scalar (not a Dist, not an ss.dur) is unit-less too.
                bad = isinstance(dist, (int, float)) and not isinstance(dist, bool)
            if bad:
                raise ValueError(
                    f"HPV genotype {self.genotype!r}: duration parameter {key!r} "
                    f"was given without time units. A unit-less duration is read "
                    f"in timesteps, not years, so at dt<1 the infectious period is "
                    f"~1/dt too short and the epidemic silently collapses. Wrap the "
                    f"magnitude in ss.years(...), e.g. "
                    f"ss.lognorm_ex(mean=ss.years(3), std=ss.years(9))."
                )
        return

    def init_pre(self, sim):
        self.male_clearance.after = ss.years(sim.t.dt_year/2) # Male clearance rounds up to a whole step, half a step on average, where the rest round to nearest
        super().init_pre(sim)
        del self.infected # So that it is read by __getattr__()
        self.add_callback('progression', self.on_progression)
        self.add_callback('transformation', self.on_transformation)
        self.add_callback('cancer_death', self.on_cancer_death)
        self.add_callback('reactivation', self.on_reactivation)
        return

    @property
    def ti_infected(self):
        """The time of the latest infection (or reactivation), which starts a course"""
        return self.ti_precin

    @property
    def ti_first_infection(self):
        """Whether, as a time, each person has ever been infected: the time of the latest"""
        return self.ti_precin

    @property
    def ti_dead_cancer(self):
        """The time of cancer-caused death, once scheduled"""
        return self.ti_exit_cancerous

    def __getattr__(self, attr):
        """``infected`` is a named sum of states; in a sim, it is the array of who is in it, as for a state"""
        if attr == 'infected':
            return self.precin | self.cin
        return super().__getattr__(attr)

    def validate_beta(self):
        """Expand a scalar pars.beta via rel_beta/transf2m/transm2f, as [f2m, m2f] (mirrors
        stisim's BaseSTI.validate_beta). An explicit dict pars.beta is passed through unchanged."""
        beta = self.pars.beta
        if not sc.isnumber(beta):
            return dict(beta)
        base = beta * self.pars.rel_beta
        betamap = {} if self.sim is None else {ss.networks.standardize_netkey(net): [0.0, 0.0] for net in self.sim.networks.keys()}
        return betamap | {'sexualnetwork': _clip_beta(base * self.pars.transf2m, base * self.pars.transm2f, self.genotype)}

    # What the transitions read (see __init__())

    def beta_f2m(self, sim):
        return self.validate_beta()['sexualnetwork'][0]

    def beta_m2f(self, sim):
        return self.validate_beta()['sexualnetwork'][1]

    def _hiv_effects(self):
        """HIV's effects on HPV (``hpv.HIVEffects``) in the sim, if any (None when no HIV)."""
        return misc.hiv_effects(self.sim)

    def _hiv_factor(self, key, uids):
        """One of HIV's effects on HPV for some people, which is 1 without HIV"""
        effects = self._hiv_effects()
        return np.ones(len(uids)) if effects is None else effects[key][uids]

    def rel_dur_precin(self, sim, uids):
        """Severity immunity shortens dur_precin, fixed at infection"""
        return 1.0 - self.sev_imm[uids]

    def rel_dur_cin(self, sim, uids):
        """age_risk's multiplier on dur_cin, by the age at infection. A smooth ramp: a hard step at 30
        would carve the cancer-age distribution into two visible modes at ~25 and ~30."""
        return _age_risk_ramp(self.sim.people.age[uids] - self._stay('precin', uids), self.pars.age_risk)

    def rel_reactivation(self, sim, uids):
        """Relative factors scale the rate, never the probability"""
        return (1.0 - self.sev_imm[uids]) * self._hiv_factor('hiv_rel_reactivation', uids)

    def rel_sev_now(self, sim, uids):
        """rel_sev is shared across genotypes via CrossImmunity; HIV co-infection scales it (gated no-op when no HIV)"""
        cross = misc.cross_immunity(sim)
        rel_sev = np.ones(len(uids)) if cross is None else np.abs(cross.rel_sev[uids])
        return rel_sev * self._hiv_factor('hiv_rel_sev', uids)

    def rel_imm(self, sim, uids):
        """HIV co-infection reduces conferred immunity (gated no-op); one dt stale, as the connector steps first"""
        return self._hiv_factor('hiv_rel_imm', uids)

    def p_sero(self, module, sim, uids):
        return np.full(len(uids), float(self.pars.sero_prob))

    def rel_nab(self, sim, uids):
        return self.sero[uids] * self.rel_imm(sim, uids)

    def _stay(self, state, uids):
        """The years people leaving a state spent in it (0 for those moved through it at once)"""
        return np.maximum(self.engine.ti - self[f'ti_{state}'].raw[uids], 0.0) * self.t.dt_year

    def p_cin(self, sim, uids):
        """P(CIN) from the years in precin; rel_sev applies separately from sev_imm, which shortened them"""
        return np.clip(compute_severity(self._stay('precin', uids), rel_sev=self.rel_sev[uids], pars=self.pars.cin_fn), 0, 1)

    def p_cancer(self, sim, uids):
        """P(cancer) from the years in CIN; sev_imm is not applied here, only rel_sev. If it was decided at CIN onset, as it was"""
        p = np.clip(compute_severity(self._stay('cin', uids), rel_sev=self.rel_sev[uids], pars=self.pars.cancer_fn), 0, 1)
        fate = self.ca_fate[uids]
        return np.where(np.isnan(fate), p, fate)

    def p_latent_precin(self, sim, uids):
        """Of the women who would clear, the share who enter latency instead (no-op unless hpv_control_prob>0)"""
        ctrl = self.pars.hpv_control_prob
        return (1.0 - self.p_cin(sim, uids)) * ctrl if ctrl else 0.0

    def p_latent_cin(self, sim, uids):
        ctrl = self.pars.hpv_control_prob
        return (1.0 - self.p_cancer(sim, uids)) * ctrl if ctrl else 0.0

    # Results

    def init_results(self):
        """The results that are not a state or a transition's: ``new_cancers`` counts detection, and
        ``sum_age_at_*`` are per-step accumulators, whose mean age is ``sum / count``"""
        super().init_results()
        self.define_results(
            ss.Result('undetected_cancerous', dtype=float, scale=True, kind='stock', label='Undetected invasive cancer'),
            ss.Result('new_cancers', dtype=float, scale=True, kind='flow', label='New detected cancers'),
            # Other names for flows, which are scaled already. WORKAROUND: as ss.outcome() these would be drawn as whole numbers from the weights of fine agents (ss.solvers.Agents.weighted_events())
            ss.Result('new_undetected_cancers', dtype=float, scale=False, kind='flow', label='New cancer onsets (pre-detection)'),
            ss.Result('new_cancer_deaths', dtype=float, scale=False, kind='flow', label='New cancer deaths'),
            ss.Result('new_reactivations', dtype=float, scale=False, kind='flow', label='New reactivations from latency'),
            ss.Result('cum_cancers', dtype=float, scale=True, kind='stock', label='Cumulative cancers'),
            ss.Result('cum_cancer_deaths', dtype=float, scale=True, kind='stock', label='Cumulative cancer deaths'),
            ss.Result('sum_age_at_cancer', dtype=float, scale=True, kind='flow', label='Sum of ages at cancer onset'),
            ss.Result('sum_age_at_cancer_death', dtype=float, scale=True, kind='flow', label='Sum of ages at cancer death'),
        )
        return

    def update_results(self):
        super().update_results()
        people = self.sim.people
        res, ti = self.results, self.ti
        res.undetected_cancerous[ti] = people.scale_flows((self.undetected_cancerous & people.alive).uids)
        res.new_undetected_cancers[ti] = res['transformation.cancerous'][ti]
        res.new_cancer_deaths[ti] = res.cancer_death[ti]
        res.new_reactivations[ti] = res.reactivation[ti]
        return

    def finalize_results(self):
        super().finalize_results()
        res = self.results
        res.cum_cancers[:] = np.cumsum(res.new_cancers)
        res.cum_cancer_deaths[:] = np.cumsum(res.new_cancer_deaths)
        return

    # What happens around the declared transitions

    def step_state(self):
        """Every genotype's moves, before any of them transmits, and detect cancers, whose onset was earlier. Reset
        rel_sus so CrossImmunity and HIV multiply, not overwrite.

        WORKAROUND: a declared disease takes its moves and its transmission together, in its ``step()``, one
        genotype at a time. A woman can be reinfected by her partner in the step she clears, so a family stepped
        that way protects her against the genotypes earlier in the order by less than against those later,
        which then favors the earlier ones. Taking the moves here is as agents only, which HPVsim is.
        """
        super().step_state()
        self.engine.step_state()
        self.run_callbacks()
        self.rel_sus[:] = 1.0
        detected = (self.undetected_cancerous & self.cancerous & (self.ti_cancer_detection <= self.ti)).uids
        if len(detected):
            self.undetected_cancerous[detected] = False
            self.results.new_cancers[self.ti] = self.sim.people.scale_flows(detected)
        return

    def step(self):
        """The transmission (see ``step_state()``)"""
        self.engine.step()
        self.engine.step_funcs()
        return

    def on_progression(self, uids):
        """Multiscale: women reaching CIN, in this step or the last one's transmission, even if their stay in it has ended already"""
        if self.pars.ms_agent_ratio > 1:
            reached = self.ti_cin.raw[uids] >= self.ti - 2 # In the engine's time (see ss.TimeArr)
            if reached.any():
                self._grow_fine_agents(uids[reached].unique())
        return

    def on_transformation(self, uids):
        """Cancer onset: the detection lag is drawn here, so a screening-lag scenario (dur_undetected>0)
        can defer new_cancers without changing biology."""
        uids = uids[self.cancerous[uids]]
        if len(uids):
            ppl = self.sim.people
            self.undetected_cancerous[uids] = True
            self.ti_cancer_detection[uids] = self.ti + self._randround(self.pars.dur_undetected.rvs(uids), uids, self._round_detection_bern)
            self.results.sum_age_at_cancer[self.ti] += float((ppl.age[uids] * ppl.scale[uids]).sum())
        return

    def on_reactivation(self, uids):
        self.ti_reactivation[uids] = self.ti
        return

    def on_cancer_death(self, uids):
        """+dt_yr: this runs before ages advance, so this is age at step end"""
        ppl = self.sim.people
        self.ti_cancer_death[uids] = self.ti
        self.results.sum_age_at_cancer_death[self.ti] += float(((ppl.age[uids] + self.t.dt_year) * ppl.scale[uids]).sum())
        return

    @staticmethod
    def _randround(values, uids, dist):
        """CRN-safe stochastic round to the nearest integer.

        Equivalent to ``sc.randround(values)`` semantically (floor + a
        Bernoulli draw on the fractional part), but routes the random draw
        through a per-decision ``ss.bernoulli`` so each agent gets a
        deterministic, per-uid draw under CRN. ``dist`` must be a dedicated
        Bernoulli created in ``__init__`` and used only at this call site.
        ``values`` and ``uids`` must align element-wise.
        """
        if len(uids) == 0:
            return np.zeros(0, dtype=int)
        values = np.maximum(values, 0.0)
        floor = np.floor(values)
        dist.set(p=values - floor)
        return (floor + dist.rvs(uids)).astype(int)

    def _grow_fine_agents(self, cin_uids):
        """Grow up to ratio-1 extra real fine cancer agents per woman reaching CIN.

        Whether cancer follows is decided here for these women (``ca_fate``), from
        the stay in CIN just scheduled, so that those transforming can be shrunk
        to scale 1/ratio before they count. For every CIN reacher, ratio-1 extra
        stays are drawn (age_risk-modified dur_cin), cancer is drawn for each
        extra, and one real fine agent is grown per extra-cancer success (full
        cross-genotype clone of the source, then this genotype's cancer-bound
        stay in CIN written). No-op at ms_agent_ratio<=1.

        RNG: the extra draws come from dedicated dists registered in __init__ --
        SEPARATE streams from the live natural-history dists, so growing fine
        agents never perturbs the real agents' draws (keeps ratio==1 bit-identical
        and incidence flat across ratios). Drawn by size (non-CRN).
        """
        ratio = int(self.pars.ms_agent_ratio)
        p = self.pars
        ppl = self.sim.people
        cancer_scale = 1.0 / ratio
        rel_sev = self.rel_sev[cin_uids]
        dt_yr = self.t.dt_year

        # Base agents: their fate from the stay they have, and shrink those that drew their own cancer.
        in_cin = self.cin[cin_uids]
        dur_cin = (self.ti_exit_cin.raw[cin_uids[in_cin]] - self.ti_cin.raw[cin_uids[in_cin]]) * dt_yr
        pcanc = compute_severity(dur_cin, rel_sev=rel_sev[in_cin], pars=p.cancer_fn)
        fate = self._fate_unif.rvs(cin_uids[in_cin]) < pcanc
        self.ca_fate[cin_uids[in_cin]] = fate
        ppl.scale[cin_uids[in_cin][fate]] = cancer_scale

        # CIN -> cancer for every extra (years), with age_risk.
        size = (len(cin_uids), ratio - 1)
        extra_dur_cin = self._extra_dur_cin.rvs(size) * self.rel_dur_cin(self.sim, cin_uids)[:, None]
        pcanc = compute_severity(extra_dur_cin, rel_sev=rel_sev[:, None] * np.ones(size), pars=p.cancer_fn)
        extra_cancer = self._extra_cancer_unif.rvs(size) < pcanc
        # Existing fine agents never spawn more fine agents (level0 guard).
        extra_cancer &= ~ppl.fine[cin_uids][:, None]
        counts = extra_cancer.sum(axis=1)
        n_new = int(counts.sum())
        if n_new == 0:
            return

        # Broadcast source uids per success and grow.
        src_uids = ss.uids(np.repeat(np.asarray(cin_uids), counts))
        new_uids = ppl.grow(n_new)

        # Full cross-genotype clone; _clone_agents also resets lifecycle states.
        _clone_agents(self.sim, src_uids, new_uids)

        # Fine-agent identity.
        ppl.fine[new_uids] = True
        ppl.scale[new_uids] = cancer_scale

        # This genotype's cancer-bound stay in CIN, in place of the source's (who may have left it already).
        left = new_uids[~self.cin[new_uids]]
        states = self.graph.states
        self.engine.move_states(left, [s for s in self.graph.people_states if s is not states.cin], states.cin)
        self.ca_fate[new_uids] = True
        self.ti_exit_cin.raw[new_uids] = self.ti_cin.raw[new_uids] + self.engine.whole_steps(new_uids, extra_dur_cin[extra_cancer])
        return

    def clear_infection(self, uids):
        """Clear the precin and CIN of this genotype, on treatment (e.g. ``hpv.tx``, or ``ss.Treatment``), to susceptible (not latency);
        cancer of it leaves no course to return to, so it ends in ``blocked``, as if of another genotype"""
        uids = ss.uids(uids)
        self.engine.force(uids, self.cure, self.cure.dst[0])
        cancerous = uids[self.cancerous[uids]]
        if len(cancerous):
            self.undetected_cancerous[cancerous] = False
            self.engine.move_states(cancerous, [self.graph.states.cancerous], self.graph.states.blocked)
        return

    def extend_cancer(self, uids, steps):
        """Delay the cancer deaths of some people by some steps (e.g. by radiation)"""
        self.ti_exit_cancerous.raw[uids] += steps
        return

    def step_die(self, uids):
        super().step_die(uids)
        self.undetected_cancerous[uids] = False
        return


def make_family(members, init_prev=None, split=None, name='hpv'):
    """The genotypes of a sim, as the strains of one disease, each with a course of its own: one cancer per person,
    the protection of ``nab_imm`` between them, and, with ``init_prev``, each person seeded getting one

    Args:
        members (list): the ``HPV`` modules, one for each genotype
        init_prev (func): the probability that each person starts infected, as ``fn(sim, uids)``, if the family seeds them
        split (list): the share of each genotype among those seeded (default: equal)
        name (str): the name of the disease
    """
    n = len(members)
    immunity = ss.immunity(due_to='nab_imm', rel_sus=np.ones((n, n)), combine='sum') # The matrix is CrossImmunity's: set in its init_pre()
    return ss.Disease(name=name, init_prev=init_prev, strains=ss.Strains(members, shared=['cancerous'], immunity=immunity, split=split, name='genotype'))

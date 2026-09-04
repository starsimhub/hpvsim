"""Per-genotype HPV disease module.

Scope note: HPVsim only models cervical cancer outcomes. HPV is also
associated with anal, oropharyngeal, penile, vaginal, and vulvar cancers
(the first three of which can occur in males); those are out of scope here.

Models the natural-history pipeline as a Starsim Infection:

    susceptible -> precin -> (clear | CIN -> (clear | cancerous -> death))

Females are eligible for the full progression; males clear from precin without
entering CIN/cancer. Clearance grants partial same-genotype immunity: per-agent
beta samples accumulate as a running max into ``nab_imm`` (humoral) and
``cell_imm`` (cell-mediated). Vaccine-conferred immunity is stored separately
in ``vax_imm`` and does NOT flow through the cross-immunity matrix — the
CSV's per-genotype ``rel_imm`` is the complete vaccine cross-protection
profile. The ``CrossImmunity`` connector reads ``nab_imm`` / ``cell_imm``
each step, matrix-multiplies to derive cross-genotype ``rel_sus`` / ``sev_imm``,
then combines with ``vax_imm`` via independent-protection paths.

Multi-genotype runs instantiate one HPV per genotype, all sharing the People
and going through the same Connector path; a 1-genotype run uses a 1×1
identity matrix on the Connector.
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
    under combined names, so the first loop over ``ppl.states`` already clones
    ALL per-agent state (People, network, demographics, disease, connector) —
    except the identity keys in ``_NO_CLONE_STATES``. The second loop over
    disease + connector ``state_list`` is therefore a redundant belt-and-
    suspenders re-copy (idempotent — same source values written twice); it is
    kept so the clone stays correct even if a starsim version stops flattening
    module states into ``people.states``. Cloning network/demographics state is
    harmless: fine agents are excluded from those subsystems, so those values
    are inert.

    Lifecycle states in ``_RESET_ON_CLONE`` (``ti_dead``/``ti_removed``/``alive``)
    are then reset to fresh-agent defaults so a clone never inherits its source's
    death/removal schedule. This is centralized here (rather than in the caller)
    so every caller of ``_clone_agents`` is correct without a separate reset step.

    src_uids and new_uids must align element-wise and be equal length.
    """
    ppl = sim.people
    for key, arr in ppl.states.items():
        if key in _NO_CLONE_STATES:
            continue
        arr[new_uids] = arr[src_uids]
    for mod in list(sim.diseases.values()) + list(sim.connectors.values()):
        for st in mod.state_list:
            st[new_uids] = st[src_uids]
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


def _normalize_genotype(key):
    """Resolve aliases (16 -> 'hpv16', 'hi5' -> 'hi5') to canonical keys."""
    s = str(key).lower().strip()
    for canonical, aliases in genotype_aliases.items():
        if s == canonical or s in aliases:
            return canonical
    raise ValueError(
        f'Unknown genotype {key!r}; valid: {list(genotype_aliases)}'
    )


class HPV(ss.Infection):
    """Per-genotype HPV disease module.

    The ``genotype`` attribute identifies which strain this instance models.
    The CrossImmunity connector reads each registered HPV's nab_imm/cell_imm
    (clearance-conferred) each step, matrix-multiplies to derive per-target
    rel_sus/sev_imm, then combines with vax_imm (vaccine-conferred, per-target
    direct) via independent-protection paths. Vaccine immunity does NOT flow
    through the cross-immunity matrix.
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
        # Natural-history defaults come from GenotypePars.
        gpars = get_genotype_pars(genotype)
        self.define_pars(
            init_prev=ss.bernoulli(p=_make_init_prev_fn(genotype)),
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
        )
        self.update_pars(pars=pars, **kwargs)
        self.pars.ms_agent_ratio = int(self.pars.ms_agent_ratio)

        # A duration without ss.years() is read in timesteps, not years.
        self._validate_duration_units()
        # ss.Infection already provides susceptible/infected/rel_sus/rel_trans/ti_infected.
        self.define_states(
            ss.FloatArr('ti_clearance', label='Time of natural clearance'),
            # Set on first infection only; ti_infected is overwritten on reinfection.
            ss.FloatArr('ti_first_infection', label='Time of first infection'),
            ss.BoolState('precin', label='Precancerous infection'),
            ss.BoolState('cin', label='Cervical intraepithelial neoplasia'),
            ss.BoolState('cancerous', label='Invasive cancer'),
            ss.BoolState('latent', label='Latent infection'),
            # Set in set_prognoses, cleared by step_state once ti_latent is stamped.
            # BoolArr, not BoolState: an internal scheduling flag, so it must not
            # auto-generate a public n_to_latent result (which would also be
            # summed into all_hpv, where it is meaningless).
            ss.BoolArr('to_latent', label='Pending transition into latency'),
            ss.FloatArr('ti_latent', label='Time latency began'),
            ss.FloatArr('ti_reactivation', label='Time of reactivation from latency'),
            ss.FloatArr('ti_cin', label='Time of CIN onset'),
            ss.FloatArr('ti_cancerous', label='Time of invasive cancer onset'),
            ss.FloatArr('ti_dead_cancer', label='Time of cancer-caused death'),
            # Running max over clearances; shortens future dur_precin by (1 - sev_imm).
            ss.FloatArr('sev_imm', label='Severity immunity', default=0.0),
            # Per-source: immunity conferred by clearing this genotype, which
            # CrossImmunity matrix-multiplies into per-target rel_sus/sev_imm.
            ss.FloatArr('nab_imm', label='Humoral immunity (clearance-conferred, source genotype)', default=0.0),
            ss.FloatArr('cell_imm', label='Cell-mediated immunity (source genotype)', default=0.0),
            # Per-target: protection against this genotype, applied directly in
            # CrossImmunity.step() and never through the cross-immunity matrix.
            ss.FloatArr('vax_imm', label='Vaccine-conferred immunity (against this/target genotype)', default=0.0),
            ss.FloatArr('txvx_sev_imm', label='Therapeutic-conferred severity immunity (against this/target genotype)', default=0.0),
        )
        # p is overwritten via .set(p=...) at each use site.
        self._cin_bern = ss.bernoulli(p=0.5)
        self._cancer_bern = ss.bernoulli(p=0.5)
        self._sero_bern = ss.bernoulli(p=0.5)
        # One rounding dist per ti_<event> so the round-up draws stay independent.
        self._round_clear_precin_bern = ss.bernoulli(p=0.5)
        self._round_cin_bern = ss.bernoulli(p=0.5)
        self._round_clear_cin_bern = ss.bernoulli(p=0.5)
        self._round_cancer_bern = ss.bernoulli(p=0.5)
        self._round_dead_bern = ss.bernoulli(p=0.5)
        # Latency entry roll and reactivation hazard get their own streams.
        self._latent_bern = ss.bernoulli(p=0.5)
        self._reactivation_bern = ss.bernoulli(p=0.5)
        # Separate RNG streams so growing fine agents doesn't perturb real agents' draws.
        # .years keeps these unitless: .rvs() returns years, not timesteps.
        def _yr(v):
            return v.years if hasattr(v, 'years') else v
        pc = self.pars.dur_precin.pars
        cn = self.pars.dur_cin.pars
        self._extra_dur_precin = ss.lognorm_ex(mean=_yr(pc['mean']), std=_yr(pc['std']))
        self._extra_dur_cin = ss.lognorm_ex(mean=_yr(cn['mean']), std=_yr(cn['std']))
        self._extra_cin_unif = ss.random()
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
        for key in ('dur_precin', 'dur_cin', 'dur_cancer', 'dur_inf_male'):
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

    def init_post(self):
        # rel_sev must be sampled before super() seeds init_prev, which reads it.
        cross = self._cross_immunity_connector()
        if cross is not None:
            cross.ensure_rel_sev(self.sim.people.alive.uids)
        super().init_post()
        return

    def _cross_immunity_connector(self):
        """Locate the CrossImmunity connector on the sim, if any.

        Returns None if no CrossImmunity is registered (single-genotype
        sims without cross-immunity get a 1x1 identity connector by
        default, but defensive against future configurations that disable
        it). When None, rel_sev stays at its default 1.0.
        """
        # Avoid circular import at module load.
        from .cross_genotype import CrossImmunity
        for c in self.sim.connectors.values():
            if isinstance(c, CrossImmunity):
                return c
        return None

    # Stocks whose auto n_* results are promoted to float and scale-weighted.
    _STOCK_STATES = ('susceptible', 'infected', 'precin', 'cin',
                     'cancerous', 'latent')

    def validate_beta(self):
        """Expand a scalar pars.beta via rel_beta/transf2m/transm2f (mirrors
        stisim's BaseSTI.validate_beta). An explicit dict pars.beta is passed
        through unchanged. A bare (unattached) module has no self.sim to
        validate networks against, so skip super() and expand/pass-through
        directly."""
        beta = self.pars.beta
        if self.sim is None:
            if sc.isnumber(beta):
                base = beta * self.pars.rel_beta
                return {'sexualnetwork': _clip_beta(
                    base * self.pars.transf2m, base * self.pars.transm2f, self.genotype,
                )}
            return dict(beta)
        betamap = super().validate_beta()
        if sc.isnumber(beta):
            base = beta * self.pars.rel_beta
            for net in betamap:
                if net == 'sexualnetwork':
                    betamap[net] = _clip_beta(
                        base * self.pars.transf2m, base * self.pars.transm2f, self.genotype,
                    )
                else:
                    betamap[net] = [0.0, 0.0]
        return betamap

    def _hiv_module(self):
        """Locate the hpv.HIV disease on the sim, if any (None when no HIV)."""
        return misc.hiv_module(self.sim)

    def init_results(self):
        """Per-step Results emitted from ``step_state``.

        ``new_cancers`` / ``new_cancer_deaths`` are realized-event counters
        (the cin -> cancerous and cancerous -> dead transitions);
        ``cum_*`` are populated as cumulative sums in ``finalize_results``.
        ``sum_age_at_*`` are per-step accumulators; mean age = ``sum / count``.

        Starsim auto-generates ``n_<state>`` results as ``dtype=int`` for every
        ``BoolState``. We promote each stock result to ``dtype=float`` here so
        that ``update_results`` can write scale-weighted values (fine agents
        carry ``scale=1/ratio`` and must count as 1/ratio, not 1).
        """
        super().init_results()
        # Must run after super() creates them.
        for state_name in self._STOCK_STATES:
            key = f'n_{state_name}'
            if key in self.results:
                res = self.results[key]
                res.dtype = float
                if res.values is not None:
                    res.values = res.values.astype(np.float64)
        self.define_results(
            ss.Result('new_cancers', dtype=float, scale=True,
                      label='New cancers'),
            ss.Result('new_cancer_deaths', dtype=float, scale=True,
                      label='New cancer deaths'),
            ss.Result('cum_cancers', dtype=float, scale=True,
                      label='Cumulative cancers'),
            ss.Result('cum_cancer_deaths', dtype=float, scale=True,
                      label='Cumulative cancer deaths'),
            ss.Result('sum_age_at_cancer', dtype=float, scale=True,
                      label='Sum of ages at cancer onset'),
            ss.Result('sum_age_at_cancer_death', dtype=float, scale=True,
                      label='Sum of ages at cancer death'),
            ss.Result('new_reactivations', dtype=float, scale=True,
                      label='New reactivations from latency'),
        )
        return

    def update_results(self):
        """Recompute per-step stock counts as scale-weighted floats.

        Starsim's ``Module.update_results`` writes ``n_<state>[ti] =
        state.sum()`` — a plain boolean count that treats every agent as one
        body regardless of its ``scale``. Fine agents carry ``scale =
        1/ms_agent_ratio``, so they must count as ``1/ratio``, not 1.

        We call ``super().update_results()`` first so the starsim pipeline
        runs normally (prevalence, new_infections, etc.), then overwrite
        each stock slot with ``people.scale_flows(uids_in_state)``.

        After the stock overwrite, ``prevalence`` is re-derived from the
        corrected scale-weighted ``n_infected`` divided by the
        scale-weighted alive-agent total. ``Infection.update_results``
        (called via super) computed ``prevalence`` from the stale plain-count
        ``n_infected``, so at ``ms_agent_ratio>1`` it would be inflated by
        roughly the ratio. Re-deriving here is a no-op at ratio==1 (where
        ``scale_flows == len``).
        """
        super().update_results()
        ti = self.ti
        ppl = self.sim.people
        auids = ppl.auids
        for state_name in self._STOCK_STATES:
            key = f'n_{state_name}'
            if key not in self.results:
                continue
            state_arr = getattr(self, state_name)
            uids_in = auids[state_arr.values]
            self.results[key][ti] = (
                ppl.scale_flows(uids_in) if len(uids_in) > 0 else 0.0
            )

        # Denominator must be alive-masked: auids still includes agents who died
        # this step, since remove_dead runs after update_results.
        if 'prevalence' in self.results and 'n_infected' in self.results:
            n_alive_sw = ppl.scale_flows(ppl.alive.uids)
            self.results['prevalence'][ti] = (
                self.results['n_infected'][ti] / n_alive_sw
                if n_alive_sw > 0 else 0.0
            )

    def finalize_results(self):
        super().finalize_results()
        res = self.results
        res.cum_cancers[:] = np.cumsum(res.new_cancers)
        res.cum_cancer_deaths[:] = np.cumsum(res.new_cancer_deaths)
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

        Defensively clamps ``values`` to >= 0 before computing the fractional
        Bernoulli probability — a negative ``frac`` would crash the Bernoulli.
        """
        if len(uids) == 0:
            return np.zeros(0, dtype=int)
        values = np.maximum(values, 0.0)
        floor = np.floor(values)
        frac = values - floor
        dist.set(p=frac)
        bumps = dist.rvs(uids)
        return (floor + bumps).astype(int)

    def _roll_latency(self, female_uids):
        """Roll hpv_control_prob for female agents about to clear.

        Called from both clearance-scheduling branches in set_prognoses (precin-
        only clearance, and post-CIN clearance) so the roll applies to any
        clearance event, matching v2.2.6 semantics. Agents who hit are flagged
        to_latent=True; step_state redirects their scheduled clearance into
        latency instead of susceptibility. No-op when hpv_control_prob==0 (the
        default), keeping the feature a true no-op unless calibrated on.
        """
        if len(female_uids) == 0 or self.pars.hpv_control_prob == 0:
            return
        p = np.full(len(female_uids), self.pars.hpv_control_prob)
        self._latent_bern.set(p=p)
        latent_uids = self._latent_bern.filter(female_uids)
        if len(latent_uids):
            self.to_latent[latent_uids] = True

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

        first_uids = uids[self.ti_first_infection.isnan[uids]]
        self.ti_first_infection[first_uids] = ti

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = ti
        self.precin[uids] = True

        # Clear stale schedules from a prior infection.
        self.ti_clearance[uids] = np.nan
        self.ti_cin[uids] = np.nan
        self.ti_cancerous[uids] = np.nan
        self.ti_dead_cancer[uids] = np.nan
        self.to_latent[uids] = False
        self.latent[uids] = False

        female_all = self.sim.people.female[uids]
        # rel_sev is shared across genotypes via CrossImmunity, sampled on first touch.
        cross = self._cross_immunity_connector()
        if cross is not None:
            cross.ensure_rel_sev(uids)
            rel_sev_uids = cross.rel_sev[uids]
        else:
            rel_sev_uids = np.ones(len(uids), dtype=float)
        # HIV co-infection scales progression severity (gated no-op when no HIV).
        hivc = self._hiv_module()
        if hivc is not None:
            rel_sev_uids = rel_sev_uids * hivc.hiv_rel_sev[uids]
        sev_imm_uids = self.sev_imm[uids]

        # 1. Precin durations: females get sample * (1 - sev_imm), males get
        #    dur_inf_male with no immunity reduction. rel_sev applies separately.
        dur_precin = p.dur_precin.rvs(uids) * (1.0 - sev_imm_uids)
        if (~female_all).any():
            male_uids = uids[~female_all]
            dur_inf_male = p.dur_inf_male.rvs(male_uids)
            dur_precin[~female_all] = dur_inf_male

        # 2. P(CIN) per female. Dists return timesteps; cin_fn expects years.
        dt_yr = self.t.dt_year
        p_cin = compute_severity(dur_precin * dt_yr,
                                   rel_sev=rel_sev_uids, pars=p.cin_fn)
        self._cin_bern.set(p=p_cin)
        cin_draw = self._cin_bern.rvs(uids)
        cin_mask = cin_draw & female_all
        cin_uids = uids[cin_mask]
        nocin_uids = uids[~cin_mask]

        # 3. Branch A: clearance from precin. Females get the CRN-safe stochastic
        #    round (_randround), males np.ceil (which biases them a dt late).
        nocin_dur = dur_precin[~cin_mask]
        nocin_female = female_all[~cin_mask]
        rounded_dur = np.empty(len(nocin_dur), dtype=int)
        if nocin_female.any():
            rounded_dur[nocin_female] = self._randround(
                nocin_dur[nocin_female],
                nocin_uids[nocin_female],
                self._round_clear_precin_bern,
            )
        if (~nocin_female).any():
            rounded_dur[~nocin_female] = np.ceil(
                nocin_dur[~nocin_female]
            ).astype(int)
        self.ti_clearance[nocin_uids] = ti + rounded_dur
        self._roll_latency(nocin_uids[nocin_female])

        if len(cin_uids) == 0:
            return

        # 4. Branch B: progression to CIN.
        self.ti_cin[cin_uids] = ti + self._randround(
            dur_precin[cin_mask], cin_uids, self._round_cin_bern,
        )
        dur_cin = p.dur_cin.rvs(cin_uids)
        # age_risk multiplier on dur_cin above the threshold age.
        ages_at_cin = self.sim.people.age[cin_uids]
        age_mod = np.ones(len(cin_uids))
        age_mod[ages_at_cin >= p.age_risk['age']] = p.age_risk['risk']
        dur_cin = dur_cin * age_mod

        # 5. P(cancer) given dur_cin. sev_imm is not applied here, only rel_sev.
        rel_sev_cin = rel_sev_uids[cin_mask]
        p_cancer = compute_severity(dur_cin * dt_yr,
                                      rel_sev=rel_sev_cin, pars=p.cancer_fn)
        self._cancer_bern.set(p=p_cancer)
        cancer_draw = self._cancer_bern.rvs(cin_uids)
        cancer_uids = cin_uids[cancer_draw]
        nocancer_uids = cin_uids[~cancer_draw]

        # 5a. Clear after CIN (no cancer).
        self.ti_clearance[nocancer_uids] = (
            self.ti_cin[nocancer_uids] + self._randround(
                dur_cin[~cancer_draw], nocancer_uids, self._round_clear_cin_bern,
            )
        )
        self._roll_latency(nocancer_uids)  # already all-female (CIN branch is female-only)

        # 5b. Progression to cancer. Guard is local: step 6 must still run over
        #     all cin_uids even when no base agent drew cancer.
        if len(cancer_uids):
            self.ti_cancerous[cancer_uids] = (
                self.ti_cin[cancer_uids] + self._randround(
                    dur_cin[cancer_draw], cancer_uids, self._round_cancer_bern,
                )
            )
            dur_cancer = p.dur_cancer.rvs(cancer_uids)
            self.ti_dead_cancer[cancer_uids] = (
                self.ti_cancerous[cancer_uids] + self._randround(
                    dur_cancer, cancer_uids, self._round_dead_bern,
                )
            )

        # 6. Multiscale: grow real fine cancer agents.
        self._grow_fine_agents(cin_uids, cancer_uids, dur_cin, age_mod,
                               rel_sev_cin, sev_imm_uids[cin_mask], dt_yr)

    def _grow_fine_agents(self, cin_uids, cancer_uids, dur_cin, age_mod,
                          rel_sev_cin, sev_imm_cin, dt_yr):
        """Grow ratio-1 extra real fine cancer agents per CIN agent.

        The transforming base agents are shrunk to scale 1/ratio; for every CIN
        reacher, ratio-1 extra trajectories are drawn (age_risk-modified
        dur_cin; CIN-conditional rejection-sampled dur_precin), cancer is drawn
        for each extra, and one real fine agent is grown per extra-cancer
        success (full cross-genotype clone of the source, then this genotype's
        cancer-bound trajectory written). No-op at ms_agent_ratio<=1.

        Units: the extra-trajectory dists (``_extra_dur_precin``/``_extra_dur_cin``)
        are built unitless from the live dists' year-parameterized mean/std, so
        ``.rvs`` returns YEARS and ``compute_severity`` receives YEARS directly.
        Scheduling via ``_randround`` needs STEPS, so durations are divided by
        ``dt_yr`` first. ``p.dur_cancer.rvs`` returns STEPS already, so it is
        passed to ``_randround`` unscaled (matches the base path in set_prognoses).

        RNG: the extra draws come from dedicated dists registered in __init__ —
        SEPARATE streams from the live natural-history dists, so growing fine
        agents never perturbs the real agents' draws (keeps ratio==1 bit-identical
        and incidence flat across ratios). Drawn by size (non-CRN).
        """
        ratio = int(self.pars.ms_agent_ratio)
        n = len(cin_uids)
        if ratio <= 1 or n == 0:
            return
        p = self.pars
        ppl = self.sim.people
        ti = self.ti
        cancer_scale = 1.0 / ratio

        # Shrink base agents that drew their own cancer.
        if len(cancer_uids):
            ppl.scale[cancer_uids] = cancer_scale

        m = ratio - 1
        size = (n, m)
        amod = age_mod[:, None]
        rel = rel_sev_cin[:, None] * np.ones(size)
        sevimm = sev_imm_cin[:, None] * np.ones(size)

        # age_risk-modified extra dur_cin (years).
        extra_dur_cin = self._extra_dur_cin.rvs(size) * amod
        # CIN-conditional (length-biased) extra dur_precin: rejection-sample.
        extra_dur_precin = self._extra_dur_precin.rvs(size) * (1.0 - sevimm)
        pending = np.ones(size, dtype=bool)
        for _ in range(64):
            if not pending.any():
                break
            cinp = compute_severity(extra_dur_precin, rel_sev=rel, pars=p.cin_fn)
            passed = (self._extra_cin_unif.rvs(size) < cinp) & pending
            pending &= ~passed
            if pending.any():
                redraw = self._extra_dur_precin.rvs(int(pending.sum())) \
                    * (1.0 - sevimm[pending])
                extra_dur_precin[pending] = redraw

        # CIN -> cancer for every extra (all are CIN now).
        pcanc = compute_severity(extra_dur_cin, rel_sev=rel, pars=p.cancer_fn)
        extra_cancer = self._extra_cancer_unif.rvs(size) < pcanc
        # Existing fine agents never spawn more fine agents (level0 guard).
        not_fine = ~ppl.fine[cin_uids]
        extra_cancer &= not_fine[:, None]
        counts = extra_cancer.sum(axis=1)
        n_new = int(counts.sum())
        if n_new == 0:
            return

        # Broadcast source uids per success and grow.
        src_uids = ss.uids(np.repeat(np.asarray(cin_uids), counts))
        new_dur_precin = extra_dur_precin[extra_cancer]   # years
        new_dur_cin = extra_dur_cin[extra_cancer]         # years
        new_uids = ppl.grow(n_new)

        # Full cross-genotype clone; _clone_agents also resets lifecycle states.
        _clone_agents(self.sim, src_uids, new_uids)

        # Fine-agent identity.
        ppl.fine[new_uids] = True
        ppl.scale[new_uids] = cancer_scale

        # This genotype's fresh cancer-bound trajectory (overwrites cloned).
        self.susceptible[new_uids] = False
        self.infected[new_uids] = True
        self.precin[new_uids] = True
        self.cin[new_uids] = False
        self.cancerous[new_uids] = False
        self.ti_infected[new_uids] = ti
        self.ti_first_infection[new_uids] = ti
        self.ti_clearance[new_uids] = np.nan
        # Durations are in years; convert to steps via /dt_yr before rounding.
        ti_cin = ti + self._randround(new_dur_precin / dt_yr, new_uids,
                                      self._round_cin_bern)
        self.ti_cin[new_uids] = ti_cin
        ti_canc = ti_cin + self._randround(new_dur_cin / dt_yr, new_uids,
                                           self._round_cancer_bern)
        self.ti_cancerous[new_uids] = ti_canc
        dur_cancer = p.dur_cancer.rvs(new_uids)  # steps (module dist)
        self.ti_dead_cancer[new_uids] = ti_canc + self._randround(
            dur_cancer, new_uids, self._round_dead_bern)
        return

    def _cancel_other_genotype_progression_for(self, uids):
        """Cancel pending cancer progression in all other genotypes for these uids.

        When this module fires cin->cancerous for uids, all OTHER HPV
        modules in the sim must have their pending cancer progression
        cancelled and their state cleared for these agents. Without this,
        agents with multi-genotype CIN can be counted as multiple cancer
        cases (one per genotype) — biologically incorrect.
        """
        if len(uids) == 0:
            return
        for module in self.sim.diseases.values():
            if not isinstance(module, HPV) or module is self:
                continue
            module.ti_cancerous[uids] = np.nan
            module.ti_dead_cancer[uids] = np.nan
            # Clear both precin and cin so nothing re-promotes on a cancerous body.
            module.precin[uids] = False
            module.cin[uids] = False
            module.ti_cin[uids] = np.nan
            # These infections won't clear naturally; the agent is dying of cancer.
            module.ti_clearance[uids] = np.nan
            module.susceptible[uids] = False
            module.infected[uids] = False

    def step_state(self):
        """Advance agents through the natural-history compartment chain.

        Order matters: clearance fires first so a just-cleared agent isn't
        re-flipped by a forward transition at the same timestep.

          1. Clearance from precin or CIN (partial-immunity path)
          2. precin -> CIN
          3. CIN -> cancerous (stops transmitting)
          4. Cancer death (via people.request_death)
        """
        self.rel_sus[:] = 1.0  # reset so CrossImmunity/HIV multiply, not overwrite
        ti = self.ti

        # --- 1. Clearance (from precin OR CIN) — partial-immunity path ---
        # infected is mutually exclusive with cancerous, so it implies precin|cin.
        cleared = (self.infected & (self.ti_clearance <= ti)).uids
        if len(cleared):
            self.infected[cleared] = False
            self.precin[cleared] = False
            self.cin[cleared] = False

            # Agents redirected into latency leave precin/cin/infected but do not
            # become susceptible and gain no clearance-conferred immunity.
            to_latent_uids = cleared[self.to_latent[cleared]]
            to_susceptible_uids = cleared[~self.to_latent[cleared]]

            self.susceptible[to_susceptible_uids] = True

            if len(to_latent_uids):
                self.latent[to_latent_uids] = True
                self.ti_latent[to_latent_uids] = ti
                self.to_latent[to_latent_uids] = False

            # Females only. sero_prob gates the first clearance; repeats always
            # take the running max.
            female = self.sim.people.female
            f_cleared = to_susceptible_uids[female[to_susceptible_uids]]
            if len(f_cleared):
                has_prior_imm = self.nab_imm[f_cleared] > 0
                first_mask  = ~has_prior_imm
                first_uids  = f_cleared[first_mask]
                repeat_uids = f_cleared[has_prior_imm]

                p = self.pars
                nab_all  = p.imm_init.rvs(f_cleared)
                cell_all = p.cell_imm_init.rvs(f_cleared)

                # HIV co-infection reduces conferred immunity (gated no-op).
                # hiv_rel_imm is one dt stale: step_state runs before the connector.
                hivc = self._hiv_module()
                if hivc is not None:
                    imm_factor = hivc.hiv_rel_imm[f_cleared]
                    nab_all = nab_all * imm_factor
                    cell_all = cell_all * imm_factor

                if len(first_uids):
                    self._sero_bern.set(p=float(p.sero_prob))
                    seroconvert = self._sero_bern.rvs(first_uids)
                    # nab_imm is gated on seroconversion; cell_imm is not, so all
                    # first-clearance females get severity protection.
                    self.nab_imm[first_uids]  = seroconvert * nab_all[first_mask]
                    self.cell_imm[first_uids] = cell_all[first_mask]

                # Running max: an HIV-reduced increment can fail to boost immunity
                # but never erases what a prior clearance conferred.
                if len(repeat_uids):
                    self.nab_imm[repeat_uids] = np.maximum(
                        self.nab_imm[repeat_uids], nab_all[has_prior_imm])
                    self.cell_imm[repeat_uids] = np.maximum(
                        self.cell_imm[repeat_uids], cell_all[has_prior_imm])

        # --- 1b. Reactivation from latent (per-timestep hazard) ---
        # ti_latent < ti, not <=: no same-step clear-into-latent-then-reactivate.
        latent_uids = (self.latent & (self.ti_latent < ti)).uids
        if len(latent_uids):
            sev_imm_uids = self.sev_imm[latent_uids]
            hivc = self._hiv_module()
            if hivc is not None:
                rel_reactivation_uids = hivc.hiv_rel_reactivation[latent_uids]
            else:
                rel_reactivation_uids = np.ones(len(latent_uids))
            # Relative factors scale the rate via scale=, never the probability:
            # the rate -> probability conversion happens once, in .to_prob().
            combined_rel = (1.0 - sev_imm_uids) * rel_reactivation_uids
            p_react = self.pars.hpv_reactivation.to_prob(self.t.dt, scale=combined_rel)
            self._reactivation_bern.set(p=p_react)
            reactivated = self._reactivation_bern.filter(latent_uids)
            if len(reactivated):
                self.latent[reactivated] = False
                self.ti_reactivation[reactivated] = ti
                # Fresh trajectory; immunity carries over. set_prognoses bumps
                # new_infections unconditionally, so back it out here.
                ppl = self.sim.people
                n_react = ppl.scale_flows(reactivated)
                self.set_prognoses(reactivated)
                self.results.new_infections[ti] -= n_react
                self.results.new_reactivations[ti] = n_react

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
            self.ti_clearance[to_cancerous] = np.nan  # cancer supersedes clearance
            ppl = self.sim.people
            ages_at_cancer = ppl.age[to_cancerous]
            w = ppl.scale[to_cancerous]
            self.results.new_cancers[ti] = ppl.scale_flows(to_cancerous)
            self.results.sum_age_at_cancer[ti] = float((ages_at_cancer * w).sum())
            self._cancel_other_genotype_progression_for(to_cancerous)

        # --- 4. Cancer death (routed through starsim's people death pipeline) ---
        to_dead = (self.cancerous & (self.ti_dead_cancer <= ti)).uids
        if len(to_dead):
            # +dt_yr: step_state runs before ages advance, so this is age at step end.
            dt_yr = self.t.dt_year
            ppl = self.sim.people
            ages_at_death = ppl.age[to_dead] + dt_yr
            w = ppl.scale[to_dead]
            self.results.new_cancer_deaths[ti] = ppl.scale_flows(to_dead)
            self.results.sum_age_at_cancer_death[ti] = float((ages_at_death * w).sum())
            self.sim.people.request_death(to_dead)

    def step_die(self, uids):
        """ Set states to False for the newly-dead. 
        IMPORTANT: if you add any new states, remember to add them here!
        """
        super().step_die(uids)
        self.precin[uids] = False
        self.cin[uids] = False
        self.cancerous[uids] = False
        self.infected[uids] = False
        self.susceptible[uids] = False
        self.latent[uids] = False
        self.to_latent[uids] = False
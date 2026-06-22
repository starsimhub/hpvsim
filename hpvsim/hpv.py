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

from .parameters import genotype_aliases, get_genotype_pars
from .seeding import _make_init_prev_fn
from .utils import compute_severity


_KNOWN_GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')


def multiscale_fine_for(sim, uids):
    """Boolean array aligned with `uids`: True where the agent is a fine
    multiscale agent in ANY HPV genotype module. `multiscale_fine` is
    per-module, so this unions across modules. Duck-typed (hasattr) to avoid
    import cycles with consumers like the network/demographics."""
    fine = np.zeros(len(uids), dtype=bool)
    for m in sim.diseases.values():
        if hasattr(m, 'multiscale_fine'):
            fine |= np.asarray(m.multiscale_fine[uids])
    return fine


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
        # Pull natural-history defaults from GenotypePars so there's a single
        # source of truth per genotype.
        gpars = get_genotype_pars(genotype)
        self.define_pars(
            init_prev=ss.bernoulli(p=_make_init_prev_fn(genotype)),
            # Sex-directional beta: p1→p2 = female→male (transf2m); p2→p1 =
            # male→female (transm2f). SexualNetwork places females in p1 and
            # males in p2 (see hpvsim/network.py:185). validate_beta accepts
            # this dict shape natively (ss.Infection.validate_beta).
            beta={
                'sexualnetwork': [
                    gpars.beta * gpars.rel_beta * gpars.transf2m,
                    gpars.beta * gpars.rel_beta * gpars.transm2f,
                ],
            },
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
            # Multiscale: number of fine cancer agents per coarse agent at the
            # CIN->cancer decision. 1 = feature off (no splitting).
            ms_agent_ratio=1,
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
            # No-op state hook to support products_dx.csv's `latent` rows. Real
            # reactivation natural history (set_prognoses branch + step_state
            # reactivation) is a post-M06 follow-on; for now nothing populates this.
            ss.BoolState('latent', label='Latent infection'),
            ss.FloatArr('ti_cin', label='Time of CIN onset'),
            ss.FloatArr('ti_cancerous', label='Time of invasive cancer onset'),
            ss.FloatArr('ti_dead_cancer', label='Time of cancer-caused death'),
            # Severity immunity, accumulated as max-of-beta-samples on each
            # clearance. Shortens future dur_precin via (1 - sev_imm) factor.
            ss.FloatArr('sev_imm', label='Severity immunity', default=0.0),
            # Clearance-conferred humoral/cell-mediated immunity. Bumped on
            # clearance; read by CrossImmunity to derive per-target rel_sus /
            # sev_imm via the cross-protection matrix. Does NOT include vaccine
            # immunity — that lives in vax_imm below so it bypasses the matrix.
            ss.FloatArr('nab_imm', label='Humoral immunity (clearance-conferred, source genotype)', default=0.0),
            ss.FloatArr('cell_imm', label='Cell-mediated immunity (source genotype)', default=0.0),
            # Vaccine-conferred immunity per target genotype. Written by
            # hpv.vx.administer(); applied directly per genotype in
            # CrossImmunity.step() WITHOUT flowing through the cross-immunity
            # matrix. The CSV's per-genotype rel_imm values are the complete
            # vaccine cross-protection profile. Combining formula (independent
            # protection paths):
            #   rel_sus = (1 - sus_imm_from_nab) * (1 - vax_imm) * (1 - txvx_imm)
            # See CrossImmunity.step() for the canonical implementation.
            ss.FloatArr('vax_imm', label='Vaccine-conferred immunity (this genotype)', default=0.0),
            ss.FloatArr(
                'txvx_imm',
                label='Therapeutic-vaccine-conferred immunity (this genotype)',
                default=0.0,
            ),
            # v2 level0/level1 tag: True for fine agents spawned by multiscale
            # splitting, so a fine agent is never re-split by THIS genotype
            # module. Per-genotype (each HPV module owns its own array, accessed
            # as self.multiscale_fine); registered here so people.grow() extends
            # it. Cross-genotype compounding is guarded/verified in the
            # multi-genotype test (plan Task 5).
            ss.BoolArr('multiscale_fine', default=False),
        )
        # Dedicated ss.random dist for the multiscale-split gate draws (CIN
        # gate and cancer gate in _multiscale_split). Using ss.random rather
        # than np.random.random ties reproducibility to Starsim's seeded RNG
        # stream rather than the global NumPy seed, which is more robust if a
        # future Starsim version stops reseeding np.random. One dist drawn
        # twice per step is fine — ss dists auto-jump between calls.
        self._ms_gate = ss.random()
        # Per-call Bernoullis whose p is overwritten via .set(p=...) at each
        # use site (placeholder p values below). _cin_bern / _cancer_bern serve
        # the core set_prognoses path ONLY; the multiscale split path draws its
        # own (CRN-free) gates via _ms_gate and does not use these.
        self._cin_bern = ss.bernoulli(p=0.5)
        self._cancer_bern = ss.bernoulli(p=0.5)
        self._sero_bern = ss.bernoulli(p=0.5)
        # Per-decision Bernoullis for CRN-safe stochastic rounding of
        # event durations. Each ``ti_<event>`` schedule gets its own dist so
        # the per-uid round-up draw is independent across decisions.
        self._round_clear_precin_bern = ss.bernoulli(p=0.5)
        self._round_cin_bern = ss.bernoulli(p=0.5)
        self._round_clear_cin_bern = ss.bernoulli(p=0.5)
        self._round_cancer_bern = ss.bernoulli(p=0.5)
        self._round_dead_bern = ss.bernoulli(p=0.5)
        return

    def init_post(self):
        # Ensure CrossImmunity has sampled rel_sev for the starting population
        # BEFORE super().init_post() runs init_prev seeding (which triggers
        # set_prognoses, which reads rel_sev). First HPV module to init_post
        # triggers the sampling; subsequent modules are no-ops because
        # rel_sev_sampled is already True for everyone.
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

    def init_results(self):
        """Per-step Results emitted from ``step_state``.

        ``new_cancers`` / ``new_cancer_deaths`` are realized-event counters
        (the cin -> cancerous and cancerous -> dead transitions);
        ``cum_*`` are populated as cumulative sums in ``finalize_results``.
        ``sum_age_at_*`` are per-step accumulators; mean age = ``sum / count``.
        """
        super().init_results()
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
        )
        return

    def finalize_results(self):
        super().finalize_results()
        res = self.results
        res.cum_cancers[:] = np.cumsum(res.new_cancers)
        res.cum_cancer_deaths[:] = np.cumsum(res.new_cancer_deaths)
        return

    def update_results(self):
        """Tally per-step results, then people-space-correct the infection
        results for multiscale fine agents.

        ``ss.Infection`` records ``new_infections`` and ``n_infected`` (hence
        ``prevalence = n_infected / n_alive``) as RAW agent counts. Multiscale
        fine cancer agents are ``infected=True`` through their CIN window and
        carry scale ``1/ratio``; the base counts each as a full body, inflating
        these per-module results (measured ~+82% prevalence, ~+114% n_infected
        at ratio=12 — the cross-genotype ``HPVTotal`` aggregator is already
        scale-weighted and stays correct). Recompute the affected results in
        people-space:

          - ``new_infections``: drop fine agents entirely — a fine agent is a
            sub-resolution of an ALREADY-counted source infection (it copies the
            source's ``ti_infected``, == its spawn step), not a new transmission
            event.
          - ``n_infected`` / ``prevalence``: scale-weight, so a fine agent at
            ``1/ratio`` contributes its weight, not a full body.

        Bit-identical to the base at ms_agent_ratio=1: with no fine agents the
        early return leaves the base raw counts untouched (and uniform scale=1
        would make the recomputation equal them anyway).

        SCOPE: this corrects the per-module HPV epidemiology (every n_<state>,
        prevalence, new_infections); the cross-genotype ``HPVTotal`` aggregator
        is already scale-weighted. NOT corrected — and a known limitation of the
        agent-overlay design — are FRAMEWORK-level demographic body counts that
        tally fine agents as whole bodies: sim ``n_alive`` (~+15% at ratio=12),
        ``ss.Deaths`` all-cause deaths (~+17-25%), and ``AgeMigration`` emigrant
        counts. Cancer-specific deaths (``new_cancer_deaths``) are scale-weighted
        and correct. For people-space epidemiology use the HPV/HPVTotal results;
        fully fixing the demographic counts would need framework-level
        scale-aware counting (a Level0Deaths and a scale-weighted n_alive).
        """
        super().update_results()
        ti = self.ti
        ppl = self.sim.people
        auids = ppl.auids
        fine = np.asarray(self.multiscale_fine[auids])
        if not fine.any():
            return  # no multiscale agents -> base raw counts are correct
        res = self.results
        scale = np.asarray(ppl.scale[auids])
        # new_infections: fine agents are not new transmission events.
        newly = np.round(np.asarray(self.ti_infected[auids])) == ti
        res.new_infections[ti] -= float(np.count_nonzero(newly & fine))
        # Every per-state body count (n_susceptible/n_infected/n_precin/n_cin/
        # n_cancerous) is a RAW count in the base; a fine agent (scale 1/ratio)
        # is counted as a full body, grossly inflating the compartments fine
        # agents occupy (measured n_cancerous +1009%, n_cin +630% at ratio=12).
        # Recompute each as the scale-weighted (people-space) sum. The cross-
        # genotype HPVTotal aggregator is already scale-weighted and unaffected.
        for state in ('susceptible', 'infected', 'precin', 'cin', 'cancerous'):
            key = 'n_' + state
            if key in res:
                vals = np.asarray(getattr(self, state)[auids])
                res[key][ti] = float((scale * vals).sum())
        alive_scale = float(scale.sum())
        res.prevalence[ti] = (res.n_infected[ti] / alive_scale) if alive_scale > 0 else 0.0
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
        # rel_sev is shared across HPV modules via the CrossImmunity connector;
        # ensure it's sampled for these uids (lazy first-touch sampling).
        cross = self._cross_immunity_connector()
        if cross is not None:
            cross.ensure_rel_sev(uids)
            rel_sev_uids = cross.rel_sev[uids]
        else:
            rel_sev_uids = np.ones(len(uids), dtype=float)
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
        p_cin = compute_severity(dur_precin * dt_yr,
                                   rel_sev=rel_sev_uids, pars=p.cin_fn)
        self._cin_bern.set(p=p_cin)
        cin_draw = self._cin_bern.rvs(uids)
        cin_mask = cin_draw & female_all
        cin_uids = uids[cin_mask]
        nocin_uids = uids[~cin_mask]

        # Schedule events with CRN-safe stochastic rounding (``_randround``)
        # for FEMALE events and ``np.ceil`` for MALE clearance.
        # Without rounding, fractional ti_<event> behaves like np.ceil at the
        # ``<= ti`` check — fine for males, but biases female timings.
        # Bias direction: np.ceil pushes male clearance to the next integer
        # step, so males clear up to one dt later than the fractional duration.

        # 3. Branch A: clearance from precin. Split male / female paths so
        #    males get np.ceil and females get the CRN-safe stochastic round.
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

        if len(cin_uids) == 0:
            return

        # 4. Branch B: progression to CIN.
        self.ti_cin[cin_uids] = ti + self._randround(
            dur_precin[cin_mask], cin_uids, self._round_cin_bern,
        )
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

        # 5b. Progression to cancer (ORIGINAL coarse/fine agents). Scheduled
        #     here exactly as pre-feature; the multiscale split below only
        #     shrinks scale + spawns ADDITIONAL fine cancer-drawers.
        if len(cancer_uids) > 0:
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

        # 6. Multiscale agent split at the CIN->cancer decision. No-op at
        #    ms_agent_ratio<=1 (early return inside). Resolves the rare cancer
        #    PATHWAY at ratio-finer granularity so the cancer/CIN2+ event-age
        #    distributions (methods Fig 5) carry ratio-x more samples. Coarse
        #    agents whose OWN cancer draw fired are shrunk to 1/ratio; ratio-1
        #    extra sub-resolutions per coarse agent each RESAMPLE their own
        #    precin/CIN durations (mirrors v2 People.set_severity, legacy
        #    people.py:280-369) and spawn a fine cancer agent at 1/ratio when
        #    they progress to cancer.
        self._multiscale_split(cin_uids, cancer_draw, rel_sev_cin, age_mod, dt_yr)
        return

    def _multiscale_split(self, cin_uids, cancer_draw, rel_sev_cin, age_mod, dt_yr):
        """Resolve the CIN->cancer pathway at ratio-finer granularity (methods Fig 5):
        more cancer/CIN2+ event-age samples without biasing the distributions.

        Each coarse CIN agent represents ``ratio`` sub-resolutions. The original
        (sub-resolution 0) keeps its scheduled timeline; if it drew cancer it is
        shrunk to ``1/ratio`` scale and stays level0 (a counted, transmitting body
        until cancer onset), else it keeps full scale and may re-split later. The
        other ``ratio-1`` sub-resolutions each resample their own ``dur_precin`` and
        ``dur_cin``; those that progress to cancer are grown as ``1/ratio`` fine
        agents. Only cancer successes are kept, so their ``dur_cin`` is length-biased
        and cancer-onset ages match single scale. The precin->CIN gate is not
        re-rolled (these agents already reached CIN). Causal-infection age is shared
        with the source. No-op at ms_agent_ratio<=1.

        Args align element-wise with ``cin_uids``: ``cancer_draw`` (the agent's own
        cancer draw), ``rel_sev_cin``, ``age_mod``.
        """
        ratio = int(self.pars.ms_agent_ratio)
        if ratio <= 1 or len(cin_uids) == 0:
            return

        ppl = self.sim.people
        p = self.pars
        ti = self.ti

        # Split only coarse CIN agents: not already fine (in any genotype) and at
        # full scale. A fine/shrunk agent must never re-split.
        coarse = (~multiscale_fine_for(self.sim, cin_uids)) & (ppl.scale[cin_uids] >= 1.0 - 1e-9)
        if not coarse.any():
            return
        coarse_uids = cin_uids[coarse]
        coarse_cancer = cancer_draw[coarse]

        # Pre-shrink snapshot so extras weight at source_scale/ratio.
        coarse_scale = ppl.scale[coarse_uids].copy()

        # Sub-resolution 0 (the original): shrink cancer-drawers to 1/ratio. Stays
        # level0 (not multiscale_fine) — a counted, transmitting body until cancer
        # onset. Non-cancer originals keep full scale (re-splittable later).
        cancer_orig_uids = coarse_uids[coarse_cancer]
        if len(cancer_orig_uids) > 0:
            ppl.scale[cancer_orig_uids] = coarse_scale[coarse_cancer] / ratio

        # The other ratio-1 sub-resolutions: draw trajectories as size-arrays and
        # grow only the cancer successes. Gates draw from self._ms_gate (an
        # ss.random dist), once per gate per step.
        m = ratio - 1
        n_block = len(coarse_uids) * m
        src = ss.uids(np.repeat(coarse_uids, m))
        rel_sev_block = np.repeat(rel_sev_cin[coarse], m)
        age_mod_block = np.repeat(age_mod[coarse], m)
        scale_block = np.repeat(coarse_scale, m)
        sev_imm_block = self.sev_imm[src]
        src_ti_cin = self.ti_cin[src]

        # CIN-conditional precin (length-biased): fresh dur_precin + CIN gate;
        # passers take the independent CIN2+ onset, the rest fall back to the
        # source's ti_cin.
        dur_precin_block = p.dur_precin.rvs(n_block) * (1.0 - sev_imm_block)
        p_cin_block = compute_severity(dur_precin_block * dt_yr,
                                       rel_sev=rel_sev_block, pars=p.cin_fn)
        cin_pass = self._ms_gate.rvs(n_block) < p_cin_block
        # dur_precin is in timesteps: ti_cin = ti + round(dur_precin).
        ti_cin_block = src_ti_cin.astype(float)
        ti_cin_block[cin_pass] = ti + sc.randround(dur_precin_block[cin_pass])

        # CIN -> cancer: resample dur_cin, draw cancer at f(dur_cin); keeping only
        # successes makes the kept dur_cin length-biased.
        dur_cin_block = p.dur_cin.rvs(n_block) * age_mod_block
        p_cancer_block = compute_severity(dur_cin_block * dt_yr,
                                          rel_sev=rel_sev_block, pars=p.cancer_fn)
        cancer_block = self._ms_gate.rvs(n_block) < p_cancer_block
        if not cancer_block.any():
            return

        # Grow ONLY the cancer successes as fine agents at source_scale/ratio.
        n_new = int(cancer_block.sum())
        new = ss.uids(ppl.grow(n_new))
        new_src = src[cancer_block]
        ppl.age[new] = ppl.age[new_src]
        ppl.female[new] = ppl.female[new_src]
        for state in ('ti_infected', 'ti_first_infection', 'rel_sus', 'rel_trans',
                      'sev_imm', 'nab_imm', 'cell_imm', 'vax_imm', 'txvx_imm'):
            getattr(self, state)[new] = getattr(self, state)[new_src]
        self.infected[new] = True
        self.susceptible[new] = False
        # Start in precin so screening detection matches the agent's stage;
        # step_state promotes non-passers (ti_cin <= ti) to cin this same step.
        self.precin[new] = True
        self.cin[new] = False
        self.multiscale_fine[new] = True  # network-excluded; never re-split
        ppl.scale[new] = scale_block[cancer_block] / ratio
        self.ti_cin[new] = ti_cin_block[cancer_block]
        new_dur_cin = dur_cin_block[cancer_block]
        # ti_cancerous scheduled from ti_cin so step_state fires it when precin->cin fires.
        self.ti_cancerous[new] = self.ti_cin[new] + sc.randround(new_dur_cin)
        dur_cancer_new = p.dur_cancer.rvs(n_new)
        self.ti_dead_cancer[new] = self.ti_cancerous[new] + sc.randround(dur_cancer_new)
        return

    def _cancel_other_genotype_progression_for(self, uids):
        """Mirror v2's check_cancer cross-genotype cancellation.

        When this module fires cin->cancerous for uids, all OTHER HPV
        modules in the sim must have their pending cancer progression
        cancelled and their state cleared for these agents. Without this,
        agents with multi-genotype CIN can be counted as multiple cancer
        cases (one per genotype) — biologically incorrect.

        Mirrors hpvsim/_v2_legacy/people.py:565-595 (check_cancer).
        """
        if len(uids) == 0:
            return
        for module in self.sim.diseases.values():
            if not isinstance(module, HPV) or module is self:
                continue
            # Cancel pending cancer progression in this other genotype
            module.ti_cancerous[uids] = np.nan
            module.ti_dead_cancer[uids] = np.nan
            # Clear all dysplasia state: v2 sets cin[:, inds] = False across
            # all genotypes; v3 splits CIN into precin + cin so we clear both
            # (and their scheduled transition times) to keep these agents from
            # re-promoting precin -> cin on the now-cancerous body.
            module.precin[uids] = False
            module.cin[uids] = False
            module.ti_cin[uids] = np.nan
            # Clear pending clearance (these infections won't clear naturally;
            # the agent is dying of cancer instead)
            module.ti_clearance[uids] = np.nan
            # Agent is no longer susceptible / infected with this genotype either
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
        # rel_sev for births/immigrants is sampled lazily by the
        # CrossImmunity connector's step() before any HPV step_infect runs,
        # and again on first-touch in set_prognoses; no per-module work needed.
        ti = self.ti

        # --- 1. Clearance (from precin OR CIN) — partial-immunity path ---
        # Returns agent to susceptible=True. nab_imm and cell_imm accumulate
        # the running max of per-agent beta samples; the CrossImmunity connector
        # reads them next step to derive rel_sus and sev_imm. ``infected`` is
        # mutually exclusive with ``cancerous`` (step 3 toggles them), so it
        # implies precin|cin.
        cleared = (self.infected & (self.ti_clearance <= ti)).uids
        if len(cleared):
            self.infected[cleared] = False
            self.susceptible[cleared] = True
            self.precin[cleared] = False
            self.cin[cleared] = False

            # Only update post-clearance immunity for females; males clear
            # without seroconverting. First-clearance immunity is gated on
            # sero_prob; non-seroconverters keep nab_imm/cell_imm = 0 and are
            # fully reinfectible on next exposure. Repeat clearances always
            # update via running max (sero_prob only gates the first event).
            female = self.sim.people.female
            f_cleared = cleared[female[cleared]]
            if len(f_cleared):
                has_prior_imm = self.nab_imm[f_cleared] > 0
                first_mask  = ~has_prior_imm
                first_uids  = f_cleared[first_mask]
                repeat_uids = f_cleared[has_prior_imm]

                p = self.pars
                nab_all  = p.imm_init.rvs(f_cleared)
                cell_all = p.cell_imm_init.rvs(f_cleared)

                if len(first_uids):
                    self._sero_bern.set(p=float(p.sero_prob))
                    seroconvert = self._sero_bern.rvs(first_uids)
                    # nab_imm (humoral) is gated on seroconversion: non-
                    # seroconverters keep 0 nab and remain fully reinfectible.
                    # cell_imm (cell-mediated severity) is NOT gated — all
                    # first-clearance females get severity protection,
                    # regardless of whether they seroconverted. Without this,
                    # non-seroconverters get no dur_precin reduction on
                    # reinfection, inflating transmission for low-sero_prob
                    # genotypes (hpv18=0.56, hi5/ohr=0.60).
                    self.nab_imm[first_uids]  = seroconvert * nab_all[first_mask]
                    self.cell_imm[first_uids] = cell_all[first_mask]

                if len(repeat_uids):
                    self.nab_imm[repeat_uids] = np.maximum(
                        self.nab_imm[repeat_uids], nab_all[has_prior_imm])
                    self.cell_imm[repeat_uids] = np.maximum(
                        self.cell_imm[repeat_uids], cell_all[has_prior_imm])

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
            scale = np.asarray(self.sim.people.scale[to_cancerous])
            ages_at_cancer = np.asarray(self.sim.people.age[to_cancerous])
            self.results.new_cancers[ti] = float(scale.sum())
            self.results.sum_age_at_cancer[ti] = float((ages_at_cancer * scale).sum())
            self._cancel_other_genotype_progression_for(to_cancerous)

        # --- 4. Cancer death (routed through starsim's people death pipeline) ---
        to_dead = (self.cancerous & (self.ti_dead_cancer <= ti)).uids
        if len(to_dead):
            # +dt to align with v2's convention: v2's check_cancer_deaths
            # fires in update_states_pre AFTER increment_age, so v2 records
            # ages_at_cancer_death = initial + (T+1)*dt at step T. v3's
            # step_state fires BEFORE finish_step's age advance, so reading
            # sim.people.age here gives initial + T*dt — one step (dt yr)
            # lower. Adding dt_yr brings the recorded value into v2's
            # convention so the parity gate compares apples-to-apples.
            # Revert by removing the +dt_yr if the underlying step-ordering
            # convention is harmonised in a future Starsim version.
            dt_yr = float(self.t.dt.years if hasattr(self.t.dt, 'years') else self.t.dt)
            scale_d = np.asarray(self.sim.people.scale[to_dead])
            ages_at_death = np.asarray(self.sim.people.age[to_dead]) + dt_yr
            self.results.new_cancer_deaths[ti] = float(scale_d.sum())
            self.results.sum_age_at_cancer_death[ti] = float((ages_at_death * scale_d).sum())
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
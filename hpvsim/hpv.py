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
            # "target genotype": vax_imm[g] is protection AGAINST genotype g,
            # already resolved per target (the CSV's rel_imm[g] is applied
            # directly). Contrast nab_imm/cell_imm above, which are "source"
            # quantities — immunity conferred BY clearing this genotype, which
            # CrossImmunity then matrix-multiplies to derive protection against
            # OTHER (target) genotypes.
            ss.FloatArr('vax_imm', label='Vaccine-conferred immunity (against this/target genotype)', default=0.0),
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
        # Per-call Bernoullis whose p is overwritten via .set(p=...) at each
        # use site (placeholder p values below).
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
        """Scale-weight the per-step infection tally (multiscale lever a).

        ``ss.Infection.update_results`` records ``new_infections`` as a RAW
        count ``np.count_nonzero(round(ti_infected) == ti)`` — so any agent
        carrying a sub-unit ``people.scale`` (under ``ms_agent_ratio>1`` an
        agent that resolved part of its mass to cancer is shrunk to a fractional
        scale) would be tallied as a FULL infection if it gets (re)infected.
        That over-counts new infections at ``ms_agent_ratio>1`` and breaks
        people-space equivalence with a single-scale run. Here we recompute the
        per-step tally as the SCALE SUM over
        the agents newly infected this step, mirroring the scale-weighting of
        the cancer tallies (Tasks 2-3). ``new_infections`` is ``scale=True`` so
        the population ``pop_scale`` is applied later in ``sim.finalize_results``
        exactly as for the unweighted base; we only replace the within-sim count
        with its scale-weighted equivalent. At ``ms_agent_ratio==1`` every scale
        is 1.0 so this reproduces the base count bit-for-bit.

        ``finalize_results`` (base) recomputes ``cum_infections`` from this
        corrected ``new_infections`` series, so cumulative infections inherit
        the fix.
        """
        # Read the seed-case count BEFORE super(): the base pops
        # ``_n_initial_cases`` off pars at ti==0, so it is unavailable after.
        ti = self.ti
        n_initial = float(self.pars.get('_n_initial_cases', 0) or 0) if ti == 0 else 0.0
        super().update_results()  # base sets new_infections (raw) + prevalence
        res = self.results
        ppl = self.sim.people
        newly = (np.round(np.asarray(self.ti_infected[ppl.auids])) == ti)
        scale = np.asarray(ppl.scale[ppl.auids])
        n_infections = float((scale * newly).sum())
        if ti == 0:
            # Mirror the base: remove the seed cases (set_prognoses(sources=-1)
            # at init on full-scale agents, so the scale-weighted count equals
            # the raw count). Subtract the same count the base would have.
            n_infections -= n_initial
        res.new_infections[ti] = n_infections
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
        #    event at ratio-finer granularity: coarse agents whose OWN cancer
        #    draw fired are shrunk to 1/ratio, and ratio-1 INDEPENDENT extra
        #    cancer draws per coarse agent each spawn a fresh fine cancer agent
        #    (also at 1/ratio) when they fire. Non-cancer coarse agents are left
        #    untouched at full scale so later-life reinfection->cancer episodes
        #    stay fully weighted. Mirrors v2 People.set_severity (legacy
        #    people.py:280-369).
        self._multiscale_split(cin_uids, cancer_draw, p_cancer, age_mod, dur_cin, dt_yr)
        return

    def _multiscale_split(self, cin_uids, cancer_draw, p_cancer, age_mod, dur_cin, dt_yr):
        """Resolve the rare CIN->cancer event at ratio-finer granularity by
        BINOMIAL fractional weighting of the ORIGINAL agent (no population growth).

        Each COARSE CIN agent (relative scale 1.0) is past the CIN gate with a
        known cancer probability ``p_cancer`` (from its own ``dur_cin``). At
        ``ms_agent_ratio = N`` the agent stands in for ``N`` people-space
        individuals at this decision. Instead of resolving the single weight-1
        cancer Bernoulli, we resolve ``N`` INDEPENDENT cancer draws at the same
        ``p_cancer`` and let the agent carry the FRACTION that progress:

          - The agent's OWN cancer draw (already taken in ``set_prognoses`` and
            scheduled by the caller) is one of the ``N`` draws. The other
            ``N - 1`` draws are resolved here as ``k_extra ~ Binomial(N-1,
            p_cancer)``. The total successes ``k = own + k_extra`` (out of ``N``).
          - If ``k > 0`` the agent BECOMES (or stays) a cancer agent carrying
            scale ``orig_scale * k / N`` — the fraction of its ``N`` people-space
            individuals that progress to cancer. If ``k == 0`` it clears.

        Why this conserves cancer mass AND reduces variance, with NO extra agents:

          - Expectation: ``E[k/N] = p_cancer`` exactly, so the expected cancer
            mass per decision equals the single-scale ``p_cancer`` — unbiased.
          - Variance: the realized fraction ``k/N`` has variance
            ``p(1-p)/N`` vs the single Bernoulli's ``p(1-p)`` — an ``N``-fold
            reduction in the rare-event sampling noise (the whole point of
            multiscale). This is achieved on the ORIGINAL agent, so the
            population size never changes.
          - No transmission perturbation: an EARLIER design grew ``N-1``
            placeholder agents per decision to draw the extra cancers on fresh
            CRN slots, then removed the non-cancer ones. Growing/removing agents
            mid-run shifts starsim's slot-based RNG for every subsequent network
            pairing and transmission draw, which systematically DEPRESSED
            transmission (measured ~-34% cancers / -40% prevalence at ratio=12).
            Resolving the extra draws as a binomial on the existing agent adds
            zero agents, so transmission is left essentially unperturbed
            (measured prevalence within a few percent of single-scale) while
            keeping cancer mass conserved. Mirrors the variance-reduction intent
            of v2 People.set_severity (legacy people.py:280-369) without v2's
            grow-extras (which over-count repeat-decider episodes as independent
            people, ignoring the once-only competing-risk nature of cancer).

        ``cancer_draw``, ``p_cancer`` and ``age_mod`` align element-wise with
        ``cin_uids``. At ``ms_agent_ratio == 1`` this is a no-op (early return),
        so single-scale runs stay bit-identical.
        """
        ratio = int(self.pars.ms_agent_ratio)
        if ratio <= 1 or len(cin_uids) == 0:
            return

        ppl = self.sim.people
        p = self.pars

        # Only refine COARSE (full-scale) CIN agents; an already-shrunk agent
        # (a previous decision left it at < 1.0) must not be re-refined (would
        # compound the scale shrink). Use the per-agent SCALE as the test — this
        # naturally guards prior cancer-fraction agents without needing a marker.
        agent_scale = np.asarray(ppl.scale[cin_uids])
        coarse = agent_scale >= (1.0 - 1e-9)
        if not coarse.any():
            return
        coarse_uids = cin_uids[coarse]
        coarse_cancer = np.asarray(cancer_draw)[coarse].astype(int)
        coarse_scale = agent_scale[coarse]
        p_coarse = np.asarray(p_cancer)[coarse]
        # The agent's OWN (age-modified) dur_cin, aligned to coarse_uids. Cancer
        # progressors are length-biased (long dur_cin -> high p_cancer), so the
        # reconciliation branches below MUST reuse this per-agent value rather
        # than resampling unconditionally from p.dur_cin (which gives far-too-
        # short durations -> cancer onset too young; distorts the by-age dist).
        dur_cin_coarse = np.asarray(dur_cin)[coarse]

        # k_extra ~ Binomial(N-1, p_cancer) — the successes among the OTHER N-1
        # sub-agents this coarse agent stands for. Drawn from a deterministic
        # per-(seed, step, genotype) numpy Generator: reproducible (same seed ->
        # same result, satisfying the reproducibility test) and independent of
        # the slot-based CRN streams used for transmission/timeline draws, so it
        # cannot perturb them. Each call gets a distinct stream via ti+genotype.
        seed = (int(self.sim.pars.rand_seed) * 2654435761
                + int(self.ti) * 40503
                + (abs(hash(self.genotype)) % 99991)) & 0x7FFFFFFF
        rng = np.random.default_rng(seed)
        k = coarse_cancer + rng.binomial(ratio - 1, p_coarse)
        is_cancer = k > 0

        # Cancer agents carry scale = orig_scale * k / N (the progressing
        # fraction of their N people-space individuals). The shrink is applied
        # to ALL k>0 agents, including those whose own draw fired (k>=1 already).
        cancer_now = coarse_uids[is_cancer]
        if len(cancer_now) > 0:
            ppl.scale[cancer_now] = coarse_scale[is_cancer] * (k[is_cancer] / ratio)

        # Reconcile scheduling against each agent's OWN draw (the caller already
        # scheduled cancer for own==1 and clearance for own==0):
        #   * newly_cancer: own draw was 0 (clearance scheduled) but k_extra>0,
        #     so it now progresses to cancer — cancel clearance, schedule the
        #     cancer timeline (resampled dur_cin->ti_cancerous->ti_dead_cancer).
        #   * still_clear: own draw was 1 (cancer scheduled) but k==0, so it now
        #     clears — cancel the cancer timeline, schedule clearance.
        # These resamplings use starsim distributions on the REAL agent uids, so
        # they remain CRN-faithful; no new population slots are created.
        newly_cancer = coarse_uids[is_cancer & (coarse_cancer == 0)]
        if len(newly_cancer) > 0:
            sel = is_cancer & (coarse_cancer == 0)
            self.ti_clearance[newly_cancer] = np.nan
            # Reuse the agent's OWN length-biased dur_cin (already age-modified)
            # — NOT an unconditional resample — so rescued cancers keep the
            # same age-at-cancer distribution as own-draw cancers.
            dur_cin_own = dur_cin_coarse[sel]
            self.ti_cancerous[newly_cancer] = (
                self.ti_cin[newly_cancer] + self._randround(
                    dur_cin_own, newly_cancer, self._round_cancer_bern,
                )
            )
            dur_cancer_new = p.dur_cancer.rvs(newly_cancer)
            self.ti_dead_cancer[newly_cancer] = (
                self.ti_cancerous[newly_cancer] + self._randround(
                    dur_cancer_new, newly_cancer, self._round_dead_bern,
                )
            )

        still_clear = coarse_uids[(~is_cancer) & (coarse_cancer == 1)]
        if len(still_clear) > 0:
            sel2 = (~is_cancer) & (coarse_cancer == 1)
            self.ti_cancerous[still_clear] = np.nan
            self.ti_dead_cancer[still_clear] = np.nan
            # Same consistency fix: reuse the agent's own dur_cin for the
            # clearance timing rather than an unconditional resample.
            dur_cin_clear = dur_cin_coarse[sel2]
            self.ti_clearance[still_clear] = (
                self.ti_cin[still_clear] + self._randround(
                    dur_cin_clear, still_clear, self._round_clear_cin_bern,
                )
            )
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
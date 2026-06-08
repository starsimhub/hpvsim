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

# Max rejection-sampling rounds for the CIN-conditional precin draw on multiscale
# extras (see _multiscale_ledger). Acceptance per round is the per-agent
# CIN-reaching probability (~25-30% typical), so ~50 rounds leaves ~1e-7
# unresolved; those fall back to the (also CIN-conditional) source precin, which
# is unbiased. This cap only trades extra diversification for compute, not
# accuracy, so a fixed value is appropriate; the loop early-exits once resolved.
_PRECIN_RESAMPLE_ROUNDS = 50


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
            # Multiscale resolution: each CIN agent's cancer pathway is resolved
            # at this many sub-resolutions (the agent's own cancer + ratio-1
            # extra sub-cancers recorded in the ledger). 1 = feature off.
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
        # Multiscale cancer-pathway LEDGER (ms_agent_ratio>1). Cancers are
        # resolved at ratio-finer granularity as scheduled DATA — no fine People
        # agents. The agent lives its single-scale life (its own cancer drives
        # the population); the ledger overlays ratio-1 EXTRA sub-cancers per
        # CIN->cancer decision purely for the RESULTS. ``_ledger_onset`` maps a
        # future onset ti -> list of pending extra events; ``_ledger_death`` maps a
        # future death ti -> realized extra cancer-deaths; ``_cancer_events``
        # accumulates realized (causal_age, cin_age, cancer_age, weight) tuples
        # for the by-event distribution analyzer (own cancers + extras).
        self._ledger_onset = {}
        self._ledger_death = {}
        self._cancer_events = []
        # Sim-shared registry (uid, sub_idx) -> onset ti of the first realized
        # cancer for that sub-individual, used by _realize_ledger to enforce
        # one cancer per sub-individual ACROSS genotypes (and across reinfection
        # episodes). Reset here so a fresh init() starts empty; all genotype
        # modules share the one dict on the sim.
        self.sim._ms_cancer_claims = {}
        return

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

        # 5b. Progression to cancer (the agent's OWN cancer). Scheduled here
        #     exactly as pre-feature; under multiscale this is sub-resolution 0
        #     (counted at weight 1/ratio in step_state) and the ledger below
        #     records the ratio-1 EXTRA sub-cancers as data.
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

        # 6. Multiscale LEDGER at the CIN->cancer decision. No-op at
        #    ms_agent_ratio<=1. Records ratio-1 EXTRA resolved sub-cancers per
        #    CIN agent into the ledger (scheduled future onset/death + ages), as
        #    DATA — no fine agents are grown. The agent's OWN cancer (drawn above)
        #    drives the population and is recorded into the ledger when it fires
        #    in step_state (weight 1/ratio). Resolves the cancer/CIN2+ event-age
        #    distributions (methods Fig 5) at ratio-x resolution.
        self._multiscale_ledger(cin_uids, rel_sev_cin, age_mod,
                                dur_precin[cin_mask], sev_imm_uids[cin_mask], dt_yr)
        return

    def _multiscale_ledger(self, cin_uids, rel_sev_cin, age_mod,
                           dur_precin_cin, sev_imm_cin, dt_yr):
        """Record ratio-1 EXTRA resolved sub-cancers per CIN agent into the
        multiscale ledger — scheduled DATA, no fine agents grown.

        The coarse agent lives its single-scale life: its OWN cancer (drawn in
        ``set_prognoses``) drives the population (death, transmission) and is
        recorded into the ledger when it fires in ``step_state`` at weight
        ``1/ratio`` (sub-resolution 0). Here we resolve the OTHER ``ratio-1``
        sub-resolutions of the same individual purely for the RESULTS: each
        resamples its own ``dur_precin`` (CIN-conditional, length-biased via the
        CIN gate) and ``dur_cin``, draws cancer at ``f(dur_cin)``, and — for the
        successes — appends ``(source_uid, causal/cin/cancer/death ages,
        onset_ti, death_ti, weight=1/ratio)`` to ``_ledger_onset[onset_ti]``. The
        event is only COUNTED at realization (``step_state``) if its source
        agent survives to its onset, so each sub-cancer shares the real source's
        competing risk (the fix for the grow design's decoupled-fate drift).

        Side numpy RNG (seeded per seed/step/genotype via crc32 — reproducible,
        process-stable), independent of the slot-keyed CRN: the extras are not
        agents, and the separate stream cannot perturb the population's draws.
        Conserves cancer mass in expectation: own ``f(d)/ratio`` +
        ``(ratio-1)*E[f(dur')]/ratio`` = ``p_bar`` per CIN agent (single-scale).
        """
        ratio = int(self.pars.ms_agent_ratio)
        n = len(cin_uids)
        if ratio <= 1 or n == 0:
            return
        import zlib
        p = self.pars
        ppl = self.sim.people
        ti = int(self.ti)
        m = ratio - 1  # extras per agent

        # rand_seed may be None (Starsim's "unseeded run" idiom); coerce to 0 so
        # the side-RNG seed arithmetic does not raise.
        rand_seed = int(self.sim.pars.rand_seed or 0)
        seed = (rand_seed * 2654435761
                + ti * 40503
                + zlib.crc32(self.genotype.encode())) & 0x7FFFFFFF
        rng = np.random.default_rng(seed)

        def _ln(dist, size):  # side-sample lognormal durations (years)
            # Same ex->im (mean/std -> mu/sigma) conversion as
            # ss.lognorm_ex.convert_ex_to_im, inlined because that method
            # mutates the dist's _pars in place (so it can't be called on the
            # live p.dur_* objects) and because we must draw on our own
            # side-RNG, not the dist's slot-keyed CRN. The formula is the
            # standard lognormal parameterization and is fixed, so it cannot
            # drift from the dist's own draws.
            mean = float(dist.pars['mean']); std = float(dist.pars['std'])
            sig2 = np.log(1.0 + (std / mean) ** 2)
            return rng.lognormal(np.log(mean) - 0.5 * sig2, np.sqrt(sig2), size)

        # Per-agent context broadcast to the n*m extras. sub_idx labels each
        # extra with a stable sub-resolution index 1..ratio-1 within its source
        # coarse agent; (source uid, sub_idx) identifies one sub-individual and
        # is the key used at realization to enforce one-cancer-per-sub-individual
        # ACROSS genotypes (the cross-genotype competition that, for the agent
        # itself = sub-resolution 0, is handled by the agent-level
        # _cancel_other_genotype_progression_for).
        infection_age = np.repeat(np.asarray(ppl.age[cin_uids], dtype=float), m)
        src_uid = np.repeat(np.asarray(cin_uids), m)
        sub_idx = np.tile(np.arange(1, ratio, dtype=int), n)
        # Per-extra weight = source's per-agent scale / ratio, captured here
        # while the source is alive. Matches the own-cancer tally's
        # ``w_own * people.scale`` convention in step_state, so own and extra
        # cancers use one weighting rule (uniform 1/ratio today, since hpvsim
        # keeps people.scale==1 and applies population scaling via the global
        # pop_scale).
        #
        # We honor people.scale read-only rather than hardcoding 1.0: that is
        # Starsim's defined population count (scale_flows == scale[inds].sum(),
        # not len), so it stays correct for any STATIC per-agent scale, not just
        # ==1. CAVEAT: it is NOT valid under DYNAMIC rescaling (people.scale
        # changing over an agent's life) — we freeze src_scale at schedule time
        # and reuse it at realization, so a source rescaled in between would be
        # counted at its stale scale. (The earlier grow design wrote
        # people.scale on spawned fine agents and so could track dynamic scale;
        # the ledger trades that for a bit-identical population. Revisit this
        # capture if dynamic rescaling is ever introduced.)
        src_scale = np.repeat(np.asarray(ppl.scale[cin_uids], dtype=float), m)
        rel_sev = np.repeat(np.asarray(rel_sev_cin, dtype=float), m)
        amod = np.repeat(np.asarray(age_mod, dtype=float), m)
        sev_imm = np.repeat(np.asarray(sev_imm_cin, dtype=float), m)
        src_precin_yr = np.repeat(np.asarray(dur_precin_cin, dtype=float), m) * dt_yr

        # CIN onset: each extra needs a precin drawn from the CIN-CONDITIONAL
        # (length-biased) distribution — exactly what a single-scale CIN-reacher
        # has (its precin is the one that passed the CIN gate on first draw).
        # Rejection-sample to get that distribution independently per extra:
        # redraw precin until it passes f_cin. This both (a) length-biases the
        # onset so cancer-onset truncation matches single-scale (unbiased count)
        # and (b) makes every extra an INDEPENDENT CIN2+-age sample, so the
        # CIN2+ distribution actually tightens (a plain single draw + fall back
        # to the source on failure left ~3/4 of extras copying the source's CIN
        # age, adding no independent information). Bounded loop over only the
        # still-unresolved indices; the ~1e-7 that never pass fall back to the
        # source's (also CIN-conditional) precin.
        size = n * m
        precin_yr = src_precin_yr.copy()
        need = np.ones(size, dtype=bool)
        for _ in range(_PRECIN_RESAMPLE_ROUNDS):
            idx = np.where(need)[0]
            if len(idx) == 0:
                break
            cand = _ln(p.dur_precin, len(idx)) * (1.0 - sev_imm[idx])
            p_cin = compute_severity(cand, rel_sev=rel_sev[idx], pars=p.cin_fn)
            passed = rng.random(len(idx)) < p_cin
            hit = idx[passed]
            precin_yr[hit] = cand[passed]
            need[hit] = False

        # CIN -> cancer: resample dur_cin, draw cancer at f(dur_cin); keep
        # successes (length-biased dur_cin -> correct, diversified onset ages).
        dcin = _ln(p.dur_cin, n * m) * amod
        keep = rng.random(n * m) < compute_severity(dcin, rel_sev=rel_sev, pars=p.cancer_fn)
        if not keep.any():
            return

        causal = infection_age[keep]
        cin_age = causal + precin_yr[keep]
        cancer_age = cin_age + dcin[keep]
        dcan = _ln(p.dur_cancer, int(keep.sum()))
        death_age = cancer_age + dcan
        onset_ti = ti + np.round((precin_yr[keep] + dcin[keep]) / dt_yr).astype(int)
        death_ti = onset_ti + np.round(dcan / dt_yr).astype(int)
        kept_uid = src_uid[keep]
        kept_sub = sub_idx[keep]
        kept_w = src_scale[keep] / ratio
        for u, j, ca, cia, cca, da, oti, dti, w in zip(
                kept_uid, kept_sub, causal, cin_age, cancer_age, death_age,
                onset_ti, death_ti, kept_w):
            self._ledger_onset.setdefault(int(oti), []).append(
                (int(u), int(j), float(ca), float(cia), float(cca), float(da), int(dti), float(w)))
        return

    def _realize_ledger(self, ti):
        """Realize the ledger's scheduled extra sub-cancers/deaths due at ``ti``.

        An extra is realized only if its source agent is "available" (a shared
        competing-risk proxy for the sub-individual's background mortality /
        emigration — see ``_sources_available``).

        Cross-genotype competition: a sub-individual ``(source uid, sub_idx)``
        may get cancer in at most ONE genotype (cancer is terminal — the person
        dies of the first one). The realized cancers are recorded in a sim-shared
        ``_ms_cancer_claims`` registry keyed by ``(uid, sub_idx)``; an extra whose
        sub-individual is already claimed (by an earlier-onset cancer in this or
        another genotype) is suppressed, mirroring single-scale's first-cancer-
        wins cancellation (``_cancel_other_genotype_progression_for``) which
        handles sub-resolution 0, the agent itself. The claim persists for the
        run, so an already-cancered sub-individual is not re-counted if its coarse
        source reinfects and reaches CIN again.

        Realized onsets add their pathway ages to ``_cancer_events`` and schedule
        the matching cancer death into ``_ledger_death``. Events whose ti falls past
        the sim window are never popped, so they are correctly truncated. Pure
        results overlay — touches no agent state.
        """
        res = self.results
        claims = self.sim._ms_cancer_claims  # shared (uid, sub_idx) -> claimed

        onsets = self._ledger_onset.pop(ti, None)
        if onsets:
            avail = self._sources_available([e[0] for e in onsets])
            n_w = 0.0
            age_w = 0.0
            for ok, (u, j, causal, cin_age, cancer_age, death_age, death_ti, w) in zip(avail, onsets):
                if not ok:
                    continue
                key = (u, j)
                if key in claims:
                    continue  # this sub-individual already got cancer (cross-genotype / earlier episode)
                claims[key] = ti
                n_w += w
                age_w += cancer_age * w
                self._cancer_events.append((causal, cin_age, cancer_age, w))
                self._ledger_death.setdefault(death_ti, []).append((u, death_age, w))
            if n_w:
                res.new_cancers[ti] += n_w
                res.sum_age_at_cancer[ti] += age_w

        deaths = self._ledger_death.pop(ti, None)
        if deaths:
            avail = self._sources_available([e[0] for e in deaths])
            n_w = 0.0
            age_w = 0.0
            for ok, (u, death_age, w) in zip(avail, deaths):
                if not ok:
                    continue
                n_w += w
                age_w += death_age * w
            if n_w:
                res.new_cancer_deaths[ti] += n_w
                res.sum_age_at_cancer_death[ti] += age_w
        return

    def _sources_available(self, uids):
        """Boolean array (aligned with ``uids``): is each ledger source agent
        "available" for its extra to realize? ``available`` = the source is still
        alive, OR it was removed by its OWN cancer (``became cancerous in any
        genotype at/before death``). The own-cancer case is excluded from the
        competing risk because a coarse agent represents ``ratio`` DIFFERENT
        people: the source dying of its own cancer says nothing about whether a
        sibling sub-individual (who got cancer independently) would be alive,
        whereas its background death / emigration IS a valid, correctly-rated
        sample of the shared hazard those siblings face. (Without the exclusion,
        late-onset extras of a cancer-drawing source are over-suppressed, ~-5%.)

        Single source of the competing-risk gate for both onset and death
        realization, evaluated only over the (small) set of event source uids
        for this step rather than the whole — growing — population array.
        """
        ppl = self.sim.people
        uids = np.asarray(uids)
        avail = np.asarray(ppl.alive.raw[uids]).copy()
        td = np.asarray(ppl.ti_dead.raw)[uids]
        died = np.isfinite(td)
        if died.any():
            cancer_before_death = np.zeros(len(uids), dtype=bool)
            for mod in self.sim.diseases.values():
                if isinstance(mod, HPV):
                    tc = np.asarray(mod.ti_cancerous.raw)[uids]
                    cancer_before_death |= np.isfinite(tc) & (tc <= td)
            avail |= died & cancer_before_death
        return avail

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
        dt_yr = float(self.t.dt.years if hasattr(self.t.dt, 'years') else self.t.dt)
        # Multiscale: the agent's OWN cancer is sub-resolution 0, counted at
        # weight 1/ratio (the ledger overlays the other ratio-1 sub-cancers).
        # At ratio==1 this is 1.0 and the ledger is empty -> bit-identical.
        ratio = int(self.pars.ms_agent_ratio)
        w_own = 1.0 / ratio

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
            # people.scale honored read-only (Starsim's scale_flows count, not
            # len): correct for any STATIC per-agent scale; hpvsim keeps it ==1.
            # See _multiscale_ledger's src_scale capture for the dynamic-scale
            # caveat the ledger shares.
            scale = np.asarray(self.sim.people.scale[to_cancerous])
            ages_at_cancer = np.asarray(self.sim.people.age[to_cancerous])
            self.results.new_cancers[ti] = w_own * float(scale.sum())
            self.results.sum_age_at_cancer[ti] = w_own * float((ages_at_cancer * scale).sum())
            # Multiscale only: record each own cancer's pathway ages (causal
            # infection, CIN2+, cancer) into the ledger event list at weight
            # w_own, so the Fig-5 distribution analyzer reads own + extra
            # sub-cancers from one place. Skipped at ratio==1 (zero overhead;
            # the analyzer reads agents directly in the single-scale case).
            if ratio > 1:
                ti_inf = np.asarray(self.ti_infected[to_cancerous], dtype=float)
                ti_cinv = np.asarray(self.ti_cin[to_cancerous], dtype=float)
                causal_ages = ages_at_cancer - (ti - ti_inf) * dt_yr
                cin_ages = ages_at_cancer - (ti - ti_cinv) * dt_yr
                for ca, cia, cca in zip(causal_ages, cin_ages, ages_at_cancer):
                    self._cancer_events.append((float(ca), float(cia), float(cca), w_own))
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
            scale_d = np.asarray(self.sim.people.scale[to_dead])
            ages_at_death = np.asarray(self.sim.people.age[to_dead]) + dt_yr
            self.results.new_cancer_deaths[ti] = w_own * float(scale_d.sum())
            self.results.sum_age_at_cancer_death[ti] = w_own * float((ages_at_death * scale_d).sum())
            self.sim.people.request_death(to_dead)

        # --- 5. Multiscale ledger: realize scheduled extra sub-cancers ---
        # The ratio-1 extra sub-resolutions resolved in _multiscale_ledger
        # become RESULTS here (no population effect — pure overlay), each gated
        # on its SOURCE agent surviving to the scheduled ti (shared competing
        # risk). Empty at ratio==1, so this is a no-op in the single-scale case.
        if self._ledger_onset or self._ledger_death:
            self._realize_ledger(ti)

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
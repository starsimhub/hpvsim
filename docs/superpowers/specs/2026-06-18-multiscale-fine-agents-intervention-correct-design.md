# Multiscale fine agents — intervention-correct redesign

**Date:** 2026-06-18
**Branch:** `m07-multiscale-investigation` (build here; rebased on the M06 test-and-treat cascade)
**Status:** approved, pre-implementation
**Supersedes:** the data-overlay approach on `m07-multiscale-ledger` (Phase-1 ledger + Phase-2 post-hoc resolver), which is intervention-blind and cannot be fixed.

---

## 1. Why this exists

`ms_agent_ratio > 1` is a variance-reduction feature: it resolves rare cervical-cancer
outcomes at higher statistical resolution. Two prior v3 attempts and the finding
that killed them:

- **Data-overlay (ledger / post-hoc resolver, `m07-multiscale-ledger`).** Extra
  sub-cancers are generated as resampled DATA, never as agents. Elegant and
  bit-identical, but the extras are computed from natural-history distributions
  **blind to interventions**. Demonstrated: with aggressive CIN screening+treatment
  the true cancer count (`ratio=1`) is ~0 (averted), but `ratio=12` reports ~105 —
  the treatment-blind extras ignore the aversion. This is unfixable without
  re-implementing every intervention against the resampled data. **Rejected.**
- **Fine agents (grow, `m07-multiscale-investigation`).** Spawns real fine agents
  on the cancer pathway at the CIN→cancer decision. Completed and de-biased
  (internal equivalence +0.4%), but interventions were never addressed and it was
  set aside (no recorded rationale) for the overlay's bit-identity.

**Decision:** revive the fine-agent approach, because real agents experience
screening/treatment natively. Two project decisions reshape it:

1. **CRN compatibility is NOT required.** No bit-identity across agent count, no
   slot-stable common-random-number guarantee. This removes the prior attempt's
   largest source of complexity (see §6).
2. **Use Starsim dists wherever possible** rather than hand-rolled numpy RNG.

The acceptance bar that both prior attempts lacked — a **multiscale + intervention
equivalence test** — is the centerpiece here.

## 2. Goal

`hpv.Sim(ms_agent_ratio=N)` resolves cancer/cancer-death (and intervention
outcomes on the cancer pathway) at ~N× sampling resolution, **correct whether or
not screening/treatment interventions are present**. `ms_agent_ratio=1` (default)
leaves all behavior unchanged.

## 3. Architecture

### 3.1 Core mechanism (reuse the investigation branch's grow, simplified)

Hook the **CIN→cancer decision** in `HPV.set_prognoses` (not infection, not
first-CIN — splitting earlier caused the −74% loss in the original spike):

- For each coarse, cancer-drawing CIN agent: shrink its per-agent `scale` to
  `1/ratio`.
- Draw `ratio−1` **independent** extra trajectories (fresh `dur_precin`,
  `dur_cin`, cancer Bernoulli, `dur_cancer`) using `ss` dists.
- `people.grow()` a fine agent **only for each cancer-success** extra (count is
  bounded by cancers, not by CIN incidence). Copy the source's demographic +
  natural-history state; set the fine agent's independently-sampled
  `ti_cancerous` / `ti_dead_cancer`; mark it `multiscale_fine`.
- A fine agent never re-splits (mirrors v2's `level0`/`level1`).

### 3.2 Fine agents are fully real for interventions (the new, critical part)

A grown fine agent carries real CIN state and a scheduled `ti_cancerous`. It is an
ordinary agent with respect to the screening/treatment interventions:

- **Eligibility:** fine agents must be *included* in screening detection and
  treatment eligibility — NOT gated out. (Demographics gate them out via the
  Level0 wrappers; interventions must not inherit that gating.) The
  implementation must verify the screening/treatment eligibility predicates select
  fine agents, and add no `~multiscale_fine` filter.
- **Effect:** treatment's `administer` clears CIN and sets `ti_cancerous = NaN`
  on the fine agent exactly as for any agent, so its cancer is averted iff it is
  in the screened+treated fraction. No special-casing.
- **Result tallies:** every intervention result that counts treated/screened
  agents (`new_cin_treated`, `new_cancer_treated`, screening counts, and any
  cascade results) must be **scale-weighted** (`people.scale[uids].sum()`), so a
  fine agent at scale `1/ratio` contributes `1/ratio`, not 1.

### 3.3 Demographics & network (reuse investigation; accept the residual)

- **Network:** fine agents are excluded from the `SexualNetwork` (kept as-is) —
  they do not transmit, so they do not perturb the epidemic.
- **Demographics:** keep the Level0 birth/death/migration exclusion (the
  `Level0Births` / `Level0Deaths` / `Level0People` wrappers and the `AgeMigration`
  level0-gating). Fine agents do not take births/deaths/migration.
- **Accepted limitation:** because fine agents are excluded from background
  mortality/migration, they do not die on their own demographic schedule — the
  documented **~±8% config-dependent competing-risk residual** in cancer counts.
  Accepted and documented; fixing it would require scale-aware demographics
  (invasive — Starsim counts raw agents), out of scope here.

### 3.4 Results audit (reuse + extend)

Keep the investigation branch's scale-weighting of all `n_<state>`/cancer tallies
(`new_cancers`, `new_cancer_deaths`, `sum_age_at_*`, `n_cancerous`, the `HPVTotal`
unions, `cum_*`). **Extend the audit to intervention results** (§3.2). The audit
list is enumerated in the plan; each scaled result gets a per-metric equivalence
assertion (a single missed result is a silent bias).

## 4. CRN-free simplifications (remove from the investigation code)

These existed only to preserve common-random-number reproducibility / bit-identity,
which is no longer required:

- **Placeholder grow/drop dance** — the prior code grew `ratio−1` placeholder
  agents to obtain fresh CRN slots, then dropped non-cancer ones. Replace with a
  direct draw of the extra trajectories and a single `grow()` of exactly the
  cancer-successes.
- **`_randround()`** CRN-safe stochastic rounder and its dedicated per-decision
  `ss.bernoulli` objects (`_round_cancer_bern`, `_round_clear_cin_bern`,
  `_round_dead_bern`, `_round_cin_bern`). Use `sc.randround()` / `np.round` or the
  dists' built-in rounding.
- **Bit-identity tests** (`test_ratio_one_is_bit_identical`,
  `test_ratio_one_still_identical_after_split_code`). Replace the `ratio=1` no-op
  guarantee with a weaker check: `ratio=1` does not register/spawn fine agents and
  produces results statistically consistent with the pre-feature code (no spawn
  path executed), rather than byte-identical.

`ss` dists: draw the extra trajectories via the live `dur_*` distributions
(`.rvs(size)` size-draws are acceptable now that perturbing the CRN stream is a
non-issue), or dedicated `ss` dist instances — not inlined numpy lognormal math.

## 5. Components / files (on the investigation branch)

- `hpvsim/hpv.py` — EDIT: simplify the `_multiscale_split` hook (drop placeholder
  dance / `_randround` / Bernoullis; draw extras via `ss` dists; grow
  cancer-successes directly). Keep scale-weighted cancer tallies.
- `hpvsim/interventions.py` — EDIT: ensure screening/treatment eligibility
  includes fine agents; scale-weight intervention result tallies.
- `hpvsim/network.py` — KEEP: fine-agent network exclusion.
- `hpvsim/demographics.py` — KEEP: Level0 wrappers (births/deaths/migration
  exclusion).
- `hpvsim/cross_genotype.py` / `analyzers.py` — KEEP/AUDIT: scale-weighted
  aggregations.
- `hpvsim/sim.py` — KEEP: forward `ms_agent_ratio`; register Level0 demographics.
- `tests/test_multiscale.py` — EDIT: drop bit-identity tests; keep mechanics +
  scale-weighting unit tests.
- `tests/test_multiscale_equivalence.py` — KEEP: internal (no-intervention)
  equivalence gate.
- `tests/test_multiscale_intervention_equivalence.py` — **NEW**: the centerpiece
  gate (§7).

## 6. Acceptance criteria

1. **`ratio=1` no-op:** with `ms_agent_ratio=1`, no fine agents are spawned and no
   split path executes; results match the pre-feature code (statistical, not
   byte-identical).
2. **Internal equivalence (no interventions):** for a cancer-bearing anchor,
   `ratio=N` at reduced `n_agents` reproduces `ratio=1` at full `n_agents` on total
   cancers, cancer deaths, and mean age at cancer/death, within tolerance,
   multi-seed, in people-space (the investigation's gate; ~+0.4% achieved).
3. **Intervention equivalence (NEW — centerpiece):** with a screening+treatment
   program active, `ratio=N` reproduces `ratio=1` on **both** realized cancers
   **and averted cancers** (i.e. the intervention's impact), within tolerance,
   multi-seed, people-space. This is the gate both prior attempts lacked.
4. **Variance reduction:** `ratio=N` shows lower cancer-incidence variance than
   `ratio=1` at equal agent count, at agent counts where transmission noise does
   not dominate (transmission-floor-aware; may be `xfail` at high N as before).
5. **Result-audit completeness:** every scale-weighted result (cancer AND
   intervention) has a per-metric equivalence assertion; the audit list is
   documented.
6. **Multi-genotype correctness:** splitting in one genotype does not corrupt other
   genotype modules; cross-genotype cancer cancellation holds.

## 7. The intervention-equivalence gate (detail)

Construct a cancer-bearing anchor with an aggressive screening+treatment program
(high coverage, CIN treatment via excision) so a large, measurable fraction of
cancers is averted. Run `ratio=1` (full N) and `ratio=N` (reduced N), multi-seed,
in people-space. Assert:

- realized cancers (post-intervention) agree within tolerance;
- averted cancers — defined as (no-intervention cancers − with-intervention
  cancers), or directly via the cancers prevented among screened+treated — agree
  within tolerance.

A broken design (e.g. the overlay) fails this dramatically (the overlay reported
~105 vs a true ~0). Tolerance is set against the §6.2 internal-equivalence
tolerance plus the accepted ~±8% residual; it must be tight enough that an
intervention-blind implementation fails it by a wide margin.

## 8. Risks

- **Branch staleness (tracked separately).** `m07-multiscale-investigation` forked
  off an older point; it must be reconciled/rebased with current main before merge.
  This is a tracked integration task, not part of the core build, but it gates
  shipping.
- **Scale-weighting audit completeness.** One missed result → silent bias.
  Mitigation: enumerate every `scale=True` result and analyzer histogram across
  cancer AND intervention outputs; per-metric equivalence assertions.
- **Intervention eligibility leakage.** If any intervention predicate inadvertently
  excludes fine agents (e.g. by inheriting a demographic level0 filter), the
  feature silently under-treats fine agents → biased averted-cancer counts.
  Mitigation: explicit test that fine agents are screened/treated at the expected
  rate.
- **Accepted ~±8% competing-risk residual** — document as a known limitation of
  the demographic-exclusion choice.

## 9. Out of scope

- Scale-aware demographics (fixing the ~±8% residual) — deferred; would fight
  Starsim's raw-count demographics.
- CRN reproducibility / bit-identity across agent counts — explicitly dropped.
- Multiscaling any outcome other than cancer / cancer-death.
- HIV-coinfection multiscale interactions.
- Dynamic pruning of spent fine agents (memory) — deferred unless long-run memory
  proves problematic.
- Reconciling the abandoned `m07-multiscale-ledger` work (it is superseded; no
  migration of the overlay code is planned).

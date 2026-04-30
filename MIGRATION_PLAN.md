# HPVsim v3.0 Migration Plan

## Overview

HPVsim v3.0 is a reimplementation of HPVsim on the [Starsim](https://starsim.org/) agent-based modeling framework. The original HPVsim (v2.x, ~16,000 LOC) uses a fully custom architecture. v3.0 inherits from Starsim's core classes (`ss.Sim`, `ss.Disease`, `ss.Network`, `ss.Intervention`, `ss.Analyzer`, `ss.Connector`) while keeping HPVsim's domain-specific logic: HPV natural history, genotype dynamics, cross-immunity, cervical cancer progression.

Migration work happens on the `v3.0-dev` branch (created off `rc2.3`). v3.0 is considered **done** when the analysis-repo suite that validated v2.3 (issues #64, #68–#73, #82–#87) reproduces on v3 with overlapping uncertainty intervals on headline results, and the migration guide is published.

## Validation criteria

Validation does **not** require identical numerical output. Results are considered equivalent when uncertainty intervals overlap across multiple seeds on headline results of the v2 analysis-repo suite. The goal is epidemiologically equivalent behavior, not bit-for-bit reproducibility.

Canonical validation set: the analysis repos defined in issue [#64](https://github.com/starsimhub/hpvsim/issues/64) and validated against v2.3 under issues #68–#73 and #82–#87:

- `hpvsim_methods_manuscript`
- `hpvsim_india`
- `hpvsim_rwanda`
- `hpvsim_1dose`
- `hpvsim_pxv_younger`
- `hpv_faster_kenya`

## Scope decisions (settled)

Carried forward from the KICKOFF_DISCUSSION notes (2026-03-25) and reconfirmed 2026-04-23:

- **Therapeutic vaccination (txvx):** Port to v3.
- **EventSchedule:** Drop (rarely used).
- **Custom `settings.py`:** Drop; rely on `ss.options`.
- **Waning immunity:** Drop (never used in any published analysis).
- **v2.x incidence-based HIV module:** Drop. Replace with STIsim's transmission-based HIV.
- **Multiscale modeling:** Low priority; not a release blocker.

Revised 2026-04-23:

- **Sexual network:** Port HPVsim's custom sexual network to Starsim's `Network` framework. Reverses an earlier decision to adopt `sti.StructuredSexual`. Rationale: reduces error sources when validating against v2.x baselines and provides a known reference for testing the STIsim network later. Lift-and-shift via `class hpv.Network(ss.Network)` is the expected starting point.

## RACI

| Role | Person(s) |
|---|---|
| **Responsible** (does the migration work) | Ryan |
| **Accountable** (owns the release decision) | Robyn, Jamie |
| **Consulted** (domain expertise, review) | Darcy, WHI |
| **Informed** (stakeholders, downstream users) | Quantium, external users |

- **PR reviews on `v3.0-dev`:** Robyn
- **Scientific validation:** TBC, likely Ryan
- **External team contact (Quantium, IARC):** Robyn

## Architecture mapping

| HPVsim v2.x (rc2.3) | HPVsim v3.0 (v3.0-dev) |
|---|---|
| Custom `BaseSim` / `Sim` | `ss.Sim` subclass (`hpv.Sim`) |
| Custom `People` / `Person` | `ss.People` subclass (`hpv.People`), stripped toward plain `ss.People` over time |
| Custom genotype handling in `Sim` | `Genotype(ss.Disease)` per genotype + `HPV(ss.Connector)` |
| Custom network (`people.py` / `population.py`) | `hpv.Network(ss.Network)` — ported from v2.x (not `sti.StructuredSexual`) |
| Custom `Intervention` base | `ss.Intervention` subclass |
| Custom `Analyzer` base | `ss.Analyzer` subclass |
| `HIVsim` class in `hiv.py` | `hpv_hiv_connector(ss.Connector)` + STIsim HIV (no lift-and-shift) |
| `immunity.py` module | Cross-immunity via `HPV` connector |
| `calibration.py` | Starsim calibration (Optuna) |
| `run.py` (MultiSim, Scenarios, Sweep) | `ss.MultiSim`, `ss.parallel()`, adapted `Scenarios` |
| `plotting.py` / `analysis.py` | Rebuilt on Starsim plotting patterns |
| Custom `settings.py` | `ss.options` |

## Milestones

Each milestone produces a user-visible demo and must meet its acceptance test before the next begins. Tests written during a milestone stay in CI from that point onward. Sub-tasks map 1:1 to GitHub issues (see "GitHub milestones and issues" below).

### M0: Foundation

**Demo:** `v3.0-dev` branch exists with CI green on a stub sim; `KICKOFF_DISCUSSION.md` + `MIGRATION_PLAN.md` committed; v2.x baseline-generation script committed (baseline files themselves are generated locally and gitignored, never committed); anchor-scenario script and ±10% comparison script committed.

**Acceptance test:** CI passes; regression harness compares a v3 run to a locally-generated v2 baseline and reports diffs.

**Sub-tasks:**
- Set up CI on `v3.0-dev` (adapted from rc2.3 CI); add a smoke-check step for the comparison CLI.
- Commit a deterministic v2.x baseline-generation script (`tests/regression/baseline.py`). Generated baseline files stay local (gitignored), never committed.
- Write the anchor scenario script (`tests/regression/anchor.py`): vanilla 4-genotype HPV sim, Nigeria, fixed seed (`0`), no interventions, 1990–2060.
- Write the ±10% comparison script (`tests/regression/compare.py`) that diffs a current run against the locally-stored v2 baseline and emits per-summary-result relative drift.
- Document how to run the regression harness in `tests/regression/README.md` and add a pointer from `tests/README.md`.

### M1: Basic transmission sim

**Demo:** Run a vanilla 4-genotype HPV sim on one country; plot aggregate HPV prevalence over time.

**Acceptance test:** Aggregate HPV prevalence trajectory overlaps v2.x uncertainty interval on the same country/pars; partnership patterns (age mixing, concurrency, duration) match v2.x.

**Sub-tasks:**
- Port HPVsim's custom sexual network as `hpv.SexualNetwork(ss.SexualNetwork)` using lift-and-shift; one class, three instances (m/c/o); cross-layer concurrency via sibling iteration.
- Add a minimal single-genotype `hpv.HPV(ss.Infection)` for HPV16 — transmission only, SIS clearance, no precin/CIN/cancer.
- Replace `hpvsim.Sim` with thin `hpv.Sim(ss.Sim)` wrapper; rely on stock `ss.People`, `ss.Pregnancy`, `ss.Deaths`. Add `hpv.data.load_country()` adapter.
- Quarantine v2 modules untouched by M01 to `hpvsim/_v2_legacy/` and v2 tests to `tests/_legacy/`.
- Add tests for partnership pattern equivalence (age-mixing matrix, concurrency distribution, partnership duration distribution) vs. v2.x.
- Add tests for HPV prevalence trajectory vs. a new M1 1-genotype HPV16 baseline.
- Multi-resolution state (`scale`/`level0`/`level1`/`cluster`) deferred indefinitely; stock `ss.People` is sufficient.

### M2: Natural history parity

**Demo:** Show single-genotype (HPV16) natural history — clearance, precin, CIN, and cancer progression — matching v2.x.

**Acceptance test:** HPV → CIN → cancer dynamics for HPV16 match v2.x's HPV16-only run within calibration tolerance, against a v2 1-genotype baseline.

**Sub-tasks:**
- Port disease progression for HPV16 into `hpv.HPV(ss.Infection)` — add precin/CIN/cancer states, port duration distributions and progression functions from v2 `parameters.get_genotype_pars('hpv16')`.
- Port `age_results` analyzer at minimum scope (enough for calibration to consume in M4).
- Port `pop_scale` / `total_pop`.
- Port age-specific migration (from v2 `people.check_migration`).
- Add tests: HPV16 CIN/cancer trajectories match v2 1-genotype baseline.
- Tighten partnership-network equivalence beyond M01 thresholds. M01 closes the n_pairs / mean_dur / concurrency_max / mixing-cosine gates (50% / 50% / ±2 / 0.85), but the casual layer still sits ~35-40% above v2 on count and ~20-25% on mean duration; the marital layer matches within 2-3%. v3 prod and a numpy-RNG variant of the network track each other (~1.5% apart) but diverge from v2 the same way, so the residual is algorithmic, not RNG-stream. Most likely culprit: cross-layer concurrency uses sibling-edge counts in v3 vs per-layer `current_partners` arrays in v2. Investigate and tighten the gate (target: each metric within 10-15% of v2 instead of 50%).

### M3: Multi-genotype and cross-immunity

**Demo:** 4-genotype HPV sim with cross-immunity matching v2's 4-genotype Nigeria baseline.

**Acceptance test:** Age-stratified HPV / CIN / cancer incidence by genotype overlaps v2.x intervals against the M0 4-genotype baseline.

**Sub-tasks:**
- Replicate `hpv.HPV(ss.Infection)` across all four genotypes `[16, 18, hi5, ohr]`; auto-instantiate via `hpv.Sim(genotypes=[...])` API.
- Add `hpv.CrossImmunity(ss.Connector)` implementing v2's cross-protection matrix.
- Wire genotype-specific natural history params (`rel_beta`, `dur_precin`, `dur_cin`, `cin_fn`, `cancer_fn`) per genotype.
- Add tests: 4-genotype prevalence + CIN + cancer trajectories match the M0 stored 4-genotype baseline.

### M4: Calibration loop

**Demo:** End-to-end calibration of one country (e.g., India) to age-stratified cancer incidence data converges and reproduces a published calibration.

**Acceptance test:** Optuna-based calibration produces a posterior consistent with v2.x calibration on the same target data.

**Sub-tasks:**
- Integrate Starsim's Optuna-based calibration with `hpv.Sim`.
- Port `compute_gof()` and likelihood functions for cancer incidence by age.
- Add calibration tests that run a small number of trials end-to-end.
- Run a full calibration for India; confirm posterior parameter ranges overlap v2.x calibration.
- Write a short calibration guide section (expanded further in M10).

### M5: Vaccination scenarios

**Demo:** Run routine + catch-up + therapeutic vaccination and show cancer incidence reduction.

**Acceptance test:** Vaccination impact trajectory overlaps v2.x intervals on `hpvsim_1dose` / `hpvsim_pxv_younger`.

**Sub-tasks:**
- Port product base classes (`dx`, `tx`, `vx`) and adapt CSV product files from v2.x `data/`.
- Port `routine_vx` intervention.
- Port `campaign_vx` intervention.
- Port `txvx` (therapeutic vaccination: `BaseTxVx`, `routine_txvx`, `campaign_txvx`, `linked_txvx`).
- Add intervention-level results tracking (number vaccinated, doses administered, by age and year).
- Add tests: vaccination scenarios reproduce `hpvsim_1dose` / `hpvsim_pxv_younger` with overlapping intervals.

### M6: Screen-and-treat cascade

**Demo:** Run a screen → triage → treat scenario on one country.

**Acceptance test:** Screening scenario overlaps v2.x intervals on `hpvsim_methods_manuscript` or equivalent.

**Sub-tasks:**
- Port screening intervention (`screen_num` style).
- Port triage logic (full screen → triage → treat cascade).
- Port treatment interventions: `treat_num` and `treat_delay`.
- Port `radiation` intervention (cancer treatment).
- Port `dynamic_pars` for time-varying parameters (e.g., condom use).
- Add tests: screening scenarios reproduce `hpvsim_methods_manuscript` with overlapping intervals.

### M7: MultiSim and scenarios

**Demo:** Run N=20 seeds, combine, produce CIs for all previous milestones; run a scenario sweep.

**Acceptance test:** Previously-matched trajectories produce uncertainty intervals from seeds; `Scenarios`-based comparison works.

**Sub-tasks:**
- Verify `ss.MultiSim` works with `hpv.Sim` (multi-seed runs, result aggregation, median + quantiles).
- Port the `Scenarios` class for parameter sweeps and intervention comparisons.
- Port the `Sweep` class for systematic parameter variation.
- Verify `ss.parallel()` works with proper random-seed handling.
- Re-run M1–M6 acceptance tests under proper uncertainty quantification (overlap intervals, not deterministic-seed equality).

### M8: HIV–HPV co-infection

**Demo:** Reproduce HIV-stratified HPV prevalence and cancer incidence on a high-HIV-burden setting.

**Acceptance test:** Reproduces `hpvsim_rwanda` outputs with overlapping intervals.

**Sub-tasks:**
- Integrate STIsim's HIV transmission model with `hpv.Sim`.
- Implement `hpv_hiv_connector(ss.Connector)` for cross-disease effects.
- Port CD4-stratified HPV progression effects (accelerated CIN, altered clearance at low CD4).
- Port ART effects on HPV (partial immune restoration slows HPV progression).
- Add HIV-stratified results (HPV prevalence by HIV status, by age group).
- Add tests: reproduces `hpvsim_rwanda` outputs with overlapping intervals.

### M9: Remaining analyzers and plotting

**Demo:** All secondary analyzers work; key paper figures reproducible via built-in plotting.

**Acceptance test:** Each analyzer matches v2.x output; `sim.plot()` and plots-by-age/genotype produce the standard figures from validation papers.

**Sub-tasks:**
- Port `snapshot` analyzer.
- Port `age_pyramid` analyzer.
- Port `age_causal_infection` analyzer.
- Port `dalys` analyzer (YLL + YLD).
- Add type-distribution results (genotype distribution of cancers).
- Implement `sim.plot()` for HPV-specific result views.
- Add plots by age group and by genotype.
- Add intervention-impact plots.
- Add calibration-result plots (data vs. fit, parameter distributions, convergence).

### M10: Release readiness

**Demo:** Migration guide published; tutorials updated; docs on MkDocs/Quarto; `pip install hpvsim==3.0.0` works.

**Acceptance test:** Full analysis-repo suite (#64 + #68–#73 + #82–#87 set) reproduces on v3 within overlapping intervals; migration guide merged; `v3.0.0` published to PyPI.

**Sub-tasks:**
- Write migration guide (v2 → v3) documenting API changes, parameter remapping, and script conversion.
- Switch docs from Sphinx to MkDocs/Quarto (issue #32).
- Update tutorials to use v3.0 API.
- Add workflow examples for calibration, multi-country comparison, HPV-HIV, vaccination impact.
- Auto-generate API reference from docstrings.
- Fix automatic download failures (issue #30).
- Split data files for faster loading (issue #12).
- Verify save/load works correctly with the new architecture.
- Strip all remaining subclass-first delegations (tracking issues from M1–M9).
- Run the full analysis-repo validation suite; confirm overlapping intervals.
- Tag v3.0.0 and publish to PyPI.

## Scope items not pinned to a milestone

| Item | Suggested home | Notes |
|---|---|---|
| Population scaling (`pop_scale` / `total_pop`) | M2 | Required for long-horizon natural history |
| Age-specific migration | M2 | Required for demographic realism in multi-decade runs |
| `radiation` intervention | M6 | Part of the full intervention cascade |
| `dynamic_pars` | M6 | Needed to vary parameters over time |
| Additional genotypes (`hr`, `lo`) | M2 | Low priority; add when natural history is wired up |
| Multiscale modeling | Unscheduled | Low priority; not a release blocker |
| Save/load | M10 or opportunistic | Not capability-blocking |
| Split data files (#12) | M10 or opportunistic | Loading performance |
| Fix automatic download failures (#30) | M0 or M10 | Infrastructure hygiene |
| Sex-specific initial prevalence | M2 | v2.x seeds initial infections differently by sex |

## Out of scope

Not ported to v3.0 (can revisit in a future release):

- **Waning immunity** — never used in published analyses.
- **v2.x incidence-based HIV** — superseded by STIsim's transmission-based HIV.
- **`EventSchedule`** — rarely used.
- **Custom `settings.py`** — superseded by `ss.options`.

## Implementation conventions

These conventions apply to every milestone; contributors should align on them from day one.

1. **Continuous runnability invariant.** `hpv.Sim().run()` must return results at every commit on `v3.0-dev`. CI enforces this; a PR that breaks the invariant is not mergeable.
2. **Dual validation gates.**
   - **Development gate (per PR).** An *anchor scenario* (vanilla natural history, no interventions, fixed seed) plus per-milestone *capability scenarios* are run against locally-stored v2.x baselines. Target: ±10% relative drift per summary result. The pinned summary-result set, established in M0, is `sim.short_summary` (total HPV infections, total cancers, total cancer deaths, mean HPV prevalence, mean cancer incidence, mean ages of infection / cancer / cancer death) plus total population. **On failure the gate is informational, not auto-blocking**: the PR carries either a fix, or an explicit note classifying the drift as expected feature-misalignment with a tracking issue for re-convergence.
   - **Release gate (per milestone acceptance test and at v3.0.0).** Overlapping uncertainty intervals against the analysis-repo suite. This is the scientific gate.
3. **Subclass-first tactic permitted as an interim.** `class Foo(ss.X)` that delegates to v2.x logic is allowed during a milestone. Every such delegation must have a tracking issue to strip it before M10. No delegations ship in v3.0.0.
4. **Lift-and-shift exclusion — HIV only.** v2.x incidence-based HIV is not lifted; STIsim's transmission-based HIV is adopted directly. The sexual network is *not* excluded — lift-and-shift of the network is the expected starting point.
5. **In-place replacement, with quarantines.** v3 work replaces `hpvsim/` in place. v2 modules untouched by the current milestone are moved to `hpvsim/_v2_legacy/`; v2 tests that exercise removed APIs are moved to `tests/_legacy/`. Active code never imports from either quarantine — quarantines exist purely as a porting reference. M10 deletes both wholesale.

## Branching and sync strategy

- `v3.0-dev` was created off `rc2.3` on 2026-04-23 and is the long-lived branch where all migration work converges.
- `rc2.3` is frozen — no further development on it. It awaits merge to `main` once approved.
- Each milestone gets its own branch off `v3.0-dev`, named after the milestone (e.g., `m01-basic-transmission-sim`). All commits for that milestone's sub-tasks land on that branch.
- When a milestone is complete (all its sub-tasks done, acceptance test green locally), open a PR from the milestone branch to `v3.0-dev`. The PR is the team review surface — Robyn (per §RACI) reviews migration work here. After merge, the milestone branch can be deleted.
- A draft PR can be opened early in the milestone so CI runs on every push to the milestone branch; flip from draft to ready-for-review when the milestone is complete.
- `v3.0-dev` is always runnable (per §Implementation conventions item 1). The milestone-PR review enforces this on each merge.
- No merges from `v3.0-dev` back into `main` until v3.0.0 release at M10. At that point, `v3.0-dev` merges to `main`.
- No further development on `main` is expected until `v3.0-dev` merges. Should a critical bug fix land on `main` (e.g., a v2.x issue affecting `rc2.3`/`main`), it gets forward-merged into `v3.0-dev` via PR. Otherwise, periodic merges from `main` are unnecessary.
- CI runs on PR open/update, on manual workflow_dispatch, and on a daily cron (against the default branch). Existing triggers in `.github/workflows/tests.yaml` — no push trigger is added.
- Old branches `rc3`, `rc3-integration`, `rc3-jc` on `origin` are left untouched as read-only references.
- M00 (Foundation) was bootstrapped as direct commits to `v3.0-dev` rather than via a milestone branch, since the planning artifacts and regression harness pre-date the milestone-branch convention. The milestone-branch model starts with M01.

## GitHub milestones and issues

A hybrid update of `starsimhub/hpvsim` milestones and migration-labeled issues is tracked in a separate GitHub issue (linked below). The update:

- Reviews the existing M00–M11 milestones and renames or restructures them to match this plan's M0–M10 partition where they don't match.
- Closes issues that no longer map, with a pointer to their replacement.
- Opens new issues for sub-tasks in this plan's milestones (one per sub-task).
- Leaves v2.3 release work in its own existing milestone on `rc2.3` / `main` — not absorbed into the migration plan.

Tracking issue: [#95](https://github.com/starsimhub/hpvsim/issues/95).

## Linked documents

- [`KICKOFF_DISCUSSION.md`](./KICKOFF_DISCUSSION.md) — RACI rationale, scope discussion, timeline, open questions from the original kick-off.
- [`docs/superpowers/specs/2026-04-23-hpvsim-starsim-port-design.md`](./docs/superpowers/specs/2026-04-23-hpvsim-starsim-port-design.md) — design spec behind this plan.
- [Old `rc3` branch](https://github.com/starsimhub/hpvsim/tree/rc3) — read-only reference for the earlier port attempt.
- [Issue #64](https://github.com/starsimhub/hpvsim/issues/64) — canonical validation-repo list.
- Issues #68–#73 and #82–#87 — v2.3 reproducibility checks against the validation repos.

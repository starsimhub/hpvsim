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
- Port `People` as `hpv.People(ss.People)` subclass using lift-and-shift; open a tracking issue to strip the subclass before M9.
- Port HPVsim's custom sexual network as `hpv.Network(ss.Network)` using lift-and-shift; open a tracking issue to refine toward Starsim idioms before M9.
- Assemble a minimal `hpv.Sim(ss.Sim)` that composes `hpv.People` and `hpv.Network` and runs without interventions.
- Add tests for partnership pattern equivalence (age-mixing matrix, concurrency distribution, partnership duration distribution) vs. v2.x.
- Add tests for HPV prevalence trajectory vs. the stored v2 baseline from M0.

### M2: Natural history parity

**Demo:** Show age- and genotype-stratified prevalence/incidence curves matching v2.x.

**Acceptance test:** Age-stratified HPV / CIN / cancer incidence by genotype overlaps v2.x intervals.

**Sub-tasks:**
- Port disease progression for each of the 4 genotypes into `Genotype(ss.Disease)`.
- Port cross-immunity via `HPV(ss.Connector)`.
- Port `age_results` analyzer at minimum viable scope (enough for calibration to consume in M3).
- Add population scaling (`pop_scale` / `total_pop`).
- Add age-specific migration (from v2.x `people.py:check_migration()`).
- Add tests: age-stratified HPV / CIN / cancer incidence by genotype match v2.x baselines.

### M3: Calibration loop

**Demo:** End-to-end calibration of one country (e.g., India) to age-stratified cancer incidence data converges and reproduces a published calibration.

**Acceptance test:** Optuna-based calibration produces a posterior consistent with v2.x calibration on the same target data.

**Sub-tasks:**
- Integrate Starsim's Optuna-based calibration with `hpv.Sim`.
- Port `compute_gof()` and likelihood functions for cancer incidence by age.
- Add calibration tests that run a small number of trials end-to-end.
- Run a full calibration for India; confirm posterior parameter ranges overlap v2.x calibration.
- Write a short calibration guide section (expanded further in M9).

### M4: Vaccination scenarios

**Demo:** Run routine + catch-up + therapeutic vaccination and show cancer incidence reduction.

**Acceptance test:** Vaccination impact trajectory overlaps v2.x intervals on `hpvsim_1dose` / `hpvsim_pxv_younger`.

**Sub-tasks:**
- Port product base classes (`dx`, `tx`, `vx`) and adapt CSV product files from v2.x `data/`.
- Port `routine_vx` intervention.
- Port `campaign_vx` intervention.
- Port `txvx` (therapeutic vaccination: `BaseTxVx`, `routine_txvx`, `campaign_txvx`, `linked_txvx`).
- Add intervention-level results tracking (number vaccinated, doses administered, by age and year).
- Add tests: vaccination scenarios reproduce `hpvsim_1dose` / `hpvsim_pxv_younger` with overlapping intervals.

### M5: Screen-and-treat cascade

**Demo:** Run a screen → triage → treat scenario on one country.

**Acceptance test:** Screening scenario overlaps v2.x intervals on `hpvsim_methods_manuscript` or equivalent.

**Sub-tasks:**
- Port screening intervention (`screen_num` style).
- Port triage logic (full screen → triage → treat cascade).
- Port treatment interventions: `treat_num` and `treat_delay`.
- Port `radiation` intervention (cancer treatment).
- Port `dynamic_pars` for time-varying parameters (e.g., condom use).
- Add tests: screening scenarios reproduce `hpvsim_methods_manuscript` with overlapping intervals.

### M6: MultiSim and scenarios

**Demo:** Run N=20 seeds, combine, produce CIs for all previous milestones; run a scenario sweep.

**Acceptance test:** Previously-matched trajectories produce uncertainty intervals from seeds; `Scenarios`-based comparison works.

**Sub-tasks:**
- Verify `ss.MultiSim` works with `hpv.Sim` (multi-seed runs, result aggregation, median + quantiles).
- Port the `Scenarios` class for parameter sweeps and intervention comparisons.
- Port the `Sweep` class for systematic parameter variation.
- Verify `ss.parallel()` works with proper random-seed handling.
- Re-run M1–M5 acceptance tests under proper uncertainty quantification (overlap intervals, not deterministic-seed equality).

### M7: HIV–HPV co-infection

**Demo:** Reproduce HIV-stratified HPV prevalence and cancer incidence on a high-HIV-burden setting.

**Acceptance test:** Reproduces `hpvsim_rwanda` outputs with overlapping intervals.

**Sub-tasks:**
- Integrate STIsim's HIV transmission model with `hpv.Sim`.
- Implement `hpv_hiv_connector(ss.Connector)` for cross-disease effects.
- Port CD4-stratified HPV progression effects (accelerated CIN, altered clearance at low CD4).
- Port ART effects on HPV (partial immune restoration slows HPV progression).
- Add HIV-stratified results (HPV prevalence by HIV status, by age group).
- Add tests: reproduces `hpvsim_rwanda` outputs with overlapping intervals.

### M8: Remaining analyzers and plotting

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

### M9: Release readiness

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
- Strip all remaining subclass-first delegations (tracking issues from M1–M8).
- Run the full analysis-repo validation suite; confirm overlapping intervals.
- Tag v3.0.0 and publish to PyPI.

## Scope items not pinned to a milestone

| Item | Suggested home | Notes |
|---|---|---|
| Population scaling (`pop_scale` / `total_pop`) | M2 | Required for long-horizon natural history |
| Age-specific migration | M2 | Required for demographic realism in multi-decade runs |
| `radiation` intervention | M5 | Part of the full intervention cascade |
| `dynamic_pars` | M5 | Needed to vary parameters over time |
| Additional genotypes (`hr`, `lo`) | M2 | Low priority; add when natural history is wired up |
| Multiscale modeling | Unscheduled | Low priority; not a release blocker |
| Save/load | M9 or opportunistic | Not capability-blocking |
| Split data files (#12) | M9 or opportunistic | Loading performance |
| Fix automatic download failures (#30) | M0 or M9 | Infrastructure hygiene |
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
3. **Subclass-first tactic permitted as an interim.** `class Foo(ss.X)` that delegates to v2.x logic is allowed during a milestone. Every such delegation must have a tracking issue to strip it before M9. No delegations ship in v3.0.0.
4. **Lift-and-shift exclusion — HIV only.** v2.x incidence-based HIV is not lifted; STIsim's transmission-based HIV is adopted directly. The sexual network is *not* excluded — lift-and-shift of the network is the expected starting point.

## Branching and sync strategy

- `v3.0-dev` was created off `rc2.3` on 2026-04-23. rc2.3 is feature-complete and will merge into `main` in due course.
- Periodically, `main` is merged into `v3.0-dev`. Normal merge commits — no rebase, no force-push.
- No merges from `v3.0-dev` back into `main` until v3.0.0 release.
- Bugs discovered on `v3.0-dev` that also affect `main` are fixed on `main` first, then merged forward.
- Old branches `rc3`, `rc3-integration`, `rc3-jc` on `origin` are left untouched as read-only references.

## GitHub milestones and issues

A hybrid update of `starsimhub/hpvsim` milestones and migration-labeled issues is tracked in a separate GitHub issue (linked below). The update:

- Reviews the existing M00–M11 milestones and renames or restructures them to match this plan's M0–M9 partition where they don't match.
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

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

**Demo:** `v3.0-dev` branch exists with CI green on a stub sim; `KICKOFF_DISCUSSION.md` + `MIGRATION_PLAN.md` committed; v2.x baseline outputs stored; anchor-scenario script and ±10% comparison script committed.

**Acceptance test:** CI passes; regression harness compares a v3 run to a stored v2 baseline and reports diffs.

**Sub-tasks:**
- Set up CI on `v3.0-dev` (adapted from rc2.3 CI).
- Generate v2.x baseline outputs for the regression harness and commit them (or commit a script that reproduces them deterministically).
- Write the anchor scenario script: vanilla 4-genotype HPV sim, one country, fixed seed, no interventions.
- Write the ±10% comparison script that diffs a v3 run against the stored v2 baseline and emits per-summary-result drift.
- Document how to run the regression harness in a `CONTRIBUTING.md` (or equivalent) section.

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

## Scope items not pinned to a milestone

## Out of scope

## Implementation conventions

## Branching and sync strategy

## GitHub milestones and issues

## Linked documents

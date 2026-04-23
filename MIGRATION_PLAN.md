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

## Scope items not pinned to a milestone

## Out of scope

## Implementation conventions

## Branching and sync strategy

## GitHub milestones and issues

## Linked documents

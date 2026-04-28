# HPVsim v2.3 → Starsim Port: Planning Spec

**Date:** 2026-04-23
**Status:** Design for review
**Scope:** This spec describes the *planning artifacts* for the HPVsim v2 → v3 port, not the port itself. It defines what the new working branch, migration plan document, and GitHub update approach look like. The migration work itself is covered by a separate implementation plan derived from this spec.

## Overview

HPVsim v3.0 will be a reimplementation of HPVsim on the [Starsim](https://starsim.org/) framework, replacing v2.x's fully custom architecture. A first attempt at this port lives on the stale `rc3` branch but has drifted out of sync with `main`. This spec designs the three artifacts needed to restart cleanly:

1. A new `v3.0-dev` branch created off `rc2.3` (which is feature-complete and will soon merge into `main`).
2. A new `MIGRATION_PLAN.md` on that branch, re-derived from scratch using the old plan as reference.
3. A hybrid GitHub milestone/issue update, deferred until the implementation plan is written.

The old `rc3` branch and its associated `MIGRATION_PLAN.md` and `KICKOFF_DISCUSSION.md` serve as references. `KICKOFF_DISCUSSION.md` is carried forward into the new branch verbatim; the old plan is not.

## Scope decisions (settled)

Carried forward from the old plan (confirmed 2026-04-23):

- **Therapeutic vaccination (txvx):** Port to v3.
- **EventSchedule:** Drop.
- **Custom `settings.py`:** Drop; rely on `ss.options`.
- **Waning immunity:** Drop (never used in any published analysis).
- **v2.x incidence-based HIV module:** Drop. Replace with STIsim's transmission-based HIV model.
- **Multiscale modeling:** Low priority; not a release blocker.

Revised 2026-04-23:

- **Sexual network:** Port HPVsim's custom network to Starsim's `Network` framework. This reverses the old plan's choice to adopt `sti.StructuredSexual`. Rationale: reduces error sources when validating against v2.x baselines and provides a known reference for testing the STIsim network later. Lift-and-shift via `class hpv.Network(ss.Network)` is the expected starting point.

## Validation criteria

Release gate: the v3 implementation reproduces the v2.x analysis-repo suite (referenced in issue #64 and validated against v2.3 under issues #68–#73, #82–#87 — `hpvsim_methods_manuscript`, `hpvsim_india`, `hpvsim_rwanda`, `hpvsim_1dose`, `hpvsim_pxv_younger`, `hpv_faster_kenya`) with **overlapping uncertainty intervals** on their headline results. Bit-for-bit reproducibility is not required.

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

## The new branch

**Name:** `v3.0-dev`.

**Creation:** Off current `rc2.3` tip. rc2.3 is feature-complete, so this is safe; no need to wait for rc2.3 → main merge.

**Initial contents:**
- rc2.3 tip code, unchanged
- `KICKOFF_DISCUSSION.md` copied verbatim from `origin/rc3`
- New `MIGRATION_PLAN.md` authored from this spec and its implementation plan
- No other code from old `rc3` is carried over at branch creation

**Sync strategy:**
- Merge `main` (which will include rc2.3) into `v3.0-dev` periodically.
- No rebases — normal merge commits. Avoids force-pushes that would affect collaborators.
- No merges from `v3.0-dev` back into `main` until v3.0.0 release.
- Bugs discovered on `v3.0-dev` that also affect `main` are fixed on `main` first, then merged forward.

**Old branches:** `rc3`, `rc3-integration`, and `rc3-jc` on `origin` are left untouched as read-only references. No renaming, no deletion. The team is informed that these are deprecated.

## Migration plan document structure

The new `MIGRATION_PLAN.md` has these sections:

| Section | Purpose |
|---|---|
| Overview | One-paragraph statement of what v3.0 is and how "done" is defined |
| Validation criteria | Overlapping-intervals definition; names the analysis-repo suite as release gate |
| Scope decisions (settled) | The settled/revised decisions listed above, with dates |
| RACI | Roles |
| Architecture mapping | Table from v2.x constructs → v3 constructs |
| Milestones | M0–M9 table with demo, acceptance test, dependencies, and sub-tasks per milestone |
| Scope items not pinned to a milestone | Appendix with suggested milestone homes |
| Out of scope | Explicit non-goals with one-line reasons |
| Implementation conventions | The five conventions listed below |
| Branching and sync strategy | Short summary of the branch plan above |
| GitHub milestones and issues | Short stub describing the deferred hybrid update |
| Linked documents | `KICKOFF_DISCUSSION.md`, old `rc3` branch, validation repos |

Sub-task granularity inside milestones matches the old plan (~5 sub-tasks per milestone, mapping 1:1 to GitHub issues when the GitHub update happens). No effort estimates and no status column — GitHub tracks status.

## Milestone structure

Organized as thin vertical slices. Each milestone produces a user-visible demo and must meet its acceptance test before the next begins. Tests written during a milestone stay in CI from that point onward.

| # | Milestone | Demo | Acceptance test |
|---|---|---|---|
| M0 | **Foundation** | `v3.0-dev` branch exists with CI green on a stub sim; `KICKOFF_DISCUSSION.md` + `MIGRATION_PLAN.md` committed; v2.x baseline-generation script committed (baseline files generated locally and gitignored); anchor-scenario script and ±10% comparison script committed | CI passes; regression harness compares a v3 run to a locally-generated v2 baseline and reports diffs |
| M1 | **Basic transmission sim** | Run a vanilla 4-genotype HPV sim on one country; plot aggregate HPV prevalence over time. Sub-tasks include porting `People` and porting the custom sexual network (subclass-first lift-and-shift allowed) | Aggregate HPV prevalence trajectory overlaps v2.x uncertainty interval on the same country/pars; partnership patterns (age mixing, concurrency, duration) match v2.x |
| M2 | **Natural history parity** | Show age- and genotype-stratified prevalence/incidence curves matching v2.x (requires disease progression, cross-immunity, `age_results` analyzer) | Age-stratified HPV / CIN / cancer incidence by genotype overlaps v2.x intervals |
| M3 | **Calibration loop** | End-to-end calibration of one country to age-stratified cancer incidence data converges and reproduces a published calibration | Optuna-based calibration on India (or equivalent) produces a posterior consistent with v2.x calibration |
| M4 | **Vaccination scenarios** | Run routine + catch-up + therapeutic vaccination and show cancer incidence reduction (product system + `routine_vx` / `campaign_vx` / `txvx`) | Vaccination impact trajectory overlaps v2.x intervals on `hpvsim_1dose` / `hpvsim_pxv_younger` |
| M5 | **Screen-and-treat cascade** | Run a screen → triage → treat scenario (dx/tx products + screening intervention + triage + `treat_num` / `treat_delay`) | Screening scenario overlaps v2.x intervals on `hpvsim_methods_manuscript` or equivalent |
| M6 | **MultiSim and scenarios** | Run N=20 seeds, combine, produce CIs for all previous milestones; run a scenario sweep (`ss.MultiSim` + `Scenarios`) | Previously-matched trajectories produce uncertainty intervals from seeds; `Scenarios`-based comparison works |
| M7 | **HIV–HPV co-infection** | Reproduce HIV-stratified HPV prevalence and cancer incidence on a high-HIV-burden setting (STIsim HIV + CD4-stratified progression + `hpv_hiv_connector`) | Reproduces `hpvsim_rwanda` outputs with overlapping intervals |
| M8 | **Remaining analyzers + plotting** | `snapshot`, `age_pyramid`, `age_causal_infection`, `dalys`, type distribution; `sim.plot()` + plots by age/genotype; intervention plots | Each analyzer matches v2.x output; key paper figures reproducible via built-in plotting |
| M9 | **Release readiness** | Migration guide published; tutorials updated; docs moved to MkDocs/Quarto; data loader fixes; API reference; v3.0.0 on PyPI | Full analysis-repo suite (#64 + #68–#73 set) reproduces on v3 within overlapping intervals; migration guide merged; `pip install hpvsim==3.0.0` works |

**Explicit sequencing notes:**
- M3 (calibration) comes before M4 (vaccination): calibration validates natural-history parity against data, which makes M4's acceptance test defensible.
- M6 (MultiSim) follows single-sim capability work: earlier milestones use deterministic seeds + stored v2.x baselines for acceptance, then are re-validated under proper uncertainty quantification at M6.
- M7 (HIV) is late because it depends on most of M1–M5 being in place; Rwanda reproduction is the validation anchor.
- No dedicated "testing" milestone — tests are added inside each milestone and gated by CI.

## Scope items not pinned to a milestone

| Item | Suggested home | Notes |
|---|---|---|
| Population scaling (`pop_scale` / `total_pop`) | M2 | Required for long-horizon natural history |
| Age-specific migration | M2 | Required for demographic realism in multi-decade runs |
| `radiation` intervention | M5 | Part of the full intervention cascade |
| `dynamic_pars` | M5 | Needed to vary parameters over time (e.g., condom use) |
| Multiscale modeling | Unscheduled | Low priority; not a release blocker |
| Save/load | M9 or opportunistic | Not capability-blocking |
| Split data files | M9 or opportunistic | Loading performance |
| Fix automatic download failures | M0 or M9 | Infrastructure hygiene |
| Additional genotypes (`hr`, `lo`) | M2 | Low priority; add when natural history is wired up |

## Out of scope for v3.0

- Waning immunity (never used in published analyses; revisit post-v3.0 if needed).
- v2.x incidence-based HIV (superseded by STIsim HIV).
- `EventSchedule` (rarely used).
- Custom `settings.py` (superseded by `ss.options`).

## Implementation conventions

These conventions apply across all milestones and are called out in the plan so contributors align on them from day one.

1. **Continuous runnability invariant.** `hpv.Sim().run()` must return results at every commit on `v3.0-dev`. CI enforces this — no PR that breaks the invariant is mergeable.
2. **Dual validation gates.**
   - **Development gate (per PR):** An *anchor scenario* (vanilla natural history, no interventions, fixed seed) plus per-milestone *capability scenarios* are run against stored v2.x baselines. Target: ±10% per summary result, where "summary result" initially means total HPV prevalence, age-standardized CIN prevalence, age-standardized cancer incidence, and total population; the exact list is pinned in M0 alongside the comparison script. On failure, the gate is **informational, not auto-blocking**: the PR must carry either a fix, or an explicit classification of the drift as expected feature misalignment with a tracking issue for re-convergence.
   - **Release gate (per milestone acceptance test and at v3.0.0):** Overlapping uncertainty intervals against the analysis-repo suite. This is the scientific gate.
3. **Subclass-first tactic permitted as an interim.** `class Foo(ss.X)` that delegates to v2.x logic is allowed during a milestone. Every such delegation must have a tracking issue to strip it before M9. No delegations ship in v3.0.0.
4. **Lift-and-shift exclusion — HIV only.** v2.x incidence-based HIV is not lifted; STIsim's transmission-based HIV is adopted directly. The sexual network is *not* excluded — lift-and-shift of the network is the expected starting point.

## Branch setup mechanics

Concrete steps to stand up `v3.0-dev`. These are the first entries in the implementation plan.

| Step | What | Who |
|---|---|---|
| 1 | `git checkout rc2.3 && git pull`; confirm up-to-date with `origin/rc2.3` tip | Ryan |
| 2 | `git checkout -b v3.0-dev` | Ryan |
| 3 | `git show origin/rc3:KICKOFF_DISCUSSION.md > KICKOFF_DISCUSSION.md` — carry forward kickoff notes verbatim | Ryan |
| 4 | Add new `MIGRATION_PLAN.md`, authored from this spec and the implementation plan | Ryan |
| 5 | Commit (e.g., "Initialize v3.0-dev branch with migration plan and kickoff notes") and push: `git push -u origin v3.0-dev` | Ryan |
| 6 | On GitHub: set branch protection rules on `v3.0-dev` matching `rc2.3`/`main` (PR required; CI required once configured) | Ryan / admin |
| 7 | Announce on team channel: new branch is `v3.0-dev`; old `rc3` / `rc3-integration` / `rc3-jc` are deprecated but left in place as read-only references | Ryan |

## GitHub update approach (deferred)

A hybrid milestone/issue update will be performed once this plan is merged:

- The existing M00–M11 milestones are reviewed and renamed/restructured to match the new M0–M9 partition where they don't match.
- Issues that no longer map to a milestone are closed with a pointer to the replacement.
- New issues are opened to cover the gap, one per sub-task in the new plan's milestones.
- v2.3 release work stays in its own existing milestone on `rc2.3`/`main` and is **not** absorbed into the migration plan.

The specific mapping is tracked in a separate issue created alongside the GitHub update, not in `MIGRATION_PLAN.md`.

## Linked documents

- `KICKOFF_DISCUSSION.md` (in repo root on `v3.0-dev`) — RACI, scope rationale, timeline, open questions
- Old `rc3` branch — read-only reference for previous port attempt
- Issue [#64](https://github.com/starsimhub/hpvsim/issues/64) and issues #68–#73 on `starsimhub/hpvsim` — canonical validation-repo set

## Out of scope for this spec

- Implementation details of any individual milestone's code (those belong in per-milestone design work during the migration).
- Time estimates or target release date (timeline is flexible).
- The migration plan's prose itself — this spec defines its shape, not its wording.
- The GitHub mapping specifics (deferred to post-implementation-plan).

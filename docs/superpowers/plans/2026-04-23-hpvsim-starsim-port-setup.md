# HPVsim v2 → Starsim Port Setup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `v3.0-dev` branch with `KICKOFF_DISCUSSION.md` carried forward from old `rc3` and a freshly authored `MIGRATION_PLAN.md`, then push to `origin`, configure branch protection, announce to the team, and open a tracking issue for the deferred GitHub milestone/issue update.

**Architecture:** All work happens on `v3.0-dev` (already created, currently contains the planning spec as its only commit). No code migration is executed by this plan — the only code-visible change is a new documentation file. The deliverables are: a branch on `origin`, two Markdown docs, configured branch protection, a team announcement, and a GitHub tracking issue. No test framework changes in this plan — that's M0 of the migration itself.

**Tech Stack:** git, gh (GitHub CLI), Markdown.

---

## Preconditions

These must be true before starting. Verify them before Task 1.

- [ ] **Confirm current branch is `v3.0-dev`**

Run: `git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol branch --show-current`
Expected: `v3.0-dev`

- [ ] **Confirm last commit is the planning spec**

Run: `git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol log --oneline -1`
Expected: one line containing `Add HPVsim v2 → Starsim port planning spec` (commit `1f4f4f44` or an amendment thereof).

- [ ] **Confirm `origin/rc3` is reachable**

Run: `git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol show origin/rc3:KICKOFF_DISCUSSION.md | head -3`
Expected: first three lines of the old KICKOFF doc (starts with `# HPVsim v3.0 Migration — Kick-off Discussion Points`).

- [ ] **Confirm `gh` CLI is authenticated for `starsimhub/hpvsim`**

Run: `gh auth status`
Expected: "Logged in to github.com" with a user that has write access to `starsimhub/hpvsim`.

---

## Task 1: Carry forward `KICKOFF_DISCUSSION.md`

**Files:**
- Create: `KICKOFF_DISCUSSION.md` (repo root)

- [ ] **Step 1: Copy the file verbatim from `origin/rc3`**

Run:
```bash
git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol show origin/rc3:KICKOFF_DISCUSSION.md > /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/KICKOFF_DISCUSSION.md
```

- [ ] **Step 2: Verify the file was created and matches `origin/rc3`'s version**

Run:
```bash
diff <(git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol show origin/rc3:KICKOFF_DISCUSSION.md) /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/KICKOFF_DISCUSSION.md
```
Expected: no output (files identical).

- [ ] **Step 3: Stage and commit**

Run:
```bash
git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol add KICKOFF_DISCUSSION.md
git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol commit -m "$(cat <<'EOF'
Carry forward KICKOFF_DISCUSSION.md from old rc3

Copied verbatim from origin/rc3 as a reference for scope rationale,
RACI discussion, and open questions from the original kick-off.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Verify commit is on v3.0-dev**

Run: `git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol log --oneline -3`
Expected: the new commit on top, previous two are "Add HPVsim v2 → Starsim port planning spec" and "Merge main into rc2.3; resolve conflicts".

---

## Task 2: Create `MIGRATION_PLAN.md` skeleton and write Overview + Validation sections

**Files:**
- Create: `MIGRATION_PLAN.md` (repo root)

- [ ] **Step 1: Create file with section headers and authored Overview and Validation sections**

Create `/c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/MIGRATION_PLAN.md` with exactly this content:

```markdown
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

## RACI

## Architecture mapping

## Milestones

## Scope items not pinned to a milestone

## Out of scope

## Implementation conventions

## Branching and sync strategy

## GitHub milestones and issues

## Linked documents
```

- [ ] **Step 2: Verify the file was created with the expected sections**

Run:
```bash
grep '^## ' /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/MIGRATION_PLAN.md
```
Expected output (12 lines, in this order):
```
## Overview
## Validation criteria
## Scope decisions (settled)
## RACI
## Architecture mapping
## Milestones
## Scope items not pinned to a milestone
## Out of scope
## Implementation conventions
## Branching and sync strategy
## GitHub milestones and issues
## Linked documents
```

- [ ] **Step 3: Commit the skeleton**

Run:
```bash
git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol add MIGRATION_PLAN.md
git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol commit -m "$(cat <<'EOF'
Draft MIGRATION_PLAN.md: Overview and Validation criteria

First section of the new migration plan. Remaining sections (Scope
decisions, RACI, Architecture, Milestones, Conventions) are added
in subsequent commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Fill in Scope decisions, RACI, and Architecture mapping

**Files:**
- Modify: `MIGRATION_PLAN.md` (repo root) — replace the three empty section stubs

- [ ] **Step 1: Replace `## Scope decisions (settled)` with authored content**

Replace the single line `## Scope decisions (settled)` with this block:

```markdown
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
```

- [ ] **Step 2: Replace `## RACI` with authored content**

Replace the single line `## RACI` with this block:

```markdown
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
```

- [ ] **Step 3: Replace `## Architecture mapping` with authored content**

Replace the single line `## Architecture mapping` with this block:

```markdown
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
```

- [ ] **Step 4: Verify the file still has all 12 section headers in order**

Run:
```bash
grep '^## ' /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/MIGRATION_PLAN.md
```
Expected: same 12 section headers in the same order as after Task 2 Step 2.

- [ ] **Step 5: Commit**

Run:
```bash
git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol add MIGRATION_PLAN.md
git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol commit -m "$(cat <<'EOF'
Draft MIGRATION_PLAN.md: Scope decisions, RACI, Architecture mapping

Records the settled scope decisions (txvx kept; EventSchedule,
settings.py, waning immunity, v2.x HIV dropped; network ported
rather than using sti.StructuredSexual), the RACI table, and the
v2.x-to-v3.0 architecture mapping.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Write the Milestones section (M0–M4)

**Files:**
- Modify: `MIGRATION_PLAN.md` — replace the `## Milestones` stub with the first half

- [ ] **Step 1: Replace `## Milestones` with the section header, preamble, and milestones M0–M4**

Replace the single line `## Milestones` with this block:

```markdown
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
```

- [ ] **Step 2: Verify the file still has all 12 top-level section headers**

Run:
```bash
grep '^## ' /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/MIGRATION_PLAN.md
```
Expected: same 12 section headers as before; new `### M0`–`### M4` sub-headers exist but don't change the top-level count.

- [ ] **Step 3: Commit**

Run:
```bash
git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol add MIGRATION_PLAN.md
git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol commit -m "$(cat <<'EOF'
Draft MIGRATION_PLAN.md: Milestones M0–M4

Defines the first five milestones as thin vertical slices with
demos, acceptance tests, and sub-tasks. M0 is Foundation; M1–M4
cover basic transmission, natural history parity, calibration,
and vaccination scenarios.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Write the Milestones section (M5–M9)

**Files:**
- Modify: `MIGRATION_PLAN.md` — append the second half of the milestones section after M4

- [ ] **Step 1: Insert M5–M9 after the M4 block**

Immediately after the `### M4` block (before `## Scope items not pinned to a milestone`), insert:

```markdown
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
```

- [ ] **Step 2: Verify all 10 milestones are present**

Run:
```bash
grep '^### M' /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/MIGRATION_PLAN.md
```
Expected:
```
### M0: Foundation
### M1: Basic transmission sim
### M2: Natural history parity
### M3: Calibration loop
### M4: Vaccination scenarios
### M5: Screen-and-treat cascade
### M6: MultiSim and scenarios
### M7: HIV–HPV co-infection
### M8: Remaining analyzers and plotting
### M9: Release readiness
```

- [ ] **Step 3: Commit**

Run:
```bash
git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol add MIGRATION_PLAN.md
git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol commit -m "$(cat <<'EOF'
Draft MIGRATION_PLAN.md: Milestones M5–M9

Completes the milestone section: screen-and-treat cascade, MultiSim
and scenarios, HIV–HPV co-infection, remaining analyzers/plotting,
and release readiness.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Write the appendix sections

**Files:**
- Modify: `MIGRATION_PLAN.md` — replace the six empty appendix stubs

- [ ] **Step 1: Replace `## Scope items not pinned to a milestone`**

Replace the single line `## Scope items not pinned to a milestone` with:

```markdown
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
```

- [ ] **Step 2: Replace `## Out of scope`**

Replace the single line `## Out of scope` with:

```markdown
## Out of scope

Not ported to v3.0 (can revisit in a future release):

- **Waning immunity** — never used in published analyses.
- **v2.x incidence-based HIV** — superseded by STIsim's transmission-based HIV.
- **`EventSchedule`** — rarely used.
- **Custom `settings.py`** — superseded by `ss.options`.
```

- [ ] **Step 3: Replace `## Implementation conventions`**

Replace the single line `## Implementation conventions` with:

```markdown
## Implementation conventions

These conventions apply to every milestone; contributors should align on them from day one.

1. **Continuous runnability invariant.** `hpv.Sim().run()` must return results at every commit on `v3.0-dev`. CI enforces this; a PR that breaks the invariant is not mergeable.
2. **Dual validation gates.**
   - **Development gate (per PR).** An *anchor scenario* (vanilla natural history, no interventions, fixed seed) plus per-milestone *capability scenarios* are run against stored v2.x baselines. Target: ±10% per summary result, where "summary result" initially means total HPV prevalence, age-standardized CIN prevalence, age-standardized cancer incidence, and total population; the exact list is pinned in M0 alongside the comparison script. **On failure the gate is informational, not auto-blocking**: the PR carries either a fix, or an explicit note classifying the drift as expected feature-misalignment with a tracking issue for re-convergence.
   - **Release gate (per milestone acceptance test and at v3.0.0).** Overlapping uncertainty intervals against the analysis-repo suite. This is the scientific gate.
3. **Subclass-first tactic permitted as an interim.** `class Foo(ss.X)` that delegates to v2.x logic is allowed during a milestone. Every such delegation must have a tracking issue to strip it before M9. No delegations ship in v3.0.0.
4. **Lift-and-shift exclusion — HIV only.** v2.x incidence-based HIV is not lifted; STIsim's transmission-based HIV is adopted directly. The sexual network is *not* excluded — lift-and-shift of the network is the expected starting point.
```

- [ ] **Step 4: Replace `## Branching and sync strategy`**

Replace the single line `## Branching and sync strategy` with:

```markdown
## Branching and sync strategy

- `v3.0-dev` was created off `rc2.3` on 2026-04-23. rc2.3 is feature-complete and will merge into `main` in due course.
- Periodically, `main` is merged into `v3.0-dev`. Normal merge commits — no rebase, no force-push.
- No merges from `v3.0-dev` back into `main` until v3.0.0 release.
- Bugs discovered on `v3.0-dev` that also affect `main` are fixed on `main` first, then merged forward.
- Old branches `rc3`, `rc3-integration`, `rc3-jc` on `origin` are left untouched as read-only references.
```

- [ ] **Step 5: Replace `## GitHub milestones and issues`**

Replace the single line `## GitHub milestones and issues` with:

```markdown
## GitHub milestones and issues

A hybrid update of `starsimhub/hpvsim` milestones and migration-labeled issues is tracked in a separate GitHub issue (linked below). The update:

- Reviews the existing M00–M11 milestones and renames or restructures them to match this plan's M0–M9 partition where they don't match.
- Closes issues that no longer map, with a pointer to their replacement.
- Opens new issues for sub-tasks in this plan's milestones (one per sub-task).
- Leaves v2.3 release work in its own existing milestone on `rc2.3` / `main` — not absorbed into the migration plan.

Tracking issue: *(to be created; linked here after creation)*.
```

- [ ] **Step 6: Replace `## Linked documents`**

Replace the single line `## Linked documents` with:

```markdown
## Linked documents

- [`KICKOFF_DISCUSSION.md`](./KICKOFF_DISCUSSION.md) — RACI rationale, scope discussion, timeline, open questions from the original kick-off.
- [`docs/superpowers/specs/2026-04-23-hpvsim-starsim-port-design.md`](./docs/superpowers/specs/2026-04-23-hpvsim-starsim-port-design.md) — design spec behind this plan.
- [Old `rc3` branch](https://github.com/starsimhub/hpvsim/tree/rc3) — read-only reference for the earlier port attempt.
- [Issue #64](https://github.com/starsimhub/hpvsim/issues/64) — canonical validation-repo list.
- Issues #68–#73 and #82–#87 — v2.3 reproducibility checks against the validation repos.
```

- [ ] **Step 7: Verify all top-level and milestone sections are present**

Run:
```bash
grep -c '^## ' /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/MIGRATION_PLAN.md
```
Expected: `12`.

Run:
```bash
grep -c '^### M' /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/MIGRATION_PLAN.md
```
Expected: `10`.

- [ ] **Step 8: Commit**

Run:
```bash
git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol add MIGRATION_PLAN.md
git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol commit -m "$(cat <<'EOF'
Draft MIGRATION_PLAN.md: appendix sections

Fills in scope items not pinned to a milestone, out-of-scope list,
implementation conventions (continuous runnability, dual validation
gates, subclass-first tactic, HIV-only lift-and-shift exclusion),
branching/sync strategy, GitHub-update stub, and linked documents.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Self-review `MIGRATION_PLAN.md` against the spec

**Files:**
- Read-only: `MIGRATION_PLAN.md`, `docs/superpowers/specs/2026-04-23-hpvsim-starsim-port-design.md`

- [ ] **Step 1: Verify every spec-required section appears**

Run:
```bash
grep '^## ' /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/MIGRATION_PLAN.md
```
Expected (exactly these 12, in order):
```
## Overview
## Validation criteria
## Scope decisions (settled)
## RACI
## Architecture mapping
## Milestones
## Scope items not pinned to a milestone
## Out of scope
## Implementation conventions
## Branching and sync strategy
## GitHub milestones and issues
## Linked documents
```
If any are missing or reordered, fix the file and amend the relevant commit (or make a follow-up commit), then re-run this step.

- [ ] **Step 2: Verify scope decisions in the plan match the spec**

Run:
```bash
grep -A2 'Therapeutic vaccination\|EventSchedule\|settings.py\|Waning immunity\|incidence-based HIV\|Multiscale modeling\|Sexual network' /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/MIGRATION_PLAN.md
```
Expected: each of the seven settled/revised decisions appears with the disposition stated in the spec (port, drop, drop, drop, drop/replace, low priority, port). If any are missing or contradict the spec, fix and commit.

- [ ] **Step 3: Verify all four implementation conventions are present**

Run:
```bash
grep -c '^[0-9]\. \*\*' /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/MIGRATION_PLAN.md
```
Expected: `4` (the four numbered conventions in `## Implementation conventions`).

- [ ] **Step 4: Verify no placeholder text remains**

Run:
```bash
grep -nE 'TODO|FIXME|\bTBD\b|\bXXX\b' /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/MIGRATION_PLAN.md
```
Expected: **no matches** (grep exits with status 1, which is normal). The `*(to be created; linked here after creation)*` line in the GitHub milestones section is an intentional placeholder that Task 11 replaces; it does not contain any of the patterns above. If any unexpected placeholders appear (e.g., a forgotten TBD), fix and commit.

- [ ] **Step 5: Verify all 10 milestone blocks include Demo, Acceptance test, and Sub-tasks**

Run:
```bash
for milestone in M0 M1 M2 M3 M4 M5 M6 M7 M8 M9; do
  echo "=== $milestone ==="
  awk "/^### $milestone:/,/^### [A-Z]|^## /" /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/MIGRATION_PLAN.md | grep -E '^\*\*Demo|^\*\*Acceptance|^\*\*Sub-tasks' | sort -u
done
```
Expected: each milestone shows `**Demo:**`, `**Acceptance test:**`, and `**Sub-tasks:**`. If any are missing, fix and commit.

---

## Task 8: Push `v3.0-dev` to `origin`

- [ ] **Step 1: Confirm current state is clean and on `v3.0-dev`**

Run:
```bash
git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol status
git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol branch --show-current
```
Expected: "nothing to commit, working tree clean" and `v3.0-dev`.

- [ ] **Step 2: Confirm branch history**

Run: `git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol log --oneline v3.0-dev ^rc2.3`
Expected: the commits added by Tasks 1–6 of this plan (6 commits: KICKOFF_DISCUSSION carry-forward, MIGRATION_PLAN sections 1/2/3/4/5) plus the spec commit. A count of 7 commits is expected.

- [ ] **Step 3: Push with upstream tracking**

**⚠ Confirm with the user before running this step — pushing to `origin` creates a remote branch visible to the team.**

Run: `git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol push -u origin v3.0-dev`
Expected: "Branch 'v3.0-dev' set up to track remote branch 'v3.0-dev' from 'origin'."

- [ ] **Step 4: Verify the branch exists on origin**

Run: `gh api repos/starsimhub/hpvsim/branches/v3.0-dev --jq '.name, .commit.sha' 2>&1`
Expected: `v3.0-dev` and a commit SHA matching the local `v3.0-dev` HEAD.

---

## Task 9: Configure branch protection on `v3.0-dev`

**Intent:** Match the protection rules already on `rc2.3` / `main`: require PR reviews and required status checks. If those rules aren't already set on `rc2.3` / `main`, skip this task and open a separate issue for the team to decide on the protection policy.

- [ ] **Step 1: Inspect protection on `rc2.3` (or `main`) as the template**

Run: `gh api repos/starsimhub/hpvsim/branches/rc2.3/protection 2>&1`
Expected: either a JSON object listing the protection rules (record which fields are set), or `"Branch not protected"`.

If the branch is not protected, **skip the remaining steps of this task** and note this in the final report; raise it as a follow-up question for the user.

- [ ] **Step 2: Apply the same protection to `v3.0-dev`**

**⚠ Confirm with the user before running this step — it modifies repository settings.**

Using the fields discovered in Step 1, construct a matching `PUT` request. For example, if `rc2.3` requires 1 approving review and PRs only:

```bash
gh api --method PUT repos/starsimhub/hpvsim/branches/v3.0-dev/protection \
  -F 'required_status_checks=null' \
  -F 'enforce_admins=false' \
  -F 'required_pull_request_reviews[required_approving_review_count]=1' \
  -F 'restrictions=null'
```

`gh api -F` passes a raw JSON value (so `null` is a real null, not the string "null") while `-f` would pass a string. Adjust the fields to match whatever `rc2.3` actually has. If uncertain, surface the question to the user rather than guessing. The GitHub UI (Settings → Branches → Add rule) is also a reasonable path if the API arguments are fiddly.

- [ ] **Step 3: Verify protection is applied**

Run: `gh api repos/starsimhub/hpvsim/branches/v3.0-dev/protection 2>&1`
Expected: a JSON object with the same fields as `rc2.3`'s protection.

---

## Task 10: Announce the new branch to the team

- [ ] **Step 1: Draft the announcement**

Use exactly this text (Markdown for Slack / GitHub Discussions; adjust spacing if the platform renders differently):

```
**HPVsim v3 migration branch is now `v3.0-dev`.**

- Branched off `rc2.3` on 2026-04-23.
- New `MIGRATION_PLAN.md` (re-derived from scratch) and `KICKOFF_DISCUSSION.md` (carried forward from old `rc3`) are in the repo root on that branch.
- Old `rc3`, `rc3-integration`, `rc3-jc` on `origin` are deprecated but **left in place as read-only references**. Do not push new work to them.
- v2.3 release work remains on `rc2.3` / `main` and is not part of the migration plan.
- Migration-labeled GitHub issues and milestones will be reconciled against the new plan shortly (tracking issue: *to be linked*).

Questions or pushback: reply in this thread or open an issue against the plan.
```

- [ ] **Step 2: Post to the team's usual channel**

**⚠ This step is performed by the user (Ryan), not by the agent** — posting to Slack / email / GitHub Discussions is outside the agent's scope. Hand off the drafted text.

- [ ] **Step 3: Record that the announcement was sent**

Once posted, note the announcement location (URL/thread link) in a brief comment on the GitHub tracking issue created in Task 11, so future readers know when/where the team was informed.

---

## Task 11: Open a GitHub tracking issue for the deferred milestone/issue update

**Files:** None in the working tree; creates a new issue on `starsimhub/hpvsim`.

- [ ] **Step 1: Draft the issue body**

Use exactly this body:

```markdown
## Scope

Reconcile the existing GitHub milestones and migration-labeled issues on this repo against the new `MIGRATION_PLAN.md` on `v3.0-dev`. The new plan uses a 10-milestone partition (M0–M9) organized as thin vertical slices with per-milestone acceptance tests; the current repo has M00–M11 milestones with different groupings.

Approach is hybrid (confirmed 2026-04-23 during planning):

- Review existing M00–M11 milestones and rename/restructure to match M0–M9 from the new plan where they don't match.
- Close issues that no longer map to a milestone, with a comment pointing to their replacement.
- Open new issues to cover sub-tasks from the new plan's milestones (roughly one per sub-task, ~50 total).
- v2.3 release work stays in its own existing milestone on `rc2.3` / `main` — not absorbed into the migration plan.

## Deliverables

- [ ] Mapping table: existing milestone/issue → new milestone (as a comment on this issue).
- [ ] Milestone renames / restructures on the repo.
- [ ] Issue closures with replacement pointers.
- [ ] Issue creations for new sub-tasks.
- [ ] Final state logged in a comment on this issue.

## References

- New plan: `MIGRATION_PLAN.md` on branch `v3.0-dev`.
- Design spec: `docs/superpowers/specs/2026-04-23-hpvsim-starsim-port-design.md` on branch `v3.0-dev`.
- Validation repos: issue #64, reproducibility checks #68–#73 and #82–#87.
```

- [ ] **Step 2: Create the issue**

**⚠ Confirm with the user before running this step — creating a GitHub issue is team-visible.**

Run:
```bash
gh issue create --repo starsimhub/hpvsim \
  --title 'Reconcile GitHub milestones and migration issues with new MIGRATION_PLAN.md' \
  --label migration \
  --body-file - <<'EOF'
## Scope

Reconcile the existing GitHub milestones and migration-labeled issues on this repo against the new `MIGRATION_PLAN.md` on `v3.0-dev`. The new plan uses a 10-milestone partition (M0–M9) organized as thin vertical slices with per-milestone acceptance tests; the current repo has M00–M11 milestones with different groupings.

Approach is hybrid (confirmed 2026-04-23 during planning):

- Review existing M00–M11 milestones and rename/restructure to match M0–M9 from the new plan where they don't match.
- Close issues that no longer map to a milestone, with a comment pointing to their replacement.
- Open new issues to cover sub-tasks from the new plan's milestones (roughly one per sub-task, ~50 total).
- v2.3 release work stays in its own existing milestone on `rc2.3` / `main` — not absorbed into the migration plan.

## Deliverables

- [ ] Mapping table: existing milestone/issue → new milestone (as a comment on this issue).
- [ ] Milestone renames / restructures on the repo.
- [ ] Issue closures with replacement pointers.
- [ ] Issue creations for new sub-tasks.
- [ ] Final state logged in a comment on this issue.

## References

- New plan: `MIGRATION_PLAN.md` on branch `v3.0-dev`.
- Design spec: `docs/superpowers/specs/2026-04-23-hpvsim-starsim-port-design.md` on branch `v3.0-dev`.
- Validation repos: issue #64, reproducibility checks #68–#73 and #82–#87.
EOF
```

(Using `--body-file -` with a single-quoted heredoc keeps backticks literal without needing to escape them through an outer double-quoted shell context.)

- [ ] **Step 3: Update `MIGRATION_PLAN.md` to link the tracking issue**

Once the issue is created, note its number (e.g., `#93`). In `MIGRATION_PLAN.md`, under `## GitHub milestones and issues`, replace:

```
Tracking issue: *(to be created; linked here after creation)*.
```

with:

```
Tracking issue: [#N](https://github.com/starsimhub/hpvsim/issues/N).
```

(substituting the actual issue number for `N`).

- [ ] **Step 4: Commit and push the link update**

Run:
```bash
git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol add MIGRATION_PLAN.md
git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol commit -m "$(cat <<'EOF'
Link MIGRATION_PLAN.md to GitHub milestone-reconciliation tracking issue

Replaces the placeholder reference with the real issue number now
that the tracking issue has been created.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git -C /c/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol push origin v3.0-dev
```

---

## Done criteria

All of the following must be true:

- [ ] `origin/v3.0-dev` exists, containing the planning spec, `KICKOFF_DISCUSSION.md`, and `MIGRATION_PLAN.md`.
- [ ] `MIGRATION_PLAN.md` has all 12 top-level sections and all 10 milestones, each with Demo / Acceptance test / Sub-tasks.
- [ ] Branch protection on `v3.0-dev` matches `rc2.3` / `main` (or has been surfaced as a follow-up question if the template wasn't set).
- [ ] Team has been informed that `v3.0-dev` exists and old `rc3` branches are read-only references.
- [ ] A GitHub tracking issue exists for the deferred milestone/issue reconciliation, and `MIGRATION_PLAN.md` links to it.

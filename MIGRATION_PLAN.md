# HPVsim v3.0 Migration Plan

## Overview

HPVsim v3.0 is a ground-up reimplementation of HPVsim on the [Starsim](https://starsim.org/) agent-based modeling framework. The original HPVsim (v2.x, ~16,000 LOC) used a fully custom architecture. The new version inherits from Starsim's `ss.Sim`, `ss.Infection`, `ss.Connector`, `ss.Intervention`, and `ss.Analyzer` classes, and leverages STIsim's `StructuredSexual` network.

This document tracks the work needed to bring HPVsim v3.0 (on the `rc3` branch) to **feature parity** with HPVsim v2.x (on the `main` branch).

### Validation criteria

Validation does **not** require identical numerical output. Results are considered equivalent if uncertainty intervals overlap across multiple seeds. The goal is epidemiologically equivalent behavior, not bit-for-bit reproducibility.

### Key decisions (agreed 2026-03-25, updated 2026-03-26)

- **MVP scope:** Core engine (Milestone 1) + Interventions (Milestone 2). Analyzers, calibration, MultiSim, HIV, plotting are post-MVP.
- **Exception:** `age_results` analyzer promoted to MVP — required for age-stratified validation.
- **Products:** Use Starsim's native `ss.Dx`/`ss.Treatment`/`ss.simple_vx` classes, adapting v2's CSV data.
- **Therapeutic vaccination (txvx):** Keep — will be ported in Milestone 2.
- **EventSchedule:** Deprecated — will not be ported.
- **Custom `settings.py`:** Replaced by `ss.options` — will not be ported.
- **Waning immunity:** Dropped — never used in any published analysis; can revisit in a future release if needed.
- **v2.x incidence-based HIV module:** Dropped — v3.0 will use STIsim's transmission-based HIV model exclusively (post-MVP).
- **Network:** Use STIsim's `StructuredSexual` for v3.0. Porting HPVsim's custom network is out of scope.
- **Population scaling & migration:** Deferred to post-MVP.
- **Multiscale modeling:** Low priority — not blocking release.
- **Testing strategy:** Generate all v2 baseline outputs upfront. Each milestone includes its own tests validated against those baselines.
- **Branch strategy:** Tag current `main` as `v2.1.0`, merge `main` into `rc3`, then merge `rc3` → `main` when ready.
- **Timeline:** No hard deadline. Agile/incremental — each milestone produces a functioning, reviewable increment.

### RACI

| Role | Person(s) |
|---|---|
| **Responsible** (doing the migration work) | Ryan |
| **Accountable** (owns the release decision) | Robyn, Jamie |
| **Consulted** (domain expertise, review) | Darcy, WHI |
| **Informed** (stakeholders, downstream users) | Quantium, external users |

- **PR reviews on `rc3`:** Robyn
- **Scientific validation:** TBC, likely Ryan
- **External team contact (Quantium, IARC):** Robyn
- **External collaborator involvement:** Will involve Quantium/IARC in validation once time estimates are finalized

### Architecture mapping

| HPVsim v2.x (original) | HPVsim v3.0 (Starsim-based) |
|---|---|
| Custom `BaseSim` / `Sim` | `ss.Sim` subclass (`hpv.Sim`) |
| Custom `People` / `Person` | `ss.People` (from Starsim) |
| Custom genotype handling in `Sim` | `Genotype(sti.BaseSTI)` per genotype + `HPV(ss.Connector)` |
| Custom network in `people.py` / `population.py` | `sti.StructuredSexual` (from STIsim) |
| Custom `Intervention` base | `ss.Intervention` subclass |
| Custom `Analyzer` base | `ss.Analyzer` subclass |
| `HIVsim` class in `hiv.py` | `hpv_hiv_connector(ss.Connector)` + STIsim HIV |
| `immunity.py` module | Cross-immunity via `HPV` connector's `update_immunity()` |
| Custom product classes (dx, tx, vx) | Starsim native `ss.Dx`, `ss.Treatment`, `ss.simple_vx` |
| `calibration.py` | Starsim `ss.Calibration` with Optuna |
| `run.py` (MultiSim, Scenarios, Sweep) | Starsim `ss.MultiSim`, `ss.parallel()` |
| `plotting.py` / `analysis.py` | To be rebuilt using Starsim patterns |

### Current state of rc3

The `rc3` branch has a working core (~2,655 LOC across 10 modules):
- **Working**: Basic disease model (4 genotypes), cross-immunity, sexual network, demographics, HPV-HIV connector stub
- **Partially working**: Vaccination (skeleton), screening/treatment (two parallel systems needing reconciliation)
- **Missing/incomplete**: Product system integration, therapeutic vaccination, advanced screening/triage workflows, analyzers (only `hiv_hpv_results`), calibration, plotting, MultiSim/Scenarios, full HIV integration, population scaling, migration

### Known issues on rc3

- `interventions.py` has two parallel intervention systems that need reconciliation (System A: v2-ported skeleton, System B: newer `screen`/`treat` classes)
- `test_hpv.py` has broken references (undefined `super_connector`, `con`, `an`)
- `test_calibration.py` uses outdated API names
- `__all__` only exports System B classes

---

## Milestones

### Milestone 0: Foundation

| # | Task | Description | Acceptance criteria |
|---|---|---|---|
| 0.1 | Merge main into rc3 | Merge `main` into `rc3` to ensure rc3 has all recent fixes, data updates, and infrastructure from main. Resolve any merge conflicts. | Merge completes cleanly. Existing rc3 tests still pass. |
| 0.2 | Tag v2 release | Tag current `main` as `v2.1.0` for reference. | Tag exists on GitHub. |
| 0.3 | Generate v2 baseline outputs | Write `tests/generate_v2_baselines.py` on `main`. Run v2 across scenarios (natural history, vaccination, screening/treatment, genotype distribution), 3 locations, seeds 0-4, years 1990-2060. Save as JSON/pickle. | Script produces baseline files with metadata. Cancer incidence in plausible range (5-50 per 100k for SSA). |
| 0.4 | Build v2-v3 comparison utility | Write `tests/compare_baselines.py`. Load v2 baselines + v3 results, compute stats (mean, 95% CI), check overlapping error bars. | Works with synthetic test data. |
| 0.5 | Stabilize rc3 test suite | Fix broken tests in `test_hpv.py`, `test_calibration.py`. Ensure `test_sim.py` passes. Mark unfixable tests as skip. | `pytest tests/test_sim.py` passes. All other test failures fixed or explicitly skipped. |
| 0.6 | Reconcile dual intervention systems | Keep System A (v2-ported skeleton matching Starsim patterns). Remove/refactor System B (`screen`/`treat`). | Single coherent class hierarchy. All classes inherit from `ss.Intervention`. |
| 0.7 | Set up CI | Configure GitHub Actions on `rc3`: run `pytest tests/` on push/PR. | CI runs and reports on PRs. |

### Milestone 1: Core Engine (MVP Part 1)

| # | Task | Description | Acceptance criteria |
|---|---|---|---|
| 1.1 | Validate natural history | Create `tests/test_natural_history.py`. Run v3 with same params/seeds/locations as v2 baselines. Compare infections, HPV prevalence, CINs, cancers, cancer incidence, deaths. | For 2+ locations, v3 cancer incidence has overlapping 95% CIs with v2 (5 seeds each). |
| 1.2 | Verify immunity dynamics | Verify rc3's simpler immunity (no waning) produces comparable results to v2. Check post-clearance immunity boost, cross-immunity, vaccine-induced immunity. | Tests verify (a) immunity increases after clearance, (b) cross-immunity reduces susceptibility, (c) results match v2 baselines. |
| 1.3 | Align result keys | Audit v2 vs v3 result keys. Add missing critical keys: per-genotype prevalence, cross-genotype totals, cancer_share_<genotype>. | v3 results contain all keys needed for baseline comparison. `test_result_consistency` passes. |
| 1.4 | Port `age_results` analyzer | Implement as `ss.Analyzer` subclass. Compute age-stratified HPV prevalence, cancer incidence, n_infected, n_cin, n_cancerous by year and age bin. | Produces age-stratified outputs. Prevalence shows expected age pattern. |
| 1.5 | Validate age-stratified outputs | Run v3 with `age_results`, compare to v2 baselines. | Age-stratified cancer incidence (10-year bins) has overlapping CIs for 2+ locations. |
| 1.6 | Parameter backward compatibility | Verify `remap_pars()` covers common v2 names (`n_years`, `burnin`, `ms_agent_ratio`, `network`, `init_hpv_prev`). | Common v2 parameter names silently remapped. `test_sim_options` passes. |
| 1.7 | Resolve core TODOs | Address the 9 existing TODOs in `hpv.py`, `sim.py`, `parameters.py`. | No remaining TODOs that block MVP. |

### Milestone 2: Interventions (MVP Part 2)

| # | Task | Description | Acceptance criteria |
|---|---|---|---|
| 2.1 | Adapt product CSVs to Starsim format | Create adapter functions mapping v2 CSV format → `ss.Dx` DataFrames. Create factory functions: `default_dx('via')`, `default_tx('ablation')`, `default_vx('bivalent')`. | `hpv.default_dx('via')` returns `ss.Dx` instance. Unit tests pass. |
| 2.2 | Implement vaccination | Complete `routine_vx` + `campaign_vx`. Set `sus_imm` on genotype modules based on vaccine `rel_imm`. Age eligibility, multi-dose support. | `routine_vx(prob=0.8, start_year=2025, product='bivalent')` reduces cancer incidence vs baseline. |
| 2.3 | Validate vaccination | Compare v3 vaccination scenarios to v2 baselines. | Cancer reduction (% 2025-2060) has overlapping CIs. |
| 2.4 | Implement screening pipeline | Complete `BaseScreening.check_eligibility()`. Implement `routine_screening` + `campaign_screening` with `ss.Dx` products. Store outcomes for downstream triage. | `routine_screening(product='hpv', prob=0.2, start_year=2025)` produces positive/negative outcomes. |
| 2.5 | Implement triage | Complete `BaseTriage`, `routine_triage`, `campaign_triage`. Chain from positive screens via eligibility lambda. | Can chain screening → triage (e.g., HPV → VIA). |
| 2.6 | Implement treatment | Complete `BaseTreatment` and `treat_num`. Treatment clears infection via `genotype.ti_clearance`. | `treat_num` with ablation clears CIN in treated agents. |
| 2.7 | Validate screening/treatment | Run 2+ WHO algorithms (VIA → ablation, HPV → ablation) against v2 baselines. | Screening-driven cancer reduction has overlapping CIs. |
| 2.8 | Implement therapeutic vaccination | Port `routine_txvx`, `campaign_txvx`, `linked_txvx`. Composite: treatment + vaccination. | `routine_txvx` applies both treatment and immunity boost. |
| 2.9 | Intervention results tracking | Track: screened, positive, treated, vaccinated, doses — by age and year. | Intervention result keys populated and nonzero. |

### Milestone 3: Analyzers and Results

| # | Task | Description |
|---|---|---|
| 3.1 | Port `snapshot` analyzer | Store People state at timepoints. |
| 3.2 | Port `age_pyramid` analyzer | Age/sex population distribution with data overlay. |
| 3.3 | Port `age_causal_infection` analyzer | Track genotype causing each cancer. |
| 3.4 | Port `cancer_detection` analyzer | Detection rates by stage. |
| 3.5 | Port `dalys` analyzer | Disability-adjusted life years (YLL + YLD). |
| 3.6 | Add type distribution results | Genotype distribution of cancers. |

### Milestone 4: Calibration

| # | Task | Description |
|---|---|---|
| 4.1 | Integrate with Starsim calibration | Use `ss.Calibration` with Optuna. Port calibration targets: cancer incidence by age, HPV prevalence by age. |
| 4.2 | Implement goodness-of-fit | Port `compute_gof()` and likelihood functions. Support fitting to cancer incidence, HPV prevalence, genotype distribution. |
| 4.3 | Write calibration guide | Document how to calibrate v3.0 to a new location (issue #11). |

### Milestone 5: MultiSim, Scenarios, and Sweeps

| # | Task | Description |
|---|---|---|
| 5.1 | Integrate with Starsim MultiSim | Ensure `ss.MultiSim` works with HPVsim. Test multiple seeds, result combination, statistics. |
| 5.2 | Port Scenarios class | Parameter sweeps and intervention comparisons. |
| 5.3 | Port Sweep class | Systematic parameter variation. |

### Milestone 6: Additional Interventions

| # | Task | Description |
|---|---|---|
| 6.1 | Implement `treat_delay` | Treatment with a delay post-diagnosis. |
| 6.2 | Implement `radiation` | Cancer treatment product. |
| 6.3 | Implement `dynamic_pars` | Runtime parameter changes. |

### Milestone 7: Population Dynamics

| # | Task | Description |
|---|---|---|
| 7.1 | Population scaling | Implement `pop_scale` / `total_pop` for representing larger populations. |
| 7.2 | Migration | Port age-specific migration logic from v2.x. |

### Milestone 8: HIV Integration

| # | Task | Description |
|---|---|---|
| 8.1 | Integrate with STIsim HIV | Replace stub `hpv_hiv_connector` with full integration. Map CD4 → HPV susceptibility/severity. |
| 8.2 | Port CD4-stratified effects | Accelerated CIN progression at low CD4, altered clearance rates. |
| 8.3 | Port ART effects on HPV | Partial immune restoration slowing HPV progression. |
| 8.4 | HIV-stratified results | HPV prevalence in HIV+ vs HIV- by age group. |
| 8.5 | Validate HPV-HIV integration | Compare against published data. |

### Milestone 9: Plotting

| # | Task | Description |
|---|---|---|
| 9.1 | Implement `sim.plot()` for HPV | Override/extend `ss.Sim.plot()` for HPV-specific results. |
| 9.2 | Plot by age group | Requires `age_results` analyzer. |
| 9.3 | Plot by genotype | Genotype breakdown. |
| 9.4 | Intervention plots | Coverage and impact over time. |
| 9.5 | Calibration plots | Data vs model fit, parameter distributions. |

### Milestone 10: Documentation and Examples

| # | Task | Description |
|---|---|---|
| 10.1 | Switch docs to MkDocs/Quarto | Issue #32. |
| 10.2 | Write migration guide (v2 → v3) | API changes, parameter remapping, script conversion. |
| 10.3 | Update tutorials | All tutorials to v3.0 API. |
| 10.4 | Add workflow examples | Screening, calibration, multi-country, vaccination impact. |

### Milestone 11: Infrastructure

| # | Task | Description |
|---|---|---|
| 11.1 | Split data files | Issue #12 — faster loading. |
| 11.2 | Fix download failures | Issue #30. |
| 11.3 | Improve save/load | Correct serialization with new architecture. |

### Milestone 12: Release

- Final integration testing across all milestones
- Merge `rc3` → `main`
- Publish v3.0.0 release

---

## Risks

1. **MRO fragility**: `HPV(ss.Connector, Genotype)` uses multiple inheritance. `super().__init__()` follows Python's MRO — fragile if Starsim's class hierarchy changes. Add explicit MRO test.
2. **Product system adaptation**: v2's products operate on 2D `[n_genotypes, n_agents]` arrays. v3's per-genotype-module architecture is different. May need a thin HPV-specific `Dx` subclass.
3. **Performance**: v2 vectorizes across genotypes in one pass. v3 loops over genotype modules sequentially. Benchmark early.
4. **Parameter drift**: Different architecture → different RNG call order → different trajectories even with same parameters. "Overlapping CIs" criterion handles this, but recalibration may be needed.
5. **Starsim/STIsim version coupling**: Actively developed dependencies. Pin versions during development.

---

## Scientific validation

Release sign-off requires replicating a minimum set of published and in-progress analyses with v3.0 and confirming overlapping uncertainty intervals.

### Minimum validation set

| # | Analysis | Rationale |
|---|---|---|
| 1 | Stuart et al. (2024), *Inferring the natural history of HPV from global cancer registries*, Sci Rep | Multi-country calibration — validates core engine + calibration |
| 2 | Stuart et al. (2026), *The role of HPV single-dose vaccination in GAVI-supported countries*, Vaccine | Validates vaccination interventions at scale |
| 3 | HPV Faster Kenya (ongoing) | Real-world ongoing use |
| 4 | Quantium team's models (IPG pilot) | External validation by non-core team |
| 5 | HPV elimination in Rwanda (under review, IARC collab) | Tests active partnerships |
| 6 | HPV Nigeria infant vaccine model (under review) | Tests infant vaccination scenarios |

---

## Key files reference

### HPVsim v3.0 (rc3 branch)
- `hpvsim/hpv.py` — Core disease model: Genotype + HPV connector, immunity, state progression
- `hpvsim/sim.py` — Sim wrapper: parameter separation, module processing, backward compat
- `hpvsim/parameters.py` — All defaults: SimPars, HPVPars, NetworkPars, ImmPars
- `hpvsim/interventions.py` — All interventions (dual system needing reconciliation)
- `hpvsim/analyzers.py` — Currently minimal (needs `age_results` for MVP)
- `hpvsim/connectors.py` — HPV-HIV connector stub
- `hpvsim/utils.py` — logf2, compute_cancer_prob, etc.
- `hpvsim/distributions.py` — beta_mean distribution
- `hpvsim/data/products_*.csv` — Product definitions (already on rc3)
- `hpvsim/data/loaders.py` — Age distribution, death rates, birth rates

### HPVsim v2.x (main branch, to be tagged v2.1.0)
- `hpvsim/sim.py` (1,395 LOC) — Full simulation logic
- `hpvsim/people.py` (1,212 LOC) — Population and disease progression
- `hpvsim/interventions.py` (1,549 LOC) — Complete intervention system with products
- `hpvsim/immunity.py` (301 LOC) — Immunity and waning
- `hpvsim/hiv.py` (896 LOC) — Full HIV model (dropping)
- `hpvsim/calibration.py` (783 LOC) — Calibration framework
- `hpvsim/analysis.py` (1,209 LOC) — Analyzers
- `hpvsim/run.py` (1,883 LOC) — MultiSim, Scenarios, Sweep
- `hpvsim/plotting.py` (944 LOC) — Visualization

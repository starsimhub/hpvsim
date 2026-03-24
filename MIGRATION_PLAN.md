# HPVsim v3.0 Migration Plan

## Overview

HPVsim v3.0 is a ground-up reimplementation of HPVsim on the [Starsim](https://starsim.org/) agent-based modeling framework. The original HPVsim (v2.x, ~16,000 LOC) used a fully custom architecture. The new version inherits from Starsim's `ss.Sim`, `ss.Infection`, `ss.Connector`, `ss.Intervention`, and `ss.Analyzer` classes, and leverages STIsim's `StructuredSexual` network.

This document tracks the work needed to bring HPVsim v3.0 (on the `rc3` branch) to **feature parity** with HPVsim v2.x (`hpvsim_orig`).

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
| `calibration.py` | Starsim calibration (to be integrated) |
| `run.py` (MultiSim, Scenarios, Sweep) | Starsim `ss.MultiSim`, `ss.parallel()` |
| `plotting.py` / `analysis.py` | To be rebuilt using Starsim patterns |

### Current state of rc3

The `rc3` branch has a working core (~2,075 LOC across 10 modules):
- **Working**: Basic disease model (4 genotypes), cross-immunity, sexual network, demographics, simplified screening/treatment, vaccination (routine + campaign), HPV-HIV connector stub
- **Missing/incomplete**: Products system, therapeutic vaccination, advanced screening/triage workflows, analyzers, calibration, plotting, MultiSim/Scenarios, full HIV integration, population scaling, migration, most tests

---

## Milestones

### Milestone 1: Core simulation engine

Ensure the core simulation loop produces epidemiologically valid results matching v2.x behavior.

| # | Issue | Priority | Description |
|---|---|---|---|
| 1.1 | Fix dt-dependent results | High | Changing `dt` should not change results (bug #13). Ensure all duration sampling and probability calculations are dt-invariant. |
| 1.2 | Add population scaling | High | Implement `pop_scale` / `total_pop` so a small agent population can represent a larger real population. Results should be scaled appropriately. |
| 1.3 | Add migration | Medium | Population sizes should match data over time. Port the age-specific migration logic from v2.x `people.py:check_migration()`. |
| 1.4 | Add additional genotypes | Low | Add `hr` (all high-risk composite) and `lo` (low-risk) genotype parameter sets, with appropriate cross-immunity entries. Currently only hpv16, hpv18, hi5, ohr are defined. |
| 1.5 | Validate natural history against v2.x | High | Run both versions with identical parameters and compare: HPV prevalence by age, CIN prevalence, cancer incidence, clearance rates. Document any intentional divergences. |
| 1.6 | Resolve TODOs in disease model | Medium | Address the 9 existing TODOs in `hpv.py`, `sim.py`, `parameters.py` (connector selection logic, parameter processing, etc.). |
| 1.7 | Implement waning immunity | Medium | v2.x has explicit waning functions (`exp_decay`, `linear_decay`) in `immunity.py`. Currently rc3 immunity is set-and-forget. Add configurable waning. |
| 1.8 | Sex-specific initial prevalence | Low | v2.x seeds initial infections differently by sex. Ensure rc3 handles this correctly. |

### Milestone 2: Interventions — products and delivery

Build the full intervention product system and delivery mechanisms.

| # | Issue | Priority | Description |
|---|---|---|---|
| 2.1 | Implement product base classes (dx, tx, vx) | High | Port the product system from v2.x `interventions.py`. Products encapsulate diagnostic sensitivity/specificity, treatment efficacy by disease state, and vaccine genotype coverage. CSV product files already exist in `data/`. |
| 2.2 | Implement therapeutic vaccination (txvx) | High | Port `BaseTxVx`, `routine_txvx`, `campaign_txvx`, `linked_txvx` from v2.x. Therapeutic vaccines target infected individuals and modify disease progression. |
| 2.3 | Implement treat_delay intervention | Medium | Port `treat_delay` from v2.x — treatment with a specified delay post-diagnosis, as opposed to `treat_num` (capacity-limited). |
| 2.4 | Implement radiation treatment | Medium | Port `radiation` intervention from v2.x for cancer treatment. |
| 2.5 | Complete triage workflow | High | The triage classes exist as skeletons. Implement the full screen → triage → treat cascade, where triage results feed into treatment eligibility. |
| 2.6 | Implement EventSchedule | Low | Port the `EventSchedule` class for ad-hoc event scheduling at specific timepoints. |
| 2.7 | Implement dynamic_pars | Low | Port `dynamic_pars` to allow parameter changes during simulation (e.g., changing condom use over time). |
| 2.8 | Add intervention results tracking | Medium | Track detailed intervention outcomes: number screened, number positive, number treated, number vaccinated, doses administered — by age and year. |
| 2.9 | Validate interventions against v2.x | High | Run identical screening + vaccination scenarios in both versions and compare outcomes. |

### Milestone 3: Analyzers and results

Port the analysis toolkit for post-simulation data extraction.

| # | Issue | Priority | Description |
|---|---|---|---|
| 3.1 | Implement age_results analyzer | High | Port `age_results` from v2.x — stratifies any result by age group and genotype. This is critical for calibration (comparing model output to age-stratified data). |
| 3.2 | Implement age_pyramid analyzer | Medium | Port `age_pyramid` — captures population age/sex distribution at specified timepoints. |
| 3.3 | Implement snapshot analyzer | Medium | Port `snapshot` — stores a copy of the People object at specified timepoints for detailed inspection. |
| 3.4 | Implement age_causal_infection analyzer | Medium | Port `age_causal_infection` — tracks the age at which causal (cancer-causing) infections were acquired. |
| 3.5 | Implement DALYs analyzer | Medium | Port `dalys` — computes disability-adjusted life years (YLL + YLD). |
| 3.6 | Add cancer incidence results | High | Ensure age-standardized cancer incidence rate (ASR) is computed correctly. The standard population weights are already in `SimPars`. |
| 3.7 | Add type distribution results | Medium | Track the genotype distribution of cancers (what fraction of cancers are caused by each genotype). |

### Milestone 4: Calibration

Port the calibration framework for fitting model parameters to data.

| # | Issue | Priority | Description |
|---|---|---|---|
| 4.1 | Port Calibration class | High | Adapt the Optuna-based `Calibration` class from v2.x to work with the Starsim architecture. Should support parallel trials and genotype-specific parameter fitting. |
| 4.2 | Implement goodness-of-fit computation | High | Port `compute_gof()` and the likelihood/fitting functions. Support fitting to: cancer incidence by age, HPV prevalence by age, genotype distribution, CIN prevalence. |
| 4.3 | Write calibration guide | Medium | Document how to calibrate HPVsim v3.0 to a new location (issue #11). |
| 4.4 | Port pre-calibration exploration | Low | v2.0 added support for parameter sweeps before calibration to identify reasonable ranges. |

### Milestone 5: Multi-sim, scenarios, and sweeps

Port the infrastructure for running and comparing multiple simulations.

| # | Issue | Priority | Description |
|---|---|---|---|
| 5.1 | Integrate with Starsim MultiSim | High | Ensure `ss.MultiSim` works correctly with HPVsim sims. Test running multiple seeds, combining results, and computing statistics (median, quantiles). |
| 5.2 | Port Scenarios class | Medium | Adapt the `Scenarios` class for running parameter sweeps and intervention comparisons. |
| 5.3 | Port Sweep class | Low | Adapt the `Sweep` class for systematic parameter variation. |
| 5.4 | Ensure parallel execution works | Medium | Verify that `ss.parallel()` works with HPVsim sims, including proper random seed handling. |

### Milestone 6: HIV integration

Build comprehensive HPV-HIV co-infection modeling.

| # | Issue | Priority | Description |
|---|---|---|---|
| 6.1 | Integrate with STIsim HIV module | High | Replace the stub `hpv_hiv_connector` with full integration using STIsim's HIV module. Map CD4 count to HPV susceptibility and severity multipliers. |
| 6.2 | Port CD4-stratified effects | High | v2.x has detailed CD4-dependent effects on HPV progression (accelerated CIN progression at low CD4, altered clearance rates). Port these. |
| 6.3 | Port ART effects on HPV | Medium | Model how ART (and CD4 reconstitution) affects HPV natural history — partial immune restoration should slow HPV progression. |
| 6.4 | Add HIV-stratified results | Medium | Add results stratified by HIV status (HPV prevalence in HIV+ vs HIV- women, by age group). The `hiv_hpv_results` analyzer is a start but needs expansion. |
| 6.5 | Validate HPV-HIV against v2.x | High | Run both versions with HIV enabled and compare: HPV prevalence by HIV status, cancer incidence in WLHIV vs general population. |

### Milestone 7: Plotting and visualization

Build plotting functions for standard result views.

| # | Issue | Priority | Description |
|---|---|---|---|
| 7.1 | Implement sim.plot() for HPV | Medium | Override or extend `ss.Sim.plot()` to show HPV-specific results: prevalence, incidence, cancer cases by genotype. |
| 7.2 | Implement plot by age group | Medium | Plot results stratified by age group (requires age_results analyzer). |
| 7.3 | Implement plot by genotype | Medium | Plot results broken down by HPV genotype. |
| 7.4 | Implement intervention plots | Low | Visualize intervention coverage and impact over time. |
| 7.5 | Implement calibration plots | Medium | Plot calibration results: data vs model fit, parameter distributions, convergence. |

### Milestone 8: Testing

Expand test coverage to match v2.x robustness.

| # | Issue | Priority | Description |
|---|---|---|---|
| 8.1 | Add intervention tests | High | Test all intervention types: screening, triage, treatment, vaccination (routine + campaign), therapeutic vaccination. |
| 8.2 | Add calibration tests | High | Test calibration workflow end-to-end. |
| 8.3 | Add MultiSim/Scenario tests | Medium | Test multi-sim running, scenario comparisons, result aggregation. |
| 8.4 | Add analyzer tests | Medium | Test all analyzer classes produce correct output. |
| 8.5 | Add regression/baseline tests | High | Port baseline tests from v2.x — run a standard sim and compare results to stored baselines to catch unintended changes. |
| 8.6 | Add network tests | Medium | Test that the StructuredSexual network produces expected partnership patterns (age-mixing, concurrency, duration). |
| 8.7 | Add HIV integration tests | Medium | Test HPV-HIV co-infection scenarios. |
| 8.8 | Add population dynamics tests | Medium | Test births, deaths, migration, population scaling. |

### Milestone 9: Documentation and examples

| # | Issue | Priority | Description |
|---|---|---|---|
| 9.1 | Switch docs from Sphinx to MkDocs/Quarto | Medium | Issue #32 — modernize documentation build system. |
| 9.2 | Write migration guide (v2 → v3) | High | Document API changes, parameter remapping, and how to convert v2.x scripts. |
| 9.3 | Update tutorials | Medium | Update all tutorials to use v3.0 API. |
| 9.4 | Add examples for common workflows | Medium | Screening algorithms (t05 exists), calibration, multi-country comparison, HPV-HIV, vaccination impact. |
| 9.5 | Generate API reference | Low | Auto-generate API docs from docstrings. |

### Milestone 10: Data and infrastructure

| # | Issue | Priority | Description |
|---|---|---|---|
| 10.1 | Split data files for faster loading | Medium | Issue #12 — current data loading is slow due to monolithic files. |
| 10.2 | Fix automatic download failures | High | Issue #30 — data downloads fail in some environments. |
| 10.3 | Add settings/options module | Low | v2.x has a rich `settings.py` with plot styling, precision control, etc. Determine what to port vs. rely on Starsim's `ss.options`. |
| 10.4 | Improve save/load | Low | Ensure sim saving and loading works correctly with the new architecture. |

---

## Priority ordering

For the initial v3.0 release, focus on these milestones in order:

1. **Milestone 1** (Core engine) — foundation everything else depends on
2. **Milestone 2** (Interventions) — needed for any policy analysis
3. **Milestone 3** (Analyzers) — needed for calibration and validation
4. **Milestone 4** (Calibration) — needed to fit to real-world data
5. **Milestone 8** (Testing) — ongoing, in parallel with above
6. **Milestone 5** (Multi-sim) — needed for uncertainty quantification
7. **Milestone 6** (HIV) — needed for sub-Saharan Africa analyses
8. **Milestone 7** (Plotting) — quality of life
9. **Milestone 9** (Documentation) — ongoing, in parallel
10. **Milestone 10** (Infrastructure) — as needed

## Key files reference

### HPVsim v3.0 (rc3 branch) — `/Users/robynstuart/gf/hpvsim/`
- `hpvsim/hpv.py` — Genotype and HPV connector classes
- `hpvsim/sim.py` — Sim class with genotype/network/demographics processing
- `hpvsim/parameters.py` — SimPars, HPVPars, NetworkPars, ImmPars
- `hpvsim/interventions.py` — screen, treat, vaccination, delivery classes
- `hpvsim/connectors.py` — HPV-HIV connector stub
- `hpvsim/analyzers.py` — HIV-HPV results analyzer
- `hpvsim/utils.py` — logf2, compute_cancer_prob, etc.
- `hpvsim/distributions.py` — beta_mean distribution
- `hpvsim/data/` — product CSVs, data loaders, downloaders

### HPVsim v2.x (original) — reference at `hpvsim_orig/`
- `hpvsim/sim.py` (1,395 LOC) — Full simulation logic
- `hpvsim/people.py` (1,212 LOC) — Population and disease progression
- `hpvsim/interventions.py` (1,549 LOC) — Complete intervention system
- `hpvsim/immunity.py` (301 LOC) — Immunity and waning
- `hpvsim/hiv.py` (896 LOC) — Full HIV model
- `hpvsim/calibration.py` (783 LOC) — Calibration framework
- `hpvsim/analysis.py` (1,209 LOC) — Analyzers
- `hpvsim/run.py` (1,883 LOC) — MultiSim, Scenarios, Sweep
- `hpvsim/plotting.py` (944 LOC) — Visualization

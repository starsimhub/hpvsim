# M09 (Part 2): Plotting layer — design

**Date:** 2026-06-08
**Branch:** `m09-plotting` (stacked off `m09-analyzers`, which is off `m07-multiscale-ledger`)
**Milestone:** M09 "Remaining analyzers and plotting" — Part 2 (plotting). Part 1
(analyzers) is its own spec/PR: `2026-06-08-m09-analyzers-design.md`.

## Scope

Build the four plotting figure families the milestone calls for, plus `.plot()`
methods on the Part-1 analyzers:

1. **HPV result views** via a `plot_sim` helper + `ss.Sim.plot` delegation.
2. **By-age and by-genotype** plots (incl. genotype-of-cancers type distribution).
3. **Intervention-impact** plots (baseline vs scenario, cancers averted).
4. **Calibration-result** plots (data-vs-fit + convergence / parameter
   distributions).

All four are buildable on this branch: its base (`m07-multiscale-ledger`)
carries the full intervention suite (`vx`, screening, triage, treatment, txvx,
`dynamic_pars`), the calibration module (`hpv.Calibration(ss.Calibration)`,
`compute_gof`, `default_eval_fn`, `AgeResults` integration), and Starsim's
`ss.MultiSim`.

**Out of scope:** image/pixel regression; new analyzers or results (Part 1 owns
those); any v2 `plot_args`/`get_default_plots`/`handle_to_plot` machinery.

## Approach

A new `hpvsim/plotting.py` module of focused functions, plus thin `.plot()`
methods on `age_pyramid` / `age_causal_infection` / `dalys` that delegate into
it. The layer **leans on Starsim's built-ins** for what they already do
(`ss.Sim.plot` for time series, `ss.MultiSim.plot` for multi-run CIs,
`ss.Calibration.plot_optuna`/`plot_final` for Optuna views) and adds only the
HPV-domain-specific figures.

Rejected alternatives: lift-and-shift v2's `plotting.py` (architecture mapping
says rebuild on Starsim patterns; duplicates `ss.Sim.plot`; strip-before-M10
debt); analyzer-only `.plot()` methods (intervention-impact and calibration
plots aren't analyzer-bound, and by-age/genotype helpers must be reusable across
sims/msims).

## Module layout

- **New** `hpvsim/plotting.py` — domain plot functions (below); exported via
  `hpvsim/__init__.py`.
- **Modify** `hpvsim/analyzers.py` — add `.plot()` to `age_pyramid`,
  `age_causal_infection`, `dalys`, each delegating to a `plotting.py` helper
  (keeps the analyzers focused on data, plotting logic in one place).
- **Modify** `hpvsim/__init__.py` — export the plotting helpers.
- **New** `tests/test_m09_plotting.py` — structural/smoke tests (Agg backend).

## Conventions (all helpers)

- Return a matplotlib `Figure`; accept optional `fig`/`ax` and `**kwargs` for
  styling. **Never call `plt.show()`** — the caller decides.
- Use Starsim time conventions: float-year x-axis, or `ss.date(...)` for any
  `axvline`/annotation (per the starsim-dev-time guidance).
- Helpers that need an analyzer or compatible inputs **validate up front and
  raise a clear `ValueError` naming the fix**; never emit a silent blank figure.

## Per-figure design

### `plot_sim(sim, which='default', fig=None, **kwargs)`

- `which='default'`: the v2 canonical 4-panel — (1) cancer incidence per 100k
  + age-standardized over time, (2) HPV / precin prevalence by age, (3) cancers
  by age, (4) genotype type distribution.
- `which='demographic'`: population size, birth/death rates.
- `which='all'`: delegate to `ss.Sim.plot()`.
- Time-series panels pull from `sim.results` (per-genotype HPV + connector
  results) via `ss.Sim.plot(key=...)`. Age panels require an `AgeResults`
  analyzer on the sim; if absent, raise `ValueError` naming the analyzer and the
  result keys it must record.

### `plot_by_age(age_results, key, years=None, kind='line', fig=None)`

- Source: `age_results.to_dataframe(key)` (t-indexed, age-bin columns). One
  series per requested year; x = age-bin label, y = value. `years=None` → all
  recorded years. `kind` in `{'line','bar'}`.

### `plot_by_genotype(sim, key='cum_cancers', normalize=False, fig=None)`

- Source: `results_by_genotype(sim, key, normalize)`. Overlaid lines (one per
  genotype, x = year); `normalize=True` → stacked area whose per-year totals are
  1.

### `plot_type_distribution(source, year=None, fig=None)`

- Source: an `AgeResults` type-dist result (`to_dataframe(normalize=True)`) or
  `results_by_genotype(..., normalize=True)`. Bar chart of each genotype's share
  of cancers at `year` (default: last recorded year).

### `plot_intervention_impact(baseline, scenario, key='cancer_incidence', labels=None, fig=None)`

- Accepts two `ss.Sim` or two `ss.MultiSim` (both same type). Top panel: both
  trajectories — median + 10/90 band when `MultiSim` (via its quantile
  machinery), single line when `Sim`. Bottom panel: averted = `baseline -
  scenario` elementwise over time. Validates the two inputs share a timevec;
  raises `ValueError` otherwise.

### `plot_calibration(calib, fig=None)`

- **Data-vs-fit** (HPV-specific): for each target DataFrame in the calibration's
  data, overlay the best-fit sim's `AgeResults` output against the observed
  values, reusing `calibration._find_age_results` / `_extract_actual`. The
  best-fit sim is obtained from the calibration's stored best result if present;
  otherwise it is rebuilt at the best parameters via the calibration's
  `build_sim` and run once. `plot_calibration` documents (and accepts) an
  optional pre-run `sim=` to skip that rebuild.
- **Convergence + parameter distributions**: thin wrappers delegating to
  `calib.plot_optuna()` / `calib.plot_final()`.

### Analyzer `.plot()` methods (in `analyzers.py`, delegate to plotting.py)

- `age_pyramid.plot(date=None)` → back-to-back male/female bars from
  `age_pyramids[date]` (default: first/only recorded date).
- `age_causal_infection.plot(fig=None)` → histograms of `age_causal` /
  `age_cin` / `age_cancer` plus the dwell-time distributions (weight-aware).
- `dalys.plot(fig=None)` → YLL / YLD stacked over `self.years` (with the DALYs
  total).

## Testing

New `tests/test_m09_plotting.py`. Force the non-interactive backend
(`matplotlib.use('Agg')`) and `plt.close('all')` after each test — headless, no
figure leakage. Assert on returned `Figure` structure, not pixels:

- `plot_sim` — `Figure` with expected axis count for `which='default'` (4) and
  `'demographic'`; raises `ValueError` when an age panel is requested but no
  `AgeResults` analyzer is present.
- `plot_by_age` — one series per requested year; tick/x count == number of age
  bins; respects `years=` subsetting.
- `plot_by_genotype` — axis line count == n_genotypes; `normalize=True` per-year
  totals == 1.
- `plot_type_distribution` — bar count == n_genotypes; normalized heights sum to
  ~1; defaults to last year.
- `plot_intervention_impact` — both a `Sim` and a `MultiSim` smoke case; the
  averted-panel series equals `baseline − scenario` elementwise; raises on
  mismatched timevecs.
- `plot_calibration` — a tiny (2–3 trial) calibration yields a data-vs-fit
  figure with one overlay per target; the convergence/param wrappers call
  through without error.
- Analyzer `.plot()` — `age_pyramid`, `age_causal_infection`, `dalys` each
  return a `Figure` with the expected axes and don't raise on a normal run.

Assertions are structural/smoke only (figure exists, right artist counts,
documented invariants: normalized sums, the averted identity) — consistent with
Part 1's targeted-equivalence rigor. No image diffing.

## Files touched

- `hpvsim/plotting.py` — new module: `plot_sim`, `plot_by_age`,
  `plot_by_genotype`, `plot_type_distribution`, `plot_intervention_impact`,
  `plot_calibration`.
- `hpvsim/analyzers.py` — `.plot()` on `age_pyramid`, `age_causal_infection`,
  `dalys` (delegating).
- `hpvsim/__init__.py` — export plotting helpers.
- `tests/test_m09_plotting.py` — new structural/smoke tests.

## Deferred / follow-up

- Image/pixel regression (not pursued).
- Any further `sim.plot()` polish beyond the canonical panel set.

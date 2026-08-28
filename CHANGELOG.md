All notable changes to the codebase are documented in this file. Changes
that may result in differences in model output, or are required in order
to run an old parameter set with the current version, are flagged with
the term "Regression information".

## Version 3.1.0 (2026-08-20)

### `starsim` floor raised to `>=3.6`

Starsim 3.6.0 makes `float()` on an `ss.TimePar` raise a `TypeError` (previously it
silently discarded the unit). Fixed the resulting breakage in `hpv.py`, `network.py`,
`demographics.py`, `calibration.py`, and `analyzers.py` by switching to `.dt_year`,
`.years`, and `sim.t.year` — the same idioms starsim itself uses internally — in
place of `float(...)` on a duration/timestep.

**Regression information**: none — these are like-for-like unit-preserving
replacements of calls that previously stripped the unit via `float()`.

### Interventions: `sex='f'` is now the default across vax + screening + treatment

Every HPV-specific intervention constructor (`routine_vx`, `campaign_vx`,
`routine_screening`, `campaign_screening`, `routine_triage`,
`campaign_triage`, `treat_num`, `treat_delay`) now defaults `sex='f'`.
Previously vaccination interventions defaulted `sex=None` (both sexes),
which required every caller building a cervical-cancer scenario to
pass `sex='f'` explicitly. Screening / triage / `treat_num` already
defaulted `sex='f'` — this change harmonises vaccination with them so
"HPV vaccination in Nigerian girls" no longer risks silently vaxing
boys because the caller forgot the `sex` kwarg.

**Regression information**: model-behavior change for existing scripts
that instantiated `routine_vx` / `campaign_vx` without `sex` and
relied on the both-sexes default. To restore, pass `sex=None`
explicitly on each call.

### Products: `ablation` + `excision` now clear precin (parity with CIN)

`hpvsim/data/products_tx.csv` — `ablation.precin` efficacy lifted from
0 → 0.936 and `excision.precin` lifted from 0 → 0.81, matching each
product's CIN efficacy. The product's `administer` method already
clears `precin` / `cin` / `cancerous` flags and schedules HPV clearance
(`ti_clearance = ti + 1`) on successful treatment, so this change lets
screen-and-treat programs actually intercept pre-cancerous lesions.

Rationale: with efficacy 0 on precin, tx_assigner-routed precin cases
consumed a slot in the treatment cascade without touching disease
state — a silent no-op that made large screening scale-ups look
implausibly weak (14% cancer reduction at 70% coverage).

**Regression information**: model-behavior change. Scenarios with
screen-and-treat interventions will now avert substantially more
cervical cancers.

### `HPV`: default `transm2f` reduced from 3.69 → 2.0

The 3.69 factor came from an older meta-analysis and turns out to be
poorly supported by the current literature. Reducing to 2.0 keeps a
moderate m→f asymmetry (m2f > f2m) without pushing the default per-act
m2f probability so close to 1 that any upward calibration knob spills
over.

Concrete impact with defaults: `beta * rel_beta * transm2f` for hpv16
drops from 0.9225 → 0.5, meaning the max safe scalar `beta` via
`route_pars` rises from ~0.271 → 0.5.

**Regression information**: this is a model-behavior change. Existing
parameter sets calibrated against the old 3.69 default will now
produce lower m→f transmission unless callers pin `transm2f=3.69` on
each genotype explicitly.

### `HPV`: per-act beta clipped to `[0, 1]` at construction

`HPV.__init__` (and the `rel_beta` recompute) now clip
`beta * rel_beta * transm2f` per direction via a new `_clip_beta`
helper that emits a `RuntimeWarning` on clip.
`parameters.apply_beta_scalar` (the `route_pars` broadcast path) uses
the same helper.

Motivation: without a clip, any parameter combination pushing per-act
transmission above 1 silently NaN'd the sim — the network's per-act
calculation `1 - (1 - p) ** acts` returns NaN for `p > 1`, so the sim
ran to completion with garbage prevalence rather than erroring. With
the new `transm2f=2.0` default the clip rarely fires; it protects
against users pinning `transm2f` higher, passing `rel_beta > 1`, or
setting per-genotype `beta` above 0.5.

### `hpv.make_calib_sims`: rerun top-N calibration trials

New public helper that reruns the top-`n` best-fit trials from a
`hpv.Calibration` (or shrunk objdict) in parallel and returns the run sims —
enabling post-calibration analysis of results the calibration itself doesn't
store (e.g. `asr_cancer_incidence` trajectory, custom by_age analyzer output).

```python
sims = hpv.make_calib_sims(calib, n=50,
                           sim_kwargs=dict(stop=2045),
                           analyzers=lambda: [hpv.by_age(['precin_prevalence'], ...)])
```

Accepts `sim_kwargs` (applied to `sim.pars` before build_fn), `analyzers`
(callable or list, appended to any analyzers already on the base sim), and
`extract_fn` (worker returns `extract_fn(sim)` instead of the full sim —
strongly recommended, since 50 full sims can be many GB pickled back to the
parent process). `hpv.plot_calibration` now uses this internally via
`extract_fn=_extract_columns`.

### `hpv.Calibration`: per-bin-scheme `by_age` analyzers

Age-stratified calibration targets can now have different bin schemes across
result names. `_setup_analyzers` groups age-stratified names by their bin-label
set and attaches one `by_age` analyzer per distinct scheme: the first stays
`all_hpv_by_age` (backward-compatible), and additional schemes get
`all_hpv_by_age_1`, `_2`, .... `_extract_columns` builds a result→analyzer map
from all `all_hpv_by_age*` analyzers, so per-target lookup is transparent to
the eval / plot paths.

Enables mixing e.g. Globocan cancer bins (16 fine bins 0-15…85+) with a coarser
HPV-prevalence survey (6 bins 19-24…64+) in the same calibration; the previous
"inconsistent bin labels" ValueError only fires now if the same *result name*
appears with conflicting bins.

### `hpv.Calibration`: single `data=` kwarg, standardized wide DataFrame

`hpv.Calibration(sim, calib_pars, data=...)` now takes a single `data=`
argument that accepts any of:

- a list of long-format CSV paths (or long-format DataFrames) — dispatched
  by column shape (age-stratified / genotype-stratified / scalar-per-year),
- a dict of pre-scoped wide DataFrames keyed by target name,
- a pre-built standardized wide DataFrame (index `t`, dot-scoped columns).

The old `datafiles=` kwarg is gone; all three forms funnel through
`hpv.data.loaders.load_calib_data`, which produces one standardized wide
DataFrame with dot-scoped columns:

    all_hpv.<name>              # scalar target, read from sim.results.all_hpv
    all_hpv.<name>.<bin>        # age-stratified, read from the auto-attached
                                #   by_age analyzer named 'all_hpv_by_age'
    by_genotype.<name>.<g>      # per-genotype distribution, computed via
                                #   hpv.results_by_genotype

`hpv.Calibration.__init__` inspects the data columns and — if any
`all_hpv.<name>.<bin>` columns are present — auto-attaches an
`all_hpv_by_age` `by_age` analyzer with edges/years derived directly from
the data. `sim.pars.stop` is auto-extended past the latest data year, so
callers don't need to redefine the sim horizon to match their targets.

`calib_pars` is now a nested dict scoped by module, with **list** leaves
`[best, low, high, step]` (step optional):

```python
calib_pars = dict(
    beta=[0.2, 0.1, 0.34],
    network=dict(m_partners_casual=[0.5, 0.1, 0.9, 0.05]),
    hi5=dict(cin_fn=dict(k=[0.15, 0.1, 0.25, 0.02])),
)
```

`sc.flattendict` collapses these to the flat dotted-key form
`ss.Calibration` expects; the flat form is rejected with a pointer to
the nested form. Optuna spec-dict leaves (`{low, high, guess, ...}`) are
also rejected — leaves must be lists.

Each `hpv.Calibration` instance now allocates its own per-run Optuna study
directory (tempdir) and defaults to `optuna.storages.JournalStorage`
(append-only per-process journals) instead of SQLite. Under ~32+
concurrent workers, SQLite's global write lock deadlocks under
`study.optimize`; JournalStorage is Optuna's recommended backend for
distributed / high-worker-count optimization.

**Regression information:** any script calling
`hpv.Calibration(datafiles=..., data=...)` with the pre-3.1 dict-of-frames
`data=` shape or the pre-3.1 flat-dotted `calib_pars` form must migrate.
`hpv.calibration.build_sim` remains the default `build_fn` and still
accepts flat dotted keys (the flattened calib_pars). CSV paths in the new
`data=` list should be long-format with columns
`year, name, [age, sex, genotype,] value`.

### `HPVTotal.asr_cancer_incidence` / `asr_cancer_mortality`

Age-standardized rates are now first-class results on `HPVTotal`, computed
internally from per-timestep WHO2000 5-year cancer / cancer-death /
female-population histograms:

- `sim.results.all_hpv.asr_cancer_incidence`
- `sim.results.all_hpv.asr_cancer_mortality`

No `age_pyramid` or other analyzer is required — `HPVTotal.init_pre`
allocates the histograms; `HPVTotal.step` accumulates them each timestep
via the private `_accumulate_asr_histograms(ti, people, hpvs)`;
`HPVTotal.finalize_results` aggregates to annual counts and computes the
WHO2000-weighted rate via the classmethod `HPVTotal.compute_asr`. Class-
level constants `WHO2000_5YR_EDGES` / `WHO2000_5YR_WEIGHTS` and the
helper `HPVTotal.who2000_weights_for_edges(edges)` expose the same
standard population for downstream use.

### `hpv.plot_calibration` rewrite: auto-inspect + top-N ribbon

`hpv.plot_calibration(calib, top_n=50, ncols=None)` now auto-discovers
targets from `calib.eval_kw['data']` (the standardized DataFrame produced
by `load_calib_data`) and lays out one panel per `(scope, name)` group:

- `all_hpv.<name>.<bin>` — age panel (`figS2`-style: per-bin bars/lines
  with a model ribbon across the top-N trials).
- `all_hpv.<name>` scalar — timeseries with a model ribbon.
- `by_genotype.<name>.<g>` — per-genotype box plots across the top-N
  trials.

Under the hood, `_run_top_n_trials(calib, n)` re-runs the top-N trials in
parallel via `sc.parallelize`, using the calibration's stored `build_fn`
and `sim` template. `top_n=50` is the default; pass `top_n=1` to plot
just the best. Old plot-config kwargs (`res_to_plot`, per-panel dicts)
are gone.

### `hpv.Calibration.shrink(n_results=100)`

Returns a lightweight ``sc.objdict`` with the top-``n_results`` trials
(by mismatch) plus the metadata needed to replot / rebuild sims: ``df``,
``best_pars``, ``eval_kw``, ``calib_pars``, ``build_fn``, ``build_kw``,
``sim``. Drop-in for ``hpv.plot_calibration`` and downstream ribbon
plotting code. Intended for committing calibration outputs to source
control: a full 1000-5000-trial ``calib.obj`` is many MB (Optuna study,
tempdir storage refs, per-trial state); the shrunken version drops all
of that.

Unlike ``stisim.Calibration.shrink``, this uses ``nsmallest(n, 'mismatch')``
(not ``iloc[0:n]``, which would return first-N chronological trials) and
doesn't require ``self.sim_results`` from ``ss.Calibration`` components
(``hpv.Calibration`` uses ``eval_fn=default_eval_fn``, not components).

### `hpv.Calibration` default `reseed=False`; top-N trial selection

Two related fixes:

- ``ss.Calibration`` defaults to ``reseed=True``, which resamples
  ``rand_seed`` from ``[0, 1_000_000]`` on every trial *as if it were a
  calibrated parameter*. For HPV/cancer this is nearly always wrong:
  cancer is a rare event, per-agent stochastic variance is large, and the
  resulting mismatch surface is dominated by seed noise -- Optuna picks
  the luckiest seed rather than the best parameters. ``hpv.Calibration``
  now defaults ``reseed=False``; callers who want per-trial reseed pass
  it explicitly.
- ``hpv.plot_calibration._run_top_n_trials`` used ``calib.df.head(n)``,
  which returns the first *N chronological* trials (essentially early
  Optuna exploration = random guesses), not the *N* best-fitting trials.
  Fixed to ``calib.df.nsmallest(n, 'mismatch')``. ``rand_seed`` is now
  also stripped from top-N par sets before rerun (in case a caller opts
  back into ``reseed=True``).

**Regression information:** any calibration produced with the pre-fix
default (``reseed=True``) has ``rand_seed`` in ``best_pars`` / ``calib.df``
and a top-N ribbon that reflects seed-noise variance, not parameter
variance. Redo the calibration on the new default before trusting the fit.

### `hpv.results_by_genotype` helper

New helper in `hpv.analyzers` (also exported at the top level):
`hpv.results_by_genotype(sim, key='cum_cancers', normalize=False)` returns
a year-indexed DataFrame of per-genotype values for a stock or flow
result, optionally normalized to sum to 1 per year (giving a per-genotype
distribution). This is the sim-level replacement for the removed
`AgeResults['cancerous_genotype_dist']` / `['cin_genotype_dist']` keys
(see the `by_age` analyzer notes below).

### `SexualNetwork` flat pars + `hpv.NetworkPars`: single source of truth

Every network knob now lives on ``self.pars`` as a flat, discoverable entry —
``m_cross_layer``, ``f_cross_layer``, ``m_partners``, ``f_partners``,
``debut``, ``layer_probs``, ``mixing``, ``acts``, ``dur_pship``,
``age_act_pars``. Defaults come from ``hpv.NetworkPars`` (new class in
``parameters.py``, mirroring ``GenotypePars``). Users open ``parameters.py``
to see every default value in the sim — the pre-3.0.1 split between
``_default_network_pars`` (in ``data/country.py``) and
``SexualNetwork.pars`` (which only held the already-shaped ``layer_pars``
and ``debut``) is gone.

Internally, ``SexualNetwork.init_pre`` calls a private ``_shape_pars`` that
builds ``self._layer_pars`` and ``self._debut`` (ss.prob-wrapped, v2 dist
dicts converted) from the flat pars. Runtime code reads the shaped cache;
the flat pars remain the single point of override.

Consequences:

- ``_shape_network_pars`` in ``data/country.py`` is deleted;
  ``_default_network_pars`` collapses to a thin wrapper over ``NetworkPars``.
- ``route_pars`` no longer has a network-rebuild special case. Flat network
  keys route to the registry like any other module par; scoped
  ``sexualnetwork.<par>[.<sub>]`` paths walk into nested Dist/dict pars via
  the standard ``apply_nested``.
- ``build_kazakhstan_sim`` in the Kazakhstan analysis project disappears
  entirely — calibration flows through the default router with keys like
  ``sexualnetwork.m_partners.c.par1`` and ``hi5.dur_cin.mean``.

### Flat parameter routing: `hpv.route_pars` + `hpv.Sim(pars={...})`

`hpv.route_pars(sim, pars)` (new; also exported from the top-level package)
routes a flat parameter dict to the right module, modeled on
`stisim.route_pars`. Bare keys are looked up in a registry built from each
module's `.pars`; scoped keys are either dotted (`hpv16.cin_fn.k`) or
nested-dict (`hpv16=dict(cin_fn=dict(k=0.7))`). `Dist` pars merge cleanly:

```python
sim = hpv.Sim(location='nigeria', genotypes=[16, 18], pars={
    'rand_seed': 42,
    'beta': 0.15,                                # broadcast to every HPV
    'm_cross_layer': 0.4,                        # SexualNetwork rebuild
    'hpv16': dict(dur_cin=6, cin_fn=dict(k=0.7)),# scoped nested dict
    'hpv18.cin_fn.k': 0.8,                       # scoped dotted key
    'cross_immunity.rel_sev.loc': 0.9,           # into CrossImmunity connector
})
```

- `hpv.Sim(pars={...})`: sim-level keys forward to `super().__init__(pars=)`
  (so pre-init state like `n_agents` picks them up); everything else routes
  after super init.
- `hpv.calibration.build_sim` is now a thin alias for `route_pars`; the old
  dotted-key-only routing is gone. Cross-immunity matrix cells are still
  reachable via `cross_immunity.<matrix>.<tgt>.<src>` (4-part legacy path).
- Scalar overrides on TimePar-wrapped `Dist` pars (e.g. `dur_cin=6` where
  the default is `mean=ss.years(5)`) auto-wrap the scalar to preserve
  units — starsim's `Dist.set()` otherwise strips the wrapper.

### `CrossImmunity` connector: pars via `define_pars`, `make_cross_immunity` method

`CrossImmunity` now registers `rel_sev` (an `ss.normal` Dist),
`cross_imm_{sus,sev}_{med,high}`, and `own_imm_hr` via `define_pars`, so
they route through the standard machinery (`hpv.route_pars`, `Pars.update`).
The old `rel_sev_loc` / `rel_sev_scale` constructor kwargs are gone; pass
`rel_sev=ss.normal(loc=..., scale=...)` or update via
`cross_immunity.rel_sev.loc=...`.

The module-level `hpv.get_cross_immunity` factory has been replaced by an
instance method: `CrossImmunity().make_cross_immunity(keys=...)` reads the
scalar med/high/own_imm_hr values from `self.pars`. Matrix-building helpers
(`_build_cross_matrix`, `_CLADE_HIGH_PAIRS`, `_FULL_OWN_IMM_KEYS`) moved
from `parameters.py` to `cross_genotype.py` alongside the class.

### `by_age` analyzer (renamed from `AgeResults`, API redesigned)

`hpv.AgeResults` has been renamed to `hpv.by_age`, matching the naming
convention of the other analyzers (`age_pyramid`, `age_causal_infection`,
`dalys`, `snapshot`). The API was also redesigned:

```python
ar = hpv.by_age('cancers')                              # one key
ar = hpv.by_age(['cancers', 'hpv_prevalence'])          # multiple keys
ar = hpv.by_age('cancers', years=2020)                  # reporting filter
ar = hpv.by_age('cancers', edges=[0,25,50,75,100])      # custom bins
```

Storage is now per-timestep as one `ss.Result` per (key, age bin), named
e.g. `sim.results.by_age.cancers_20_25`. This means:

- Count and flow keys are declared `scale=True`, so starsim's
  `finalize_results` multiplies by `sim.pars.pop_scale` automatically
  (matches the v2 `cancers` / `cins` semantics).
- Prevalence keys are declared `scale=False` (ratios in [0,1] should not
  be scaled).
- Individual per-bin Results can be annualized via `ss.Result.annualize()`.
- Convenience 2D arrays are populated on the analyzer after finalize:
  `sim.analyzers.by_age.cancers` is shape `(npts, n_bins)`.
- `to_dataframe(key)` annualizes each per-bin Result and returns a
  year-indexed DataFrame with age-bin columns.

**Removed keys** (`normalize=` argument on `to_dataframe` also removed):

- `cancerous_genotype_dist`, `cin_genotype_dist` — per-genotype-by-age
  distributions. Use `hpv.results_by_genotype(sim, key='cum_cancers')`
  at the sim level instead.
- `cancer_incidence`, `cin_incidence` — per-100k rates. Compute from
  `cancers` / an at-risk denominator externally.

**Removed storage shape**: prevalence keys used to store `(n_bins, 2)`
= `(num, denom)` per bin so `ss.BetaBinomial` calibration components
could pull raw counts. Now storing the ratio directly. No one uses the
BetaBinomial component today.

### `AgeResults`-era `cancers` / `cins` semantics still hold

Cancers / cins keys were changed in v3.0 → v3.0.1 from a prevalent-stock
snapshot (raw agent counts) to an annual event flow multiplied by
`sim.pars.pop_scale` — restoring v2 semantics. Use `n_cancerous` /
`n_cin` for the prevalent-stock snapshots.

**Regression information (AgeResults / by_age):** any calibration or
plot comparing `AgeResults['cancers']` to absolute annual case-count
data (e.g. Globocan cancer-cases-by-age) was silently off by both a
stock-vs-flow semantic shift and a `pop_scale` factor. It now matches.
Scripts asserting exact prevalent stock counts should migrate to
`n_cancerous`. Downstream `hpv.AgeResults` references must be renamed
to `hpv.by_age` and the `result_args=` API replaced with the positional-
keys form above.

### `age_pyramid` / `dalys` / `age_causal_infection` scaled to real pop

Three analyzers previously stored per-agent counts weighted only by
`people.scale` (multiscale weight, not `pop_scale`), so their outputs
were in agent-scale units. They now multiply by `sim.pars.pop_scale` at
finalize to emit real-population magnitudes, matching `sim.results.*`
and `by_age` FLOW/COUNT semantics:

- `age_pyramid.age_pyramids[date]` values × pop_scale.
- `dalys.yll`, `dalys.yld`, `dalys.dalys` × pop_scale (removes the
  "callers multiply" docstring caveat).
- `age_causal_infection.weights` × pop_scale.

### `HPVTotal.prevalence` and `HIVStratifiedResults.hpv_prevalence_*`

These `ss.Result` fields were declared with the default `scale=True`, so
starsim's finalize multiplied them by `pop_scale` — but they are ratios
in [0,1] that should not be scaled. Now declared `scale=False`. Under
the v3.0 default (`pop_scale=1`) the bug was invisible; the auto-populate
change below exposes it.

### `hpv.Calibration.worker` crash tolerance

`ss.Calibration.worker` (in starsim) calls `study.optimize` bare. When
one worker hits a transient Optuna storage error (e.g. SQLite lock under
heavy parallelism), Optuna's own error handler raises
`assert False, 'Should not reach.'`, taking the whole run down.
`hpv.Calibration` now overrides `worker()` to wrap `study.optimize` in
try/except — a single worker failure logs and returns None; the
remaining workers complete. Matches `stisim.Calibration.worker`.

### Demographics API additions

- `hpv.Sim()` bare (no `location` supplied) is now a valid natural-history
  playground: uniform ages 0-60 (starsim default), no births/deaths/
  migration modules, location-agnostic default sexual network, `pop_scale=1`.
  Emits an `ss.warn` describing the auto-configuration.
- `hpv.Sim(location='<name>')` — when `total_pop` is not passed, it is now
  auto-populated from the sum of the UN WPP per-age counts at the sim start
  year (matches `stisim.Sim.process_demographics`). Previous v3.0 behavior
  left `pop_scale=1` even with a location.
- `hpv.Sim(location='<name>', datafolder='<path>')` — a new `datafolder`
  kwarg accepts user-supplied CSVs (`age_data.csv`, `birth_rate.csv`,
  `death_rate.csv`, `pop_total.csv`) for sub-national / custom locations.
  Missing indicators emit a warning and fall back to the bundled UN WPP
  data for the caller's `location`.
- `hpv.demo(name=None, run=True, plot=True, **kwargs)` — new factory
  returning a canonical example sim. Currently only `'nigeria'` is
  registered (defaults on no-arg). Follows the `hivsim.demo` pattern
  (extensible via `hpvsim.examples.EXAMPLES`).

**Regression information:** any script using `hpv.Sim(location='<name>',
...)` without passing `total_pop` explicitly will now see `pop_scale > 1`
and downstream results scaled to real-population magnitudes. Any code that
was implicitly relying on `pop_scale=1` (comparing raw agent counts to
real-scale targets) must either pass `total_pop=n_agents` to preserve the
old behavior, or divide finalized `sim.results` by `sim.pars.pop_scale`
before use.

### Miscellaneous fixes

- `HIVStratifiedResults.hpv_prevalence_with_hiv` / `hpv_prevalence_no_hiv`
  are now declared with `scale=False`. Previously the numerator was
  multiplied by `pop_scale` at finalize but the denominator (a ratio) was
  not, so the reported "prevalence" was inflated by `pop_scale`. Under
  the old v3.0 default (`pop_scale=1`) the bug was invisible; the
  demographics changes above expose it and this release fixes it.

## Version 3.0.0 (2026-07-16)

HPVsim v3 is a ground-up migration onto [Starsim](https://docs.starsim.org).
The disease model, sexual network, demographics, interventions, and analyzers
are now Starsim modules, and `hpv.Sim` wraps `starsim.Sim`. The natural-history
model, genotypes, and interventions are preserved; the API changed
substantially. See the [migration guide](docs/migration.qmd) for a full v2→v3
walkthrough.

- Rebuilt on Starsim (`starsim>=3.5`); requires Python ≥ 3.10.
- Multi-genotype HPV with cross-immunity, natural-history progression
  (precin/CIN/cancer), sexual network, births/deaths/age-specific migration,
  vaccination, screening, and test-and-treat cascades all reimplemented as
  Starsim modules.
- Multiscale modeling (`ms_agent_ratio`) grows real fine agents rather than
  scheduling extras, giving an intervention-correct, unbiased cancer level.
- Analyzers (`snapshot`, `age_pyramid`, `age_causal_infection`, `dalys`,
  `AgeResults`, per-genotype results) and built-in plotting ported.
- HIV–HPV co-infection via a transmission-based HIV module (built on STIsim).
  Adding an `hpv.HIV` disease auto-wires the `hpv_hiv_connector` (raising HPV
  susceptibility/severity by CD4 stratum) and an `HIVStratifiedResults`
  analyzer. See the migration guide for the API.
- *Regression information*: v3 uses Starsim's RNG framework and does not share
  a stream with v2; results are not bit-identical to v2 even with the same
  seed. Validate on overlapping uncertainty intervals, not exact values.
- *Regression information*: the `hpv.Sim` constructor no longer takes a
  positional parameter dict (the first positional argument is `location`); pass
  parameters as keyword arguments or `hpv.Sim(**pars)`. `end` is now `stop`; the
  pooled `'hr'` genotype shorthand is replaced by `hi5`/`ohr`.
- *Regression information*: results are organized by module
  (`sim.results.hpv16.cum_infections`, aggregate `sim.results.all_hpv.*`)
  rather than one flat dict; `sim.short_summary` and the top-level
  `hpv.save`/`hpv.load`/`hpv.MultiSim` helpers are removed (use `sim.save()` /
  `ss.load()` / `ss.MultiSim`).
- Not ported to v3.0: waning immunity, `EventSchedule`, and custom
  `settings.py` (superseded by `ss.options`).

## Version 2.3.0 (2026-04-20)

- Fixes dt-dependent results by scaling partnership formation rates to
  per-timestep probabilities; `layer_probs` and cross-layer defaults
  converted to annual probabilities.
- *Regression information*: If workflows from v2.2.6 or earlier override default `layer_probs`, `f_cross_layer`, or `m_cross_layer` values
  and have timesteps not equal to 1 year, then the probabilities must be converted to annual probabilities instead of per-timestep probabilities using this formula: `1 - (1 - prob) ** dt`
- *Regression information:* baseline model outputs change; baselines have been regenerated.
- Fixes `precins` flow never being incremented; removes redundant `dysplasias` flow (was an alias of `cins`).
- Adds test coverage for previously untested code paths.
- Vaccine immunity is now sterilizing (all-or-nothing) rather than leaky (per-contact). `imm_init` sets the probability of sterilizing immunity; non-sterilizing recipients get leaky protection at the `imm_init` level. Default is 0.95.
- Adds per-timestep transmission logging (`sim._transmission_log`) for downstream analysis of transmission chains.
- Calibration now supports resuming from an existing database via `keep_db=True`, running only the remaining trials.
- Calibration workers catch exceptions instead of crashing the entire run.
- Fixes `res_to_plot` indexing bug in `Calibration.plot()`.
- *Regression information*: vaccine efficacy will differ from previous versions due to the immunity model change.

## Version 2.2.7 (2026-04-22)

- Fix cancer treatment results always being blank: `BaseTreatment.check_eligibility` was excluding cancer patients (preventing radiation from running), `BaseTreatment.apply` was writing to CIN fields for cancer treatment, and `cum_cancer_treated` cumsum used the wrong source array
- *Github info* PR [94](https://github.com/starsimhub/hpvsim/pull/94), issue [91](https://github.com/starsimhub/hpvsim/issues/91)

## Version 2.2.6 (2026-04-17)

- Reconcile different copies of repository
- *Github info* PR [75](https://github.com/starsimhub/hpvsim/pull/75)

## Version 2.2.5 (2025-10-27)

- Small bugfix for campaign vaccination
- *Github info* PR
  [689](https://github.com/starsimhub/hpvsim_orig/pull/689)

## Version 2.2.4 (2025-08-20)

- Fixes a bug in analyzer results for cancer by age and HIV status
- *Github info* PR
  [687](https://github.com/starsimhub/hpvsim_orig/pull/687)

## Version 2.2.3 (2025-06-27)

- Small bugfixes and changes to HIV module parameterization
- *Github info* PR
  [685](https://github.com/starsimhub/hpvsim_orig/pull/685)

## Version 2.2.2 (2025-06-20)

- Bugfix to allow running simulations beyond 2100
- *Github info* PR
  [681](https://github.com/starsimhub/hpvsim_orig/pull/681)

## Version 2.2.1 (2025-05-29)

- Bugfix for running calibrations to prevent interventions being
  reinitialized
- *Github info* PR
  [678](https://github.com/starsimhub/hpvsim_orig/pull/678)

## Version 2.2.0 (2025-05-23)

- Refresh results: ensure all main results are populated, remove cancer
  detection results, and fix bug with HPV prevalence calculations
- Updates to docs
- *Github info* PR
  [673](https://github.com/starsimhub/hpvsim_orig/pull/673)

## Version 2.1.0 (2025-03-25)

- Updates how HPV prognoses are re-evaluated for WLWH
- Fixes CD4 reconstitution trajectory so that it plateaus before
  quadratic starts decreasing
- Fixes ART coverage so that it's now by age, sex, and time
- Fixes assignment of HIV mortality based upon ART coverage
- Removes HIV-mortality from background mortality
- Small fix to enable calibration to HIV-stratified data
- Adds a more robust data downloading method and renamed `get_data()` to
  `download_data()`; updated data version to 1.4
- *Github info* PR [652](https://github.com/amath-idm/hpvsim/pull/652)

## Version 2.0.2 (2024-03-05)

- Modifies DALY analyzer to output YLLL, YLD and DALYs
- *Github info* PR [659](https://github.com/amath-idm/hpvsim/pull/659)

## Version 2.0.1 (2024-02-14)

- Adds in relative transmissibility attribute to people that can be
  modified by vaccination or treatment
- *Github info* PR [643](https://github.com/amath-idm/hpvsim/pull/658)

## Version 2.0.0 (2023-11-29)

- Simplifies natural history model by compressing CIN grades
- Changes the way HPV progression is modeled so that there is a
  probability of developing CIN based upon duration of precin and
  probability of cancer based upon duration of cancer (based upon
  Rodriguez et al.
  <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3705579/>)
- Adds support for pre-calibration explorations
- Improvements to networks, including clustering functionality, support
  for different distributions for male and female partners and for
  differing concurrency rates, and changes to default partnership
  durations
- Exposes a parameter for specifying the sex ratio of a population
- Fixes plotting issue with tutorial
- Updates filtering for tests that are not genotype-specific
- *Github info* PR [643](https://github.com/amath-idm/hpvsim/pull/643)

## Version 1.2.7 (2023-09-22)

- Updates `sim.summary` to have more useful information
- *Github info* PR [618](https://github.com/amath-idm/hpvsim/pull/618)

## Version 1.2.6 (2023-09-22)

- Fixes plotting issue with MultiSims and Jupyter notebooks
- Allows scenarios to be run fully in parallel
- *Github info* PR [614](https://github.com/amath-idm/hpvsim/pull/614)

## Version 1.2.5 (2023-09-21)

- Fixes file path when run via Jupyter
- *Github info* PR [610](https://github.com/amath-idm/hpvsim/pull/610)

## Version 1.2.4 (2023-09-19)

- Fixes Matplotlib regression in plotting
- *Github info* PR [609](https://github.com/amath-idm/hpvsim/pull/609)

## Version 1.2.3 (2023-08-30)

- Updates data loading to be much more efficient
- *Github info* PR [604](https://github.com/amath-idm/hpvsim/pull/604)

## Version 1.2.2 (2023-08-11)

- Improved tests and included `conda` environment specification
- *Github info* PR [598](https://github.com/amath-idm/hpvsim/pull/598)

## Version 1.2.1 (2023-07-09)

- Updated data files being used
- *Github info* PR [586](https://github.com/amath-idm/hpvsim/pull/586)

## Version 1.2.0 (2023-05-31)

- Changes to improve run speed, most notably changes to how migration is
  applied
- Additional tests to ensure consistency between calibration results,
  age analyzer results, and sim results
- Updates to natural history to prevent people progressing too quickly
  to cancer
- *Github info* PR [576](https://github.com/amath-idm/hpvsim/pull/576)

## Version 1.1.5 (2023-03-23)

- Adds cross-protection functionality to t-cell immunity and adds
  <span class="title-ref">sev_imm</span> attribute to people
- *Github info* PR [564](https://github.com/amath-idm/hpvsim/pull/564)

## Version 1.1.4 (2023-03-15)

- Fixes bug that caused location data to be loaded twice
- *Github info* PR [546](https://github.com/amath-idm/hpvsim/pull/546)

## Version 1.1.3 (2023-03-14)

- Fixes bug that misses some ways you can specify sex for vaccination
- *Github info* PR [555](https://github.com/amath-idm/hpvsim/pull/555)

## Version 1.1.2 (2023-03-13)

- Fixes bug that never computed cancer deaths by age
- *Github info* PR [554](https://github.com/amath-idm/hpvsim/pull/554)

## Version 1.1.1 (2023-03-01)

- Sets time to and date of HIV death for those not on ART and who fail
  on ART
- Moves all HIV attributes, parameters, and results into hivsim class
  instance
- Merges HIV results with sim.results at conclusion of simulation
- Adds HIV pars as an argument to calibration as well as HIV-specific
  results to age-results analyzer
- Allows for flexible severity growth functions
- *Github info* PR [542](https://github.com/amath-idm/hpvsim/pull/542)

## Version 1.1.0 (2023-02-16)

- Moves all HIV functionality into hiv.py
- Establishes new class HIVsim, which is defined by a set of parameters
  and methods for updating a people object
- Bug fix for setting people.sev wrong on day of infection
- *Github info* PR [526](https://github.com/amath-idm/hpvsim/pull/526)

## Version 1.0.1 (2023-02-09)

- Fixes computation of dur_episomal by adjusting for dt
- *GitHub info*: PR [527](https://github.com/amath-idm/hpvsim/pull/527)

## Version 1.0.0 (2023-01-31)

- Official release!
- *GitHub info*: PR [521](https://github.com/amath-idm/hpvsim/pull/521)

## Version 0.4.17 (2023-01-31)

- Adds a tutorial on calibration
- Small changes to parameter values
- *GitHub info*: PR [520](https://github.com/amath-idm/hpvsim/pull/520)

## Version 0.4.16 (2023-01-30)

- Change to natural history, including computation of transformation
  based upon time with dysplasia
- Addition of cellular immunity to moderate progression in a secondary
  infection
- Default parameter changes and some small typo/bug fixes
- *GitHub info*: PR [513](https://github.com/amath-idm/hpvsim/pull/513)

## Version 0.4.15 (2023-01-13)

- Fixed bug in intervention and analyzer initialization
- *GitHub info*: PR [511](https://github.com/amath-idm/hpvsim/pull/511)

## Version 0.4.14 (2023-01-11)

- Add Sweep class
- *GitHub info*: PR [431](https://github.com/amath-idm/hpvsim/pull/431)

## Version 0.4.13 (2023-01-09)

- Dysplasia percentages are now tracked throughout agent lifetimes, and
  CIN grades are defined as properties based on these percentages
- Removes all genotypes aside from HPV 16, 18 and a composite 'other
  high risk' genotype from the defaults
- *GitHub info*: PR [507](https://github.com/amath-idm/hpvsim/pull/507)

## Version 0.4.12 (2023-01-02)

- Adds documentation and examples for screening algorithms.
- *GitHub info*: PR [505](https://github.com/amath-idm/hpvsim/pull/505)

## Version 0.4.11 (2022-12-21)

- Adds colposcopy and cytology testing options, along with default
  values for screening sensitivity and specificity.
- Adds a clearance probability for treatment to control the % of treated
  women who also clear their infection
- Removes use_multiscale parameter and sets ms_agent_ratio to 1 by
  default
- *GitHub info*: PR [497](https://github.com/amath-idm/hpvsim/pull/497)

## Version 0.4.10 (2022-12-19)

- Change the seed used for running simulations to avoid having random
  processes in the model run sometimes being correlated with population
  attributes
- Deprecate `Sim.set_seed()` - use `hpu.set_seed()` instead
- Added `hpvsim.rootdir` to provide a convenient absolute path to the
- Added equality operator for <span class="title-ref">Result</span>
  objects
- Exporting simulation results to JSON now includes 2D results (e.g., by
  genotype)
- `age_pyramid` and `age_results` analyzer argument changed from
  `datafile` to `data` since this input supports both passing in a
  filename or a dataframe
- *GitHub info*: PR [485](https://github.com/amath-idm/hpvsim/pull/485)

## Version 0.4.9 (2022-12-16)

- Added in high- and low-grade lesions to type distribution results
- Changes default duration and rate of dysplasia for hr HPVs
- *GitHub info*: PR [479](https://github.com/amath-idm/hpvsim/pull/482)

## Version 0.4.8 (2022-12-14)

- Small bug fix to re-enable plots of cytology outcomes by genotype
- *GitHub info*: PR [484](https://github.com/amath-idm/hpvsim/pull/484)

## Version 0.4.7 (2022-12-13)

- Migration is now modeled by finding mismatches between the modeled
  population size by age and data on population sizes by age
  (previously, this adjustment was done for the overall population
  rather than by age bucket).
- *GitHub info*: PR [479](https://github.com/amath-idm/hpvsim/pull/479)

## Version 0.4.6 (2022-12-12)

- Changes to several default parameters: default genotypes are now 16,
  18, and other high-risk; and default hpv control prob is now 0.
- Results now capture infections by age and type distributions.
- Adds age of cancer to analyzer
- Changes to default plotting styles
- Various bugfixes: prevents immunity values from exceeding 1, ensures
  people with cancer aren't given second cancers
- *GitHub info*: PR [458](https://github.com/amath-idm/hpvsim/pull/458)

## Version 0.4.5 (2022-12-06)

- Removes default screening products pending review
- *GitHub info*: PR [464](https://github.com/amath-idm/hpvsim/pull/464)

## Version 0.4.4 (2022-12-05)

- Changes to progression to cancer -- no longer based on clinical
  cutoffs, now stochastically applied by genotype to CIN3 agents
- *GitHub info*: PR [430](https://github.com/amath-idm/hpvsim/pull/430)

## Version 0.4.3 (2022-12-01)

- Fixes bug with population growth function
- *GitHub info*: PR [459](https://github.com/amath-idm/hpvsim/pull/459)

## Version 0.4.2 (2022-11-21)

- Changes to parameterization of immunity
- *GitHub info*: PR [425](https://github.com/amath-idm/hpvsim/pull/425)

## Version 0.4.1 (2022-11-21)

- Fixes age of migration
- Adds scale parameter for vital dynamics
- *GitHub info*: PR [423](https://github.com/amath-idm/hpvsim/pull/423)

## Version 0.4.0 (2022-11-16)

- Adds merge method for scenarios and fixes printing bugs
- *GitHub info*: PR [422](https://github.com/amath-idm/hpvsim/pull/422)

## Version 0.3.9 (2022-11-15)

- Simplifies genotype initialization, adds checks for HIV runs.
- Since the last release, changes were also made to virological
  clearance rates for people receiving treatment - previously all
  treated people would clear infection, but now some may control
  latently instead.
- *GitHub info*: PRs [421](https://github.com/amath-idm/hpvsim/pull/421)
  and [420](https://github.com/amath-idm/hpvsim/pull/420)

## Version 0.3.8 (2022-11-02)

- Store treatment properties as part of sim.people
- *GitHub info*: PR [413](https://github.com/amath-idm/hpvsim/pull/413)

## Version 0.3.7 (2022-11-01)

- Fix to ensure consistent results for the number of txvx doses
- *GitHub info*: PR [411](https://github.com/amath-idm/hpvsim/pull/411)

## Version 0.3.6 (2022-11-01)

- Fix bug related to screening eligibility. NB, this has a sizeable
  impact on results - screening strategies will be much more effective
  after this fix.
- *GitHub info*: PR [396](https://github.com/amath-idm/hpvsim/pull/396)

## Version 0.3.5 (2022-10-31)

- Store stocks related to interventions
- *GitHub info*: PR [395](https://github.com/amath-idm/hpvsim/pull/395)

## Version 0.3.4 (2022-10-31)

- Bugfixes for therapeutic vaccination
- *GitHub info*: PR [394](https://github.com/amath-idm/hpvsim/pull/394)

## Version 0.3.3 (2022-10-30)

- Changes to therapeautic vaccine efficacy assumptions
- *GitHub info*: PR [393](https://github.com/amath-idm/hpvsim/pull/393)

## Version 0.3.2 (2022-10-26)

- Additional tutorials and minor release tidying
- *GitHub info*: PR [380](https://github.com/amath-idm/hpvsim/pull/380)

## Version 0.3.1 (2022-10-26)

- Fixes bug with screening
- Increases coverage of baseline test
- *GitHub info*: PR [373](https://github.com/amath-idm/hpvsim/pull/373)

## Version 0.3.0 (2022-10-26)

- Implements multiscale modeling
- Minor release tidying
- *GitHub info*: PR [365](https://github.com/amath-idm/hpvsim/pull/365)

## Version 0.2.11 (2022-10-25)

- Changes the way dates of HPV clearance are assigned to use durations
  sampled
- *GitHub info*: PR [374](https://github.com/amath-idm/hpvsim/pull/374)

## Version 0.2.10 (2022-10-24)

- Fixes bug with treatment
- *GitHub info*: PR [354](https://github.com/amath-idm/hpvsim/pull/354)

## Version 0.2.9 (2022-10-18)

- Prevents infectious people from being passed to People.infect()
- Fixes bugs with initialization within scenario runs
- Remove ununsed prevalence results
- *GitHub info*: PR [338](https://github.com/amath-idm/hpvsim/pull/345)

## Version 0.2.8 (2022-10-17)

- Fixes bug with intervention year interpolation
- Changes reactivation probabilities to annual, not per time step
- Refactor prognoses calls
- *GitHub info*: PR [338](https://github.com/amath-idm/hpvsim/pull/338)

## Version 0.2.7 (2022-10-14)

- Adds robust relative paths via `hpv.datadir`
- *GitHub info*: PR [333](https://github.com/amath-idm/hpvsim/pull/333)

## Version 0.2.6 (2022-10-12)

- Removes Numba since slower for small sims and only 10% faster for
  large sims.
- Moves functions from `utils.py` into `people.py`, `sim.py`, and
  `population.py`.
- *GitHub info*: PR [326](https://github.com/amath-idm/hpvsim/pull/326)

## Version 0.2.5 (2022-10-07)

- Adds people filtering (NB: not used, and later removed).
- Fixes bug with `print(sim)` not working.
- Adds baseline tests.
- *GitHub info*: PR [310](https://github.com/amath-idm/hpvsim/pull/310)

## Version 0.2.4 (2022-10-07)

- Changes to dysplasia progression parameterization
- Adds a new implementation of HPV natural history for HIV positive
  women
- Note: HIV was added since the previous version
- *GitHub info*: PR [304](https://github.com/amath-idm/hpvsim/pull/304)

## Version 0.2.3 (2022-09-01)

- Adds a `use_migration` parameter that activates immigration/emigration
  to ensure population sizes line up with data.
- Adds simple data versioning.
- *GitHub info*: PR [279](https://github.com/amath-idm/hpvsim/pull/279)

## Version 0.2.2 (2022-08-22)

- Separates out the `Calibration` class into a separate file and to no
  longer inherit from `Analyzer`. Functionality is unchanged.
- *GitHub info*: PR [255](https://github.com/amath-idm/hpvsim/pull/255)

## Version 0.2.1 (2022-08-19)

- Improves calibration to enable support for MySQL.
- Fixes plotting bug.
- *GitHub info*: PR [253](https://github.com/amath-idm/hpvsim/pull/253)

## Version 0.2.0 (2022-08-19)

- Fixed tests and data loading logic.
- *GitHub info*: PR [251](https://github.com/amath-idm/hpvsim/pull/251)

## Version 0.1.0 (2022-08-01)

- Updated calibration.
- *GitHub info*: PR [215](https://github.com/amath-idm/hpvsim/pull/215)

## Version 0.0.3 (2022-07-18)

- Updated data loading scripts.
- *GitHub info*: PR [156](https://github.com/amath-idm/hpvsim/pull/156)

## Version 0.0.2 (2022-06-15)

- Made into a Python module.
- *GitHub info*: PR [64](https://github.com/amath-idm/hpvsim/pull/64)

## Version 0.0.1 (2022-04-04)

- Initial version.

All notable changes to the codebase are documented in this file. Changes that may result in differences in model output, or are required in order to run an old parameter set with the current version, are flagged with the term "Regression information".

## Version 3.2.0 (2026-09-02)

- Restores female HPV latency, dropped in the v2→v3 rewrite. Opt-in via `hpv_control_prob`; a no-op at its default of 0. Once enabled it has a large effect — latency withholds women from their genotype's cancer pathway, roughly halving cancer counts at the default reactivation hazard — and `hpv_reactivation` carries v2's never-fitted placeholder value of 0.025/year, so fit it for your setting rather than relying on the default.
- Unifies HIV co-infection under `hpv.HIV`, with `hpv.HIV_transmit` and `hpv.HIV_incidence` leaves and new `hpv.Sim(model_hiv=...)` sugar. CD4 stratum now also scales latency reactivation.
- `hpv.Sim` accepts bare hpvsim kwargs as well as `pars=`, so `hpv.Sim(beta=0.15)` works. Adds `nw_pars=` and `imm_pars=`.
- Nested parameter overrides now merge rather than replace sibling keys (`cin_fn`, `cancer_fn`, `age_risk`, `age_act_pars_*`).
- Fixes `hpv.SimPars().dt`, which was 0.5 while `hpv.Sim` used 0.25.
- `hpv.HIV_incidence` and `hpv.data.reshape_art_coverage` now accept age-banded HIV and ART inputs, such as the 5-year bands UNAIDS and Spectrum emit. Banded incidence previously raised `IndexError`, and banded ART coverage silently left most ages untreated. Year lookup is now nearest-year rather than exact-match.
- `hpv.by_age` now accepts the population denominators `n_alive`, `n_females` and `n_males`, so crude age-specific rates no longer need a second analyzer. These keys were declared but unimplemented, and previously raised `ValueError`.
- Vaccination interventions now define `new_vaccinated`, `new_doses`, `cum_vaccinated` and `cum_doses`, so counting doses no longer needs a manual `vaccinated.sum() * pop_scale`.
- Naming an intervention after its own product (`hpv.routine_vx(product='bivalent', name='vx')`) now raises a clear error at construction instead of an opaque `AttributeError: Module vx already added` at init.
- Fixes `hpv.dx(df=...)` and `hpv.tx(df=...)` crashing at `sim.init()` with `TypeError: attribute name must be string, not 'NoneType'`.
- Adds a user guide, refreshes the tutorials, and fixes the docs build, broken since v3.1.0.
- *Regression information*: treatment and therapeutic-vaccine interventions honour `sex=` correctly. `sex=['f','m']` previously restricted silently to women, and `hpv.BaseTxVx` ignored `sex=` altogether. `sex='f'` and `sex=None` are unchanged.
- *Regression information*: the `age_range` upper bound is now exclusive for treatment and txvx, matching vaccination, screening and triage. Agents alive at sim start are unaffected (float ages never land exactly on the bound), but agents born during the run lose one timestep of eligibility at the top of the band — up to `dt/(hi-lo)` of eligible person-time.
- *Regression information*: treating a woman who had entered the latency branch now returns her to susceptible rather than leaving her latent. Only affects runs with `hpv_control_prob > 0`.
- *Regression information*: `reshape_art_coverage` leaves the top age band open-ended, so ART now reaches agents above the last data row (over-80s in a band ending at 80) who previously received none.
- *Regression information*: the internal `n_to_latent` result is no longer published on genotypes or on `all_hpv`. It was scheduling bookkeeping, unscaled, and meaningless once summed across genotypes. `n_latent` is unchanged. Anything indexing `all_hpv` results positionally will shift by one.
- *Regression information*: `stisim` is now optional; HIV users need `pip install hpvsim[hiv]`.
- *Regression information*: `hpv.hiv_incidence`, `hpv.hiv_art`, `hpv_hiv_connector`, and `HIVStratifiedResults` are removed. Use `hpv.HIV_incidence`/`hpv.HIV_transmit` or `model_hiv=`.
- *Regression information*: `HPV.pars.beta` is now a scalar; use `HPV.validate_beta()` for the per-network dict.
- *Regression information*: per-cell `cross_immunity.<matrix>.<tgt>.<src>` calibration keys are removed; use the scalar `CrossImmunity` pars.

### Known issues

These predate v3.2 and are not introduced by it, but they are documented here because they affect how results should be interpreted.

- **Results depend on `dt`.** The same parameters give substantially different epidemiology at different timesteps: HPV prevalence is 0.001, 0.044 and 0.200 at `dt` of 1.0, 0.5 and 0.25, and cervical cancer counts scale similarly. The values do not converge as `dt` falls. Treat `dt=0.25` (the default) as the reference, keep `dt` fixed across any comparison, and recalibrate if you change it — a calibration is specific to the `dt` it was fitted at. The likely cause is that `layer_probs` specifies the fraction of an age band that is partnered, a stock, but is consumed as a per-year partnership formation rate.
- **Multiscale over-reports agent counts.** With `ms_agent_ratio > 1`, agents grown to follow extra cancer trajectories carry a reduced per-agent weight, but `n_alive` and the intervention flow results (`n_screened`, `new_cin_treated`, `new_doses`) count them at full weight. The error grows with both the ratio and the length of the run, because it compounds: exact at ratio 1, about 1.01x at ratio 5, and at ratio 100 roughly 1.3x after one year rising to 3.4x over a sixty-year run. This is a reporting bug only — the underlying dynamics are correct. `sum(people.scale)` is conserved and grows at the right rate, so `sum(people.scale) * pop_scale` recovers the true population, and results whose denominators are already scale-weighted, including `asr_cancer_incidence` and the per-genotype disease results, are unaffected.
- **Two interventions cannot share one product.** Products are registered as modules keyed by name, so `hpv.routine_vx(product='bivalent')` alongside `hpv.campaign_vx(product='bivalent')` raises `AttributeError: Module vx already added`. This blocks the routine-plus-catch-up pattern; use a single intervention with a time-varying `prob`, or distinct product types, until this is fixed.
- **`hpv.tx` on a latent infection is a silent no-op.** A therapeutic-vaccine product with `latent` rows will report a woman as treated and set her clearance time, but she is not `infected`, so clearance never fires and she stays latent. Does not affect `ablation` or `excision`.

## Version 3.1.0 (2026-08-20)
Adds flat parameter routing, a redesigned calibration workflow, and real-population scaling by default; requires `starsim>=3.6`.

- `hpv.route_pars(sim, pars)` and `hpv.Sim(pars={...})` route flat, dotted, or nested keys to the right module. `SexualNetwork` pars are now flat (defaults in `hpv.NetworkPars`), and `CrossImmunity` pars use `define_pars`.
- `hpv.Calibration` takes a single `data=` argument (CSVs, DataFrames, or dicts) and nested `calib_pars` with `[best, low, high, step]` leaves. Adds `shrink()`, `hpv.make_calib_sims()`, and a rewritten `hpv.plot_calibration()`.
- `hpv.AgeResults` is now `hpv.by_age`, with a positional-keys API and one `ss.Result` per (key, age bin). The genotype-distribution and per-100k-rate keys are removed; use `hpv.results_by_genotype()`.
- New WHO2000-standardized `all_hpv.asr_cancer_incidence` / `asr_cancer_mortality` results (no analyzer needed).
- Demographics: `total_pop` is auto-populated from UN WPP when `location` is given, and `datafolder=` loads custom CSVs. `hpv.demo()` returns an example sim; bare `hpv.Sim()` is a natural-history playground.
- Prevalence results are now `scale=False`; `age_pyramid`, `dalys`, and `age_causal_infection` outputs are scaled by `pop_scale`.
- *Regression information*: `hpv.Sim(location=...)` without `total_pop` now gives `pop_scale > 1`, so results are at real-population scale; pass `total_pop=n_agents` for the old behavior.
- *Regression information*: default `transm2f` drops from 3.69 to 2.0, per-act beta is clipped to [0, 1] (higher values previously gave silent NaNs), and `ablation`/`excision` now clear precin, so screen-and-treat scenarios avert more cancers; recalibrate existing parameter sets.
- *Regression information*: all vaccination, screening, and treatment interventions now default to `sex='f'`; pass `sex=None` for both sexes.
- *Regression information*: `hpv.Calibration(datafiles=...)` and flat dotted `calib_pars` are removed; the default is now `reseed=False`, so redo calibrations that fit seed noise under the old `reseed=True`.

## Version 3.0.0 (2026-07-16)
HPVsim v3 is a ground-up migration onto [Starsim](https://docs.starsim.org). The disease model, sexual network, demographics, interventions, and analyzers are now Starsim modules, and `hpv.Sim` wraps `starsim.Sim`. The natural-history model, genotypes, and interventions are preserved; the API changed substantially. See the [migration guide](docs/migration.qmd) for a full v2→v3 walkthrough.

- Rebuilt on Starsim (`starsim>=3.5`); requires Python ≥ 3.10.
- Multi-genotype HPV with cross-immunity, natural history (precin/CIN/cancer), sexual network, demographics, vaccination, screening, and test-and-treat cascades all reimplemented as Starsim modules.
- Multiscale modeling (`ms_agent_ratio`) grows real fine agents rather than scheduling extras, giving an intervention-correct, unbiased cancer level.
- Analyzers (`snapshot`, `age_pyramid`, `age_causal_infection`, `dalys`, `AgeResults`, per-genotype results) and built-in plotting ported.
- HIV–HPV co-infection via a transmission-based HIV module built on STIsim, raising HPV susceptibility and severity by CD4 stratum. (Redesigned in v3.2.)
- *Regression information*: v3 uses Starsim's RNG framework and does not share a stream with v2; results are not bit-identical to v2 even with the same seed. Validate on overlapping uncertainty intervals, not exact values.
- *Regression information*: the `hpv.Sim` constructor no longer takes a positional parameter dict (the first positional argument is `location`); pass parameters as keyword arguments or `hpv.Sim(**pars)`. `end` is now `stop`; the pooled `'hr'` genotype shorthand is replaced by `hi5`/`ohr`.
- *Regression information*: results are organized by module (`sim.results.hpv16.cum_infections`, aggregate `sim.results.all_hpv.*`) rather than one flat dict; `sim.short_summary` and the top-level `hpv.save`/`hpv.load`/`hpv.MultiSim` helpers are removed (use `sim.save()` / `ss.load()` / `ss.MultiSim`).
- Not ported to v3.0: waning immunity, `EventSchedule`, and custom `settings.py` (superseded by `ss.options`).

## Version 2.3.0 (2026-04-20)
- Fixes dt-dependent results by scaling partnership formation rates to per-timestep probabilities; `layer_probs` and cross-layer defaults converted to annual probabilities.
- *Regression information*: If workflows from v2.2.6 or earlier override default `layer_probs`, `f_cross_layer`, or `m_cross_layer` values and have timesteps not equal to 1 year, then the probabilities must be converted to annual probabilities instead of per-timestep probabilities using this formula: `1 - (1 - prob) ** dt`
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
- *Github info* PR [689](https://github.com/starsimhub/hpvsim_orig/pull/689)

## Version 2.2.4 (2025-08-20)
- Fixes a bug in analyzer results for cancer by age and HIV status
- *Github info* PR [687](https://github.com/starsimhub/hpvsim_orig/pull/687)

## Version 2.2.3 (2025-06-27)
- Small bugfixes and changes to HIV module parameterization
- *Github info* PR [685](https://github.com/starsimhub/hpvsim_orig/pull/685)

## Version 2.2.2 (2025-06-20)
- Bugfix to allow running simulations beyond 2100
- *Github info* PR [681](https://github.com/starsimhub/hpvsim_orig/pull/681)

## Version 2.2.1 (2025-05-29)
- Bugfix for running calibrations to prevent interventions being reinitialized
- *Github info* PR [678](https://github.com/starsimhub/hpvsim_orig/pull/678)

## Version 2.2.0 (2025-05-23)
- Refresh results: ensure all main results are populated, remove cancer detection results, and fix bug with HPV prevalence calculations
- Updates to docs
- *Github info* PR [673](https://github.com/starsimhub/hpvsim_orig/pull/673)

## Version 2.1.0 (2025-03-25)
- Updates how HPV prognoses are re-evaluated for WLWH
- Fixes CD4 reconstitution trajectory so that it plateaus before quadratic starts decreasing
- Fixes ART coverage so that it's now by age, sex, and time
- Fixes assignment of HIV mortality based upon ART coverage
- Removes HIV-mortality from background mortality
- Small fix to enable calibration to HIV-stratified data
- Adds a more robust data downloading method and renamed `get_data()` to `download_data()`; updated data version to 1.4
- *Github info* PR [652](https://github.com/amath-idm/hpvsim/pull/652)

## Version 2.0.2 (2024-03-05)
- Modifies DALY analyzer to output YLLL, YLD and DALYs
- *Github info* PR [659](https://github.com/amath-idm/hpvsim/pull/659)

## Version 2.0.1 (2024-02-14)
- Adds in relative transmissibility attribute to people that can be modified by vaccination or treatment
- *Github info* PR [643](https://github.com/amath-idm/hpvsim/pull/658)

## Version 2.0.0 (2023-11-29)
- Simplifies natural history model by compressing CIN grades
- Changes the way HPV progression is modeled so that there is a probability of developing CIN based upon duration of precin and probability of cancer based upon duration of cancer (based upon Rodriguez et al. <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3705579/>)
- Adds support for pre-calibration explorations
- Improvements to networks, including clustering functionality, support for different distributions for male and female partners and for differing concurrency rates, and changes to default partnership durations
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
- Changes to improve run speed, most notably changes to how migration is applied
- Additional tests to ensure consistency between calibration results, age analyzer results, and sim results
- Updates to natural history to prevent people progressing too quickly to cancer
- *Github info* PR [576](https://github.com/amath-idm/hpvsim/pull/576)

## Version 1.1.5 (2023-03-23)
- Adds cross-protection functionality to t-cell immunity and adds <span class="title-ref">sev_imm</span> attribute to people
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
- Sets time to and date of HIV death for those not on ART and who fail on ART
- Moves all HIV attributes, parameters, and results into hivsim class instance
- Merges HIV results with sim.results at conclusion of simulation
- Adds HIV pars as an argument to calibration as well as HIV-specific results to age-results analyzer
- Allows for flexible severity growth functions
- *Github info* PR [542](https://github.com/amath-idm/hpvsim/pull/542)

## Version 1.1.0 (2023-02-16)
- Moves all HIV functionality into hiv.py
- Establishes new class HIVsim, which is defined by a set of parameters and methods for updating a people object
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
- Change to natural history, including computation of transformation based upon time with dysplasia
- Addition of cellular immunity to moderate progression in a secondary infection
- Default parameter changes and some small typo/bug fixes
- *GitHub info*: PR [513](https://github.com/amath-idm/hpvsim/pull/513)

## Version 0.4.15 (2023-01-13)
- Fixed bug in intervention and analyzer initialization
- *GitHub info*: PR [511](https://github.com/amath-idm/hpvsim/pull/511)

## Version 0.4.14 (2023-01-11)
- Add Sweep class
- *GitHub info*: PR [431](https://github.com/amath-idm/hpvsim/pull/431)

## Version 0.4.13 (2023-01-09)
- Dysplasia percentages are now tracked throughout agent lifetimes, and CIN grades are defined as properties based on these percentages
- Removes all genotypes aside from HPV 16, 18 and a composite 'other high risk' genotype from the defaults
- *GitHub info*: PR [507](https://github.com/amath-idm/hpvsim/pull/507)

## Version 0.4.12 (2023-01-02)
- Adds documentation and examples for screening algorithms.
- *GitHub info*: PR [505](https://github.com/amath-idm/hpvsim/pull/505)

## Version 0.4.11 (2022-12-21)
- Adds colposcopy and cytology testing options, along with default values for screening sensitivity and specificity.
- Adds a clearance probability for treatment to control the % of treated women who also clear their infection
- Removes use_multiscale parameter and sets ms_agent_ratio to 1 by default
- *GitHub info*: PR [497](https://github.com/amath-idm/hpvsim/pull/497)

## Version 0.4.10 (2022-12-19)
- Change the seed used for running simulations to avoid having random processes in the model run sometimes being correlated with population attributes
- Deprecate `Sim.set_seed()` - use `hpu.set_seed()` instead
- Added `hpvsim.rootdir` to provide a convenient absolute path to the
- Added equality operator for <span class="title-ref">Result</span> objects
- Exporting simulation results to JSON now includes 2D results (e.g., by genotype)
- `age_pyramid` and `age_results` analyzer argument changed from `datafile` to `data` since this input supports both passing in a filename or a dataframe
- *GitHub info*: PR [485](https://github.com/amath-idm/hpvsim/pull/485)

## Version 0.4.9 (2022-12-16)
- Added in high- and low-grade lesions to type distribution results
- Changes default duration and rate of dysplasia for hr HPVs
- *GitHub info*: PR [479](https://github.com/amath-idm/hpvsim/pull/482)

## Version 0.4.8 (2022-12-14)
- Small bug fix to re-enable plots of cytology outcomes by genotype
- *GitHub info*: PR [484](https://github.com/amath-idm/hpvsim/pull/484)

## Version 0.4.7 (2022-12-13)
- Migration is now modeled by finding mismatches between the modeled population size by age and data on population sizes by age (previously, this adjustment was done for the overall population rather than by age bucket).
- *GitHub info*: PR [479](https://github.com/amath-idm/hpvsim/pull/479)

## Version 0.4.6 (2022-12-12)
- Changes to several default parameters: default genotypes are now 16, 18, and other high-risk; and default hpv control prob is now 0.
- Results now capture infections by age and type distributions.
- Adds age of cancer to analyzer
- Changes to default plotting styles
- Various bugfixes: prevents immunity values from exceeding 1, ensures people with cancer aren't given second cancers
- *GitHub info*: PR [458](https://github.com/amath-idm/hpvsim/pull/458)

## Version 0.4.5 (2022-12-06)
- Removes default screening products pending review
- *GitHub info*: PR [464](https://github.com/amath-idm/hpvsim/pull/464)

## Version 0.4.4 (2022-12-05)
- Changes to progression to cancer -- no longer based on clinical cutoffs, now stochastically applied by genotype to CIN3 agents
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
- Since the last release, changes were also made to virological clearance rates for people receiving treatment - previously all treated people would clear infection, but now some may control latently instead.
- *GitHub info*: PRs [421](https://github.com/amath-idm/hpvsim/pull/421) and [420](https://github.com/amath-idm/hpvsim/pull/420)

## Version 0.3.8 (2022-11-02)
- Store treatment properties as part of sim.people
- *GitHub info*: PR [413](https://github.com/amath-idm/hpvsim/pull/413)

## Version 0.3.7 (2022-11-01)
- Fix to ensure consistent results for the number of txvx doses
- *GitHub info*: PR [411](https://github.com/amath-idm/hpvsim/pull/411)

## Version 0.3.6 (2022-11-01)
- Fix bug related to screening eligibility. NB, this has a sizeable impact on results - screening strategies will be much more effective after this fix.
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
- Changes the way dates of HPV clearance are assigned to use durations sampled
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
- Removes Numba since slower for small sims and only 10% faster for large sims.
- Moves functions from `utils.py` into `people.py`, `sim.py`, and `population.py`.
- *GitHub info*: PR [326](https://github.com/amath-idm/hpvsim/pull/326)

## Version 0.2.5 (2022-10-07)
- Adds people filtering (NB: not used, and later removed).
- Fixes bug with `print(sim)` not working.
- Adds baseline tests.
- *GitHub info*: PR [310](https://github.com/amath-idm/hpvsim/pull/310)

## Version 0.2.4 (2022-10-07)
- Changes to dysplasia progression parameterization
- Adds a new implementation of HPV natural history for HIV positive women
- Note: HIV was added since the previous version
- *GitHub info*: PR [304](https://github.com/amath-idm/hpvsim/pull/304)

## Version 0.2.3 (2022-09-01)
- Adds a `use_migration` parameter that activates immigration/emigration to ensure population sizes line up with data.
- Adds simple data versioning.
- *GitHub info*: PR [279](https://github.com/amath-idm/hpvsim/pull/279)

## Version 0.2.2 (2022-08-22)
- Separates out the `Calibration` class into a separate file and to no longer inherit from `Analyzer`. Functionality is unchanged.
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

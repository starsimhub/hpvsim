---
title: "What's new"
---

## Unreleased (HPVsim 4)

**HPVsim runs on Starsim 4.** The natural-history model, genotypes, network, and products are unchanged in meaning; how you deliver interventions, fit to data, and read results follows Starsim 4.

- **Interventions are Starsim's deliveries of HPVsim's products.** `hpv.routine_vx`, `campaign_vx`, `routine_screening`, `campaign_screening`, `routine_triage`, `campaign_triage`, `treat_num`, `treat_delay`, `routine_txvx`, `campaign_txvx`, and `linked_txvx` are gone. Use `ss.vaccinate(hpv.vx('bivalent'), where=..., coverage=..., start=...)`, `ss.screen(hpv.dx('via'), ...)`, and `ss.treat(hpv.tx('ablation'), ...)`; who is eligible is a condition such as `ss.ppl.female & (ss.ppl.age >= 30) & (ss.ppl.age < 50)`, and a step of a cascade is triggered by a result of the step before, e.g. `who=screening.positive & hpv.no_cancer`. Each person is vaccinated once, and a treatment's outcomes are `cured`, `failed`, and `unneeded`.
- **Calibration is `sim.fit()`.** `hpv.Calibration` and `hpv.make_calib_sims` are gone: give the parameters to fit as `ss.par(value, bounds=[low, high])` and the data to `hpv.Sim(data=...)`, then `sim.fit(gof=hpv.calibration.gof, method='optuna', n_trials=...)`. Parameters inside dict parameters can be fitted by their dotted names (e.g. `cancer_fn.transform_prob`). `hpv.plot_calibration()` takes the fit.
- **Results are named after their states**: `infected`, `cin`, `alive` (were `n_infected`, `n_cin`, `n_alive`), here and in `hpv.by_age`. Every count is weighted by each agent's scale, so `alive` is right under multiscale. A delivery's results are `delivered`, `eligible`, `coverage`, and one per outcome, e.g. `sim.results.screening.positive`. `res.annualize()` is `res.resample('year')`.
- **Running many sims**: a MultiSim's results have the runs as their first axis (`msim.results.all_hpv.cum_cancers.median()`, `.quantiles()`) instead of `msim.median()`, and `ss.parallel(*sims, n_runs=...)` runs each scenario with several seeds.
- `hpv.Sim(copy_inputs=...)` is gone: the sim copies the modules it is given, so read them back from the sim.

*Regression information*: every script that builds interventions or calibrates needs updating; see the [migration guide](migration.qmd) and Starsim's [guide to version 4](https://docs.starsim.org/migration/v3_v4.html).

{{< include ../CHANGELOG.md >}}

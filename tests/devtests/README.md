# Developer tests

Statistical acceptance gates and parameter-recovery checks that are too expensive for the `tests/` suite. Each one needs many seeds, many agents, or a real Optuna calibration, so together they take on the order of 15-20 minutes rather than the seconds the unit suite is budgeted for.

They are *not* redundant with `tests/`: the unit suite checks each mechanism directly (and cheaply) on a single run, while these check the population-level statistical properties that only emerge over many seeds. Run them before opening a PR that touches multiscale, natural history, HIV, or calibration, and in the nightly job.

## Usage

    cd devtests
    pytest                       # everything (~15-20 min)
    pytest -n auto               # in parallel, if pytest-xdist is installed
    pytest test_multiscale_grow_gates.py   # one gate

Dependencies are the same as the unit suite: `pip install -r ../tests/requirements.txt`.

## What lives here

| File | Gate |
| --- | --- |
| `test_multiscale_grow_gates.py` | Cancer incidence flat across `ms_agent_ratio`; intervention-averted fraction equal across ratios; mean-age-at-cancer variance shrinks as the ratio rises. |
| `test_analyzers_multiscale_gates.py` | `age_causal_infection` mean age and `dalys` totals are unbiased across `ms_agent_ratio`, seed-averaged. |
| `test_age_results_subannual.py` | `cancer_incidence` accumulates across every sub-step of a calendar year at `dt<1`. |
| `test_hiv_incidence_import.py` | The Rwanda incidence importer builds a plausible HIV epidemic and wires CD4 trajectories. |
| `test_hiv_coinfection_gate.py` | HIV+ agents carry higher HPV prevalence and more cancers than HIV- agents. |
| `test_calibration_recovery.py` | Synthetic two-parameter recovery through the `data=` + `compute_gof` Optuna path. |
| `test_plot_calibration.py` | `hpv.plot_calibration` renders age-axis and year-axis targets from a real `Calibration` run. |

## Adding a test here

Put a test in `devtests/` only if it is a *statistical* gate — one whose assertion needs many seeds or a large population to be meaningful. If a single small run can check the mechanism, it belongs in `tests/` instead. Keep the `@pytest.mark.slow` marker so `pytest -m "not slow"` still works if these are ever collected alongside the unit suite.

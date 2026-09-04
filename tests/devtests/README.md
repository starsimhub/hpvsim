# Developer tests

> **Slated for deletion in v3.3 (test cleanup).** The whole folder costs ~15-20
> minutes, and the mechanisms these gates cover are also checked cheaply in
> `tests/`. The plan is to keep the unit coverage and drop these. If a gate here
> is the only thing covering a behaviour you care about, move that coverage into
> `tests/` before 3.3.

Statistical acceptance gates and parameter-recovery checks that are too expensive for the `tests/` suite. Each one needs many seeds, many agents, or a real Optuna calibration, so together they take on the order of 15-20 minutes rather than the seconds the unit suite is budgeted for.

They are *not* redundant with `tests/`: the unit suite checks each mechanism directly (and cheaply) on a single run, while these check the population-level statistical properties that only emerge over many seeds. Run them before opening a PR that touches multiscale, natural history, HIV, or calibration, and in the nightly job.

## Usage

    cd devtests
    pytest                       # everything (~15-20 min)
    pytest -n auto               # in parallel, if pytest-xdist is installed
    pytest test_multiscale_grow_gates.py   # one gate

Dependencies are the same as the unit suite: `pip install -r ../tests/requirements.txt`.

## What lives here

(Five further gates listed here previously were removed before 3.2.)

| File | Gate |
| --- | --- |
| `test_multiscale_grow_gates.py` | Cancer incidence flat across `ms_agent_ratio`; intervention-averted fraction equal across ratios; mean-age-at-cancer variance shrinks as the ratio rises. |
| `test_analyzers_multiscale_gates.py` | `age_causal_infection` mean age and `dalys` totals are unbiased across `ms_agent_ratio`, seed-averaged. |

## Adding a test here

Put a test in `devtests/` only if it is a *statistical* gate — one whose assertion needs many seeds or a large population to be meaningful. If a single small run can check the mechanism, it belongs in `tests/` instead. Keep the `@pytest.mark.slow` marker so `pytest -m "not slow"` still works if these are ever collected alongside the unit suite.

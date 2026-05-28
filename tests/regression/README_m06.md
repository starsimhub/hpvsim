# M06 parity baselines

The M06 PR ships two anchor scenarios and three parity tests:

| Anchor PARS | v2 baseline generator | Short-summary parity | Trajectory parity |
|---|---|---|---|
| `anchor_screen_treat.py` | `multi_seed_v2_screen_treat.py` | `test_m06_screen_treat_parity.py` | `test_m06_trajectory_parity.py` |
| `anchor_txvx_routine.py` | `multi_seed_v2_txvx.py` | `test_m06_txvx_parity.py` | (none — short-summary only) |

The v2 baseline JSONs are **gitignored** (covered by the `tests/regression/v[23]_*.json` glob in `.gitignore`):

- `v2_seeds_n30_screen_treat.json` (short-summary)
- `v2_seeds_n30_screen_treat_traj.json` (trajectory)
- `v2_seeds_n30_txvx.json` (short-summary)

## Regenerating baselines

From an environment with the v2.3 release installed:

```bash
python tests/regression/multi_seed_v2_screen_treat.py
python tests/regression/multi_seed_v2_txvx.py
```

Each generator runs 30 seeds × full sim. Expect ~30-60 minutes each.

## Running the parity tests

```bash
pytest tests/test_m06_screen_treat_parity.py \
       tests/test_m06_txvx_parity.py \
       tests/test_m06_trajectory_parity.py \
       -m slow -v
```

All three are marked `@pytest.mark.slow` and excluded from CI's
`-m 'not slow'` run; they execute locally before the M06 PR opens.

If the v2 baseline JSON is absent, the parity test is skipped with a
clear message pointing back to the corresponding generator script.

## Multi-intervention name collisions

Each intervention in the anchor PARS scripts is constructed with an
explicit `name=`:
- `routine_screening(name='primary', ...)`
- `routine_triage(name='colpo', ...)`
- `treat_num(name='excision_rx', ...)`
- `routine_txvx(name='txvx', ...)`

These names are referenced by cross-intervention eligibility callbacks
(e.g. `lambda s: s.interventions['primary'].outcomes['positive']`).
Do not rename them without updating callers.

## M05 lessons embedded in the v2 generators

1. **Counting cadence** — all per-agent flow counters (`screened`,
   `cin_treated`, `tx_vaccinated`, etc.) are alive-masked before summing.
   See `multi_seed_v2_screen_treat.py:_run_one_seed`.
2. **Person-years** — `cancer_incidence_2030_2060` uses
   `annual_dt = sim.resfreq * sim.pars['dt']` as the per-step person-years
   contribution; v2 stores annual results so this comes out to 1.0
   under the default `resfreq=4, dt=0.25`.
3. **Per-step vs flow counters** — the trajectory test uses the
   `_FirstEventLogger` analyzer to compute v2-equivalent flow counters
   from v3 per-intervention BoolArr state, snapshotting `.raw` pre/post
   each step.

## Test pre-PR gate

Before opening the M06 PR:

```bash
pytest tests/ -m "not slow" -x -q             # CI suite (no parity)
pytest tests/test_m06_*_parity.py -m slow -v  # short-summary parity
pytest tests/test_m06_trajectory_parity.py -m slow -v  # trajectory parity
```

All must be green. The CRN-perturbation guard
(`test_no_cascade_baseline_unchanged`) runs as part of the CI suite.

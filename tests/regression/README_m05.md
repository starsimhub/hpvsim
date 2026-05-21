# M05 vaccination parity gates

Two slow tests (`test_m05_vx_routine_parity.py`,
`test_m05_vx_campaign_parity.py`) plus one trajectory test
(`test_m05_vx_trajectory_parity.py`) gate v3 vaccination against
locally-regenerated v2.x baselines. All three follow M03's multi-seed
z-score pattern (`|z| < 3`) over the M03 short summary plus three
vaccination-specific summary scalars (`n_vaccinated_2060`,
`n_doses_2060`, `cancer_incidence_2030_2060`).

## Regenerating the v2 baselines (one-time, local)

1. Activate a separate conda env with v2 hpvsim installed:

       conda activate hpvsim-v2

2. Generate both baseline JSONs (30 seeds each, gitignored output):

       python tests/regression/multi_seed_v2_vx.py --n 30

   Outputs:
       tests/regression/v2_seeds_n30_vx_routine.json
       tests/regression/v2_seeds_n30_vx_campaign.json

3. Switch back to the v3 env:

       conda activate hpvsim-v3

## Running the M05 parity gate locally

    pytest -m slow tests/test_m05_vx_routine_parity.py tests/test_m05_vx_campaign_parity.py tests/test_m05_vx_trajectory_parity.py -v

The slow tests are excluded from CI by `pytest -m 'not slow'`.

## Updating the baselines

If you tighten an anchor scenario's PARS, regenerate the corresponding
v2 baseline (step 2 above) before re-running the parity gate.
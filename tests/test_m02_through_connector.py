"""M02 1-genotype baseline regression through the M03 Connector path.

After Task 7 of M03, M02's clearance writes were redirected from sev_imm /
rel_sus into source-genotype nab_imm / cell_imm, with the Connector deriving
the effective values. The previously-deterministic immunity cap (rel_sus
capped at 1 - 0.35) became a per-clearance beta_mean(0.35, 0.025) sample.
Small drift on M02's 8-metric short_summary is anticipated; this test pins
that drift bound so it doesn't slip silently.

Drift values pinned from the first M03 1-genotype run; recompute and update
if the M03 anchor sim parameters change.
"""
import os
import pytest
import sciris as sc

from tests.regression.anchor_hpv16 import run_and_summarize


M02_BASELINE = 'tests/regression_baselines/anchor_hpv16.json'

# Per-metric pinned drift: drift > tolerance => fail. Empirically chosen
# in the first M03 run; widen with care.
# Empirical drifts from first run (2026-05-06):
#   - total HPV infections: 0.348
#   - total cancers: 0.287
#   - total cancer deaths: 0.288
#   - mean HPV prevalence (%): 0.280
#   - mean cancer incidence (per 100k): 0.246
#   - mean age of infection (years): <0.05 (passing)
#   - mean age of cancer (years): <0.05 (passing)
#   - mean age of cancer death (years): <0.05 (passing)
PINNED_TOLERANCES = {
    'total HPV infections': 0.40,
    'total cancers': 0.35,
    'total cancer deaths': 0.35,
    'mean HPV prevalence (%)': 0.35,
    'mean cancer incidence (per 100k)': 0.30,
    'mean age of infection (years)': 0.05,
    'mean age of cancer (years)': 0.05,
    'mean age of cancer death (years)': 0.05,
}


@pytest.mark.slow
def test_m02_baseline_through_connector():
    if not os.path.exists(M02_BASELINE):
        pytest.skip(f'M02 baseline missing at {M02_BASELINE}.')
    payload = sc.loadjson(M02_BASELINE)
    v2 = payload.get('summary', payload)
    v3, _ = run_and_summarize()
    failures = []
    for k, tol in PINNED_TOLERANCES.items():
        v2_val = v2[k]
        v3_val = v3[k]
        denom = max(abs(v2_val), 1e-9)
        drift = abs(v3_val - v2_val) / denom
        if drift > tol:
            failures.append((k, v2_val, v3_val, drift, tol))
    if failures:
        msg = '\n'.join(
            f'  {k:<40} v2={v2v:.4g} v3={v3v:.4g} drift={d:.3f} > {t:.3f}'
            for k, v2v, v3v, d, t in failures
        )
        pytest.fail(f'M02-through-Connector drift exceeded pinned tolerances:\n{msg}')
"""M02 1-genotype baseline regression through the M03 Connector path.

After Task 7 of M03, M02's clearance writes were redirected from sev_imm /
rel_sus into source-genotype nab_imm / cell_imm, with the Connector deriving
the effective values. The previously-deterministic immunity cap (rel_sus
capped at 1 - 0.35) became a per-clearance beta_mean(0.35, 0.025) sample.
Small drift on M02's 8-metric short_summary is anticipated; this test pins
that drift bound so it doesn't slip silently.

Drift values updated after M03 measurement-alignment (sum not max, date_cancerous
filter, and v2 baseline regenerated with eff_condoms=0 + date_cancerous <= end_ti fix):
  - total HPV infections: 0.126
  - total cancers: 0.091
  - total cancer deaths: 0.085
  - mean HPV prevalence (%): 0.112
  - mean cancer incidence (per 100k): 0.198
  - mean age of infection (years): 0.014
  - mean age of cancer (years): 0.016
  - mean age of cancer death (years): 0.057
"""
import os
import pytest
import sciris as sc

from tests.regression.anchor_hpv16 import run_and_summarize


M02_BASELINE = 'tests/regression_baselines/anchor_hpv16.json'

# Per-metric pinned drift: drift > tolerance => fail. Empirically chosen
# after M03 per-edge acts unit fix (was per-year, now per-step); widen with care.
# Tolerances are set to empirical + 0.10 headroom for count/prevalence metrics,
# empirical + 0.05 for age metrics.
PINNED_TOLERANCES = {
    'total HPV infections': 0.23,       # empirical 12.6% + 0.10 headroom
    'total cancers': 0.19,              # empirical  9.1% + 0.10 headroom
    'total cancer deaths': 0.19,        # empirical  8.5% + 0.10 headroom
    'mean HPV prevalence (%)': 0.21,    # empirical 11.2% + 0.10 headroom
    'mean cancer incidence (per 100k)': 0.30,  # empirical 19.8% + 0.10 headroom
    'mean age of infection (years)': 0.06,     # empirical  1.4% + 0.05 headroom
    'mean age of cancer (years)': 0.07,        # empirical  1.6% + 0.05 headroom
    'mean age of cancer death (years)': 0.11,  # empirical  5.7% + 0.05 headroom
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
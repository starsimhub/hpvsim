"""M03 development gate: 40-entry short_summary parity vs. v2 4-genotype baseline.

Fails any entry that drifts >10% relative to v2. Skipped if the local baseline
JSON is missing (regenerate via tests/regression/baseline_v23.py with
``--genotypes four`` from a v2 hpvsim env).
"""
import os
import pytest
import sciris as sc

from tests.regression.anchor_4genotype import run_and_summarize


BASELINE_PATH = 'tests/regression_baselines/anchor_4genotype.json'
RELATIVE_TOLERANCE = 0.10


@pytest.mark.slow
def test_short_summary_parity_4genotype():
    if not os.path.exists(BASELINE_PATH):
        pytest.skip(
            f'Baseline missing at {BASELINE_PATH}. Regenerate via '
            f"`python tests/regression/baseline_v23.py --genotypes four` "
            f'from a v2 hpvsim environment.'
        )
    payload = sc.loadjson(BASELINE_PATH)
    # Match the JSON shape produced by baseline_v23.regen_4genotype: top-level
    # may be either {'metadata': ..., 'summary': {...}} (M02 convention) or a
    # flat 40-entry dict — handle both.
    v2_summary = payload.get('summary', payload)
    v3_summary, _ = run_and_summarize()

    drifts = {}
    for k, v2_val in v2_summary.items():
        if k not in v3_summary:
            # Skip non-summary keys (like 'total population') that exist in
            # the M02 baseline shape but aren't part of the M03 short_summary.
            if k == 'total population':
                continue
            drifts[k] = (v2_val, None, 'missing in v3')
            continue
        v3_val = v3_summary[k]
        denom = max(abs(v2_val), 1e-9)
        rel_drift = abs(v3_val - v2_val) / denom
        if rel_drift > RELATIVE_TOLERANCE:
            drifts[k] = (v2_val, v3_val, rel_drift)

    if drifts:
        rows = '\n'.join(
            f'  {k:<50} v2={v2:.4g}  v3={v3 if v3 is not None else "MISSING":<10}  drift={d}'
            for k, (v2, v3, d) in drifts.items()
        )
        pytest.fail(
            f'M03 short_summary drift > {RELATIVE_TOLERANCE:.0%} on '
            f'{len(drifts)} of {len(v2_summary)} entries:\n{rows}'
        )
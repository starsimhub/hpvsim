"""HIV-HPV coinfection direction gate.

Runs the full Rwanda HIV+HPV anchor sim, which is far too expensive for the
unit suite. The connector's own arithmetic (CD4 stratification, rel_sus /
rel_sev multipliers) is unit-tested in ``tests/test_hiv_connector.py``; this
asserts the emergent population-level direction.
"""
import numpy as np
import pytest

from tests.regression.anchor_hiv_hpv import build_sim


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_hiv_positive_have_higher_hpv_and_cancer():
    sim = build_sim(seed=0)
    sim.run()
    res = sim.results.hivstratifiedresults
    sl = slice(-40, None)  # last 10 years at dt=0.25
    assert np.nanmean(res['hpv_prevalence_with_hiv'][sl]) > \
           np.nanmean(res['hpv_prevalence_no_hiv'][sl])
    assert res['cancers_with_hiv'].sum() > 0

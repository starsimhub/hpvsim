import pytest
import numpy as np
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


def test_cd4_high_bins_to_gt200():
    from hpvsim.hiv import hpv_hiv_connector
    c = hpv_hiv_connector()
    assert list(c._cd4_stratum(np.array([594.0, 800.0]))) == [1, 1]

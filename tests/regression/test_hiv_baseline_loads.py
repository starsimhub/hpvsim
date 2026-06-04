"""Sanity test: the v2 Rwanda HIV-stratified parity target loads sanely.

Fast: only loads the artifact/published target (no sim run). Asserts the
expected metric keys and that HIV+ cervical-cancer incidence exceeds HIV-
incidence, matching the published ~33 vs ~13 per 100k.
"""
import pytest

from tests.regression.baseline_hiv_v2 import load_hiv_baseline, published_target

METRICS = ['cancer_incidence_no_hiv', 'cancer_incidence_with_hiv']


def test_load_returns_expected_keys():
    t = load_hiv_baseline()
    assert t['metrics'] == METRICS
    assert set(t['aggregate']) == set(METRICS)
    assert set(t['by_age']) == set(METRICS)


def test_aggregate_values_positive_and_banded():
    t = load_hiv_baseline()
    for metric in METRICS:
        v = t['aggregate'][metric]
        assert v['value'] > 0, metric
        assert v['low'] <= v['value'] <= v['high'], metric
        assert v['published'] > 0, metric
        assert v['source']  # provenance string present


def test_hiv_positive_exceeds_negative():
    t = load_hiv_baseline()
    pos = t['aggregate']['cancer_incidence_with_hiv']
    neg = t['aggregate']['cancer_incidence_no_hiv']
    # Model target band
    assert pos['value'] > neg['value']
    # Published 2017 registry (~33 HIV+ vs ~13.1 HIV-)
    assert pos['published'] > neg['published']
    assert neg['published'] == pytest.approx(13.1, abs=0.1)
    assert pos['published'] == pytest.approx(33.0, abs=0.5)


def test_published_in_plausible_range():
    t = load_hiv_baseline()
    # HIV+ target in a plausible cervical-cancer incidence range (per 100k).
    pos = t['aggregate']['cancer_incidence_with_hiv']['value']
    neg = t['aggregate']['cancer_incidence_no_hiv']['value']
    assert 20.0 < pos < 60.0
    assert 8.0 < neg < 25.0


def test_by_age_present_and_monotone_shape():
    t = load_hiv_baseline()
    for metric in METRICS:
        by_age = t['by_age'][metric]
        assert set(by_age) == {'25-35', '35-45', '45-55', '55+'}
        for label, v in by_age.items():
            assert v['value'] > 0, (metric, label)
            assert v['published'] > 0, (metric, label)
        # HIV+ by-age incidence should exceed HIV- in the peak band.
    pos45 = t['by_age']['cancer_incidence_with_hiv']['45-55']['value']
    neg45 = t['by_age']['cancer_incidence_no_hiv']['45-55']['value']
    assert pos45 > neg45


def test_published_target_helper_standalone():
    pub = published_target()
    assert pub['aggregate']['cancer_incidence_with_hiv'] > \
        pub['aggregate']['cancer_incidence_no_hiv']

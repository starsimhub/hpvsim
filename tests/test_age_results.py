"""Unit tests for hpv.AgeResults analyzer."""
import numpy as np
import pytest
import sciris as sc

import hpvsim as hpv


def test_age_results_importable_and_constructible():
    """hpv.AgeResults exists at top level and accepts a result_args dict."""
    ar = hpv.AgeResults(
        result_args=sc.objdict(
            cancers=sc.objdict(
                years=[2020],
                edges=np.array([0., 20., 40., 60., 100.]),
            ),
        ),
    )
    assert isinstance(ar, hpv.AgeResults)
    # result_args stored as objdict whether passed as dict or objdict
    assert 'cancers' in ar.result_args
    assert list(ar.result_args.cancers.years) == [2020]

"""Unit tests for hpv.Calibration and helpers."""
import numpy as np
import pandas as pd
import pytest
import sciris as sc

import hpvsim as hpv


def test_calibration_importable():
    """hpv.Calibration exists at top level and is an ss.Calibration."""
    import starsim as ss
    sim = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0)
    calib_pars = dict(beta=dict(low=0.10, high=0.30, guess=0.20))
    calib = hpv.Calibration(sim, calib_pars, total_trials=2, debug=True)
    assert isinstance(calib, ss.Calibration)
"""Tests for hpv.Sim's parameter-routing surface: the user-facing
equivalence between bare kwargs, pars=, and calib_pars."""
import numpy as np
import pandas as pd
import pytest
import sciris as sc
import starsim as ss

import hpvsim as hpv


def make_sim(**kw):
    return hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0, **kw)


def test_expanddict():
    """expanddict converts flat dotted keys to nested dicts, merging shared prefixes."""
    assert hpv.expanddict({'beta': 0.15}) == {'beta': 0.15}
    assert hpv.expanddict({'hpv16.cin_fn.k': 0.77}) == {'hpv16': {'cin_fn': {'k': 0.77}}}
    assert hpv.expanddict({'hpv16.cin_fn.k': 0.77, 'hpv16.beta': 0.3}) == {
        'hpv16': {'cin_fn': {'k': 0.77}, 'beta': 0.3}
    }

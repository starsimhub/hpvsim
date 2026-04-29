"""Tests for hpvsim.network.SexualNetwork."""

import numpy as np
import pytest
import starsim as ss

from hpvsim.network import SexualNetwork


def test_known_layers_accepted():
    for layer in ('m', 'c'):
        net = SexualNetwork(layer=layer)
        assert net.layer == layer


def test_unknown_layer_rejected():
    with pytest.raises(ValueError, match="m.*c"):
        SexualNetwork(layer='x')


def test_n_partners_elsewhere_with_no_siblings_returns_zeros():
    """One-layer-only sim: helper returns all zeros."""
    net = SexualNetwork(layer='m')
    sim = ss.Sim(networks=[net], n_agents=200, diseases=None,
                 dur=ss.years(1), dt=ss.years(0.5), verbose=0,
                 copy_inputs=False)
    sim.init()
    n = net._n_partners_elsewhere()
    assert n.shape == (len(sim.people),)
    assert (n == 0).all()


def test_n_partners_elsewhere_filters_non_hpv_networks():
    """An ss.RandomNet sibling should NOT be counted."""
    hpv_m = SexualNetwork(layer='m')
    rand = ss.RandomNet(n_contacts=5)
    sim = ss.Sim(networks=[hpv_m, rand], n_agents=200, diseases=None,
                 dur=ss.years(1), dt=ss.years(0.5), verbose=0,
                 copy_inputs=False)
    sim.init()
    sim.run_one_step()
    n = hpv_m._n_partners_elsewhere()
    assert (n == 0).all(), \
        f'isinstance filter failed - {n.sum()} non-zero entries from non-hpv siblings'


def test_n_partners_elsewhere_counts_sibling_hpv_networks():
    """A sibling SexualNetwork instance contributes its edge endpoints."""
    hpv_m = SexualNetwork(layer='m')
    hpv_c = SexualNetwork(layer='c')
    sim = ss.Sim(networks=[hpv_m, hpv_c], n_agents=200, diseases=None,
                 dur=ss.years(1), dt=ss.years(0.5), verbose=0,
                 copy_inputs=False)
    sim.init()
    hpv_c.append(p1=ss.uids([0, 1]), p2=ss.uids([2, 3]),
                 beta=np.array([1.0, 1.0]),
                 dur=np.array([10.0, 10.0]),
                 acts=np.array([100, 100]))
    n = hpv_m._n_partners_elsewhere()
    assert n[0] == 1 and n[1] == 1 and n[2] == 1 and n[3] == 1
    assert n[4:].sum() == 0
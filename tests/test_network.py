"""Tests for hpvsim.network.SexualNetwork."""

import numpy as np
import pytest
import starsim as ss
import sciris as sc
import hpvsim

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
                 acts=np.array([100, 100]),
                 start_ti=np.array([0.0, 0.0]))
    n = hpv_m._n_partners_elsewhere()
    assert n[0] == 1 and n[1] == 1 and n[2] == 1 and n[3] == 1
    assert n[4:].sum() == 0





def _layered_sim(layers=('m', 'c'), n_agents=2000, n_steps=4):
    """Build a Sim with SexualNetwork instances configured from Nigeria data."""
    country = hpvsim.data.load_country('nigeria')
    networks = [SexualNetwork(layer=k, pars=country['network_pars'][k])
                for k in layers]
    sim = ss.Sim(
        networks=networks, n_agents=n_agents, diseases=None,
        dur=ss.years(n_steps * 0.5), dt=ss.years(0.5),
        rand_seed=0, verbose=0,
        copy_inputs=False,
    )
    return sim


def test_pairs_form_after_a_few_steps():
    sim = _layered_sim()
    sim.run()
    for net in sim.networks():
        if isinstance(net, SexualNetwork):
            assert len(net) > 0, f'layer {net.layer} formed no pairs'


def test_pairs_dissolve_via_stock_end_pairs():
    sim = _layered_sim(layers=('c',), n_steps=20)
    sim.run()
    net = sim.networks()[0]
    assert (net.edges.dur > 0).all() or len(net) == 0


def test_pair_endpoints_are_male_female():
    sim = _layered_sim()
    sim.run()
    people = sim.people
    for net in sim.networks():
        if not isinstance(net, SexualNetwork):
            continue
        if len(net) == 0:
            continue
        f_at_p1 = people.female[net.edges.p1]
        f_at_p2 = people.female[net.edges.p2]
        assert (f_at_p1 ^ f_at_p2).all(), \
            f'layer {net.layer} has same-sex pairs'


def test_cross_layer_concurrency_filter():
    """With cross_layer=0, no agent should appear in both m and c."""
    country = hpvsim.data.load_country('nigeria')
    nets = []
    for k in ('m', 'c'):
        pars = sc.dcp(country['network_pars'][k])
        pars['cross_layer'] = {
            'm': ss.prob(0.0, ss.years(1)),
            'f': ss.prob(0.0, ss.years(1)),
        }
        nets.append(SexualNetwork(layer=k, pars=pars))
    sim = ss.Sim(networks=nets, n_agents=2000, diseases=None,
                 dur=ss.years(2), dt=ss.years(0.5), rand_seed=0, verbose=0,
                 copy_inputs=False)
    sim.run()
    m_net, c_net = nets
    if len(m_net) == 0 or len(c_net) == 0:
        return  # not enough sample
    m_members = set(m_net.members.tolist())
    c_members = set(c_net.members.tolist())
    overlap = m_members & c_members
    assert len(overlap) / max(1, len(m_members | c_members)) < 0.01, \
        f'cross_layer=0 violated: {len(overlap)} agents in both layers'


def test_age_mixing_assortativity():
    """Sampled pairs concentrate on/near the mixing-matrix diagonal."""
    sim = _layered_sim(layers=('m',), n_agents=5000, n_steps=10)
    sim.run()
    net = sim.networks()[0]
    if len(net) < 100:
        return
    people = sim.people
    f_at_p1 = people.female[net.edges.p1]
    f_uids = np.where(f_at_p1, net.edges.p1, net.edges.p2)
    m_uids = np.where(f_at_p1, net.edges.p2, net.edges.p1)
    f_ages = people.age[f_uids]
    m_ages = people.age[m_uids]
    bins = np.arange(0, 81, 5)
    f_bins = np.digitize(f_ages, bins) - 1
    m_bins = np.digitize(m_ages, bins) - 1
    n_bins = len(bins) - 1
    obs = np.zeros((n_bins, n_bins))
    for fb, mb in zip(f_bins, m_bins):
        if 0 <= fb < n_bins and 0 <= mb < n_bins:
            obs[fb, mb] += 1
    if obs.sum() == 0:
        return
    obs_d = obs / obs.sum()
    diag = np.trace(obs_d) + np.trace(obs_d, offset=1) + np.trace(obs_d, offset=-1)
    far_off = obs_d[0, -1] + obs_d[-1, 0]
    assert diag > 5 * far_off, \
        f'mixing not assortative: diag {diag:.3f} not >> far {far_off:.3f}'
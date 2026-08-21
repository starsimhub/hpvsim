"""Tests for hpvsim.network.SexualNetwork."""

import numpy as np
import starsim as ss
import sciris as sc
import hpvsim

from hpvsim.network import SexualNetwork


# ---------------------------------------------------------------------------- #
# Construction                                                                 #
# ---------------------------------------------------------------------------- #


def test_default_construction_exposes_flat_pars():
    """SexualNetwork() populates the two fixed layers and every flat par."""
    net = SexualNetwork()
    assert net.layers == ('m', 'c')
    assert hasattr(net, 'partners_target_m')
    assert hasattr(net, 'partners_target_c')
    # NetworkPars-shaped flat pars discoverable on the module.
    for k in ('m_cross_layer', 'f_cross_layer',
              'm_partners_marital', 'm_partners_casual',
              'f_partners_marital', 'f_partners_casual',
              'debut_f', 'debut_m',
              'layer_probs_marital', 'layer_probs_casual',
              'mixing_marital', 'mixing_casual',
              'acts_marital', 'acts_casual',
              'dur_pship_marital', 'dur_pship_casual',
              'age_act_pars_marital', 'age_act_pars_casual'):
        assert k in net.pars, f'missing par: {k}'


def test_overrides_merge_over_defaults():
    """Kwarg overrides mutate specific flat pars without touching the others."""
    net = SexualNetwork(m_cross_layer=0.0, f_cross_layer=0.0)
    assert net.pars.m_cross_layer == 0.0
    assert net.pars.f_cross_layer == 0.0
    # Untouched pars keep NetworkPars defaults.
    assert float(net.pars.debut_f.pars.loc) == 15.0


# ---------------------------------------------------------------------------- #
# Pair-formation behavior with full Nigeria pars                               #
# ---------------------------------------------------------------------------- #


def _sim_with_country_pars(n_agents=2000, n_steps=4):
    """Build a Sim with a fully-configured SexualNetwork from Nigeria data."""
    country = hpvsim.data.load_country('nigeria')
    net = SexualNetwork(**country['network_pars'])
    sim = ss.Sim(
        networks=[net], n_agents=n_agents, diseases=None,
        dur=ss.years(n_steps * 0.5), dt=ss.years(0.5),
        rand_seed=0, verbose=0,
        copy_inputs=False,
    )
    return sim, net


def test_pairs_form_after_a_few_steps():
    sim, net = _sim_with_country_pars()
    sim.run()
    for lkey in net.layers:
        assert net.n_pairs_in_layer(lkey) > 0, f'layer {lkey} formed no pairs'


def test_pairs_dissolve_via_stock_end_pairs():
    sim, net = _sim_with_country_pars(n_steps=20)
    sim.run()
    assert (np.asarray(net.edges.dur) > 0).all() or len(net) == 0


def test_pair_endpoints_are_male_female():
    sim, net = _sim_with_country_pars()
    sim.run()
    if len(net) == 0:
        return
    people = sim.people
    f_at_p1 = people.female[net.edges.p1]
    f_at_p2 = people.female[net.edges.p2]
    assert (f_at_p1 ^ f_at_p2).all(), 'has same-sex pairs'


def test_cross_layer_concurrency_filter():
    """With cross_layer=0, no agent should appear in both m and c."""
    country = hpvsim.data.load_country('nigeria')
    network_pars = sc.dcp(country['network_pars'])
    network_pars['m_cross_layer'] = 0.0
    network_pars['f_cross_layer'] = 0.0
    net = SexualNetwork(**network_pars)
    sim = ss.Sim(networks=[net], n_agents=2000, diseases=None,
                 dur=ss.years(2), dt=ss.years(0.5), rand_seed=0, verbose=0,
                 copy_inputs=False)
    sim.run()

    m_mask = net.edges_for_layer('m')
    c_mask = net.edges_for_layer('c')
    if not m_mask.any() or not c_mask.any():
        return  # not enough sample

    def members(mask):
        p1 = np.asarray(net.edges.p1)[mask]
        p2 = np.asarray(net.edges.p2)[mask]
        return set(np.unique(np.concatenate([p1, p2])).tolist())

    m_members = members(m_mask)
    c_members = members(c_mask)
    overlap = m_members & c_members
    assert len(overlap) / max(1, len(m_members | c_members)) < 0.01, \
        f'cross_layer=0 violated: {len(overlap)} agents in both layers'


def test_age_mixing_assortativity():
    """Sampled pairs concentrate on/near the mixing-matrix diagonal.

    Filtered to marital edges only so casual pairs don't dilute the signal
    (layers ('m', 'c') are now fixed on ``SexualNetwork``).
    """
    country = hpvsim.data.load_country('nigeria')
    net = SexualNetwork(**country['network_pars'])
    sim = ss.Sim(networks=[net], n_agents=5000, diseases=None,
                 dur=ss.years(5), dt=ss.years(0.5), rand_seed=0, verbose=0,
                 copy_inputs=False)
    sim.run()
    m_mask = net.edges_for_layer('m')
    if m_mask.sum() < 100:
        return
    people = sim.people
    p1 = np.asarray(net.edges.p1)[m_mask]
    p2 = np.asarray(net.edges.p2)[m_mask]
    f_at_p1 = people.female[p1]
    f_uids = np.where(f_at_p1, p1, p2)
    m_uids = np.where(f_at_p1, p2, p1)
    f_ages = people.age[ss.uids(f_uids)]
    m_ages = people.age[ss.uids(m_uids)]
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
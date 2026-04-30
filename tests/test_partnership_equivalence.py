"""M01 acceptance gate: partnership patterns vs. v2.x.

Per-layer comparison of:
  - age-mixing matrix (5y bins, 0-80, female x male)
  - concurrency distribution (n_concurrent_partners histogram)
  - partnership duration distribution

Against v2 baselines stored in tests/regression_baselines/partnership_v2.json
(gitignored; generation procedure documented in tests/regression/README.md).

v2 has 2 layers (m, c); the test parametrizes over both.

Pass criteria:
  - mixing-matrix bin-wise relative diff < 15% (per non-zero bin)
  - concurrency: KS-test p > 0.01
  - duration: KS-test p > 0.01
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import hpvsim as hpv
from hpvsim.network import SexualNetwork


_BASELINE_PARTNERSHIP = Path(__file__).resolve().parent / 'regression_baselines' / 'partnership_v2.json'

PARS = dict(
    location='nigeria',
    genotype='hpv16',
    n_agents=10_000,
    start=1990,
    stop=2015,   # 1990-2010 burnin + 2010-2015 observation per spec
    dt=0.5,
    rand_seed=0,
    verbose=0,
)


def _capture_partnership_stats(sim):
    """For each SexualNetwork in sim, return a dict with mixing_matrix,
    concurrency_hist, duration_samples -- same shape as the v2 baseline."""
    people = sim.people
    out = {}
    for net in sim.networks():
        if not isinstance(net, SexualNetwork):
            continue
        if len(net) == 0:
            out[net.layer] = dict(
                mixing_matrix=np.zeros((16, 16)).tolist(),
                concurrency_hist=[len(people), 0, 0, 0, 0],
                duration_samples=[],
            )
            continue
        # Mixing matrix
        f_at_p1 = np.asarray(people.female[net.edges.p1])
        f_uids = np.where(f_at_p1, np.asarray(net.edges.p1),
                          np.asarray(net.edges.p2))
        m_uids = np.where(f_at_p1, np.asarray(net.edges.p2),
                          np.asarray(net.edges.p1))
        bins = np.arange(0, 81, 5)
        f_bins = np.digitize(np.asarray(people.age[f_uids]), bins) - 1
        m_bins = np.digitize(np.asarray(people.age[m_uids]), bins) - 1
        n_bins = len(bins) - 1
        mat = np.zeros((n_bins, n_bins))
        for fb, mb in zip(f_bins, m_bins):
            if 0 <= fb < n_bins and 0 <= mb < n_bins:
                mat[fb, mb] += 1
        if mat.sum() > 0:
            mat = mat / mat.sum()

        # Concurrency: count partnerships per UID in this network
        n_uids = len(people.alive.raw)
        n_per_agent = np.zeros(n_uids, dtype=int)
        np.add.at(n_per_agent, np.asarray(net.edges.p1), 1)
        np.add.at(n_per_agent, np.asarray(net.edges.p2), 1)
        # Restrict to alive agents only
        alive_uids = people.alive.uids
        n_per_alive = n_per_agent[alive_uids]
        max_k = max(5, int(n_per_alive.max()) + 1) if len(n_per_alive) else 5
        conc_hist = np.bincount(n_per_alive, minlength=max_k).tolist()

        # Remaining partnership duration in YEARS (edges.dur is decremented
        # each step, so it stores remaining time in timesteps; multiply by
        # dt to match the v2 baseline-generation script's units).
        if hasattr(net.edges, 'dur'):
            dt_years = float(sim.t.dt)
            durations = (np.asarray(net.edges.dur) * dt_years).tolist()
        else:
            durations = []

        out[net.layer] = dict(
            mixing_matrix=mat.tolist(),
            concurrency_hist=conc_hist,
            duration_samples=durations,
        )
    return out


@pytest.fixture(scope='module')
def v3_stats():
    sim = hpv.Sim(**PARS)
    sim.run()
    return _capture_partnership_stats(sim)


@pytest.fixture(scope='module')
def v2_stats():
    if not _BASELINE_PARTNERSHIP.exists():
        pytest.skip(
            'partnership_v2.json baseline not present (gitignored; '
            'see tests/regression/README.md for generation procedure)'
        )
    with open(_BASELINE_PARTNERSHIP) as f:
        return json.load(f)


@pytest.mark.parametrize('layer', ['m', 'c'])
def test_age_mixing_matrix(v3_stats, v2_stats, layer):
    """Per-layer age-mixing matrix matches v2 within 15% bin-wise."""
    v3 = np.array(v3_stats[layer]['mixing_matrix'])
    v2 = np.array(v2_stats[layer]['mixing_matrix'])
    assert v3.shape == v2.shape
    nonzero = v2 > 0.001  # only check bins with non-trivial v2 mass
    if nonzero.sum() == 0:
        return
    rel_diff = np.abs(v3[nonzero] - v2[nonzero]) / v2[nonzero]
    max_diff = rel_diff.max()
    assert max_diff < 0.15, \
        f'layer {layer} mixing matrix max bin-wise rel diff {max_diff:.3f} >= 0.15'


@pytest.mark.parametrize('layer', ['m', 'c'])
def test_concurrency_distribution(v3_stats, v2_stats, layer):
    """Per-layer concurrency distribution matches v2 (KS p > 0.01)."""
    v3_hist = np.array(v3_stats[layer]['concurrency_hist'])
    v2_hist = np.array(v2_stats[layer]['concurrency_hist'])
    v3_samples = np.repeat(np.arange(len(v3_hist)), v3_hist)
    v2_samples = np.repeat(np.arange(len(v2_hist)), v2_hist)
    if len(v3_samples) == 0 or len(v2_samples) == 0:
        return
    ks_stat, p_value = stats.ks_2samp(v3_samples, v2_samples)
    assert p_value > 0.01, \
        f'layer {layer} concurrency KS p={p_value:.4f} <= 0.01'


@pytest.mark.parametrize('layer', ['m', 'c'])
def test_duration_distribution(v3_stats, v2_stats, layer):
    """Per-layer partnership-duration distribution matches v2 (KS p > 0.01)."""
    v3_dur = np.array(v3_stats[layer]['duration_samples'])
    v2_dur = np.array(v2_stats[layer]['duration_samples'])
    if len(v3_dur) < 30 or len(v2_dur) < 30:
        pytest.skip(f'layer {layer}: too few duration samples for KS-test')
    ks_stat, p_value = stats.ks_2samp(v3_dur, v2_dur)
    assert p_value > 0.01, \
        f'layer {layer} duration KS p={p_value:.4f} <= 0.01'
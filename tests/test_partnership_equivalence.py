"""M02 acceptance gate: partnership patterns vs. v2.x.

Per-layer comparison of:
  - active pair count (m: 10%, c: 15%)
  - mean partnership duration in years (m: 5%, c: 25%)
  - max concurrency (within +/-1)
  - age-mixing matrix shape via cosine similarity (>= 0.90)

Against v2 baselines stored in tests/regression_baselines/partnership_v2.json
(gitignored; generation procedure documented in tests/regression/README.md).

v2 has 2 layers (m, c); the test parametrizes over both.

These thresholds were tightened from the M01 release (50% / 50% / +/-2 / 0.85)
after the M02 network fixes brought all metrics within ~5% of v2 (8-seed mean):
  - shared per-agent debut across SexualNetwork layers
  - all-layer dissolve-then-create step ordering
  - AgeMigration in default Sim demographics
  - single multi-layer SexualNetwork (refactor — exposes one ss.Network in
    sim.networks() with edges tagged by layer_id)

The casual layer's mean-duration gate is set wider because casual durations
are sampled from lognormal(mean=1y, std=2y) — heavy right tail — and the
active-pair sample at end-of-sim is small (~600 pairs), so the survivor-
biased mean has 8-21% per-seed variance even when the underlying
distribution matches. The 8-seed mean lands within ~5% of v2.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import starsim as ss

import hpvsim as hpv
from hpvsim.network import SexualNetwork


_BASELINE_PARTNERSHIP = Path(__file__).resolve().parent / 'regression_baselines' / 'partnership_v2.json'

PARS = dict(
    location='nigeria',
    genotype='hpv16',
    n_agents=10_000,
    start=1990,
    stop=2015,   # 1990-2010 burnin + 2010-2015 observation per spec
    dt=0.25,     # Match v2 default and v3 standard
    rand_seed=0,
    verbose=0,
)


def _capture_partnership_stats(sim):
    """Capture per-layer partnership stats from the (single) hpv.SexualNetwork
    in sim. Returns a dict ``{layer: {mixing_matrix, concurrency_hist,
    duration_samples}}`` with the same shape as the v2 baseline.
    """
    people = sim.people
    out = {}
    nets = [n for n in sim.networks() if isinstance(n, SexualNetwork)]
    if not nets:
        return out
    net = nets[0]  # single multi-layer network
    dt_years = float(sim.t.dt)
    ti_now = float(sim.t.ti)
    bins = np.arange(0, 81, 5)
    n_bins = len(bins) - 1

    for lkey in net.layers:
        layer_mask = net.edges_for_layer(lkey)
        if not layer_mask.any():
            out[lkey] = dict(
                mixing_matrix=np.zeros((n_bins, n_bins)).tolist(),
                concurrency_hist=[len(people), 0, 0, 0, 0],
                duration_samples=[],
            )
            continue

        p1 = np.asarray(net.edges.p1)[layer_mask]
        p2 = np.asarray(net.edges.p2)[layer_mask]
        start_ti = np.asarray(net.edges.start_ti)[layer_mask]
        remaining = np.asarray(net.edges.dur)[layer_mask]

        # Mixing matrix at FORMATION TIME (matches v2's age_f/age_m in
        # to_df, stored at pair creation). edges.start_ti is the timestep
        # when the pair formed; subtract from current age to recover age-
        # at-formation.
        f_at_p1 = np.asarray(people.female[ss.uids(p1)])
        f_uids = np.where(f_at_p1, p1, p2)
        m_uids = np.where(f_at_p1, p2, p1)
        years_since = (ti_now - start_ti) * dt_years
        f_age_form = np.asarray(people.age[ss.uids(f_uids)]) - years_since
        m_age_form = np.asarray(people.age[ss.uids(m_uids)]) - years_since
        f_bins = np.digitize(f_age_form, bins) - 1
        m_bins = np.digitize(m_age_form, bins) - 1
        mat = np.zeros((n_bins, n_bins))
        for fb, mb in zip(f_bins, m_bins):
            if 0 <= fb < n_bins and 0 <= mb < n_bins:
                mat[fb, mb] += 1
        if mat.sum() > 0:
            mat = mat / mat.sum()

        # Concurrency: count partnerships per UID in this layer.
        n_uids = len(people.alive.raw)
        n_per_agent = np.zeros(n_uids, dtype=int)
        np.add.at(n_per_agent, p1, 1)
        np.add.at(n_per_agent, p2, 1)
        alive_uids = people.alive.uids
        n_per_alive = n_per_agent[alive_uids]
        max_k = max(5, int(n_per_alive.max()) + 1) if len(n_per_alive) else 5
        conc_hist = np.bincount(n_per_alive, minlength=max_k).tolist()

        # Original partnership duration in YEARS at formation (matches v2's
        # stored ``dur`` in to_df). edges.dur is REMAINING timesteps
        # (decremented each step); reconstruct original by adding elapsed.
        original_ts = remaining + (ti_now - start_ti)
        durations = (original_ts * dt_years).tolist()

        out[lkey] = dict(
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


def _rel(v3, v2, scale=None):
    """Relative difference, with safe handling of v2≈0."""
    scale = scale if scale is not None else max(abs(v2), 1e-9)
    return abs(v3 - v2) / scale


def _hist_to_samples(hist):
    return np.repeat(np.arange(len(hist)), hist)


def _pair_count(hist):
    """Total active pairs from a concurrency histogram (sum k * count_k / 2)."""
    h = np.asarray(hist)
    return int((h * np.arange(len(h))).sum() // 2)


def _cosine_similarity(a, b):
    a, b = a.flatten(), b.flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# Per-layer tolerances. Marital is stable (long-lived pairs, large active set,
# tight per-seed variance). Casual is high-variance (lognormal heavy-tail
# durations, small active set, ~8-21% per-seed variance on the mean dur).
# These reflect 8-seed worst-case envelopes with ~1.5x headroom.
_PAIR_COUNT_TOL = {'m': 0.10, 'c': 0.15}
_MEAN_DUR_TOL = {'m': 0.05, 'c': 0.25}
_CONCURRENCY_MAX_DELTA = 1
_MIXING_COSINE_FLOOR = 0.90


@pytest.mark.parametrize('layer', ['m', 'c'])
def test_active_pair_count(v3_stats, v2_stats, layer):
    """Active pair count matches v2 within layer-specific tolerance."""
    v3_n = _pair_count(v3_stats[layer]['concurrency_hist'])
    v2_n = _pair_count(v2_stats[layer]['concurrency_hist'])
    if v2_n < 10:
        pytest.skip(f'layer {layer}: v2 has only {v2_n} active pairs')
    rel = _rel(v3_n, v2_n)
    tol = _PAIR_COUNT_TOL[layer]
    assert rel < tol, \
        f'layer {layer} active pair count v3={v3_n} vs v2={v2_n} (rel diff {rel:.2%} >= {tol:.0%})'


@pytest.mark.parametrize('layer', ['m', 'c'])
def test_concurrency_max(v3_stats, v2_stats, layer):
    """Max simultaneous partnerships per agent in this layer matches v2 within ±1.

    The max is determined purely by partner-count distributions and concurrency
    rules, so it's robust to alive-count differences. ±1 absorbs single-run
    Monte Carlo tail noise.
    """
    v3_hist = np.asarray(v3_stats[layer]['concurrency_hist'])
    v2_hist = np.asarray(v2_stats[layer]['concurrency_hist'])
    v3_max = int(np.argmax(v3_hist[::-1] > 0)) if v3_hist.sum() > 0 else 0
    v3_max = len(v3_hist) - 1 - v3_max if v3_hist.sum() > 0 else 0
    v2_max = int(np.argmax(v2_hist[::-1] > 0)) if v2_hist.sum() > 0 else 0
    v2_max = len(v2_hist) - 1 - v2_max if v2_hist.sum() > 0 else 0
    v2_n = _pair_count(v2_stats[layer]['concurrency_hist'])
    if v2_n < 100:
        pytest.skip(f'layer {layer}: v2 has only {v2_n} active pairs (max-concurrency tail unstable at end-of-sim)')
    assert abs(v3_max - v2_max) <= _CONCURRENCY_MAX_DELTA, \
        f'layer {layer} max concurrency v3={v3_max} vs v2={v2_max} (delta > {_CONCURRENCY_MAX_DELTA})'


@pytest.mark.parametrize('layer', ['m', 'c'])
def test_mean_duration(v3_stats, v2_stats, layer):
    """Mean partnership duration (years) matches v2 within layer-specific tolerance."""
    v3_dur = np.asarray(v3_stats[layer]['duration_samples'])
    v2_dur = np.asarray(v2_stats[layer]['duration_samples'])
    if len(v3_dur) < 30 or len(v2_dur) < 30:
        pytest.skip(f'layer {layer}: too few duration samples ({len(v3_dur)} vs {len(v2_dur)})')
    v3_mean, v2_mean = float(v3_dur.mean()), float(v2_dur.mean())
    rel = _rel(v3_mean, v2_mean)
    tol = _MEAN_DUR_TOL[layer]
    assert rel < tol, \
        f'layer {layer} mean dur v3={v3_mean:.2f}y vs v2={v2_mean:.2f}y (rel diff {rel:.2%} >= {tol:.0%})'


@pytest.mark.parametrize('layer', ['m', 'c'])
def test_mixing_matrix_shape(v3_stats, v2_stats, layer):
    """Age-mixing matrix shape (cosine similarity) matches v2 above 0.90.

    Cosine similarity tolerates Monte Carlo noise per-bin while still
    catching gross structural mismatch (e.g., wrong age preferences).
    Skips when v2 has too few active pairs to estimate a stable shape.
    """
    v3 = np.array(v3_stats[layer]['mixing_matrix'])
    v2 = np.array(v2_stats[layer]['mixing_matrix'])
    assert v3.shape == v2.shape
    v2_n = _pair_count(v2_stats[layer]['concurrency_hist'])
    if v2_n < 100:
        pytest.skip(f'layer {layer}: v2 has only {v2_n} active pairs (too few for stable mixing-matrix shape)')
    sim = _cosine_similarity(v3, v2)
    assert sim > _MIXING_COSINE_FLOOR, \
        f'layer {layer} mixing matrix cosine similarity {sim:.3f} <= {_MIXING_COSINE_FLOOR}'
"""Diagnostic: run the v3 anchor with LegacyRngSexualNetwork (v2 RNG plumbing)
and compare partnership stats against the v2 baseline.

If the legacy-RNG sim aligns more closely with v2 than the production
SexualNetwork does, the divergence between v2 and v3 is RNG-plumbing-driven
(not algorithmic). If the alignment is unchanged or worse, an algorithmic
difference is responsible.

Usage: python tests/compare_legacy_rng.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import starsim as ss

# Make the legacy-RNG fixture importable.
sys.path.insert(0, str(Path(__file__).parent))

from _legacy_rng_network import LegacyRngSexualNetwork  # noqa: E402

import hpvsim as hpv  # noqa: E402
from hpvsim.network import SexualNetwork  # noqa: E402


def build_sim(network_cls, location='nigeria', n_agents=10_000,
              start=1990, stop=2015, dt=0.5, rand_seed=0):
    """Build a Sim with the given SexualNetwork class for both layers."""
    np.random.seed(rand_seed)  # v2-style global seeding for legacy network
    country = hpv.data.load_country(location)
    if network_cls is LegacyRngSexualNetwork:
        # Legacy network reads pars directly from v2; ignore country['network_pars']
        nets = [LegacyRngSexualNetwork(layer=k, location=location)
                for k in ('m', 'c')]
    else:
        nets = [SexualNetwork(layer=k, pars=country['network_pars'][k])
                for k in ('m', 'c')]
    return hpv.Sim(
        location=location, n_agents=n_agents, start=start, stop=stop,
        dt=dt, rand_seed=rand_seed, networks=nets, verbose=0,
    )


def capture_stats(sim):
    """Capture per-layer partnership stats matching test_partnership_equivalence."""
    people = sim.people
    dt_years = float(sim.t.dt)
    ti_now = float(sim.t.ti)
    bins = np.arange(0, 81, 5)
    n_bins = len(bins) - 1

    out = {}
    for net in sim.networks():
        if not isinstance(net, SexualNetwork):
            continue
        if len(net) == 0:
            out[net.layer] = dict(n_pairs=0, mean_age_f=0.0, mean_age_m=0.0,
                                  mean_dur=0.0, max_concurrency=0)
            continue

        f_at_p1 = np.asarray(people.female[net.edges.p1])
        f_uids = np.where(f_at_p1, np.asarray(net.edges.p1),
                          np.asarray(net.edges.p2))
        m_uids = np.where(f_at_p1, np.asarray(net.edges.p2),
                          np.asarray(net.edges.p1))
        years_since = (ti_now - np.asarray(net.edges.start_ti)) * dt_years
        f_age_form = np.asarray(people.age[f_uids]) - years_since
        m_age_form = np.asarray(people.age[m_uids]) - years_since

        n_per_agent = np.zeros(len(people.alive.raw), dtype=int)
        np.add.at(n_per_agent, np.asarray(net.edges.p1), 1)
        np.add.at(n_per_agent, np.asarray(net.edges.p2), 1)
        n_per_alive = n_per_agent[people.alive.uids]

        original_ts = np.asarray(net.edges.dur) + (ti_now - np.asarray(net.edges.start_ti))
        out[net.layer] = dict(
            n_pairs=len(net),
            mean_age_f=float(f_age_form.mean()),
            mean_age_m=float(m_age_form.mean()),
            mean_dur=float((original_ts * dt_years).mean()),
            max_concurrency=int(n_per_alive.max()) if len(n_per_alive) else 0,
            alive_count=int(len(people.alive.uids)),
        )
    return out


def main():
    base = Path(__file__).parent / 'regression_baselines' / 'partnership_v2.json'
    if not base.exists():
        print('partnership_v2.json baseline missing; generate it first.')
        return 1
    with open(base) as f:
        v2 = json.load(f)

    print('Running v3 with PRODUCTION SexualNetwork (Starsim Dist plumbing)...')
    sim_prod = build_sim(SexualNetwork)
    sim_prod.run()
    prod = capture_stats(sim_prod)

    print('Running v3 with LegacyRngSexualNetwork (v2 RNG plumbing)...')
    sim_legacy = build_sim(LegacyRngSexualNetwork)
    sim_legacy.run()
    legacy = capture_stats(sim_legacy)

    print('\n=== Comparison ===\n')
    fmt = '{:>20}  {:>12}  {:>12}  {:>12}'
    for layer in ('m', 'c'):
        print(f'--- layer {layer} ---')
        print(fmt.format('metric', 'v2 baseline', 'v3 prod', 'v3 legacy-RNG'))
        # v2 stats
        v2_layer = v2[layer]
        v2_n = sum(np.asarray(v2_layer['concurrency_hist']) *
                   np.arange(len(v2_layer['concurrency_hist']))) // 2
        v2_dur = np.asarray(v2_layer['duration_samples'])
        v2_dur_mean = float(v2_dur.mean()) if len(v2_dur) else 0.0
        # v2 has no separate mean_age_f/mean_age_m; mixing matrix gives shape only
        print(fmt.format('n_pairs', int(v2_n),
                         prod[layer]['n_pairs'], legacy[layer]['n_pairs']))
        print(fmt.format('mean_dur (yrs)', f'{v2_dur_mean:.2f}',
                         f'{prod[layer]["mean_dur"]:.2f}',
                         f'{legacy[layer]["mean_dur"]:.2f}'))
        v2_max_c = max((i for i, c in enumerate(v2_layer['concurrency_hist']) if c > 0), default=0)
        print(fmt.format('max_concurrency', v2_max_c,
                         prod[layer]['max_concurrency'],
                         legacy[layer]['max_concurrency']))
        print()
    print(f'alive count: v3 prod={prod["m"]["alive_count"]}, '
          f'v3 legacy={legacy["m"]["alive_count"]}, '
          f'v2 from anchor (per concurrency_hist sum, m layer)='
          f'{sum(v2["m"]["concurrency_hist"])}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
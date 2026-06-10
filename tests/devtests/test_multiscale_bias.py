"""
Regression test for the multiscale cancer-incidence deflation bug.

Background
----------
`People.set_severity` (the multiscale CIN->cancer resolver) is called only on
women who have ALREADY reached CIN, so the base agent is unconditionally CIN.
The buggy code re-rolled the precin->CIN gate for each of the `ms_agent_ratio-1`
spawned extras and zeroed their cancer probability if they "failed" -- while
weighting every extra equally with the base. That deflates expected cancer
weight per CIN woman by `[1 + (n_extra-1)*P(cin)] / n_extra`, so total cancer
incidence shrinks as `ms_agent_ratio` grows instead of staying flat.

This test pins the invariant the multiscale technique is supposed to guarantee:
**total cancers (in people-space) must be (statistically) invariant to
`ms_agent_ratio`.**
"""

import numpy as np
import sciris as sc
import hpvsim as hpv


SEEDS = list(range(6))
RATIOS = [1, 10]
TOL = 0.08  # caught regressions: full bug ~-72%, age_risk omission ~-11%; true residual ~1%
# Equal total_pop AND equal n_agents across arms => identical people-space, so a
# raw comparison of scale-weighted cancer totals isolates the multiscale bias
# (not a population/discretization difference).
BASE_PARS = dict(
    total_pop      = 20e3,
    n_agents       = 20e3,
    start          = 1975,
    n_years        = 50,
    genotypes      = [16, 18],
    verbose        = 0,
)


def _total_cancers_people(ms_ratio, seed):
    """Scale-weighted total cancers over the whole run, in people-space."""
    sim = hpv.Sim(BASE_PARS, ms_agent_ratio=ms_ratio, rand_seed=seed)
    sim.run()
    return float(np.sum(np.array(sim.results['cancers'].values)))


def _mean_over_seeds(ms_ratio):
    return np.mean([_total_cancers_people(ms_ratio, sd) for sd in SEEDS])


def test_total_cancers_flat_across_ms_agent_ratio():
    """Total people-space cancers must not depend on ms_agent_ratio."""
    means = {r: _mean_over_seeds(r) for r in RATIOS}
    base = means[1]
    for r in RATIOS:
        rel = abs(means[r] - base) / base
        assert rel < TOL, (
            f"ms_agent_ratio={r} total cancers {means[r]:.0f} deviates "
            f"{rel*100:.1f}% from ms_agent_ratio=1 baseline {base:.0f} "
            f"(>{TOL*100:.0f}%): multiscale is biasing cancer incidence. means={means}"
        )


if __name__ == '__main__':
    T = sc.timer()
    means = {r: _mean_over_seeds(r) for r in RATIOS}
    base = means[1]
    print(f'{"ms_ratio":>10} {"cancers(people)":>16} {"rel.vs.ratio1":>14}')
    for r in RATIOS:
        print(f'{r:>10} {means[r]:>16.0f} {(means[r]/base-1)*100:>13.1f}%')
    test_total_cancers_flat_across_ms_agent_ratio()
    print('PASSED')
    T.toc('Done')
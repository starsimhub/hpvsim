"""Seed-averaged multiscale-equivalence gates for the analyzers.

SLATED FOR DELETION IN v3.3 (test cleanup). These gates cost ~15-20 min for
the whole folder, which is why they sit outside ``tests/``. The mechanisms
they cover are also checked cheaply in the unit suite, so the plan is to keep
the unit coverage and drop these rather than keep paying for them on every
release. If a gate here is the ONLY thing covering a behaviour you care
about, move that coverage into ``tests/`` before 3.3.

Both quantities below are means over rare events (age at cancer, DALYs from
young-onset cancer), so a single seed is far too noisy to assert on — they need
4-6 seeds per ratio, i.e. ~20 full sims. The cheap per-run invariants those
analyzers must satisfy are covered in ``tests/test_analyzers.py``.
"""
import numpy as np
import pytest
import hpvsim as hpv


@pytest.mark.slow
def test_age_causal_infection_grow_unbiased():
    """Grow's fine agents don't bias mean age-at-cancer vs ratio==1 base agents.

    A SINGLE-seed comparison is too fragile here: the whole natural-history age
    distribution reshuffles across RNG streams (e.g. between starsim versions),
    so at n_agents=3000 one seed's ratio=1-vs-ratio=3 mean-age gap swings up to
    ~2.5yr on noise alone. We therefore compare the SEED-AVERAGED mean age,
    matching the multi-seed methodology of test_multiscale_grow_gates, which
    validates the rigorous ratio-independence properties (incidence-flat,
    variance-shrinks, ratio==1 bit-identical).
    """
    def run(ratio, seed):
        aci = hpv.age_causal_infection(start=2000)
        sim = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1990, stop=2040,
                      n_agents=3000, rand_seed=seed, ms_agent_ratio=ratio, analyzers=[aci])
        sim.run()
        a = sim.analyzers['age_causal_infection']
        return np.nanmean(a.age_cancer), len(a.age_cancer)
    seeds = range(6)
    res1 = [run(1, s) for s in seeds]
    res3 = [run(3, s) for s in seeds]
    ages1 = [m for m, _ in res1]
    ages3 = [m for m, _ in res3]
    # grow yields ~ratio x more resolved samples every seed
    assert all(n3 > n1 for (_, n1), (_, n3) in zip(res1, res3))
    # seed-averaged mean age at cancer agrees within 2 years
    assert abs(np.mean(ages3) - np.mean(ages1)) < 2.0


@pytest.mark.slow
def test_dalys_grow_overlaps_single_scale():
    """Mean total DALYs converge across ms_agent_ratio (multiscale-equivalence).

    DALYs are dominated by rare young-onset cancers, so single-seed totals are
    high-variance; the unbiased quantity is the mean over seeds. On grow, extra
    cancers are real fine agents at scale 1/ratio, so the scale-weighted DALY
    total matches the ratio==1 base-agent path in expectation. (This only holds
    because both paths count REALIZED cancers — the step path gates on
    cancerous & alive, not a bare ti_cancerous==ti time-match, which would
    overcount agents who die before onset.)
    """
    seeds = [1, 2, 3, 4]

    def mean_total(ratio):
        tots = []
        for s in seeds:
            d = hpv.dalys(start=2000)
            sim = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1990, stop=2040,
                          n_agents=5000, rand_seed=s, ms_agent_ratio=ratio, analyzers=[d])
            sim.run(verbose=0)
            tots.append(sim.analyzers['dalys'].dalys.sum())
        return float(np.mean(tots))

    m1 = mean_total(1)
    m3 = mean_total(3)
    assert abs(m3 - m1) / m1 < 0.15

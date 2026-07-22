import numpy as np
import pytest
import sciris as sc
import starsim as ss
import hpvsim as hpv
from hpvsim.hpv import HPV

PYRAMID_EDGES = np.array([0., 15., 50., 100.])


def _hpv_mods(sim):
    return [d for d in sim.diseases.values() if isinstance(d, HPV)]


@pytest.fixture(scope='module')
def base_sim():
    """One single-scale (ratio=1), 2-genotype sim with the full analyzer stack,
    run once and shared (read-only) across the single-scale analyzer tests.

    Carries age_causal_infection / dalys / age_pyramid / snapshot so a single
    run backs the ordering, DALY-decomposition, pyramid-histogram, per-genotype
    and no-fine-agent assertions. 3000 agents to 2040 keeps robust cancer signal.
    Tests that MUTATE people (snapshot deep-copy) use their own sim instead.
    """
    sim = hpv.Sim(genotypes=['hpv16', 'hpv18'], location='nigeria',
                  start=1990, stop=2040, n_agents=3000, rand_seed=1,
                  analyzers=[hpv.age_causal_infection(start=2000),
                             hpv.dalys(start=2000, life_expectancy=84),
                             hpv.age_pyramid(edges=PYRAMID_EDGES, timepoints=[2010]),
                             hpv.snapshot(timepoints=[2010])])
    sim.run()
    return sim


@pytest.fixture(scope='module')
def grow_sim():
    """One multiscale (ratio=5) sim, run once and shared across the grow tests.

    ratio=5 materializes real fine cancer agents; 5000 agents over 1970-2030
    gives enough realized cancers for the analyzer-vs-engine consistency check.
    """
    aci = hpv.age_causal_infection()
    sim = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1970, stop=2030,
                  n_agents=5000, rand_seed=1, ms_agent_ratio=5, analyzers=[aci])
    sim.run(verbose=0)
    return sim


def test_grow_spawns_fine_cancer_agents_at_ratio_gt1(grow_sim):
    """At ratio>1 the grow engine materializes real fine cancer agents.

    (Replaces the M09 ledger tuple test: the grow engine has no _cancer_events
    ledger — extra cancers are real agents in sim.people with fine=True and the
    shrunk multiscale scale 1/ratio.)
    """
    ratio = int(grow_sim.diseases['hpv16'].pars.ms_agent_ratio)
    ppl = grow_sim.people
    fine = np.asarray(ppl.fine.raw, dtype=bool)
    assert fine.sum() > 0, 'expected fine agents grown at ratio>1'
    # Fine agents carry the shrunk multiscale scale.
    assert np.allclose(np.asarray(ppl.scale.raw)[fine], 1.0 / ratio)
    # At least one fine agent reached cancer in some genotype.
    any_cancer = np.zeros(len(fine), dtype=bool)
    for m in _hpv_mods(grow_sim):
        any_cancer |= np.asarray(m.cancerous.raw, dtype=bool)
    assert (fine & any_cancer).any(), 'expected fine cancer agents'


def test_no_fine_agents_at_ratio_one(base_sim):
    """At ms_agent_ratio==1 no fine agents are grown (single-scale fast path)."""
    assert not np.asarray(base_sim.people.fine.raw, dtype=bool).any()


def test_grow_analyzer_matches_engine_cancer_total(grow_sim):
    """Consistency gate: age_causal scaled event count == engine cum_cancers.

    On grow, extra cancers are real fine agents, so the analyzer (live-agent
    path, weighted by people.scale) reproduces the engine's own realized cancer
    accounting to within boundary/same-tick-death noise (~1-2% observed). This
    is the acceptance test the abandoned ledger path used to paper over.
    """
    aci = grow_sim.analyzers['age_causal_infection']
    engine_total = float(hpv.results_by_genotype(grow_sim, key='cum_cancers').iloc[-1].sum())
    analyzer_total = float(aci.weights.sum())
    assert engine_total > 0
    assert abs(analyzer_total - engine_total) / engine_total < 0.05
    # Raw fine-agent samples greatly exceed the scaled total at ratio>1.
    assert len(aci.age_cancer) > analyzer_total


def test_resolve_date_ticks_nearest():
    from hpvsim.analyzers import _resolve_date_ticks
    sim = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=2018, stop=2022, n_agents=300)
    sim.init()
    out = _resolve_date_ticks(sim, [2020, ss.date(2021)])
    # Keys are ss.date; values are tick indices whose year matches the request.
    keys = list(out.keys())
    assert all(isinstance(k, ss.date) for k in keys)
    tol = float(sim.t.dt) / 2 + 1e-9
    assert abs(sim.timevec[out[keys[0]]].years - 2020) <= tol
    assert abs(sim.timevec[out[keys[1]]].years - 2021) <= tol


def test_snapshot_records_and_get_coerces():
    # Own sim (this test mutates sim.people, so it cannot share a fixture).
    snap = hpv.snapshot(timepoints=[2000, 2010])
    sim = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1995, stop=2012,
                  n_agents=250, analyzers=[snap])
    sim.run()
    s = sim.analyzers['snapshot']
    # Two snapshots, keyed by ss.date.
    assert len(s.snapshots) == 2
    assert all(isinstance(k, ss.date) for k in s.snapshots.keys())
    # get() coerces int / str / ss.date to the same entry.
    p_int = s.get(2010)
    p_str = s.get('2010')
    p_date = s.get(ss.date(2010))
    assert p_int is p_str is p_date
    # Deep copy: mutating the live sim's people after the run leaves the snapshot intact.
    assert len(p_int.age) > 0
    snap_ages_before = np.asarray(p_int.age.values).copy()
    sim.people.age[:] = -999.0
    assert np.allclose(np.asarray(p_int.age.values), snap_ages_before)
    assert not np.allclose(np.asarray(p_int.age.values), -999.0)
    # default get() returns the first snapshot
    assert s.get() is s.snapshots[list(s.snapshots.keys())[0]]


def test_age_pyramid_bins_sum_to_alive(base_sim):
    a = base_sim.analyzers['age_pyramid']
    assert len(a.age_pyramids) == 1
    date = list(a.age_pyramids.keys())[0]
    assert isinstance(date, ss.date)
    arr = a.age_pyramids[date]                  # (nbins, 2): col0=male, col1=female
    assert arr.shape == (len(a.bins), 2)

    # Ground truth: recompute the scale-weighted age histograms from the
    # snapshot of People taken at the SAME tick. This independently verifies
    # the male/female column assignment, the alive mask, and scale-weighting.
    ppl = base_sim.analyzers['snapshot'].get(2010)
    alive = ppl.alive.values
    ages = ppl.age.values
    female = ppl.female.values
    scale = getattr(ppl, 'scale', None)
    w = scale.values if scale is not None else None

    def hist(mask):
        ww = w[mask] if w is not None else None
        return np.histogram(ages[mask], bins=PYRAMID_EDGES, weights=ww)[0]

    exp_male = hist(alive & ~female)
    exp_female = hist(alive & female)
    assert np.allclose(arr[:, 0], exp_male), 'male column mismatch (possible male/female swap)'
    assert np.allclose(arr[:, 1], exp_female), 'female column mismatch (possible male/female swap)'

    # Named invariant: total binned count == scale-weighted alive within [0, 100).
    in_range = alive & (ages >= PYRAMID_EDGES[0]) & (ages < PYRAMID_EDGES[-1])
    expected_total = float(w[in_range].sum()) if w is not None else float(in_range.sum())
    assert abs(arr.sum() - expected_total) < 1e-6

    # tidy-frame sanity
    df = a.to_dataframe()
    assert set(df['sex']) == {'male', 'female'}
    assert (df['count'] >= 0).all()
    assert df['count'].sum() > 0


def test_age_causal_infection_single_scale(base_sim):
    a = base_sim.analyzers['age_causal_infection']
    assert len(a.age_cancer) > 0
    # Ordering: causal infection precedes CIN precedes cancer, on average.
    assert np.nanmean(a.age_causal) < np.nanmean(a.age_cin) < np.nanmean(a.age_cancer)
    # Dwell times non-negative and consistent: total ~= precin + cin.
    assert np.all(a.dwelltime['total'] >= -1e-9)
    assert np.allclose(a.dwelltime['total'],
                       a.dwelltime['precin'] + a.dwelltime['cin'], atol=1e-6)
    # At ratio==1 every event has weight 1 (people.scale is all 1).
    assert np.allclose(a.weights, 1.0)


@pytest.mark.slow
def test_age_causal_infection_grow_unbiased():
    """Grow's fine agents don't bias mean age-at-cancer vs ratio==1 base agents.

    A SINGLE-seed comparison is too fragile here: the whole natural-history age
    distribution reshuffles across RNG streams (e.g. between starsim versions),
    so at n_agents=3000 one seed's ratio=1-vs-ratio=3 mean-age gap swings up to
    ~2.5yr on noise alone. We therefore compare the SEED-AVERAGED mean age,
    matching the multi-seed methodology of the slow acceptance gate
    (test_multiscale_grow_acceptance), which validates the rigorous
    ratio-independence properties (incidence-flat, variance-shrinks, ratio==1
    bit-identical). This is the fast unit-level guard on the mean only.
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


def test_dalys_basic_and_av_disutility(base_sim):
    a = base_sim.analyzers['dalys']
    # av_disutility matches the v2 GBD2017 constant exactly.
    expected = 0.288*0.05 + 0.049*0.85 + 0.451*0.09 + 0.54*0.01
    assert np.isclose(a.av_disutility, expected)
    # DALYs decompose into YLL + YLD, all non-negative, indexed by year.
    assert len(a.dalys) == len(a.years)
    assert np.allclose(a.dalys, a.yll + a.yld)
    assert (a.yll >= 0).all() and (a.yld >= 0).all()
    assert a.dalys.sum() > 0
    assert a.years[0] == 2000


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


def test_results_by_genotype_stacks_and_normalizes(base_sim):
    df = hpv.results_by_genotype(base_sim, key='cum_cancers')
    assert list(df.columns) == ['hpv16', 'hpv18']
    assert len(df) == len(base_sim.timevec)
    # cum_cancers is non-decreasing in time per genotype.
    assert (df['hpv16'].diff().dropna() >= -1e-9).all()
    # Normalized rows sum to 1 where any cancers exist.
    ndf = hpv.results_by_genotype(base_sim, key='cum_cancers', normalize=True)
    row_sums = ndf.sum(axis=1)
    nonzero = row_sums[df.sum(axis=1) > 0]
    assert np.allclose(nonzero, 1.0)


def test_public_api_exports():
    for name in ['snapshot', 'age_pyramid', 'age_causal_infection', 'dalys',
                 'results_by_genotype', 'AgeResults']:
        assert hasattr(hpv, name), f'hpv.{name} not exported'


def test_start_filter_excludes_earlier_onsets():
    """`start` excludes cancer onsets before that year in dalys + age_causal.

    A later `start` must capture strictly fewer DALYs and fewer counted cancers
    than an earlier `start`. Both windows are measured from ONE run by attaching
    two differently-named analyzer pairs (early=1990, late=2025), so the
    comparison is on an identical trajectory with a single sim.
    """
    d_early = hpv.dalys(start=1990, name='dalys_early')
    d_late = hpv.dalys(start=2025, name='dalys_late')
    a_early = hpv.age_causal_infection(start=1990, name='aci_early')
    a_late = hpv.age_causal_infection(start=2025, name='aci_late')
    sim = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1990, stop=2040,
                  n_agents=3000, rand_seed=1,
                  analyzers=[d_early, d_late, a_early, a_late])
    sim.run()
    d0 = sim.analyzers['dalys_early']
    d1 = sim.analyzers['dalys_late']
    a0 = sim.analyzers['aci_early']
    a1 = sim.analyzers['aci_late']
    assert d0.years[0] == 1990 and d1.years[0] == 2025
    assert 0 < d1.dalys.sum() < d0.dalys.sum()
    assert 0 < len(a1.age_cancer) < len(a0.age_cancer)


def test_age_pyramid_die_on_past_end():
    """age_pyramid raises when a timepoint is past sim end and die=True."""
    ap = hpv.age_pyramid(timepoints=[2099], die=True)
    sim = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=2000, stop=2010,
                  n_agents=200, analyzers=[ap])
    with pytest.raises(ValueError):
        sim.run()

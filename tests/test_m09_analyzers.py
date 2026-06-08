import numpy as np
import sciris as sc
import starsim as ss
import hpvsim as hpv
from hpvsim.hpv import HPV

CFG = dict(location='nigeria', genotypes=['hpv16'], start=1990, stop=2030, n_agents=2000)


def _hpv_mods(sim):
    return [d for d in sim.diseases.values() if isinstance(d, HPV)]


def test_ledger_event_tuple_is_enriched_6tuple():
    """At ratio>1 each _cancer_events row is (onset_ti, causal, cin, cancer, death, w)."""
    sim = hpv.Sim(ms_agent_ratio=3, rand_seed=1, **CFG)
    sim.run()
    events = []
    for m in _hpv_mods(sim):
        events += list(m._cancer_events)
    assert len(events) > 0, 'expected some cancer events at ratio>1'
    for ev in events:
        assert len(ev) == 6, f'expected 6-tuple, got {len(ev)}'
        onset_ti, causal, cin_age, cancer_age, death_age, w = ev
        assert onset_ti == int(onset_ti) and onset_ti >= 0
        assert causal <= cin_age <= cancer_age <= death_age + 1e-9
        assert w > 0


def test_ledger_empty_at_ratio_one():
    """At ms_agent_ratio==1 the event ledger is never populated (fast path)."""
    sim = hpv.Sim(ms_agent_ratio=1, rand_seed=1, **CFG)
    sim.run()
    for m in _hpv_mods(sim):
        assert len(m._cancer_events) == 0


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
    snap = hpv.snapshot(timepoints=[2000, 2010])
    sim = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1995, stop=2015,
                  n_agents=500, analyzers=[snap])
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


def test_age_pyramid_bins_sum_to_alive():
    edges = np.array([0., 15., 50., 100.])
    ap = hpv.age_pyramid(timepoints=[2010], edges=edges)
    snap = hpv.snapshot(timepoints=[2010])
    sim = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=2000, stop=2012,
                  n_agents=800, analyzers=[ap, snap])
    sim.run()
    a = sim.analyzers['age_pyramid']
    assert len(a.age_pyramids) == 1
    date = list(a.age_pyramids.keys())[0]
    assert isinstance(date, ss.date)
    arr = a.age_pyramids[date]                  # (nbins, 2): col0=male, col1=female
    assert arr.shape == (len(a.bins), 2)

    # Ground truth: recompute the scale-weighted age histograms from the
    # snapshot of People taken at the SAME tick. This independently verifies
    # the male/female column assignment, the alive mask, and scale-weighting.
    ppl = sim.analyzers['snapshot'].get(2010)
    alive = ppl.alive.values
    ages = ppl.age.values
    female = ppl.female.values
    scale = getattr(ppl, 'scale', None)
    w = scale.values if scale is not None else None

    def hist(mask):
        ww = w[mask] if w is not None else None
        return np.histogram(ages[mask], bins=edges, weights=ww)[0]

    exp_male = hist(alive & ~female)
    exp_female = hist(alive & female)
    assert np.allclose(arr[:, 0], exp_male), 'male column mismatch (possible male/female swap)'
    assert np.allclose(arr[:, 1], exp_female), 'female column mismatch (possible male/female swap)'

    # Named invariant: total binned count == scale-weighted alive within [0, 100).
    in_range = alive & (ages >= edges[0]) & (ages < edges[-1])
    expected_total = float(w[in_range].sum()) if w is not None else float(in_range.sum())
    assert abs(arr.sum() - expected_total) < 1e-6

    # tidy-frame sanity
    df = a.to_dataframe()
    assert set(df['sex']) == {'male', 'female'}
    assert (df['count'] >= 0).all()
    assert df['count'].sum() > 0


def test_age_causal_infection_single_scale():
    aci = hpv.age_causal_infection(start=2000)
    sim = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1990, stop=2040,
                  n_agents=3000, rand_seed=1, analyzers=[aci])
    sim.run()
    a = sim.analyzers['age_causal_infection']
    assert len(a.age_cancer) > 0
    # Ordering: causal infection precedes CIN precedes cancer, on average.
    assert np.nanmean(a.age_causal) < np.nanmean(a.age_cin) < np.nanmean(a.age_cancer)
    # Dwell times non-negative and consistent: total ~= precin + cin.
    assert np.all(a.dwelltime['total'] >= -1e-9)
    assert np.allclose(a.dwelltime['total'],
                       a.dwelltime['precin'] + a.dwelltime['cin'], atol=1e-6)
    # At ratio==1 every event has weight 1.
    assert np.allclose(a.weights, 1.0)


def test_age_causal_infection_ledger_unbiased():
    """Mean ages at ratio>1 (ledger) overlap ratio==1 (agents)."""
    def run(ratio):
        aci = hpv.age_causal_infection(start=2000)
        sim = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1990, stop=2040,
                      n_agents=3000, rand_seed=2, ms_agent_ratio=ratio, analyzers=[aci])
        sim.run()
        a = sim.analyzers['age_causal_infection']
        return np.nanmean(a.age_cancer), len(a.age_cancer)
    mean1, n1 = run(1)
    mean3, n3 = run(3)
    assert n3 > n1                      # ledger yields ~ratio x more samples
    assert abs(mean3 - mean1) < 2.0     # mean age at cancer within 2 years


def test_dalys_basic_and_av_disutility():
    d = hpv.dalys(start=2000, life_expectancy=84)
    sim = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1990, stop=2040,
                  n_agents=3000, rand_seed=1, analyzers=[d])
    sim.run()
    a = sim.analyzers['dalys']
    # av_disutility matches the v2 GBD2017 constant exactly.
    expected = 0.288*0.05 + 0.049*0.85 + 0.451*0.09 + 0.54*0.01
    assert np.isclose(a.av_disutility, expected)
    # DALYs decompose into YLL + YLD, all non-negative, indexed by year.
    assert len(a.dalys) == len(a.years)
    assert np.allclose(a.dalys, a.yll + a.yld)
    assert (a.yll >= 0).all() and (a.yld >= 0).all()
    assert a.dalys.sum() > 0
    assert a.years[0] == 2000


def test_dalys_ledger_overlaps_single_scale():
    def total(ratio):
        d = hpv.dalys(start=2000)
        sim = hpv.Sim(genotypes=['hpv16'], location='nigeria', start=1990, stop=2040,
                      n_agents=3000, rand_seed=2, ms_agent_ratio=ratio, analyzers=[d])
        sim.run()
        return sim.analyzers['dalys'].dalys.sum()
    t1 = total(1)
    t3 = total(3)
    # Weighted DALYs total within 40% across ratios.
    # The ledger path at ratio>1 stores own+extras each at weight 1/ratio, so the
    # weighted event count should equal the ratio==1 count in expectation.  In
    # practice the two sims have different cancer dynamics (different RNG streams),
    # so total-DALY equivalence is loose (~25-35% spread at n=3000).
    assert abs(t3 - t1) / t1 < 0.40

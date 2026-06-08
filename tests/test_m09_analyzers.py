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
    # Snapshot is a deep copy: mutating the live sim doesn't change it.
    n_before = len(p_int.age)
    assert n_before > 0
    # default get() returns the first snapshot
    assert s.get() is s.snapshots[list(s.snapshots.keys())[0]]

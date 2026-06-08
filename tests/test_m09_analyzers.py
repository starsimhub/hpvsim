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

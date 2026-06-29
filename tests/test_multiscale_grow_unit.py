import sys, os
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
import hpvsim as hpv
import starsim as ss
import numpy as np

def test_ratio_param_and_fine_state_exist():
    assert hpv.__file__.startswith(WT), f'wrong hpvsim loaded: {hpv.__file__}'
    # fine state exists on People and defaults all-False. Checked at ratio==1
    # (grow is a no-op) so no fine agents are spawned during init seeding.
    sim1 = hpv.Sim(location='nigeria', n_agents=500, start=2000, stop=2002,
                   ms_agent_ratio=1)
    sim1.init()
    assert 'fine' in sim1.people.states
    assert not sim1.people.fine.values.any()
    # ratio propagated to every genotype HPV module
    sim = hpv.Sim(location='nigeria', n_agents=500, start=2000, stop=2002,
                  ms_agent_ratio=10)
    sim.init()
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            assert int(dis.pars.ms_agent_ratio) == 10

def test_results_are_float_and_scale_weighted_noop_at_ratio1():
    # ratio==1: scale_flows == len, so counts are unchanged but dtype is float.
    sim = hpv.Sim(location='nigeria', n_agents=2000, start=1990, stop=2020,
                  ms_agent_ratio=1, rand_seed=1)
    sim.run()
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            r = dis.results
            assert r.new_cancers.values.dtype == np.float64
            assert r.new_cancer_deaths.values.dtype == np.float64
            # ratio==1 keeps everyone at scale 1.0 → integer-valued counts
            nc = r.new_cancers.values
            assert np.allclose(nc, np.round(nc))
    assert not sim.people.fine.values.any()
    assert np.allclose(sim.people.scale.values, 1.0)

def test_clone_agents_copies_people_and_module_state():
    import numpy as np
    from hpvsim.hpv import _clone_agents
    sim = hpv.Sim(location='nigeria', n_agents=400, start=2000, stop=2001,
                  genotypes=[16, 18], ms_agent_ratio=1, rand_seed=2)
    sim.init()
    ppl = sim.people
    src = ppl.auids[:5]
    # mark distinct source values we can check after cloning
    ppl.age[src] = np.array([11., 22., 33., 44., 55.])
    new = ppl.grow(len(src))
    _clone_agents(sim, src, new)
    assert np.allclose(ppl.age[new], ppl.age[src])
    # uid must NOT be overwritten by the clone
    assert not np.array_equal(np.asarray(ppl.uid[new]), np.asarray(ppl.uid[src]))
    # module states copied for every genotype
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            assert np.array_equal(np.asarray(dis.susceptible[new]),
                                  np.asarray(dis.susceptible[src]))

def test_grow_creates_fine_cancer_agents_at_ratio():
    import numpy as np
    n0 = 6000
    sim = hpv.Sim(location='nigeria', n_agents=n0, start=1980, stop=2025,
                  ms_agent_ratio=10, rand_seed=3)
    sim.run()
    ppl = sim.people
    # multiscale grew real fine agents...
    assert ppl.fine.values.any(), 'no fine agents were grown'
    # ...all fine agents carry scale 1/ratio
    fine_uids = ppl.auids[ppl.fine[ppl.auids]]
    assert np.allclose(ppl.scale[fine_uids], 1.0/10)
    # ...and every fine agent is cancer-bound in exactly the genotype that grew
    # it (cancerous now, or scheduled): at least one HPV module flags them.
    flagged = np.zeros(len(fine_uids), dtype=bool)
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            sched = ~np.isnan(np.asarray(dis.ti_cancerous[fine_uids]))
            flagged |= (np.asarray(dis.cancerous[fine_uids]) | sched)
    assert flagged.all()

def test_fine_agents_excluded_from_network():
    sim = hpv.Sim(location='nigeria', n_agents=6000, start=1980, stop=2025,
                  ms_agent_ratio=10, rand_seed=4)
    sim.run()
    ppl = sim.people
    fine_uids = ppl.auids[ppl.fine[ppl.auids]]
    assert len(fine_uids) > 0
    # No edge in any sexual-network layer touches a fine agent.
    net = [n for n in sim.networks.values()
           if isinstance(n, hpv.SexualNetwork)][0]
    edges = net.edges
    fine_set = set(np.asarray(fine_uids).tolist())
    p1 = set(np.asarray(edges.p1).tolist())
    p2 = set(np.asarray(edges.p2).tolist())
    assert fine_set.isdisjoint(p1) and fine_set.isdisjoint(p2)

def test_fine_agents_not_emigrated():
    """White-box behavioral test: AgeMigration._emigrate never receives a fine uid.

    Approach: monkeypatch _emigrate to capture, at call-time, which of the
    candidate band_uids are fine.  After the run, assert (a) fine agents existed
    (non-vacuous) and (b) none of the uids passed to _emigrate were fine at that
    moment.

    Why non-vacuous: with the OLD ``snap_uids = people.auids.copy()`` line, fine
    agents are counted in the pyramid and their uids appear in band_uids, so when
    a band is over-target they CAN be passed to _emigrate.  This test captures
    fine-ness AT CALL TIME (before request_removal), so it genuinely fails RED if
    the exclusion filter is removed.
    """
    from hpvsim.demographics import AgeMigration
    import numpy as np

    # Each entry is True if any uid in that _emigrate call was fine at call-time.
    any_fine_emigrated = []
    fine_existed = []          # Track whether fine agents existed when _emigrate fired

    sim = hpv.Sim(location='nigeria', n_agents=6000, start=1980, stop=2025,
                  ms_agent_ratio=10, rand_seed=6)
    sim.init()

    # Find the AgeMigration instance so we can wrap its _emigrate method.
    mig = [d for d in sim.demographics.values()
           if isinstance(d, AgeMigration)][0]
    ppl = sim.people
    orig_emigrate = mig._emigrate

    def recording_emigrate(band_uids, n):
        # Query fine state AT THIS MOMENT (before request_removal removes agents).
        if len(band_uids) > 0 and 'fine' in ppl.states:
            fine_flags = np.asarray(ppl.fine[band_uids])
            any_fine_emigrated.append(fine_flags.any())
            # Also record whether any fine agents exist right now (non-vacuous guard).
            all_alive = ppl.auids
            fine_existed.append(np.asarray(ppl.fine[all_alive]).any())
        return orig_emigrate(band_uids, n)

    mig._emigrate = recording_emigrate
    sim.run()

    # Non-vacuous guard: _emigrate must have fired at least once while fine agents
    # existed, proving the scenario was actually exercised.
    assert any(fine_existed), (
        '_emigrate never fired while fine agents existed — test is vacuous'
    )

    # Core behavioral assertion: none of the band_uids passed to _emigrate were
    # fine at call-time.
    assert not any(any_fine_emigrated), (
        'AgeMigration emigrated at least one fine agent (snapshot filter missing)'
    )


def test_fine_agents_do_not_drive_births():
    from hpvsim.demographics import Births
    # A fine parent dropped from birth_uids == excluded from the pool, because
    # births are an independent per-agent Bernoulli.
    sim = hpv.Sim(location='nigeria', n_agents=3000, start=2000, stop=2002,
                  ms_agent_ratio=10, rand_seed=5)
    sim.init()
    births = [d for d in sim.demographics.values() if isinstance(d, Births)]
    assert births, 'default births class should be hpv.Births'
    ppl = sim.people
    # Force a couple of agents fine, then confirm get_births never returns them.
    some = ppl.auids[:50]
    ppl.fine[some] = True
    b = births[0]
    for _ in range(20):
        uids = b.get_births()
        assert not np.asarray(ppl.fine[uids]).any()

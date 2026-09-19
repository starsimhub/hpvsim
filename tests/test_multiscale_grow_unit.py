import sys, os
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
import hpvsim as hpv
import starsim as ss
import numpy as np
import pytest


# Module-scoped runs: the tests below are read-only on post-run state. dt=0.5 over
# 1990-2020 at n=3000 is the cheapest window that grows enough fine agents to make
# the non-vacuity assertions in each test hold.
class _GrownRun:
    """A finished ms_agent_ratio=10 sim plus what AgeMigration did during it."""
    def __init__(self, sim):
        self.sim = sim
        self.fine_in_emigrate = []   # per _emigrate call: was any candidate fine?
        self.fine_existed = []       # per _emigrate call: did any fine agent exist?
        self.hazard_removed = []     # per _emigrate_fine call: how many were removed?


@pytest.fixture(scope='module')
def grown():
    """One ms_agent_ratio=10 run with both AgeMigration emigration paths recorded."""
    from hpvsim.demographics import AgeMigration
    sim = hpv.Sim(location='nigeria', n_agents=3000, start=1990, stop=2020,
                  dt=0.5, ms_agent_ratio=10, rand_seed=6)
    sim.init()
    run = _GrownRun(sim)
    mig = [d for d in sim.demographics.values() if isinstance(d, AgeMigration)][0]
    ppl = sim.people
    orig_emigrate, orig_emigrate_fine = mig._emigrate, mig._emigrate_fine

    def recording_emigrate(band_uids, n):
        # Query fine state now, before request_removal removes agents.
        if len(band_uids) > 0 and 'fine' in ppl.states:
            run.fine_in_emigrate.append(bool(np.asarray(ppl.fine[band_uids]).any()))
            run.fine_existed.append(bool(np.asarray(ppl.fine[ppl.auids]).any()))
        return orig_emigrate(band_uids, n)

    def recording_emigrate_fine(fine_band_uids, p):
        orig_emigrate_fine(fine_band_uids, p)
        # Count fine uids newly marked for removal by this call.
        if len(fine_band_uids):
            just = fine_band_uids[np.asarray(ppl.ti_removed[fine_band_uids]) == sim.ti]
            run.hazard_removed.append(len(just))

    mig._emigrate = recording_emigrate
    mig._emigrate_fine = recording_emigrate_fine
    sim.run()
    return run


@pytest.fixture(scope='module')
def grown_sim(grown):
    """The finished ms_agent_ratio=10 sim, for tests that don't need the records."""
    return grown.sim


@pytest.fixture(scope='module')
def ratio1_sim():
    """One ms_agent_ratio==1 run (n=2000, 1990-2010) — no fine agents."""
    sim = hpv.Sim(location='nigeria', n_agents=2000, start=1990, stop=2010,
                  dt=0.5, ms_agent_ratio=1, rand_seed=1)
    sim.run()
    return sim

def test_ratio_param_and_fine_state_exist():
    assert hpv.__file__.startswith(WT), f'wrong hpvsim loaded: {hpv.__file__}'
    # Checked at ratio==1, where grow is a no-op, so nothing is seeded fine.
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

def test_results_are_float_and_scale_weighted_noop_at_ratio1(ratio1_sim):
    # Float storage is kept for ratio>1 compatibility, but with all scale==1 the
    # weighting collapses to raw counts.
    sim = ratio1_sim
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            r = dis.results
            assert r.new_cancers.values.dtype == np.float64
            assert r.new_cancer_deaths.values.dtype == np.float64
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
    assert not np.array_equal(np.asarray(ppl.uid[new]), np.asarray(ppl.uid[src]))
    # Module states copied for every genotype
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            assert np.array_equal(np.asarray(dis.susceptible[new]),
                                  np.asarray(dis.susceptible[src]))

def test_grow_creates_fine_cancer_agents_at_ratio(grown_sim):
    import numpy as np
    sim = grown_sim
    ppl = sim.people
    assert ppl.fine.values.any(), 'no fine agents were grown'
    fine_uids = ppl.auids[ppl.fine[ppl.auids]]
    assert np.allclose(ppl.scale[fine_uids], 1.0/10)
    # Every fine agent is cancer-bound (cancerous or scheduled) in some genotype.
    flagged = np.zeros(len(fine_uids), dtype=bool)
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            sched = ~np.isnan(np.asarray(dis.ti_cancerous[fine_uids]))
            flagged |= (np.asarray(dis.cancerous[fine_uids]) | sched)
    assert flagged.all()

def test_fine_agents_excluded_from_network(grown_sim):
    sim = grown_sim
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

def test_fine_agents_excluded_from_pyramid_emigration(grown):
    """AgeMigration._emigrate, the pyramid-target path, never receives a fine uid.

    Counting fine agents as whole bodies would over-fill the cancer-age bands.
    They face a separate per-band hazard instead (see
    test_fine_agents_face_emigration_hazard). The ``grown`` fixture wraps
    _emigrate to record which candidate band_uids were fine at call time.
    """
    assert any(grown.fine_existed), (
        '_emigrate never fired while fine agents existed — test is vacuous'
    )
    assert not any(grown.fine_in_emigrate), (
        'AgeMigration emigrated at least one fine agent (snapshot filter missing)'
    )


def test_fine_agents_face_emigration_hazard(grown):
    """AgeMigration._emigrate_fine's per-band Bernoulli hazard fires at ratio>1.

    Excluding fine agents from emigration entirely would over-realize cancer
    relative to single scale: a coarse source can emigrate before its cancer
    fires, but its fine peers otherwise cannot.
    """
    assert grown.sim.people.fine.values.any() or grown.hazard_removed, 'no fine agents grown'
    assert sum(grown.hazard_removed) > 0, (
        'no fine agent was emigrated via the per-band hazard: fine agents must face '
        'the same emigration competing-risk as coarse agents, or cancer is over-realized'
    )


def test_fine_agents_do_not_drive_births():
    from hpvsim.demographics import Births
    # Births are an independent per-agent Bernoulli, so dropping a fine parent
    # from birth_uids is the same as excluding it from the pool.
    sim = hpv.Sim(location='nigeria', n_agents=3000, start=2000, stop=2002,
                  ms_agent_ratio=10, rand_seed=5)
    sim.init()
    births = [d for d in sim.demographics.values() if isinstance(d, Births)]
    assert births, 'default births class should be hpv.Births'
    ppl = sim.people
    some = ppl.auids[:50]
    ppl.fine[some] = True
    b = births[0]
    for _ in range(20):
        uids = b.get_births()
        assert not np.asarray(ppl.fine[uids]).any()


def test_stock_prevalence_scale_weighted(grown_sim):
    """Fine agents (scale=1/ratio) count as 1/ratio in n_* stock results.

    Every per-genotype stock and the HPVTotal union stocks must be float-typed
    and match a scale_flows recompute at the final step.
    """
    sim = grown_sim

    ppl = sim.people
    fine_uids = ppl.auids[np.asarray(ppl.fine[ppl.auids], dtype=bool)]
    assert len(fine_uids) > 0, 'no fine agents grown — test is vacuous'

    # sim.ti after run() is the index of the last completed step (npts - 1).
    ti_last = sim.ti
    # Stored results carry pop_scale (applied at finalize); scale_flows does not.
    pop_scale = sim.pars.pop_scale

    # ---- Per-genotype HPV module stocks ----
    for dis in sim.diseases.values():
        if not isinstance(dis, hpv.HPV):
            continue
        r = dis.results
        for state_name in ('susceptible', 'infected', 'precin', 'cin',
                           'cancerous', 'latent'):
            key = f'n_{state_name}'
            assert key in r, f'{dis.name}: expected result {key}'
            assert r[key].dtype == float, (
                f'{dis.name}.{key}: dtype={r[key].dtype}, expected float'
            )
            assert r[key].values.dtype == np.float64, (
                f'{dis.name}.{key}: values.dtype={r[key].values.dtype}, expected float64'
            )
            # Post-run people state still matches what was stored at sim.ti.
            state_arr = getattr(dis, state_name)
            uids_in = ppl.auids[np.asarray(state_arr[ppl.auids], dtype=bool)]
            expected = ppl.scale_flows(uids_in) if len(uids_in) > 0 else 0.0
            stored = float(r[key].values[ti_last]) / pop_scale
            assert np.isclose(stored, expected, rtol=1e-6), (
                f'{dis.name}.{key} at ti={ti_last}: stored/pop_scale={stored:.4f}, '
                f'scale_flows={expected:.4f}'
            )

    # ---- HPVTotal union-based stocks ----
    hpvt = sim.results.get('all_hpv', None)
    assert hpvt is not None, 'HPVTotal analyzer result block missing'
    for key in ('n_infected', 'n_precin', 'n_cin', 'n_cancerous'):
        assert hpvt[key].dtype == float, (
            f'hpvtotal.{key}: dtype={hpvt[key].dtype}, expected float'
        )
        assert hpvt[key].values.dtype == np.float64, (
            f'hpvtotal.{key}: values.dtype={hpvt[key].values.dtype}, expected float64'
        )
    # Index via auids, not alive.values: grow() extends uids past n_agents slots.
    auids = ppl.auids
    alive_mask = np.asarray(ppl.alive[auids], dtype=bool)
    any_infected = np.zeros(len(auids), dtype=bool)
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            any_infected |= np.asarray(dis.infected[auids], dtype=bool)
    any_infected &= alive_mask
    inf_uids = auids[any_infected]
    expected_hpvt_inf = ppl.scale_flows(inf_uids) if len(inf_uids) > 0 else 0.0
    stored_hpvt_inf = float(hpvt['n_infected'].values[ti_last]) / pop_scale
    assert np.isclose(stored_hpvt_inf, expected_hpvt_inf, rtol=1e-6), (
        f'hpvtotal.n_infected at ti={ti_last}: stored/pop_scale={stored_hpvt_inf:.4f}, '
        f'scale_flows={expected_hpvt_inf:.4f}'
    )


def test_per_genotype_prevalence_scale_weighted(grown_sim):
    """Per-genotype prevalence must be re-derived from the scale-weighted n_infected.

    HPV.update_results divides the scale-weighted n_infected by
    scale_flows(alive_uids), so stored_prev * n_alive_sw must land within 5% of
    n_infected_sw (population turnover) and well below the unweighted count.
    """
    sim = grown_sim

    ppl = sim.people
    auids = ppl.auids
    fine_uids = auids[np.asarray(ppl.fine[auids], dtype=bool)]
    assert len(fine_uids) > 0, 'no fine agents grown — test is vacuous'

    ti_last = sim.ti
    n_alive_sw = ppl.scale_flows(auids)
    assert n_alive_sw > 0

    for dis in sim.diseases.values():
        if not isinstance(dis, hpv.HPV):
            continue
        r = dis.results
        assert 'prevalence' in r, f'{dis.name}: prevalence result missing'

        stored_prev = float(r['prevalence'].values[ti_last])
        # n_infected has scale=True, prevalence is a ratio: divide out pop_scale.
        n_infected_sw = float(r['n_infected'].values[ti_last]) / sim.pars.pop_scale

        # rtol=5% absorbs one step's births/deaths between update_results and here.
        back_calc_numerator = stored_prev * n_alive_sw
        assert np.isclose(back_calc_numerator, n_infected_sw, rtol=0.05), (
            f'{dis.name}.prevalence back-calc numerator mismatch at ti={ti_last}: '
            f'stored_prev*n_alive_sw={back_calc_numerator:.4f}, '
            f'n_infected_sw={n_infected_sw:.4f} (rtol=5%)'
        )

        infected_arr = dis.infected
        plain_n_infected = float(np.asarray(infected_arr[auids], dtype=bool).sum())
        infected_fine = int(np.asarray(infected_arr[fine_uids], dtype=bool).sum())
        assert infected_fine > 0, (
            f'{dis.name}: no fine agents are infected, so scale-weighting is '
            f'indistinguishable from a plain count (test is vacuous)'
        )

        # Counting fine agents as 1 rather than 1/ratio inflates prevalence.
        plain_prev = plain_n_infected / n_alive_sw
        assert stored_prev < plain_prev * 0.95, (
            f'{dis.name}: stored_prev={stored_prev:.6f} is not measurably '
            f'below plain_prev={plain_prev:.6f}: prevalence must be scale-weighted, '
            f'not a plain count. infected_fine={infected_fine}'
        )


def test_per_genotype_prevalence_ratio1_unchanged(ratio1_sim):
    """At ms_agent_ratio==1 the prevalence re-derive is a no-op.

    All agents have scale=1.0, so scale_flows == plain count and the
    back-calculated numerator matches n_infected_sw within 5%.
    """
    sim = ratio1_sim

    ppl = sim.people
    assert not ppl.fine.values.any(), 'unexpected fine agents at ratio==1'

    ti_last = sim.ti
    auids = ppl.auids
    n_alive_sw = ppl.scale_flows(auids)
    assert n_alive_sw > 0

    for dis in sim.diseases.values():
        if not isinstance(dis, hpv.HPV):
            continue
        r = dis.results
        stored_prev = float(r['prevalence'].values[ti_last])
        # n_infected carries pop_scale (applied at finalize); n_alive_sw does not.
        n_infected_sw = float(r['n_infected'].values[ti_last]) / sim.pars.pop_scale
        back_calc_numerator = stored_prev * n_alive_sw
        assert np.isclose(back_calc_numerator, n_infected_sw, rtol=0.05), (
            f'{dis.name}.prevalence ratio==1 back-calc mismatch at ti={ti_last}: '
            f'stored_prev*n_alive_sw={back_calc_numerator:.4f}, '
            f'n_infected_sw={n_infected_sw:.4f}'
        )

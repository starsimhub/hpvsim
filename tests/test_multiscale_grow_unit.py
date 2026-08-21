import sys, os
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
import hpvsim as hpv
import starsim as ss
import numpy as np
import pytest


# Module-scoped shared runs. The multiscale-grow invariant tests below are
# read-only on post-run state, so they share one expensive run each instead of
# rebuilding it per test.
#
# ``grown`` also carries the AgeMigration call records, so the two white-box
# emigration tests read them off the same run rather than each paying for their
# own ms_agent_ratio=10 sim. dt=0.5 over 1990-2020 at n=3000 is the cheapest
# window that still grows a few hundred fine agents, infects most of them, and
# fires the per-band fine-emigration hazard tens of times — see the non-vacuity
# assertions in each test.
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
        # Query fine state AT THIS MOMENT (before request_removal removes agents).
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

def test_results_are_float_and_scale_weighted_noop_at_ratio1(ratio1_sim):
    # ratio==1: no fine agents, all people.scale=1. Float storage is preserved
    # for compatibility with the ratio>1 case, but the underlying weighting
    # collapses to raw counts. (The "counts are integer" invariant this used to
    # spot-check via np.allclose(nc, round(nc)) is now implicit in the
    # people.scale==1 assertion below, once pop_scale scaling is applied at
    # finalize.)
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
    # uid must NOT be overwritten by the clone
    assert not np.array_equal(np.asarray(ppl.uid[new]), np.asarray(ppl.uid[src]))
    # module states copied for every genotype
    for dis in sim.diseases.values():
        if isinstance(dis, hpv.HPV):
            assert np.array_equal(np.asarray(dis.susceptible[new]),
                                  np.asarray(dis.susceptible[src]))

def test_grow_creates_fine_cancer_agents_at_ratio(grown_sim):
    import numpy as np
    sim = grown_sim
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
    """Fine agents are excluded from the pyramid-target emigration path.

    AgeMigration._emigrate is the pyramid-TARGET path (removes excess real bodies
    to hit the age x sex target). Fine agents must NOT go through it — counting
    them as whole bodies over-fills cancer-age bands and causes catastrophic
    over-emigration. They instead face an INDEPENDENT per-band hazard via
    _emigrate_fine (see test_fine_agents_face_emigration_hazard), so this test
    asserts only that the pyramid-target path never receives a fine uid.

    White-box: the ``grown`` fixture wraps _emigrate to capture, at call-time,
    which of the candidate band_uids were fine. Non-vacuous: with the snapshot
    filter removed (``snap_uids = people.auids.copy()``), fine uids appear in
    band_uids and this fails RED.
    """
    # Non-vacuous guard: _emigrate must have fired at least once while fine agents
    # existed, proving the scenario was actually exercised.
    assert any(grown.fine_existed), (
        '_emigrate never fired while fine agents existed — test is vacuous'
    )

    # Core behavioral assertion: none of the band_uids passed to _emigrate were
    # fine at call-time.
    assert not any(grown.fine_in_emigrate), (
        'AgeMigration emigrated at least one fine agent (snapshot filter missing)'
    )


def test_fine_agents_face_emigration_hazard(grown):
    """Fine agents DO face an independent per-band emigration hazard.

    Task 9 fix: fine agents are excluded from the pyramid target, but if they
    were excluded from emigration entirely they would over-realize cancer vs
    single scale (the coarse source can emigrate before its cancer fires, but
    its fine peers otherwise cannot). AgeMigration._emigrate_fine applies a
    per-band Bernoulli hazard to fine agents. The ``grown`` fixture wraps it to
    count removals; this asserts it fires at ms_agent_ratio>1.

    Non-vacuous: asserts fine agents existed AND at least one was emigrated via
    the hazard over the run.
    """
    assert grown.sim.people.fine.values.any() or grown.hazard_removed, 'no fine agents grown'
    assert sum(grown.hazard_removed) > 0, (
        'no fine agent was emigrated via the per-band hazard — fine agents are '
        'not facing emigration competing-risk (Task 9 incidence inflation returns)'
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


def test_stock_prevalence_scale_weighted(grown_sim):
    """Fine agents (scale=1/ratio) must count as 1/ratio in n_* stock results.

    Checks dtype==float AND that scale_flows-recomputed values match the stored
    result at the final time step, for every stock that had fine agents in it.
    Also checks HPVTotal union-based stocks are similarly scale-weighted.
    Ratio==1 is covered by the regression gate tests (scale_flows==count).
    """
    sim = grown_sim

    ppl = sim.people
    # Must have grown some fine agents for the test to be non-vacuous.
    fine_uids = ppl.auids[np.asarray(ppl.fine[ppl.auids], dtype=bool)]
    assert len(fine_uids) > 0, 'no fine agents grown — test is vacuous'

    # sim.ti after run() is the index of the last completed step (npts - 1).
    ti_last = sim.ti
    # Stored results have pop_scale applied at finalize; scale_flows is
    # agent-scale (people.scale-weighted, no pop_scale). Divide stored by
    # pop_scale to compare on the same scale.
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
            # dtype must be float (not int)
            assert r[key].dtype == float, (
                f'{dis.name}.{key}: dtype={r[key].dtype}, expected float'
            )
            assert r[key].values.dtype == np.float64, (
                f'{dis.name}.{key}: values.dtype={r[key].values.dtype}, expected float64'
            )
            # Value must match scale_flows-weighted recompute at final step.
            # Post-run people state matches what was stored at sim.ti (last step).
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
    # Verify the union n_infected in HPVTotal matches scale_flows across modules.
    # Use auids + starsim accessors (not alive.values array) to avoid UID/index
    # confusion when ppl.grow() has grown agents beyond initial n_agents slots.
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

    At ms_agent_ratio=10, fine agents (scale=1/10) are grown for cancer-bound
    trajectories. Before the fix, Infection.update_results computed:
        prevalence = plain_count_n_infected / n_alive_sw
    where plain_count_n_infected treats every fine agent as 1 (not 1/ratio),
    inflating prevalence when many fine agents are infected.

    After the fix, HPV.update_results re-derives prevalence from the already
    scale-weighted n_infected divided by scale_flows(alive_uids), so the
    back-calculated numerator (stored_prev * n_alive_sw) should be close to
    n_infected_sw — not the plain count.

    Key invariant tested: back_calc_n_infected ≈ n_infected_sw (within 5%
    for population-turnover drift), AND significantly less than plain_n_infected.
    Both checks must be non-vacuous (fine agents actually infected).
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
        # n_infected has scale=True (multiplied by pop_scale at finalize);
        # prevalence has scale=False (a ratio, unchanged). Divide n_infected by
        # pop_scale to compare on the agent scale of n_alive_sw.
        n_infected_sw = float(r['n_infected'].values[ti_last]) / sim.pars.pop_scale

        # Back-calculate the numerator from stored prevalence and post-run
        # alive count. Small drift (<2%) is expected because n_alive_sw
        # changes by one step's deaths/births between update_results and here.
        back_calc_numerator = stored_prev * n_alive_sw
        assert np.isclose(back_calc_numerator, n_infected_sw, rtol=0.05), (
            f'{dis.name}.prevalence back-calc numerator mismatch at ti={ti_last}: '
            f'stored_prev*n_alive_sw={back_calc_numerator:.4f}, '
            f'n_infected_sw={n_infected_sw:.4f} (rtol=5%)'
        )

        # Verify the fix is non-vacuous: fine agents must be infected so
        # pre-fix behavior would have inflated prevalence by ~ratio.
        infected_arr = dis.infected
        plain_n_infected = float(np.asarray(infected_arr[auids], dtype=bool).sum())
        infected_fine = int(np.asarray(infected_arr[fine_uids], dtype=bool).sum())
        assert infected_fine > 0, (
            f'{dis.name}: no fine agents are infected — '
            f'pre-fix inflation cannot be demonstrated (test is vacuous)'
        )

        # Pre-fix behavior: prevalence = plain_n_infected / n_alive_sw.
        # Since fine agents are infected (counted as 1 each pre-fix, but should
        # be 1/ratio), the pre-fix value is significantly larger.
        plain_prev = plain_n_infected / n_alive_sw
        assert stored_prev < plain_prev * 0.95, (
            f'{dis.name}: stored_prev={stored_prev:.6f} is NOT significantly '
            f'below plain_prev={plain_prev:.6f} — fix may not have taken effect. '
            f'infected_fine={infected_fine}'
        )


def test_per_genotype_prevalence_ratio1_unchanged(ratio1_sim):
    """At ms_agent_ratio==1 the prevalence re-derive is a no-op.

    At ratio=1 all agents have scale=1.0, so scale_flows == plain count and
    n_infected_sw == n_infected. The back-calculated numerator from stored
    prevalence must match n_infected_sw closely (within 5% for population drift).
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
        # Divide n_infected by pop_scale (applied at finalize) so it matches
        # n_alive_sw's agent-scale.
        n_infected_sw = float(r['n_infected'].values[ti_last]) / sim.pars.pop_scale
        # Back-calculate: stored_prev * n_alive_sw should ≈ n_infected_sw.
        back_calc_numerator = stored_prev * n_alive_sw
        assert np.isclose(back_calc_numerator, n_infected_sw, rtol=0.05), (
            f'{dis.name}.prevalence ratio==1 back-calc mismatch at ti={ti_last}: '
            f'stored_prev*n_alive_sw={back_calc_numerator:.4f}, '
            f'n_infected_sw={n_infected_sw:.4f}'
        )

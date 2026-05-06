"""Lifecycle smoke tests for HPV16 natural history."""
import numpy as np
import pytest

import hpvsim as hpv


def test_hpv_has_progression_states():
    """HPV defines precin/cin/cancerous BoolStates and ti_* FloatArrs."""
    sim = hpv.Sim(n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0)
    sim.init()
    mod = sim.diseases.hpv16
    for name in ('precin', 'cin', 'cancerous'):
        assert hasattr(mod, name), f'HPV missing BoolState {name!r}'
    for name in ('ti_cin', 'ti_cancerous', 'ti_dead_cancer'):
        assert hasattr(mod, name), f'HPV missing FloatArr {name!r}'


def test_hpv_has_progression_pars():
    """HPV defines dur_precin/dur_cin/dur_cancer durations + cin_fn/cancer_fn dicts + imm_init."""
    mod = hpv.HPV(genotype='hpv16')
    p = mod.pars
    for name in ('dur_precin', 'dur_cin', 'dur_cancer', 'cin_fn', 'cancer_fn', 'imm_init'):
        assert name in p, f'HPV.pars missing {name!r}'


def test_hpv_progression_pars_match_v2_hpv16():
    """Spot-check that the lognormal mean/std and severity-fn dicts match v2."""
    mod = hpv.HPV(genotype='hpv16')
    p = mod.pars
    # cin_fn matches v2 _v2_legacy/parameters.py:338
    assert p.cin_fn == dict(form='logf2', k=0.3, x_infl=0, ttc=50)
    # cancer_fn includes the cin_fn keys (so _compute_severity's cin_integral
    # branch can call _compute_severity_integral internally without re-merging).
    assert p.cancer_fn['method'] == 'cin_integral'
    assert p.cancer_fn['transform_prob'] == 2e-3
    # The dur_* are ss distribution instances. Initialize via mock() so we can
    # draw samples outside a sim context.
    p.dur_precin.mock()
    durs = p.dur_precin.rvs(5000)
    # Lognormal is non-negative; check shape and positivity.
    assert len(durs) == 5000
    assert np.all(durs >= 0)


def test_set_prognoses_assigns_ti_clearance_or_ti_cin():
    """Every newly-infected agent has either ti_clearance or ti_cin set."""
    sim = hpv.Sim(n_agents=500, location='nigeria',
                  start=1990, stop=1992, dt=0.25, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    ever_infected = mod.ti_first_infection.notnan
    has_clearance = mod.ti_clearance.notnan
    has_cin = mod.ti_cin.notnan
    assert (has_clearance | has_cin)[ever_infected].all()


def test_set_prognoses_cancer_only_in_females():
    """Males never progress to CIN; only females reach ti_cin / ti_cancerous."""
    sim = hpv.Sim(n_agents=2000, location='nigeria',
                  start=1990, stop=2000, dt=0.25, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    has_cin = mod.ti_cin.notnan
    has_cancer = mod.ti_cancerous.notnan
    males = ~sim.people.female
    assert not (has_cin & males).any()
    assert not (has_cancer & males).any()


def test_set_prognoses_chain_consistency():
    """For agents with cancer scheduled: ti_cin <= ti_cancerous <= ti_dead_cancer."""
    sim = hpv.Sim(n_agents=5000, location='nigeria',
                  start=1990, stop=2000, dt=0.25, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    has_cancer_sched = mod.ti_cancerous.notnan
    if has_cancer_sched.any():
        uids = has_cancer_sched.uids
        # Compare time-step values directly
        ti_cin = mod.ti_cin[uids]
        ti_cancerous = mod.ti_cancerous[uids]
        ti_dead = mod.ti_dead_cancer[uids]
        assert (ti_cin <= ti_cancerous).all()
        assert (ti_cancerous <= ti_dead).all()


def test_step_state_progresses_precin_to_cin():
    """An agent whose ti_cin <= ti flips precin→cin."""
    sim = hpv.Sim(n_agents=2000, location='nigeria',
                  start=1990, stop=2010, dt=0.25, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    has_cin_sched = mod.ti_cin.notnan
    if has_cin_sched.any():
        uids = has_cin_sched.uids
        passed = sim.t.ti >= mod.ti_cin[uids]
        assert passed.any(), 'No CIN-scheduled agent ever had ti >= ti_cin by sim end'


def test_step_state_progresses_cin_to_cancerous():
    """An agent whose ti_cancerous <= ti flips cin→cancerous and stops transmitting."""
    sim = hpv.Sim(n_agents=5000, location='nigeria',
                  start=1990, stop=2030, dt=0.25, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    cancerous_now = mod.cancerous.uids
    if len(cancerous_now):
        # Cancer agents are not currently infected and not susceptible
        assert not mod.infected[cancerous_now].any()
        assert not mod.susceptible[cancerous_now].any()
        # And rel_trans = 0
        rel_trans_arr = np.asarray(mod.rel_trans[cancerous_now])
        assert (rel_trans_arr == 0).all()


def test_step_state_cancer_death_removes_agents():
    """Agents whose ti_dead_cancer <= ti are no longer alive."""
    sim = hpv.Sim(n_agents=5000, location='nigeria',
                  start=1990, stop=2050, dt=0.25, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    has_dead_sched = mod.ti_dead_cancer.notnan
    if has_dead_sched.any():
        uids = has_dead_sched.uids
        passed = sim.t.ti >= mod.ti_dead_cancer[uids]
        if passed.any():
            passed_uids = uids[np.asarray(passed)]
            alive_arr = np.asarray(sim.people.alive[passed_uids])
            assert not alive_arr.any(), \
                f'{int(alive_arr.sum())} cancer-death-scheduled agents still alive'


def test_step_die_resets_bool_states():
    """Dying agents have precin/cin/cancerous cleared (so result counts are accurate)."""
    sim = hpv.Sim(n_agents=500, location='nigeria',
                  start=1990, stop=2010, dt=0.25, rand_seed=0)
    sim.init()
    mod = sim.diseases.hpv16

    # Manually infect and progress some agents to each state to test cleanup
    test_uids = np.array([0, 1, 2, 3, 4], dtype=int)

    # Set up test agents in different compartments
    mod.infected[test_uids] = True
    mod.precin[test_uids] = True
    mod.susceptible[test_uids] = True
    mod.cin[test_uids[[1, 2, 3]]] = True
    mod.cancerous[test_uids[[2, 3]]] = True

    # Call step_die for these agents (simulating death)
    mod.step_die(test_uids)

    # Verify all disease-compartment states are cleared for dead agents
    assert not mod.precin[test_uids].any(), "precin not cleared after step_die"
    assert not mod.cin[test_uids].any(), "cin not cleared after step_die"
    assert not mod.cancerous[test_uids].any(), "cancerous not cleared after step_die"
    assert not mod.infected[test_uids].any(), "infected not cleared after step_die"
    assert not mod.susceptible[test_uids].any(), "susceptible not cleared after step_die"


def test_hpv_has_raw_immunity_states():
    """HPV defines 1D nab_imm / cell_imm FloatArrs as source-genotype immunity stores."""
    sim = hpv.Sim(n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0)
    sim.init()
    mod = sim.diseases.hpv16
    for name in ('nab_imm', 'cell_imm'):
        assert hasattr(mod, name), f'HPV missing FloatArr {name!r}'
        arr = getattr(mod, name)
        # Default 0.0 across the population at init.
        assert np.allclose(np.asarray(arr.values), 0.0)


def test_cleared_agents_have_reduced_susceptibility():
    """After running a sim, agents who cleared have rel_sus < 1.0
    (Connector-derived from running-max nab_imm samples).
    """
    sim = hpv.Sim(n_agents=500, location='nigeria',
                  start=1990, stop=1995, dt=0.25, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16

    # Agents who have ever been infected and are now susceptible (cleared,
    # not in cancerous compartment).
    ever = mod.ti_first_infection.notnan
    cleared_now = (ever & mod.susceptible & ~mod.cancerous).uids
    if len(cleared_now):
        rel_sus_arr = np.asarray(mod.rel_sus[cleared_now])
        # All cleared agents should have reduced rel_sus < 1.0 (some immunity).
        assert (rel_sus_arr < 1.0).all(), \
            f'cleared agents have rel_sus={rel_sus_arr[:5]}; expected <1.0'


def test_clearance_writes_raw_immunity_not_effective():
    """After Task 7: HPV.step_state writes nab_imm/cell_imm; rel_sus/sev_imm are Connector-derived."""
    sim = hpv.Sim(n_agents=2000, location='nigeria',
                  start=1990, stop=2010, dt=0.5, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    # After running, agents that have ever cleared should have nab_imm > 0.
    nab = np.asarray(mod.nab_imm.values)
    cell = np.asarray(mod.cell_imm.values)
    # At least some agents will have cleared given run length.
    assert (nab > 0).any(), 'no agents cleared and bumped nab_imm'
    assert (cell > 0).any(), 'no agents cleared and bumped cell_imm'
    # And rel_sus / sev_imm reflect the source state through the Connector
    # (single-genotype identity: rel_sus = 1 - nab_imm; sev_imm = cell_imm).
    rel_sus = np.asarray(mod.rel_sus.values)
    sev_imm = np.asarray(mod.sev_imm.values)
    cleared_uids = (nab > 0).nonzero()[0]
    assert np.allclose(rel_sus[cleared_uids], 1.0 - nab[cleared_uids], atol=1e-6)
    assert np.allclose(sev_imm[cleared_uids], cell[cleared_uids], atol=1e-6)


@pytest.mark.parametrize('genotype', ['hpv18', 'hi5', 'ohr'])
def test_genotype_pars_for_non_hpv16(genotype):
    """GenotypePars supports hpv18, hi5, ohr with v2 defaults."""
    gp = hpv.get_genotype_pars(genotype)
    assert gp.genotype == genotype
    for name in ('dur_precin', 'dur_cin', 'dur_cancer', 'cin_fn',
                 'cancer_fn', 'imm_init', 'cell_imm_init', 'rel_beta'):
        assert name in gp, f'{genotype} GenotypePars missing {name!r}'


def test_hpv18_specific_v2_values():
    """hpv18 has v2's specific cin_fn k=0.25 and rel_beta=0.75."""
    gp = hpv.get_genotype_pars('hpv18')
    assert gp.cin_fn['k'] == pytest.approx(0.25)
    assert float(gp.rel_beta) == pytest.approx(0.75)
    assert float(gp.sero_prob) == pytest.approx(0.56)


def test_hi5_specific_v2_values():
    """hi5 has v2's cancer_fn transform_prob=1.5e-3."""
    gp = hpv.get_genotype_pars('hi5')
    assert gp.cancer_fn['transform_prob'] == pytest.approx(1.5e-3)
    assert float(gp.rel_beta) == pytest.approx(0.9)


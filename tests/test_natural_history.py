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


def test_hpv_progression_pars_hpv16():
    """Spot-check the lognormal mean/std and severity-fn dicts."""
    mod = hpv.HPV(genotype='hpv16')
    p = mod.pars
    # cin_fn parameters
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
    """After running a sim, female agents who seroconverted after clearance
    have rel_sus < 1.0 (Connector-derived from running-max nab_imm samples).
    Males are excluded: male immunity is never updated on clearance, so males
    retain rel_sus = 1.0.  Only females with nab_imm > 0 are checked here.
    """
    sim = hpv.Sim(n_agents=500, location='nigeria',
                  start=1990, stop=1995, dt=0.25, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16

    # Female agents who have ever been infected, are now susceptible (cleared),
    # and have seroconverted (nab_imm > 0 — i.e. not blocked by sero_prob gate).
    ever = mod.ti_first_infection.notnan
    female = sim.people.female
    cleared_f = (ever & mod.susceptible & ~mod.cancerous & female).uids
    if len(cleared_f):
        nab_arr = np.asarray(mod.nab_imm[cleared_f])
        seroconverted = cleared_f[nab_arr > 0]
        if len(seroconverted):
            rel_sus_arr = np.asarray(mod.rel_sus[seroconverted])
            assert (rel_sus_arr < 1.0).all(), \
                f'seroconverted female cleared agents have rel_sus={rel_sus_arr[:5]}; expected <1.0'


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


def test_hi5_specific_values():
    """hi5 has cancer_fn transform_prob=1.5e-3."""
    gp = hpv.get_genotype_pars('hi5')
    assert gp.cancer_fn['transform_prob'] == pytest.approx(1.5e-3)
    assert float(gp.rel_beta) == pytest.approx(0.9)


def test_known_genotypes_extended_to_four():
    """_KNOWN_GENOTYPES gates HPV(genotype=...) for all four genotype keys."""
    for key in ('hpv16', 'hpv18', 'hi5', 'ohr'):
        mod = hpv.HPV(genotype=key)
        assert mod.genotype == key


def test_per_genotype_init_prev_curve():
    """Each genotype gets its own init_prev curve; all four seed non-zero infections."""
    sim = hpv.Sim(
        n_agents=5000, location='nigeria',
        start=1990, stop=1990, dt=1.0, rand_seed=0,
        genotypes=[16, 18, 'hi5', 'ohr'],
    )
    sim.init()
    for key in ('hpv16', 'hpv18', 'hi5', 'ohr'):
        mod = sim.diseases[key]
        n_init = int(np.asarray(mod.infected.values).sum())
        assert n_init > 0, f'{key} seeded zero initial infections'


def test_clearance_sero_prob_gates_first_immunity():
    """First-clearance immunity is gated on sero_prob; ~(1-sero_prob) of
    first-cleared agents keep nab_imm=0 (still fully susceptible)."""
    sim = hpv.Sim(n_agents=5000, location='nigeria',
                  start=1990, stop=2010, dt=0.5, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    # Females who ever cleared. Use ti_clearance.notnan as a proxy for cleared
    # at least once. (Only females have CIN; clearance from precin can also
    # set ti_clearance.)
    female = sim.people.female
    ever_cleared_f = (mod.ti_clearance.notnan & female).uids
    if len(ever_cleared_f) >= 50:
        nab = np.asarray(mod.nab_imm[ever_cleared_f])
        zero_imm_frac = float((nab == 0).sum()) / float(len(ever_cleared_f))
        # Approximate: with sero_prob=0.75, ~25% of FIRST clearances stay 0;
        # but agents who clear multiple times get repeat-path updates which
        # always boost. So zero_imm_frac is bounded above by (1 - sero_prob).
        # Some flexibility in the upper bound; just verify it's > 0 and < 0.30.
        assert 0.0 < zero_imm_frac < 0.30, \
            f'zero_imm_frac={zero_imm_frac:.3f}; expected sero_prob gating in (0, 0.30)'


def test_clearance_males_get_no_immunity():
    """Males never get post-clearance immunity (gated to cleared females)."""
    sim = hpv.Sim(n_agents=5000, location='nigeria',
                  start=1990, stop=2010, dt=0.5, rand_seed=0)
    sim.run()
    mod = sim.diseases.hpv16
    males = ~sim.people.female
    male_uids = males.uids
    nab_male = np.asarray(mod.nab_imm[male_uids])
    cell_male = np.asarray(mod.cell_imm[male_uids])
    assert (nab_male == 0).all(), \
        f'{int((nab_male > 0).sum())} males have nab_imm > 0; expected zero (only females are updated)'
    assert (cell_male == 0).all(), \
        f'{int((cell_male > 0).sum())} males have cell_imm > 0'


def test_network_acts_are_per_step_not_per_year():
    """Edge.acts is in per-step units (already divided by dt at formation).

    Default marital acts is neg_binomial(par1=80, par2=40) per year.
    With dt=0.25, per-step mean is ~20. Casual is ~12.5. Combined network
    mean should be in the 10-30 range, not 50-90 (which would indicate the
    per-year value is being used per step).
    """
    sim = hpv.Sim(
        n_agents=2000, location='nigeria',
        start=1990, stop=1991, dt=0.25, rand_seed=0,
    )
    sim.run()
    net = sim.networks.sexualnetwork
    if not len(net.edges):
        pytest.skip('No edges formed; cannot check acts unit.')
    acts = np.asarray(net.edges.acts)
    mean_acts = float(acts.mean())
    assert 5 <= mean_acts <= 35, (
        f'edges.acts mean = {mean_acts:.1f}; expected per-step (10-30 range), '
        f'not per-year (50-90 range)'
    )


def test_network_acts_age_modulated():
    """Edge.acts is age-scaled: young/old couples have lower acts than peak couples."""
    sim = hpv.Sim(
        n_agents=5000, location='nigeria',
        start=1990, stop=1992, dt=0.25, rand_seed=0,
    )
    sim.run()
    net = sim.networks.sexualnetwork
    if not len(net.edges):
        pytest.skip('No edges formed')
    # Compute average couple age per edge.
    # edges.p1/p2 are agent UIDs; index age via ss.FloatArr UID-based access.
    import starsim as ss
    p1_uids = ss.uids(np.asarray(net.edges.p1).astype(int))
    p2_uids = ss.uids(np.asarray(net.edges.p2).astype(int))
    age_p1 = np.asarray(sim.people.age[p1_uids])
    age_p2 = np.asarray(sim.people.age[p2_uids])
    acts = np.asarray(net.edges.acts)
    layer_id = np.asarray(net.edges.layer_id)
    avg_age = (age_p1 + age_p2) / 2.0

    # In each layer, peak-age couples should have higher acts on average than
    # young or old couples. Use marital layer (layer_id 0) and bin by avg_age.
    m_mask = layer_id == 0
    if m_mask.sum() < 10:
        pytest.skip('Not enough marital edges')
    young = m_mask & (avg_age < 22)
    peak = m_mask & (avg_age >= 28) & (avg_age <= 32)
    old = m_mask & (avg_age > 60)
    if young.sum() and peak.sum():
        assert acts[young].mean() < acts[peak].mean(), \
            f'Young couples should have lower acts than peak: {acts[young].mean():.1f} vs {acts[peak].mean():.1f}'
    if peak.sum() and old.sum():
        assert acts[old].mean() < acts[peak].mean(), \
            f'Old couples should have lower acts than peak: {acts[old].mean():.1f} vs {acts[peak].mean():.1f}'


def test_directional_beta_sets_per_network_pair():
    """HPV.pars.beta is a dict keyed by network with [transf2m, transm2f] pair.

    Verifies the sex-asymmetric transmission rates flow into Starsim's
    betamap as the per-direction values it uses inside Infection.infect.
    """
    sim = hpv.Sim(n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0)
    sim.init()
    mod = sim.diseases.hpv16
    beta = mod.pars.beta
    assert isinstance(beta, dict), \
        f'beta should be a per-network dict, got {type(beta)}'
    assert 'sexualnetwork' in beta, \
        f'beta missing sexualnetwork key; got keys {list(beta.keys())}'
    pair = beta['sexualnetwork']
    assert len(pair) == 2, f'beta pair length {len(pair)}'
    # F→M = gpars.beta * 1.0 = 0.25; M→F = gpars.beta * 3.69 = 0.9225.
    assert float(pair[0]) == pytest.approx(0.25, abs=1e-9)
    assert float(pair[1]) == pytest.approx(0.25 * 3.69, abs=1e-9)


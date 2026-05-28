"""Unit tests for hpv.treat_delay — integer-ti scheduler."""
import numpy as np
import starsim as ss
import hpvsim as hpv


def _four_genotype_sim_with(intv):
    return hpv.Sim(
        n_agents=500, start=2020, stop=2025, location='nigeria',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        interventions=[intv],
    )


def test_treat_delay_zero_delay_treats_same_step():
    intv = hpv.treat_delay(name='rx', product='excision', prob=1.0, delay=0)
    sim = _four_genotype_sim_with(intv)
    sim.init()
    # Force CIN on several agents
    uids = sim.people.alive.uids[:5]
    sim.diseases['hpv16'].cin[uids] = True
    sim.run_one_step()
    live = sim.interventions['rx']
    assert live.cin_treated.uids.size > 0


def test_treat_delay_two_year_delay_fires_2_years_later():
    """With dt=0.25 and delay=2.0 years, eligible agents enqueued at ti=T
    fire at ti=T + round(2.0/0.25) = T+8."""
    intv = hpv.treat_delay(name='rx', product='excision', prob=1.0, delay=2.0)
    sim = _four_genotype_sim_with(intv)
    sim.init()
    uids = sim.people.alive.uids[:5]
    sim.diseases['hpv16'].cin[uids] = True
    initial_ti = sim.ti
    # Step once — schedules treatment for ti = initial_ti + round(2/dt)
    sim.run_one_step()
    live = sim.interventions['rx']
    assert live.cin_treated.uids.size == 0  # not yet treated
    # Step forward until just before the due ti
    target_ti = initial_ti + int(round(2.0 / sim.t.dt_year))
    while sim.ti < target_ti:
        sim.run_one_step()
    # On this exact step the queue fires
    sim.run_one_step()
    assert sim.interventions['rx'].cin_treated.uids.size > 0


def test_treat_delay_queue_drains_after_fire():
    intv = hpv.treat_delay(name='rx', product='excision', prob=1.0, delay=0)
    sim = _four_genotype_sim_with(intv)
    sim.init()
    uids = sim.people.alive.uids[:3]
    sim.diseases['hpv16'].cin[uids] = True
    sim.run_one_step()
    live = sim.interventions['rx']
    # After firing, the queue entry for the current ti should be gone
    assert sim.ti not in live.scheduler or len(live.scheduler[sim.ti]) == 0


def test_treat_delay_fractional_delay_rounds_to_nearest_step():
    """At dt=0.25, delay=0.5 yr → round(0.5/0.25) = 2 steps. Fire requires
    `run_one_step()` to be called WHEN sim.ti == due_ti (not after the step
    that scheduled it)."""
    intv = hpv.treat_delay(name='rx', product='excision', prob=1.0, delay=0.5)
    sim = _four_genotype_sim_with(intv)
    sim.init()
    uids = sim.people.alive.uids[:3]
    sim.diseases['hpv16'].cin[uids] = True
    initial_ti = sim.ti
    sim.run_one_step()  # step at ti=initial; enqueues for due_ti=initial+2; advances to initial+1
    live = sim.interventions['rx']
    assert live.cin_treated.uids.size == 0  # not yet fired
    sim.run_one_step()  # step at ti=initial+1; still not due
    assert live.cin_treated.uids.size == 0
    sim.run_one_step()  # step at ti=initial+2; FIRE
    assert live.cin_treated.uids.size > 0


def test_treat_delay_non_integer_rounding_edge_case():
    """delay=0.3 yr / dt=0.25 = 1.2; round(1.2) = 1 step."""
    intv = hpv.treat_delay(name='rx', product='excision', prob=1.0, delay=0.3)
    sim = _four_genotype_sim_with(intv)
    sim.init()
    uids = sim.people.alive.uids[:3]
    sim.diseases['hpv16'].cin[uids] = True
    initial_ti = sim.ti
    sim.run_one_step()  # step at ti=initial; enqueues for due_ti=initial+1
    live = sim.interventions['rx']
    assert live.cin_treated.uids.size == 0
    sim.run_one_step()  # step at ti=initial+1; FIRE
    assert live.cin_treated.uids.size > 0

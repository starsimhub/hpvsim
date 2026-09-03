"""Integration smoke tests for hpv.routine_txvx / campaign_txvx / linked_txvx."""
import numpy as np
import pytest
import starsim as ss
import hpvsim as hpv


def _four_genotype_sim_with(intvs):
    return hpv.Sim(
        n_agents=500, start=2020, stop=2025, location='nigeria',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        interventions=intvs,
    )


def test_routine_txvx_flips_tx_vaccinated():
    intv = hpv.routine_txvx(
        name='txvx',
        product='txvx1',
        prob=0.9,
        age_range=[25, 26],
        start_year=2021,
        end_year=2023,
    )
    sim = _four_genotype_sim_with([intv])
    sim.run()
    live = sim.interventions['txvx']
    assert live.tx_vaccinated.uids.size > 0


def test_campaign_txvx_runs():
    intv = hpv.campaign_txvx(
        name='campaign',
        product='txvx1',
        prob=0.5,
        age_range=[25, 30],
        years=[2022],
    )
    sim = _four_genotype_sim_with([intv])
    sim.run()
    live = sim.interventions['campaign']
    assert live.tx_vaccinated.uids.size > 0


def test_linked_txvx_requires_eligibility():
    with pytest.raises(ValueError, match='eligibility'):
        hpv.linked_txvx(product='txvx1', prob=0.5)


def test_linked_txvx_fires_only_on_eligibility_callback():
    """linked_txvx with eligibility callback only dosed where callback yields uids."""
    screen = hpv.routine_screening(
        name='primary', product='via', prob=1.0,
        age_range=[30, 50], sex='f',
        start_year=2021, end_year=2022,
    )
    linked = hpv.linked_txvx(
        name='linked',
        product='txvx1',
        prob=1.0,
        eligibility=lambda s: s.interventions['primary'].outcomes['positive'],
    )
    sim = _four_genotype_sim_with([screen, linked])
    sim.run()
    # linked.tx_vaccinated count <= screen.screened count (positives are a subset)
    assert sim.interventions['linked'].tx_vaccinated.uids.size <= sim.interventions['primary'].screened.uids.size


def test_routine_txvx_string_product_resolves_to_txvx():
    intv = hpv.routine_txvx(
        name='txvx', product='txvx1', prob=0.5,
        age_range=[25, 26], start_year=2021, end_year=2022,
    )
    sim = _four_genotype_sim_with([intv])
    sim.init()
    live = sim.interventions['txvx']
    assert live.product.__class__.__name__ == 'txvx'
    assert live.product.pars.name == 'txvx1'


def test_routine_txvx_age_range_upper_bound_exclusive():
    """age_range=[lo, hi) — an agent exactly at hi is not eligible."""
    intv = hpv.routine_txvx(name='txvx_program', product='txvx1', prob=1.0,
                            age_range=[30, 40], start_year=2021, end_year=2022)
    sim = _four_genotype_sim_with([intv])
    sim.init()
    uids = sim.people.female.uids[:3]
    for g in ('hpv16', 'hpv18', 'hi5', 'ohr'):
        sim.diseases[g].cancerous[uids] = False
    sim.people.age[uids] = [30.0, 39.9, 40.0]
    eligible = sim.interventions['txvx_program'].check_eligibility()
    assert uids[0] in eligible
    assert uids[1] in eligible
    assert uids[2] not in eligible


def test_linked_txvx_handles_empty_eligibility_callback():
    """linked_txvx must run without errors when the eligibility callback
    returns no uids (the typical 'screen fired no positives this step' path).
    No doses delivered, no exception raised.
    """
    # Eligibility callback that always returns empty
    linked = hpv.linked_txvx(
        name='linked',
        product='txvx1',
        prob=1.0,
        eligibility=lambda s: ss.uids(),
    )
    sim = _four_genotype_sim_with([linked])
    sim.run()
    live = sim.interventions['linked']
    assert live.tx_vaccinated.uids.size == 0

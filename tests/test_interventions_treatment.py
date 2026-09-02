"""Integration smoke tests for hpv.treat_num and its HPV-specific eligibility."""
import numpy as np
import starsim as ss
import hpvsim as hpv


def _four_genotype_sim_with(intvs):
    return hpv.Sim(
        n_agents=500, start=2020, stop=2025, location='nigeria',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        interventions=intvs,
    )


def test_treat_num_treat_cancer_flag_set_for_radiation():
    treat = hpv.treat_num(name='cancer_rx', product=hpv.radiation(), prob=1.0)
    sim = _four_genotype_sim_with([treat])
    sim.init()
    assert sim.interventions['cancer_rx'].treat_cancer is True


def test_treat_num_treat_cancer_flag_unset_for_excision():
    treat = hpv.treat_num(name='cin_rx', product='excision', prob=1.0)
    sim = _four_genotype_sim_with([treat])
    sim.init()
    assert sim.interventions['cin_rx'].treat_cancer is False


def test_treat_num_excludes_cancer_when_not_treat_cancer():
    """A non-cancer treat_num must NOT make cancerous agents eligible."""
    treat = hpv.treat_num(name='cin_rx', product='excision', prob=1.0)
    sim = _four_genotype_sim_with([treat])
    sim.init()
    uids = sim.people.alive.uids[:5]
    sim.diseases['hpv16'].cancerous[uids] = True
    live = sim.interventions['cin_rx']
    eligible = live.check_eligibility()
    for u in uids:
        assert u not in eligible


def test_treat_num_only_treats_cancer_when_treat_cancer():
    """A treat_num(radiation) must ONLY treat cancerous agents.

    Cancer treatments default to sex='f' (cervical cancer program), so the
    state-gating check uses female UIDs to isolate it from the sex filter.
    """
    treat = hpv.treat_num(name='cancer_rx', product=hpv.radiation(), prob=1.0)
    sim = _four_genotype_sim_with([treat])
    sim.init()
    uids = sim.people.female.uids[:5]
    sim.diseases['hpv16'].cancerous[uids] = True
    live = sim.interventions['cancer_rx']
    eligible = live.check_eligibility()
    for u in uids:
        assert u in eligible
    other = sim.people.female.uids[10]
    assert other not in eligible


def test_treat_num_string_product_resolves_to_tx():
    """treat_num(product='excision') should resolve via hpv.tx."""
    treat = hpv.treat_num(name='rx', product='excision', prob=0.0)
    sim = _four_genotype_sim_with([treat])
    sim.init()
    live = sim.interventions['rx']
    assert live.product.__class__.__name__ == 'tx'
    assert live.product.name == 'excision'


def test_treat_num_both_sexes_includes_males():
    """sex=None, ['f','m'], 'fm' and [0,1] all mean both sexes; sex='f' excludes men.

    One sim per sex spec: two interventions sharing a product name cannot
    coexist in a sim (both products register on sim.people under that name).
    """
    eligible = {}
    for sex in (None, ['f', 'm'], 'fm', [0, 1], 'f'):
        treat = hpv.treat_num(name='rx', product='excision', prob=1.0, sex=sex)
        sim = _four_genotype_sim_with([treat])
        sim.init()
        males = set(int(u) for u in sim.people.male.uids)
        uids = set(int(u) for u in sim.interventions['rx'].check_eligibility())
        eligible[str(sex)] = uids
        if sex == 'f':
            assert not (uids & males), 'sex="f" must exclude men'
        else:
            assert uids & males, f'sex={sex!r} must include men'
    both = [v for k, v in eligible.items() if k != 'f']
    assert all(v == both[0] for v in both), 'all both-sexes specs must agree'


def test_treat_num_age_range_upper_bound_exclusive():
    """age_range=[lo, hi) — an agent exactly at hi is not eligible."""
    treat = hpv.treat_num(name='rx', product='excision', prob=1.0, age_range=[30, 40])
    sim = _four_genotype_sim_with([treat])
    sim.init()
    uids = sim.people.female.uids[:3]
    for g in ('hpv16', 'hpv18', 'hi5', 'ohr'):
        sim.diseases[g].cancerous[uids] = False
    sim.people.age[uids] = [30.0, 39.9, 40.0]
    eligible = sim.interventions['rx'].check_eligibility()
    assert uids[0] in eligible
    assert uids[1] in eligible
    assert uids[2] not in eligible


def test_treat_num_capacity_respected():
    """With max_capacity=5, no more than 5 agents are treated per step."""
    treat = hpv.treat_num(
        name='rx',
        product='excision',
        prob=1.0,
        max_capacity=5,
        eligibility=lambda s: s.people.alive.uids[:50],
    )
    sim = _four_genotype_sim_with([treat])
    sim.init()
    chosen = sim.people.alive.uids[:50]
    sim.diseases['hpv16'].cin[chosen] = True
    sim.run()
    live = sim.interventions['rx']
    n_steps = len(sim.timevec)
    assert live.cin_treated.uids.size <= n_steps * 5

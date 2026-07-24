"""Unit tests for the intervention eligibility helpers."""
import numpy as np
import starsim as ss
import hpvsim as hpv
from hpvsim.interventions import _compose_screening_eligibility
from hpvsim.utils import any_genotype_cancer


def _two_genotype_sim():
    sim = hpv.Sim(
        n_agents=300, start=2020, stop=2022, location='nigeria',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18')],
    )
    sim.init()
    return sim


def test_screening_eligibility_default_is_female_alive():
    sim = _two_genotype_sim()
    elig = _compose_screening_eligibility(age_range=None, sex='f', extra=None, debut_age=None)
    uids = elig(sim)
    for u in uids[:10]:
        assert sim.people.female[u]
        assert sim.people.alive[u]


def test_screening_eligibility_age_range_filter():
    sim = _two_genotype_sim()
    elig = _compose_screening_eligibility(age_range=[30, 50], sex='f', extra=None, debut_age=None)
    uids = elig(sim)
    for u in uids[:10]:
        a = sim.people.age[u]
        assert 30 <= a < 50


def test_screening_eligibility_debut_age_filter():
    sim = _two_genotype_sim()
    elig = _compose_screening_eligibility(age_range=None, sex='f', extra=None, debut_age=15)
    uids = elig(sim)
    for u in uids[:10]:
        assert sim.people.age[u] >= 15


def test_screening_eligibility_extra_callback_intersection():
    sim = _two_genotype_sim()
    chosen = sim.people.alive.uids[:5]
    elig = _compose_screening_eligibility(
        age_range=None, sex='f', extra=lambda s: chosen, debut_age=None,
    )
    uids = elig(sim)
    for u in uids:
        assert u in chosen
        assert sim.people.female[u]


def test_any_genotype_cancer_ors_across_modules():
    sim = _two_genotype_sim()
    uids = sim.people.alive.uids[:5]
    sim.diseases['hpv18'].cancerous[uids] = True
    cancer = any_genotype_cancer(sim)
    for u in uids:
        assert cancer[u]
    # An agent not flagged cancerous on either genotype
    other = sim.people.alive.uids[10]
    assert not cancer[other]

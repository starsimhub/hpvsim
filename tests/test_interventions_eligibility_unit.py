"""Unit tests for the condition on cancer that treatments use."""
import numpy as np
import starsim as ss
import hpvsim as hpv
from hpvsim.utils import any_genotype_cancer


def _two_genotype_sim():
    sim = hpv.Sim(
        n_agents=300, start=2020, stop=2022, location='nigeria',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18')],
    )
    sim.init()
    return sim


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

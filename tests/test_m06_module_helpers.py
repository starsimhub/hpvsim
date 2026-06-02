"""Unit tests for the shared sim-introspection helpers in hpvsim.utils."""
import hpvsim as hpv
from hpvsim.utils import iter_hpv_modules, find_genotype_module


def _two_genotype_sim():
    sim = hpv.Sim(
        n_agents=100, start=2020, stop=2021, location='nigeria',
        diseases=[hpv.HPV(genotype='hpv16'), hpv.HPV(genotype='hpv18')],
    )
    sim.init()
    return sim


def test_iter_hpv_modules_returns_hpv_only():
    sim = _two_genotype_sim()
    mods = list(iter_hpv_modules(sim))
    assert {m.genotype for m in mods} == {'hpv16', 'hpv18'}


def test_find_genotype_module_returns_match():
    sim = _two_genotype_sim()
    m = find_genotype_module(sim, 'hpv16')
    assert m is not None
    assert m.genotype == 'hpv16'


def test_find_genotype_module_returns_none_for_unknown():
    sim = _two_genotype_sim()
    assert find_genotype_module(sim, 'unknown_genotype') is None

"""Unit tests for module-level helpers shared across products."""
import hpvsim as hpv
from hpvsim.products import _iter_hpv_modules, _find_genotype_module


def _two_genotype_sim():
    sim = hpv.Sim(
        n_agents=100, start=2020, stop=2021, location='nigeria',
        diseases=[hpv.HPV(genotype='hpv16'), hpv.HPV(genotype='hpv18')],
    )
    sim.init()
    return sim


def test_iter_hpv_modules_returns_hpv_only():
    sim = _two_genotype_sim()
    mods = list(_iter_hpv_modules(sim))
    assert {m.genotype for m in mods} == {'hpv16', 'hpv18'}


def test_find_genotype_module_returns_match():
    sim = _two_genotype_sim()
    m = _find_genotype_module(sim, 'hpv16')
    assert m is not None
    assert m.genotype == 'hpv16'


def test_find_genotype_module_returns_none_for_unknown():
    sim = _two_genotype_sim()
    assert _find_genotype_module(sim, 'unknown_genotype') is None
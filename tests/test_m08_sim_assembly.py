# tests/test_m08_sim_assembly.py
import pytest
import starsim as ss
import hpvsim as hpv
from hpvsim.hiv import hpv_hiv_connector, HIVStratifiedResults


def _tiny(**kw):
    return dict(n_agents=200, start=2000, stop=2002, dt=0.25, location='nigeria', **kw)


def test_genotypes_plus_other_disease_merges():
    """A non-HPV disease passed via diseases= merges with genotype-built HPV."""
    other = ss.SIS()  # any non-HPV ss.Disease as a stand-in
    sim = hpv.Sim(**_tiny(genotypes=[16, 18], diseases=[other]))
    sim.init()
    hpv_mods = [d for d in sim.diseases.values() if isinstance(d, hpv.HPV)]
    assert len(hpv_mods) == 2                      # genotypes still built
    assert any(isinstance(d, ss.SIS) for d in sim.diseases.values())  # other merged in


def test_hpv_instance_override_still_works():
    """diseases=[HPV,...] override path is unchanged (no genotypes=)."""
    sim = hpv.Sim(**_tiny(diseases=[hpv.HPV(genotype='hpv16'), hpv.HPV(genotype='hpv18')]))
    sim.init()
    hpv_mods = [d for d in sim.diseases.values() if isinstance(d, hpv.HPV)]
    assert len(hpv_mods) == 2


def test_hpv_instances_plus_genotypes_raises():
    """Specifying the HPV set two ways still raises."""
    with pytest.raises(ValueError, match='genotypes='):
        hpv.Sim(**_tiny(genotypes=[16], diseases=[hpv.HPV(genotype='hpv16')]))


def test_hiv_autowires_connector_and_analyzer():
    sim = hpv.Sim(n_agents=200, start=2000, stop=2001, dt=0.25, location='nigeria',
                  genotypes=[16, 18], diseases=[hpv.HIV(beta_m2f=0.0)])
    sim.init()
    assert any(isinstance(c, hpv_hiv_connector) for c in sim.connectors.values())
    assert any(isinstance(a, HIVStratifiedResults) for a in sim.analyzers.values())


def test_no_hiv_no_autowire():
    sim = hpv.Sim(n_agents=200, start=2000, stop=2001, dt=0.25, location='nigeria',
                  genotypes=[16, 18])
    sim.init()
    assert not any(isinstance(c, hpv_hiv_connector) for c in sim.connectors.values())
    assert not any(isinstance(a, HIVStratifiedResults) for a in sim.analyzers.values())

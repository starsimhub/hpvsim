"""Test that HPV gains `latent` BoolState and `txvx_imm` FloatArr in M06."""
import numpy as np
import starsim as ss
import hpvsim as hpv


def _tiny_sim():
    return hpv.Sim(
        location='nigeria',
        n_agents=200, start=2020, stop=2022,
        genotypes=['hpv16'],
        rand_seed=0,
    )


def test_hpv_has_latent_boolstate():
    sim = _tiny_sim()
    sim.init()
    mod = sim.diseases['hpv16']
    assert hasattr(mod, 'latent'), "HPV module should have a `latent` state"
    # No-op default: nobody should be latent right after init.
    assert mod.latent.uids.size == 0
    assert isinstance(mod.latent, ss.BoolState)


def test_hpv_has_txvx_imm_floatarr():
    sim = _tiny_sim()
    sim.init()
    mod = sim.diseases['hpv16']
    assert hasattr(mod, 'txvx_imm'), "HPV module should have `txvx_imm` FloatArr"
    assert isinstance(mod.txvx_imm, ss.FloatArr)
    # All defaults to zero — no agents have been txvx-vaccinated.
    assert np.all(mod.txvx_imm.values == 0.0)


def test_latent_state_stays_zero_through_short_run():
    """No-op latent: nothing populates it. After a 2yr run, still zero."""
    sim = _tiny_sim()
    sim.run()
    assert sim.diseases['hpv16'].latent.uids.size == 0

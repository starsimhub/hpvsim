"""Save/load round-trip tests for hpv.Sim and ss.MultiSim.

These pin down that a run hpv.Sim (and a MultiSim of them) survives a
save/load cycle with its results intact. The v2 versions lived in the
now-deleted legacy suite (tests/_legacy/test_misc.py, test_run.py) and
exercised the removed top-level hpv.save/hpv.load facade; v3 saves via
the starsim-native sim.save / ss.load and the hpvsim.misc wrappers, so
these are rewritten against the v3 surface.

Tiny n_agents and short dur keep each test fast.
"""
import numpy as np
import pytest

import starsim as ss
import hpvsim as hpv
from hpvsim import misc


def _tiny_sim(rand_seed=0, label=None):
    return hpv.Sim(
        n_agents=300,
        location='nigeria',
        genotypes=[16],
        start=2000,
        stop=2003,
        dt=0.25,
        rand_seed=rand_seed,
        verbose=0,
        label=label,
    )


def _cum_infections(sim):
    return np.asarray(sim.results.hpv16.cum_infections[:])


def test_sim_save_load_roundtrip(tmp_path):
    """Default sim.save() (shrink=True) + ss.load preserves the sim.

    hpv.Sim.shrink defaults size_limit=None, so the default save skips
    starsim's per-module size check — which otherwise trips on the
    CrossImmunity connector and HPVTotal analyzer (each references the shared
    disease modules; the budget double-counts them). Dist shrinking still
    runs, so the saved file is small.
    """
    sim = _tiny_sim()
    sim.run()
    ref = _cum_infections(sim)

    fp = str(tmp_path / 'sim.sim')
    sim.save(fp)
    assert (tmp_path / 'sim.sim').exists()

    loaded = ss.load(fp)
    assert isinstance(loaded, hpv.Sim)
    assert np.allclose(_cum_infections(loaded), ref)


def test_sim_save_load_via_misc(tmp_path):
    """The hpvsim.misc.save / misc.load wrappers round-trip a run sim."""
    sim = _tiny_sim()
    sim.run()
    ref = _cum_infections(sim)

    fp = str(tmp_path / 'sim.obj')
    misc.save(fp, sim)
    assert (tmp_path / 'sim.obj').exists()

    loaded = misc.load(fp)
    assert isinstance(loaded, hpv.Sim)
    assert np.allclose(_cum_infections(loaded), ref)


def test_multigenotype_default_save(tmp_path):
    """Default sim.save() works for a multi-genotype sim.

    Guards the hpv.Sim.shrink size_limit=None override: without it, starsim's
    per-module size check raises on CrossImmunity / HPVTotal for >1 genotype.
    """
    sim = hpv.Sim(n_agents=300, location='nigeria', genotypes=[16, 18, 'hi5', 'ohr'],
                  start=2000, stop=2005, dt=0.25, rand_seed=0, verbose=0)
    sim.run()

    fp = str(tmp_path / 'multi.sim')
    sim.save(fp)  # default shrink=True; must not raise
    loaded = ss.load(fp)
    assert isinstance(loaded, hpv.Sim)
    assert np.allclose(_cum_infections(loaded), _cum_infections(sim))


def test_multisim_save_load_roundtrip(tmp_path):
    """A run ss.MultiSim of hpv.Sims round-trips with per-sim results intact."""
    msim = ss.MultiSim(_tiny_sim(), n_runs=3).run(verbose=0)
    ref = [_cum_infections(s) for s in msim.sims]

    fp = str(tmp_path / 'msim.obj')
    misc.save(fp, msim)
    assert (tmp_path / 'msim.obj').exists()

    loaded = misc.load(fp)
    assert isinstance(loaded, ss.MultiSim)
    assert len(loaded.sims) == len(msim.sims)
    for got, want in zip(loaded.sims, ref):
        assert np.allclose(_cum_infections(got), want)

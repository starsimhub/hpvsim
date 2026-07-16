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
    """sim.save(shrink=False) + ss.load preserves the sim and its results.

    shrink=False is used because starsim's shrink size-check trips on the
    CrossImmunity connector and HPVTotal analyzer for multi-genotype sims:
    each holds references to the (shared) disease modules, which the
    per-module budget counts against them even though pickle serializes the
    disease modules once. shrink=False sidesteps the check and saves the full,
    re-runnable sim; the actual file stays small (~1 MB).
    """
    sim = _tiny_sim()
    sim.run()
    ref = _cum_infections(sim)

    fp = str(tmp_path / 'sim.sim')
    sim.save(fp, shrink=False)
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

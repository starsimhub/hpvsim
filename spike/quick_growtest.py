"""Clean isolation: does mid-run growth of INERT, network-excluded agents
depress transmission among the real agents, holding init identical?

Baseline and test arms start from the SAME 6000-agent init + same seed, so the
real agents' initial draws are identical. The test arm grows K inert agents
(scale 0, never infected, multiscale_fine -> network-excluded) at one mid-run
step. Any divergence afterwards is purely the act of growing. Sweeping K shows
whether the effect scales with agent count (per-capita-style) or is fixed.
"""
import numpy as np
import sciris as sc
import starsim as ss
import hpvsim as hpv

CFG = dict(location='nigeria', genotypes=['hpv16'], start=1990, stop=2040,
           dt=0.25, total_pop=1e6, n_agents=6000, verbose=0)
GROW_TI = 80  # ~year 2010


class GrowInert(ss.Analyzer):
    """At GROW_TI, grow K inert network-excluded agents once."""
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.done = False

    def step(self):
        sim = self.sim
        if self.done or sim.ti != GROW_TI or self.k == 0:
            return
        self.done = True
        new = ss.uids(sim.people.grow(self.k))
        sim.people.scale[new] = 0.0
        for m in sim.diseases.values():
            if hasattr(m, 'multiscale_fine'):
                m.multiscale_fine[new] = True
                m.susceptible[new] = False
                m.infected[new] = False


def _build_demog(kind):
    """Fresh demographics per sim (modules can't be shared across sims)."""
    if kind == 'default':
        return None
    if kind == 'none':
        return []
    if kind == 'no-migration':
        from hpvsim.demographics import ScaleWeightedBirths
        return [ScaleWeightedBirths(), ss.Deaths()]
    if kind == 'migration-only':
        return [hpv.AgeMigration()]
    raise ValueError(kind)


def run(k, seed=0, kind='default'):
    kw = dict(CFG)
    demog = _build_demog(kind)
    if demog is not None:
        kw['demographics'] = demog
    s = hpv.Sim(rand_seed=seed, analyzers=[GrowInert(k)], **kw)
    s.run()
    return np.asarray(s.diseases.hpv16.results.new_infections, dtype=float)


if __name__ == '__main__':
    T = sc.timer()
    for kind in ('default', 'no-migration', 'migration-only', 'none'):
        print(f'--- {kind} ---')
        base = run(0, kind=kind)
        base_after = base[GROW_TI:].sum()
        print(f'  baseline new_infections[after grow] = {base_after:.0f}')
        for k in (1000, 4000):
            arr = run(k, kind=kind)
            after = arr[GROW_TI:].sum()
            pre_identical = np.array_equal(base[:GROW_TI], arr[:GROW_TI])
            print(f'  K={k:>5}: rel.change={(after-base_after)/base_after*100:+.1f}%  '
                  f'pre-grow identical={pre_identical}')
    T.toc('runtime')

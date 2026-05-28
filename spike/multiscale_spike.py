"""THROWAWAY SPIKE — multiscale feasibility on the v3/Starsim HPVsim port.

Not production code. Not a test. Purpose: produce empirical evidence for the
multiscale design doc. Validates four things:

  R1. grow() mid-run + state copy works across the multi-genotype module set
      without corrupting array lengths / leaving NaNs.
  R2. Per-agent `scale` weighting of cancer counts reconciles with Starsim's
      global `pop_scale` multiply (Result(scale=True)).
  R3. Reproducibility (slot-keyed CRN) survives a variable agent count — two
      same-seed runs give identical scaled cancer totals.
  EQ. Internal equivalence: multiscale (few coarse agents, ms_ratio>1)
      reproduces single-scale (many agents) cancer person-totals, ideally at
      lower variance per unit cost.

Mechanism (faithful to v2 _v2_legacy/people.py:280-369, simplified to the
single-genotype anchor): at set_prognoses, each coarse agent that entered the
CIN pathway is split into `ms_ratio` fine agents at scale 1/ms_ratio. The
original is shrunk; ms_ratio-1 copies are spawned via people.grow() and given
*independent* natural-history trajectories via a second (base) set_prognoses
call. Cancer is then resolved at ms_ratio-finer granularity.

Run:  python spike/multiscale_spike.py
"""

import numpy as np
import starsim as ss
import hpvsim as hpv


# Natural-history trajectory states cloned onto spawned fine agents (base
# ss.Infection states + HPV-specific). cancerous is False at split time.
_TRAJECTORY_STATES = [
    'susceptible', 'infected', 'rel_sus', 'rel_trans', 'ti_infected',
    'ti_clearance', 'ti_first_infection', 'precin', 'cin', 'cancerous',
    'latent', 'ti_cin', 'ti_cancerous', 'ti_dead_cancer',
    'sev_imm', 'nab_imm', 'cell_imm', 'vax_imm', 'txvx_imm',
]


# ---------------------------------------------------------------------------
# Multiscale HPV module
# ---------------------------------------------------------------------------
class MultiscaleHPV(hpv.HPV):
    """HPV genotype module that splits CIN-pathway agents into fine sub-agents."""

    def __init__(self, genotype='hpv16', ms_ratio=1, **kwargs):
        super().__init__(genotype=genotype, **kwargs)
        self.ms_ratio = int(ms_ratio)
        self._coarse_scale = 1.0  # default Starsim relative scale

    def set_prognoses(self, uids, sources=None):
        super().set_prognoses(uids, sources)
        if self.ms_ratio <= 1 or len(uids) == 0:
            return
        # Skip during init seeding — keep growth confined to the stepping loop.
        if not getattr(self.sim, 'initialized', False):
            return

        ppl = self.sim.people
        # CIN-pathway agents are the multiscale target (ti_cin scheduled).
        cand = uids[self.ti_cin.notnan[uids]]
        if len(cand) == 0:
            return
        # Only split coarse originals, never already-fine agents.
        cand = cand[np.asarray(ppl.scale[cand]) >= self._coarse_scale * 0.999]
        if len(cand) == 0:
            return

        ratio = self.ms_ratio
        # 1. Shrink the originals.
        ppl.scale[cand] = ppl.scale[cand] / ratio

        # 2. Spawn (ratio-1) copies per original.
        new_uids = ss.uids(ppl.grow(len(cand) * (ratio - 1)))
        src = ss.uids(np.repeat(np.asarray(cand), ratio - 1))

        # 3. Copy demographic identity (male is a read-only ~female property).
        ppl.age[new_uids] = ppl.age[src]
        ppl.female[new_uids] = ppl.female[src]
        ppl.scale[new_uids] = ppl.scale[src]  # already the shrunk 1/ratio value

        # 4. Clone the source's FULL natural-history trajectory onto the copies.
        #    Deterministic cloning conserves expected cancer mass exactly
        #    (ratio copies x 1/ratio scale == original's 1.0). Independent
        #    downstream cancer draws (the variance-reduction half of the
        #    technique, and the subtle part) are deliberately NOT done here;
        #    see the design doc's risk section.
        for nm in _TRAJECTORY_STATES:
            getattr(self, nm)[new_uids] = getattr(self, nm)[src]


# ---------------------------------------------------------------------------
# Scale-aware cancer counter (UID-based; robust to a growing population)
# ---------------------------------------------------------------------------
class CancerScaleAnalyzer(ss.Analyzer):
    """Accumulate scale-weighted cancer onsets, independent of module internals."""

    def __init__(self, genotype='hpv16', **kwargs):
        super().__init__(**kwargs)
        self.genotype = genotype
        self._seen = set()
        self.scaled_new_cancers = 0.0   # agent-equivalents
        self.raw_new_cancers = 0        # raw agent count (for contrast)

    def step(self):
        cur = np.asarray(self.sim.diseases[self.genotype].cancerous.uids)
        new = [int(u) for u in cur if int(u) not in self._seen]
        if new:
            nu = ss.uids(np.array(new))
            self.scaled_new_cancers += float(np.asarray(self.sim.people.scale[nu]).sum())
            self.raw_new_cancers += len(new)
            self._seen.update(new)

    @property
    def total_cancers_people(self):
        return self.scaled_new_cancers * float(self.sim.pars.pop_scale)


# ---------------------------------------------------------------------------
# Sim builders — both use the diseases= path so seeding is identical
# ---------------------------------------------------------------------------
# Fixed total people across both sims so pop_scale (=total_pop/n_agents)
# makes them model the SAME population at different agent resolutions.
TOTAL_POP = 1e6
BASE = dict(location='nigeria', start=1990, stop=2060, dt=0.25, verbose=0,
            total_pop=TOTAL_POP)


def _az(sim):
    for a in sim.analyzers.values():
        if isinstance(a, CancerScaleAnalyzer):
            return a
    raise RuntimeError(f'analyzer not found in {list(sim.analyzers.keys())}')


def make_single(n_agents, seed):
    return hpv.Sim(n_agents=n_agents, diseases=[hpv.HPV(genotype='hpv16')],
                   rand_seed=seed, analyzers=[CancerScaleAnalyzer()], **BASE)


def make_multi(n_agents, seed, ms_ratio):
    return hpv.Sim(n_agents=n_agents,
                   diseases=[MultiscaleHPV(genotype='hpv16', ms_ratio=ms_ratio)],
                   rand_seed=seed, analyzers=[CancerScaleAnalyzer()], **BASE)


def cancers_of(sim):
    az = _az(sim)
    return dict(scaled=az.scaled_new_cancers, raw=az.raw_new_cancers,
                people=az.total_cancers_people, n_agents_final=len(sim.people),
                pop_scale=float(sim.pars.pop_scale))


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def run():
    print('=' * 78)
    print('SPIKE: multiscale feasibility on v3/Starsim HPVsim')
    print('=' * 78)

    print('\n[R1] grow()+state-copy integrity (multiscale, n=3000, ratio=10)')
    s1 = make_multi(3000, seed=0, ms_ratio=10); s1.run()
    ppl = s1.people
    lengths = dict(scale=len(ppl.scale.raw), age=len(ppl.age.raw),
                   cancerous=len(s1.diseases.hpv16.cancerous.raw))
    print(f'   raw array lengths (should all match): {lengths}')
    n_nan = int(np.isnan(np.asarray(ppl.age[ppl.auids])).sum())
    print(f'   NaNs in active ages: {n_nan}')
    c1 = cancers_of(s1)
    print(f'   final n_agents(active)={c1["n_agents_final"]}, pop_scale={c1["pop_scale"]:.3g}')
    print(f'   scaled cancers(agent-equiv)={c1["scaled"]:.1f}  raw={c1["raw"]}  '
          f'people={c1["people"]:.0f}')

    print('\n[R3] reproducibility: same seed twice -> identical?')
    s1b = make_multi(3000, seed=0, ms_ratio=10); s1b.run()
    c1b = cancers_of(s1b)
    same = np.isclose(c1['scaled'], c1b['scaled']) and c1['raw'] == c1b['raw']
    print(f'   run A scaled={c1["scaled"]:.6f} raw={c1["raw"]}')
    print(f'   run B scaled={c1b["scaled"]:.6f} raw={c1b["raw"]}')
    print(f'   IDENTICAL: {same}')

    def stat(x):
        return f'mean={x.mean():.0f}  std={x.std(ddof=1):.0f}  CV={x.std(ddof=1)/x.mean():.3f}'

    print('\n[EQ] internal equivalence + controls (cancers in people-space)')
    seeds = range(6)
    configs = {
        'single  N=20000        ': lambda sd: make_single(20000, sd),
        'multi   N=20000 ratio=1 ': lambda sd: make_multi(20000, sd, 1),
        'multi   N=3000  ratio=1 ': lambda sd: make_multi(3000, sd, 1),
        'multi   N=3000  ratio=10': lambda sd: make_multi(3000, sd, 10),
    }
    base = None
    for label, mk in configs.items():
        vals = np.array([cancers_of(_run(mk(sd)))['people'] for sd in seeds])
        if base is None:
            base = vals.mean()
        bias = (vals.mean() - base) / base
        print(f'   {label}: {stat(vals)}  bias_vs_single={bias:+.1%}')

    print('\n' + '=' * 78 + '\nDONE\n' + '=' * 78)


def _run(sim):
    sim.run()
    return sim


if __name__ == '__main__':
    run()

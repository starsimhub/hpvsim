"""Verify the v3 ported Rwanda network reproduces the DHS/v2 partnership pattern,
and connect it to old-age HPV acquisition (the HIV- 55+ cancer driver).

For each 5-yr female age band over 2010-2019 it reports:
  - realized marital / casual participation (fraction of alive women in a
    partnership) vs the INPUT annual layer_probs (the DHS-derived target).
  - HPV acquisition rate (new infections per 100 woman-years) -- is old-age
    acquisition really high, and does it track participation?

If realized participation ~ the input layer_probs, the network faithfully
implements the (DHS-anchored) v2 calibration -> old-age exposure is 'real', not
a v3 bug. If realized >> input, v3 over-partners older women (a fixable bug).

Run: .venv/Scripts/python.exe tests/regression/diag_network_oldage.py [n_seeds=4] [n_agents=10000]
"""
import sys
from pathlib import Path

import numpy as np
import starsim as ss

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hpvsim.hpv import HPV  # noqa: E402
from tests.regression import rwanda_calib as rc  # noqa: E402
from tests.regression.rwanda_calib import build_rwanda_sim  # noqa: E402

_EDGES = list(range(15, 80, 5)) + [200]      # 15-20,...,70-75,75+
_LABELS = [f'{lo}' for lo in _EDGES[:-1]]
_WIN = (2010, 2019)


def _input_participation():
    """Female annual participation per _EDGES band from the input layer_probs."""
    lp = rc._layer_probs_annual()
    out = {}
    for key in ('m', 'c'):
        ages = lp[key][0]; fpart = lp[key][1]      # row0=age bounds, row1=female
        out[key] = [float(np.interp(lo, ages, fpart)) for lo in _EDGES[:-1]]
    return out


class _Probe(ss.Analyzer):
    def init_pre(self, sim):
        self.hpv = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        super().init_pre(sim)
        n = len(sim.t.timevec); nb = len(_LABELS)
        self.yr = np.floor(np.asarray(sim.t.timevec, float)).astype(int)[:n]
        self.alive = np.zeros((nb, n)); self.mar = np.zeros((nb, n))
        self.cas = np.zeros((nb, n)); self.newinf = np.zeros((nb, n))

    def step(self):
        ti = self.sim.ti; ppl = self.sim.people
        auids = np.asarray(ppl.auids)
        # Exclude fine multiscale agents: they are grown at the cancer decision
        # and excluded from the sexual network by design, so they contribute 0
        # to participation/acquisition but would inflate the alive denominator
        # in the cancer-age bands this diagnostic studies (ratio=5 default).
        fine = (ppl.fine.values if 'fine' in ppl.states
                else np.zeros(ppl.female.values.shape, bool))
        fem = ppl.female.values & ppl.alive.values & ~fine
        age = ppl.age.values
        e = self.sim.networks['sexualnetwork'].edges
        lid = np.asarray(e.layer_id); p1 = np.asarray(e.p1); p2 = np.asarray(e.p2)
        part = {}
        for key, val in (('m', 0), ('c', 1)):
            eps = np.concatenate([p1[lid == val], p2[lid == val]])
            part[key] = np.isin(auids, eps)
        newinf = np.zeros(fem.shape, bool)
        for m in self.hpv:
            newinf |= (m.infected.values & (m.ti_infected.values == ti))
        for bi, (lo, hi) in enumerate(zip(_EDGES[:-1], _EDGES[1:])):
            b = fem & (age >= lo) & (age < hi)
            self.alive[bi, ti] = b.sum()
            self.mar[bi, ti] = (b & part['m']).sum()
            self.cas[bi, ti] = (b & part['c']).sum()
            self.newinf[bi, ti] = (b & newinf).sum()


def main(n_seeds=4, n_agents=10000):
    agg = {k: np.zeros(len(_LABELS)) for k in ('alive', 'mar', 'cas', 'newinf')}
    for s in range(n_seeds):
        sim = build_rwanda_sim(seed=s, n_agents=n_agents, start=1960, stop=2020)
        pr = _Probe()
        sim.pars['analyzers'] = list(sim.pars.get('analyzers') or []) + [pr]
        sim.run(verbose=0)
        a = [x for x in sim.analyzers.values() if isinstance(x, _Probe)][0]
        w = (a.yr >= _WIN[0]) & (a.yr <= _WIN[1])
        for k in agg:
            agg[k] += getattr(a, k)[:, w].sum(axis=1)

    inp = _input_participation()
    print(f'\nv3 ported Rwanda network: realized vs INPUT (DHS) female participation, '
          f'+ HPV acquisition, ages 15-75+ ({n_seeds} seeds, n={n_agents}, {_WIN}).')
    print(f'{"age":>4} | {"marital realized/input":>24} | {"casual realized/input":>22} '
          f'| {"HPV acq /100wy":>14}')
    dt = rc._DT
    for bi, lab in enumerate(_LABELS):
        al = agg['alive'][bi]
        if al == 0:
            continue
        mr = agg['mar'][bi] / al; cr = agg['cas'][bi] / al
        acq = agg['newinf'][bi] / (al * dt) * 100   # per 100 woman-years
        print(f'{lab:>4} | {mr:>10.2f} / {inp["m"][bi]:<10.2f} | '
              f'{cr:>9.2f} / {inp["c"][bi]:<9.2f} | {acq:>12.1f}')
    print('\nRead: if realized ~ input across ages (esp 50-75), the network faithfully '
          'implements the DHS calibration. If HPV acquisition stays high at 55+, that is '
          'the old-age exposure feeding the HIV- a55 cancer overshoot.')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(n_seeds=int(a[0]) if len(a) > 0 else 4,
         n_agents=int(a[1]) if len(a) > 1 else 10000)
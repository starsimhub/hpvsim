"""Quick probe: print per-step partnership counts and per-step new infections
for the v3 4-genotype anchor sim, to compare against v2's network statistics.

Run from a v3 env at the repo root:

    python tests/regression/probe_partnerships.py
"""

import sys
from pathlib import Path

import numpy as np
import sciris as sc
import starsim as ss

import hpvsim as hpv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_4genotype import PARS  # noqa: E402


class NetProbe(ss.Analyzer):
    """Capture per-step partnership counts (per layer) and partnership acts
    summary statistics; also captures per-step new infections summed across
    all genotypes.
    """

    def __init__(self):
        super().__init__()
        self.rows = []

    def step(self):
        sim = self.sim
        ti = int(sim.ti)
        try:
            year = float(sim.t.timevec[ti])
        except Exception:
            year = float(ti)
        net = sim.networks['sexualnetwork']
        edges = net.edges

        row = {'ti': ti, 'year': year, 'n_edges_total': len(edges.p1)}

        for lkey in net.layers:
            mask = net.edges_for_layer(lkey)
            n = int(mask.sum())
            row[f'{lkey}_n_pairs'] = n
            if n:
                acts = np.asarray(edges.acts)[mask]
                row[f'{lkey}_acts_mean'] = float(acts.mean())
                row[f'{lkey}_acts_p50'] = float(np.percentile(acts, 50))
                row[f'{lkey}_acts_p90'] = float(np.percentile(acts, 90))
            else:
                row[f'{lkey}_acts_mean'] = 0.0
                row[f'{lkey}_acts_p50'] = 0.0
                row[f'{lkey}_acts_p90'] = 0.0

        # Sum per-step new infections across genotypes.
        new_inf_total = 0
        for g in ('hpv16', 'hpv18', 'hi5', 'ohr'):
            if g in sim.diseases:
                new_inf_total += int(sim.diseases[g].results.new_infections[ti])
        row['new_inf_total'] = new_inf_total

        self.rows.append(row)


def main():
    pars = sc.dcp(PARS)
    probe = NetProbe()
    sim = hpv.Sim(analyzers=[probe], copy_inputs=False, **pars)
    sim.run()

    rows = probe.rows
    # Print sample years.
    print(f'{"year":>6}  {"m_pairs":>8}  {"c_pairs":>8}  {"m_acts":>8}  {"c_acts":>8}  {"new_inf":>8}')
    for r in rows:
        if int(r['year']) in (1990, 1991, 1992, 1995, 2000, 2010, 2030, 2059):
            if abs(r['year'] - int(r['year'])) < 0.1:
                print(f'{r["year"]:>6.1f}  {r["m_n_pairs"]:>8}  {r["c_n_pairs"]:>8}  '
                      f'{r["m_acts_mean"]:>8.2f}  {r["c_acts_mean"]:>8.2f}  '
                      f'{r["new_inf_total"]:>8}')


if __name__ == '__main__':
    main()
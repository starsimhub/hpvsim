"""v2 mirror of probe_partnerships.py — partnership counts and acts stats.

Run with the v2.3 env from anywhere; the script knows where to import from.
"""

import sys
from pathlib import Path

import numpy as np
import sciris as sc
import hpvsim as hpv
import hpvsim.analysis as hpa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_v23 import PARS_4GENOTYPE  # noqa: E402


class NetProbe(hpa.Analyzer):
    def __init__(self, label=None):
        super().__init__(label=label)
        self.rows = []

    def apply(self, sim):
        people = sim.people
        try:
            year = float(sim.yearvec[sim.t])
        except Exception:
            year = float(sim.t)

        row = {'ti': int(sim.t), 'year': year}

        # v2's contacts are dict[layer_name] -> {f, m, acts, dur, ...}
        for lkey in people.contacts:
            layer = people.contacts[lkey]
            n = len(layer['f'])
            # Per-step acts = stored acts × dt.
            acts_per_step = np.asarray(layer['acts']) * float(sim.pars['dt'])
            row[f'{lkey}_n_pairs'] = int(n)
            if n:
                row[f'{lkey}_acts_mean'] = float(acts_per_step.mean())
                row[f'{lkey}_acts_p50'] = float(np.percentile(acts_per_step, 50))
                row[f'{lkey}_acts_p90'] = float(np.percentile(acts_per_step, 90))
            else:
                row[f'{lkey}_acts_mean'] = 0.0
                row[f'{lkey}_acts_p50'] = 0.0
                row[f'{lkey}_acts_p90'] = 0.0

        # Per-step new infections summed across genotypes.
        if 'infections_by_genotype' in sim.results:
            inf_bg = np.asarray(sim.results['infections_by_genotype'])  # (n_g, n_t)
            t = min(int(sim.t), inf_bg.shape[1] - 1)
            row['new_inf_total'] = int(inf_bg[:, t].sum())
        else:
            row['new_inf_total'] = 0

        self.rows.append(row)


def main():
    pars = sc.dcp(PARS_4GENOTYPE)
    probe = NetProbe(label='NetProbe')
    sim = hpv.Sim(pars=pars, analyzers=probe)
    sim.run()
    ran = sim.get_analyzer('NetProbe')

    print(f'{"year":>6}  {"m_pairs":>8}  {"c_pairs":>8}  {"m_acts":>8}  {"c_acts":>8}  {"new_inf":>8}')
    for r in ran.rows:
        if abs(r['year'] - int(r['year'])) < 0.1 and int(r['year']) in (1990, 1991, 1992, 1995, 2000, 2010, 2030, 2059):
            print(f'{r["year"]:>6.1f}  {r.get("m_n_pairs", 0):>8}  {r.get("c_n_pairs", 0):>8}  '
                  f'{r.get("m_acts_mean", 0):>8.2f}  {r.get("c_acts_mean", 0):>8.2f}  '
                  f'{r["new_inf_total"]:>8}')


if __name__ == '__main__':
    main()
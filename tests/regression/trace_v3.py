"""Per-step instrumented trace of the v3 4-genotype anchor sim.

Captures, for every timestep:
  * alive / alive-female counts
  * per-genotype: n_inf, n_inf_f, n_sus_f, n_immune_f (nab_imm > 0),
    nab_imm distribution among alive females (mean / mean of positives /
    p50/p75/p90/p95/p99), mean rel_sus and sev_imm among alive females,
    and new infections this step split into first vs reinfection
    (using ti_first_infection == ti as the first-infection flag).

Run from a v3 env at the repo root:

    python tests/regression/trace_v3.py
    python tests/regression/trace_v3.py --out tests/regression/v3_trace.csv

Mirrors the metric set captured by ``trace_v2.py`` so the two CSVs can be
diffed directly.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import sciris as sc
import starsim as ss

import hpvsim as hpv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_4genotype import PARS  # noqa: E402

GENOTYPE_KEYS = ('hpv16', 'hpv18', 'hi5', 'ohr')

DEFAULT_OUT = Path(__file__).resolve().parent / 'v3_trace.csv'


class HPVTrace(ss.Analyzer):
    """Capture per-step nab_imm distribution + reinfection counts."""

    def __init__(self, genotype_keys):
        super().__init__()
        self.genotype_keys = tuple(genotype_keys)
        self.rows = []

    def step(self):
        sim = self.sim
        ti = int(sim.ti)
        people = sim.people
        # raw to get UID-aligned full-size arrays (np.asarray on FloatArr/BoolState
        # returns the active-uid slice, which has a different length than .raw).
        female = np.asarray(people.female.raw).astype(bool)
        alive = np.asarray(people.alive.raw).astype(bool)

        try:
            year = float(sim.t.timevec[ti])
        except Exception:
            year = float(ti)

        row = {
            'ti': ti,
            'year': year,
            'n_alive': int(alive.sum()),
            'n_alive_f': int((alive & female).sum()),
        }

        for g in self.genotype_keys:
            mod = sim.diseases[g]

            # Use ``raw`` (UID-indexed, full-size) for cross-state masking.
            # ``values`` would only return active-uid slices and we want to
            # combine flags from multiple FloatArr/BoolState objects.
            inf_arr = np.asarray(mod.infected.raw).astype(bool)
            sus_arr = np.asarray(mod.susceptible.raw).astype(bool)
            nab_raw = np.asarray(mod.nab_imm.raw)
            cell_raw = np.asarray(mod.cell_imm.raw)
            rel_sus_raw = np.asarray(mod.rel_sus.raw)
            sev_imm_raw = np.asarray(mod.sev_imm.raw)
            ti_inf = np.asarray(mod.ti_infected.raw)
            ti_first = np.asarray(mod.ti_first_infection.raw)

            alive_f = alive & female
            n_alive_f = int(alive_f.sum())

            n_inf = int((alive & inf_arr).sum())
            n_inf_f = int((alive_f & inf_arr).sum())
            n_sus_f = int((alive_f & sus_arr).sum())

            nab_f = nab_raw[alive_f]
            cell_f = cell_raw[alive_f]
            rel_sus_f = rel_sus_raw[alive_f]
            sev_imm_f = sev_imm_raw[alive_f]

            n_immune_f = int((nab_f > 0).sum())

            if len(nab_f):
                qs = np.percentile(nab_f, [50, 75, 90, 95, 99])
                pos = nab_f[nab_f > 0]
                mean_pos = float(pos.mean()) if len(pos) else 0.0
                mean_nab_f = float(nab_f.mean())
                mean_cell_f = float(cell_f.mean())
                mean_rel_sus_f = float(rel_sus_f.mean())
                mean_sev_imm_f = float(sev_imm_f.mean())
            else:
                qs = [0.0] * 5
                mean_pos = 0.0
                mean_nab_f = 0.0
                mean_cell_f = 0.0
                mean_rel_sus_f = 0.0
                mean_sev_imm_f = 0.0

            new_mask = (ti_inf == ti) & alive
            # nan-safe: treat missing ti_first as "not first this step"
            first_mask = new_mask & (ti_first == ti)
            re_mask = new_mask & ~first_mask

            row.update({
                f'{g}.n_inf':              n_inf,
                f'{g}.n_inf_f':            n_inf_f,
                f'{g}.n_sus_f':            n_sus_f,
                f'{g}.n_immune_f':         n_immune_f,
                f'{g}.mean_nab_f':         mean_nab_f,
                f'{g}.mean_nab_f_pos':     mean_pos,
                f'{g}.nab_p50':            float(qs[0]),
                f'{g}.nab_p75':            float(qs[1]),
                f'{g}.nab_p90':            float(qs[2]),
                f'{g}.nab_p95':            float(qs[3]),
                f'{g}.nab_p99':            float(qs[4]),
                f'{g}.mean_cell_f':        mean_cell_f,
                f'{g}.mean_rel_sus_f':     mean_rel_sus_f,
                f'{g}.mean_sev_imm_f':     mean_sev_imm_f,
                f'{g}.new_inf_this_step':  int(new_mask.sum()),
                f'{g}.first_inf_this_step': int(first_mask.sum()),
                f'{g}.re_inf_this_step':   int(re_mask.sum()),
            })

        self.rows.append(row)


def write_csv(rows, path):
    if not rows:
        print(f'No rows captured; skipping write to {path}.')
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {len(rows)} rows to {path}')


def run_trace(out_path):
    pars = sc.dcp(PARS)
    trace = HPVTrace(GENOTYPE_KEYS)
    aggregator = hpv.sim.Aggregate()
    # copy_inputs=False so we can read trace.rows directly after run
    sim = hpv.Sim(analyzers=[aggregator, trace], copy_inputs=False, **pars)
    sim.run()
    write_csv(trace.rows, out_path)
    return sim, trace


def main(argv=None):
    p = argparse.ArgumentParser(description='Trace v3 4-genotype anchor sim.')
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    run_trace(args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
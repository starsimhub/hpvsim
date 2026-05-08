"""Per-step instrumented trace of the v2.3 4-genotype anchor sim.

Mirrors ``trace_v3.py``: same column layout, same metric semantics, so the
two CSVs can be diffed directly.

Differences vs v3:
  * Tracks ``ever_infected[g, person]`` inside the analyzer (v2 has no
    per-agent ti_first_infection); reinfection vs first-infection split is
    derived from a per-step transition snapshot.
  * Reads ``people.peak_imm`` for the running-max of post-clearance samples
    (v2's source-of-truth) AND ``people.nab_imm`` for current immunity
    (peak_imm with optional waning applied; identical to peak_imm when
    ``use_waning=False``). v3's ``nab_imm`` is the running max — it is
    semantically equivalent to v2's ``peak_imm``, not v2's ``nab_imm``.
    We dump BOTH columns in the v2 trace so the comparison can use either.
  * mean_rel_sus_f reported as 1 - sus_imm[g] mean across alive females,
    matching v3's CrossImmunity Connector output.

Run from a v2.3 env at the repo root:

    python tests/regression/trace_v2.py
    python tests/regression/trace_v2.py --out tests/regression/v2_trace.csv

DO NOT run inside the v3 env.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import sciris as sc
import hpvsim as hpv
import hpvsim.analysis as hpa  # v2's Analyzer base class

# Pull the same PARS_4GENOTYPE the baseline regenerator uses, so the trace
# runs on a config equivalent to the v3 anchor.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_v23 import PARS_4GENOTYPE  # noqa: E402

GENOTYPE_KEYS = ('hpv16', 'hpv18', 'hi5', 'ohr')

DEFAULT_OUT = Path(__file__).resolve().parent / 'v2_trace.csv'


class HPVTrace(hpa.Analyzer):
    """Per-step trace of nab_imm distribution + reinfection split."""

    def __init__(self, genotype_keys, label=None):
        super().__init__(label=label)
        self.genotype_keys = tuple(genotype_keys)
        self.rows = []
        # Boolean arrays sized to current population. Grow lazily.
        self.ever_infected = None        # shape (n_g, n_people)
        self.prev_infectious = None      # shape (n_g, n_people)

    def initialize(self, sim=None):
        super().initialize(sim)
        return

    def _grow_to(self, n, n_g):
        """Grow tracking arrays to accommodate ``n`` agents."""
        if self.ever_infected is None:
            self.ever_infected = np.zeros((n_g, n), dtype=bool)
            self.prev_infectious = np.zeros((n_g, n), dtype=bool)
            return
        cur = self.ever_infected.shape[1]
        if n > cur:
            ext_e = np.zeros((n_g, n), dtype=bool)
            ext_e[:, :cur] = self.ever_infected
            self.ever_infected = ext_e
            ext_p = np.zeros((n_g, n), dtype=bool)
            ext_p[:, :cur] = self.prev_infectious
            self.prev_infectious = ext_p

    def apply(self, sim):
        people = sim.people
        n = len(people)
        n_g = sim['n_genotypes']
        self._grow_to(n, n_g)

        alive = np.asarray(people.alive).astype(bool)
        female = np.asarray(people.is_female).astype(bool)
        alive_f = alive & female

        infectious = np.asarray(people.infectious).astype(bool)   # (n_g, n)
        susceptible = np.asarray(people.susceptible).astype(bool)
        peak_imm = np.asarray(people.peak_imm)                     # (n_g, n)
        nab_imm = np.asarray(people.nab_imm)                       # (n_g, n)
        cell_imm = np.asarray(people.cell_imm)
        sus_imm = np.asarray(people.sus_imm)
        sev_imm = np.asarray(people.sev_imm)

        # Per-step new infection events: ``date_infectious[g, person] == sim.t``.
        # date_infectious is overwritten on every new infection event (v2's
        # _v2_legacy/people.py:983), so this catches *all* new infections this
        # step, including same-step clear-and-reinfect events that a
        # transition snapshot (curr_inf & ~prev_inf) would miss.
        date_inf = np.asarray(people.date_infectious)               # (n_g, n)
        became = (date_inf == int(sim.t)) & alive[None, :]

        try:
            year = float(sim.yearvec[sim.t])
        except Exception:
            year = float(sim.t)

        row = {
            'ti': int(sim.t),
            'year': year,
            'n_alive': int(alive.sum()),
            'n_alive_f': int(alive_f.sum()),
        }

        gen_map = sim.pars.get('genotype_map', {})
        # Map each requested key to its row in the per-genotype arrays.
        # gen_map is {idx: 'hpv16', ...}; reverse it once.
        idx_by_key = {v: k for k, v in gen_map.items()}

        for g_key in self.genotype_keys:
            if g_key not in idx_by_key:
                # genotype not in this sim — fill zeros so column layout matches v3
                row.update({f'{g_key}.{k}': 0
                            for k in (
                                'n_inf', 'n_inf_f', 'n_sus_f', 'n_immune_f',
                                'mean_nab_f', 'mean_nab_f_pos',
                                'nab_p50', 'nab_p75', 'nab_p90', 'nab_p95', 'nab_p99',
                                'mean_cell_f', 'mean_rel_sus_f', 'mean_sev_imm_f',
                                'new_inf_this_step', 'first_inf_this_step',
                                're_inf_this_step',
                                # extras
                                'mean_peak_imm_f', 'mean_peak_imm_f_pos',
                            )})
                continue
            g = idx_by_key[g_key]

            inf_g = infectious[g]
            sus_g = susceptible[g]
            # Per-target rel_sus = 1 - sus_imm (matches v3's Connector output).
            rel_sus_g = 1.0 - sus_imm[g]
            sev_imm_g = sev_imm[g]

            # v3's nab_imm = running max of imm_init samples (no waning yet); v2's
            # equivalent is peak_imm. Report both so we can pick the right one.
            peak_g = peak_imm[g]
            nab_g = nab_imm[g]
            cell_g = cell_imm[g]

            # Slice to alive females.
            peak_f = peak_g[alive_f]
            nab_f = nab_g[alive_f]
            cell_f = cell_g[alive_f]
            rel_sus_f = rel_sus_g[alive_f]
            sev_imm_f = sev_imm_g[alive_f]

            n_inf = int((alive & inf_g).sum())
            n_inf_f = int((alive_f & inf_g).sum())
            n_sus_f = int((alive_f & sus_g).sum())
            n_immune_f = int((peak_f > 0).sum())  # ever-cleared+seroconverted

            if len(peak_f):
                qs = np.percentile(peak_f, [50, 75, 90, 95, 99])
                pos = peak_f[peak_f > 0]
                mean_pos = float(pos.mean()) if len(pos) else 0.0
                mean_peak_f = float(peak_f.mean())
                mean_nab_f = float(nab_f.mean())
                nab_pos = nab_f[nab_f > 0]
                mean_nab_pos = float(nab_pos.mean()) if len(nab_pos) else 0.0
                mean_cell_f = float(cell_f.mean())
                mean_rel_sus_f = float(rel_sus_f.mean())
                mean_sev_imm_f = float(sev_imm_f.mean())
            else:
                qs = [0.0] * 5
                mean_pos = 0.0
                mean_peak_f = 0.0
                mean_nab_f = 0.0
                mean_nab_pos = 0.0
                mean_cell_f = 0.0
                mean_rel_sus_f = 0.0
                mean_sev_imm_f = 0.0

            # Reinfection split via ever_infected snapshot.
            new_mask = became[g] & alive
            first_mask = new_mask & ~self.ever_infected[g, :n]
            re_mask = new_mask & self.ever_infected[g, :n]

            row.update({
                f'{g_key}.n_inf':              n_inf,
                f'{g_key}.n_inf_f':            n_inf_f,
                f'{g_key}.n_sus_f':            n_sus_f,
                f'{g_key}.n_immune_f':         n_immune_f,
                # default *_nab_f columns mirror v3's nab_imm semantics ==
                # "running-max of sampled per-clearance immunity" == v2's peak_imm.
                f'{g_key}.mean_nab_f':         mean_peak_f,
                f'{g_key}.mean_nab_f_pos':     mean_pos,
                f'{g_key}.nab_p50':            float(qs[0]),
                f'{g_key}.nab_p75':            float(qs[1]),
                f'{g_key}.nab_p90':            float(qs[2]),
                f'{g_key}.nab_p95':            float(qs[3]),
                f'{g_key}.nab_p99':            float(qs[4]),
                f'{g_key}.mean_cell_f':        mean_cell_f,
                f'{g_key}.mean_rel_sus_f':     mean_rel_sus_f,
                f'{g_key}.mean_sev_imm_f':     mean_sev_imm_f,
                f'{g_key}.new_inf_this_step':  int(new_mask.sum()),
                f'{g_key}.first_inf_this_step': int(first_mask.sum()),
                f'{g_key}.re_inf_this_step':   int(re_mask.sum()),
                # Extras unique to v2 trace (after-waning current immunity).
                f'{g_key}.mean_peak_imm_f':    mean_peak_f,
                f'{g_key}.mean_peak_imm_f_pos': mean_pos,
            })

            # Update ever_infected with current infectious (so any agent
            # currently infectious will count as ever-infected next step).
            self.ever_infected[g, :n] |= inf_g

        # Snapshot for next step's transition detection.
        self.prev_infectious[:, :n] = infectious

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
    pars = sc.dcp(PARS_4GENOTYPE)
    trace = HPVTrace(GENOTYPE_KEYS, label='HPVTrace')
    sim = hpv.Sim(pars=pars, analyzers=trace)
    sim.run()
    # v2 deepcopies analyzers into sim.analyzers; fetch the one that ran.
    ran = sim.get_analyzer('HPVTrace')
    write_csv(ran.rows, out_path)
    return sim, ran


def main(argv=None):
    p = argparse.ArgumentParser(description='Trace v2.3 4-genotype anchor sim.')
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    run_trace(args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
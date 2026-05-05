"""v3 per-step trace + side-by-side comparison vs v2.

Captures detailed per-step state and per-call compute_severity statistics,
writes ``v3_trace.csv`` to the project root, and prints aggregate stats.
Pair with ``hpvsim_v23_frozen/trace_v2.py`` (writes ``v2_trace.csv``) for
matching v2 numbers, then pass ``--compare`` to see them side-by-side.

Per-step (annual snapshot) columns captured:
  year                    -- year of snapshot
  n_alive                 -- total alive population
  n_susceptible_f         -- alive females who are HPV-susceptible
  n_infectious            -- alive HPV-infected (precin OR cin compartments)
  n_precin                -- alive in precin compartment
  n_cin_active            -- alive in CIN compartment
  n_cancerous             -- alive in cancerous compartment
  n_immune_f              -- alive females with rel_sus < 1 OR sev_imm > 0
                             (post-clearance immune set)
  cum_infections          -- cumulative new infection events
  cum_cin_scheduled       -- cumulative ti_cin SET events (progression to
                             CIN scheduled)
  cum_cancer_scheduled    -- cumulative ti_cancerous SET events
  mean_rel_sus_f          -- mean rel_sus across alive females
  mean_sev_imm_f          -- mean sev_imm across alive females
  hpv_prev                -- prevalence (n_infectious / n_alive)

Run:
    python tests/regression/trace_v3.py [--compare]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Add project root so the relative import below works regardless of cwd.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import hpvsim as hpv
import hpvsim.hpv as hpv_mod
import starsim as ss
from tests.regression.anchor_hpv16 import PARS


# ---------------------------------------------------------------------------- #
# State capture                                                                #
# ---------------------------------------------------------------------------- #


class _StateAnalyzer(ss.Analyzer):
    """Captures per-step disease state at the END of each step. Annual
    snapshots are extracted post-run from the per-step buffer.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.snapshots = []  # one dict per timestep

    def step(self):
        sim = self.sim
        people = sim.people
        mod = sim.diseases.hpv16

        alive = np.asarray(people.alive).astype(bool)
        female = np.asarray(people.female).astype(bool)

        # Compartment booleans (UID-aware via Arr)
        susceptible = np.asarray(mod.susceptible).astype(bool)
        precin = np.asarray(mod.precin).astype(bool)
        cin = np.asarray(mod.cin).astype(bool)
        cancerous = np.asarray(mod.cancerous).astype(bool)
        infected = np.asarray(mod.infected).astype(bool)

        rel_sus = np.asarray(mod.rel_sus)
        sev_imm = np.asarray(mod.sev_imm)

        f_alive = alive & female
        n_alive = int(alive.sum())
        n_sus_f = int((susceptible & f_alive).sum())
        n_precin = int((precin & alive).sum())
        n_cin = int((cin & alive).sum())
        n_cancer = int((cancerous & alive).sum())
        n_inf = int((infected & alive).sum())
        # Immune: females with rel_sus < ~1 (post-clearance immunity reduced).
        # Tolerance for floating compare.
        n_immune_f = int((f_alive & ((rel_sus < 0.999) | (sev_imm > 0.001))).sum())

        rel_sus_mean = float(rel_sus[f_alive].mean()) if n_alive else 0.0
        sev_imm_mean = float(sev_imm[f_alive].mean()) if n_alive else 0.0

        self.snapshots.append({
            'ti': float(sim.t.ti),
            'year': float(sim.t.now('year')),
            'n_alive': n_alive,
            'n_susceptible_f': n_sus_f,
            'n_infectious': n_inf,
            'n_precin': n_precin,
            'n_cin_active': n_cin,
            'n_cancerous': n_cancer,
            'n_immune_f': n_immune_f,
            'mean_rel_sus_f': rel_sus_mean,
            'mean_sev_imm_f': sev_imm_mean,
        })


def _wrap_compute_severity():
    """Monkey-patch _compute_severity to log per-call stats."""
    log = []
    orig = hpv_mod._compute_severity

    def traced(t, rel_sev=None, pars=None):
        out = orig(t, rel_sev=rel_sev, pars=pars)
        is_cancer = pars is not None and pars.get('method') == 'cin_integral'
        log.append({
            'kind': 'cancer' if is_cancer else 'cin',
            'n': len(t) if hasattr(t, '__len__') else 1,
            'mean_t': float(np.mean(t)),
            'mean_rs': float(np.mean(rel_sev)) if rel_sev is not None else 1.0,
            'mean_p': float(np.mean(out)),
        })
        return out

    hpv_mod._compute_severity = traced
    return log, orig


def _wrap_set_prognoses():
    """Monkey-patch HPV.set_prognoses to count CIN-scheduled / cancer-scheduled
    transitions. Each call increments cumulative counters by the number of
    ti_cin / ti_cancerous values that newly become finite.
    """
    counters = dict(cum_inf=0, cum_cin=0, cum_cancer=0)
    orig = hpv_mod.HPV.set_prognoses

    def traced(self, uids, sources=None):
        # cum_inf increments by the number of UIDs entering set_prognoses.
        counters['cum_inf'] += len(uids)
        # Count finite ti_cin / ti_cancerous BEFORE — note set_prognoses
        # resets these to NaN at the top, then sets them for selected uids.
        # So a clean way to count newly-scheduled is: count finite ti_cin
        # for THESE UIDs after the call (already finite or just set).
        out = orig(self, uids, sources)
        ti_cin_after = np.asarray(self.ti_cin[uids])
        ti_cancer_after = np.asarray(self.ti_cancerous[uids])
        counters['cum_cin'] += int(np.isfinite(ti_cin_after).sum())
        counters['cum_cancer'] += int(np.isfinite(ti_cancer_after).sum())
        return out

    hpv_mod.HPV.set_prognoses = traced
    return counters, orig


# ---------------------------------------------------------------------------- #
# Aggregation helpers                                                          #
# ---------------------------------------------------------------------------- #


def _aggregate(calls, key):
    """Length-weighted mean of `key` across `calls`."""
    n = sum(c['n'] for c in calls)
    if n == 0:
        return 0.0
    return sum(c[key] * c['n'] for c in calls) / n


# ---------------------------------------------------------------------------- #
# Run                                                                          #
# ---------------------------------------------------------------------------- #


def run_v3_trace(out_path='v3_trace.csv'):
    """Run the M02 anchor sim with all hooks; write per-year metrics to CSV."""
    cs_log, orig_cs = _wrap_compute_severity()
    sp_counters, orig_sp = _wrap_set_prognoses()

    # Subclass that snapshots cumulative counters (tracked outside the analyzer
    # so we can read them after starsim deep-copies the analyzer instance).
    class _Analyzer(_StateAnalyzer):
        def step(self):
            super().step()
            self.snapshots[-1].update({
                'cum_infections': sp_counters['cum_inf'],
                'cum_cin_scheduled': sp_counters['cum_cin'],
                'cum_cancer_scheduled': sp_counters['cum_cancer'],
            })

    try:
        sim = hpv.Sim(**PARS, analyzers=[_Analyzer()])
        sim.run()
    finally:
        hpv_mod._compute_severity = orig_cs
        hpv_mod.HPV.set_prognoses = orig_sp

    # Retrieve the actual analyzer from sim (it's a deep copy of what we passed in).
    analyzer = next(a for a in sim.analyzers() if isinstance(a, _StateAnalyzer))
    snapshots = analyzer.snapshots
    res = sim.results.hpv16
    n_total = len(snapshots)
    qi_per_year = np.linspace(0, n_total - 1, 71).astype(int)

    rows = []
    for i, yr in enumerate(np.linspace(1990, 2060, 71)):
        qi = qi_per_year[i]
        s = snapshots[qi]
        rows.append({
            'year': float(yr),
            'n_alive': s['n_alive'],
            'n_susceptible_f': s['n_susceptible_f'],
            'n_infectious': s['n_infectious'],
            'n_precin': s['n_precin'],
            'n_cin_active': s['n_cin_active'],
            'n_cancerous': s['n_cancerous'],
            'n_immune_f': s['n_immune_f'],
            'cum_infections': s.get('cum_infections', 0),
            'cum_cin_scheduled': s.get('cum_cin_scheduled', 0),
            'cum_cancer_scheduled': s.get('cum_cancer_scheduled', 0),
            'mean_rel_sus_f': s['mean_rel_sus_f'],
            'mean_sev_imm_f': s['mean_sev_imm_f'],
            'hpv_prev': s['n_infectious'] / max(s['n_alive'], 1),
        })

    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    cin_calls = [c for c in cs_log if c['kind'] == 'cin']
    cancer_calls = [c for c in cs_log if c['kind'] == 'cancer']
    summary = dict(
        n_cin_calls=len(cin_calls),
        cin_total_n=sum(c['n'] for c in cin_calls),
        cin_mean_t=_aggregate(cin_calls, 'mean_t'),
        cin_mean_rel_sev=_aggregate(cin_calls, 'mean_rs'),
        cin_mean_prob=_aggregate(cin_calls, 'mean_p'),
        n_cancer_calls=len(cancer_calls),
        cancer_total_n=sum(c['n'] for c in cancer_calls),
        cancer_mean_t=_aggregate(cancer_calls, 'mean_t'),
        cancer_mean_rel_sev=_aggregate(cancer_calls, 'mean_rs'),
        cancer_mean_prob=_aggregate(cancer_calls, 'mean_p'),
        cum_infections=sp_counters['cum_inf'],
        cum_cin_scheduled=sp_counters['cum_cin'],
        cum_cancer_scheduled=sp_counters['cum_cancer'],
    )
    print(f'Wrote {out_path} with {len(rows)} annual rows')
    print('\nv3 compute_severity aggregates:')
    print(f'  cin: {summary["n_cin_calls"]} calls, total n={summary["cin_total_n"]}')
    print(f'    mean t={summary["cin_mean_t"]:.3f}y, rel_sev={summary["cin_mean_rel_sev"]:.3f}, '
          f'cin_prob={summary["cin_mean_prob"]*100:.2f}%')
    print(f'  cancer: {summary["n_cancer_calls"]} calls, total n={summary["cancer_total_n"]}')
    print(f'    mean t={summary["cancer_mean_t"]:.3f}y, rel_sev={summary["cancer_mean_rel_sev"]:.3f}, '
          f'cancer_prob={summary["cancer_mean_prob"]*100:.4f}%')
    print(f'\nv3 cumulative event counters:')
    print(f'  set_prognoses calls (cum_infections incl init): {summary["cum_infections"]}')
    print(f'  CIN events scheduled (ti_cin set):              {summary["cum_cin_scheduled"]}')
    print(f'  Cancer events scheduled (ti_cancerous set):     {summary["cum_cancer_scheduled"]}')
    return rows, summary


# ---------------------------------------------------------------------------- #
# Compare                                                                      #
# ---------------------------------------------------------------------------- #


def compare_traces(v2_path='v2_trace.csv', v3_path='v3_trace.csv'):
    def load(p):
        with open(p) as f:
            return list(csv.DictReader(f))
    v2 = load(v2_path)
    v3 = load(v3_path)

    keys = ['n_alive', 'n_infectious', 'hpv_prev']
    extra_v3_only = ['n_precin', 'n_cin_active', 'n_immune_f',
                     'cum_infections', 'cum_cin_scheduled',
                     'cum_cancer_scheduled', 'mean_rel_sus_f', 'mean_sev_imm_f']
    extra_in_v2 = [k for k in extra_v3_only if v2 and k in v2[0]]

    print(f'{"year":>6}', end='')
    for k in keys:
        print(f'  {"v2_"+k:>13} {"v3_"+k:>13} {"rel":>7}', end='')
    print()
    indices = [0, 5, 10, 20, 30, 40, 50, 60, 70]
    for i in indices:
        if i >= min(len(v2), len(v3)):
            continue
        yr = float(v2[i]['year'])
        print(f'{yr:>6.0f}', end='')
        for k in keys:
            v2v = float(v2[i][k])
            v3v = float(v3[i][k])
            rel = (v3v - v2v) / max(abs(v2v), 1e-9) * 100
            fmt = '13.3f' if 'prev' in k else '13.0f'
            print(f'  {v2v:{fmt}} {v3v:{fmt}} {rel:+6.1f}%', end='')
        print()

    if extra_in_v2:
        print('\nExtra metrics (v2-vs-v3):')
        print(f'{"year":>6}', end='')
        for k in extra_in_v2:
            print(f'  {"v2_"+k:>13} {"v3_"+k:>13}', end='')
        print()
        for i in indices:
            if i >= min(len(v2), len(v3)):
                continue
            yr = float(v2[i]['year'])
            print(f'{yr:>6.0f}', end='')
            for k in extra_in_v2:
                print(f'  {float(v2[i][k]):>13.0f} {float(v3[i][k]):>13.0f}', end='')
            print()
    else:
        print('\nv3-only metrics (v2_trace.csv missing these columns):')
        print(f'{"year":>6}', end='')
        for k in extra_v3_only:
            print(f'  {k:>20}', end='')
        print()
        for i in indices:
            if i >= len(v3):
                continue
            yr = float(v3[i]['year'])
            print(f'{yr:>6.0f}', end='')
            for k in extra_v3_only:
                v = float(v3[i][k])
                fmt = '20.4f' if 'mean' in k else '20.0f'
                print(f'  {v:{fmt}}', end='')
            print()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='v3_trace.csv')
    p.add_argument('--compare', action='store_true',
                   help='After tracing, load v2_trace.csv (must exist in cwd) '
                        'and print side-by-side comparison')
    args = p.parse_args()
    run_v3_trace(args.out)
    if args.compare:
        v2_path = 'v2_trace.csv'
        if not Path(v2_path).exists():
            print(f'\nNo {v2_path} found; skipping comparison.')
        else:
            print()
            compare_traces(v2_path, args.out)
# HPVsim v3.0 — M0 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the regression harness on `v3.0-dev` so every later milestone has a deterministic v2-vs-current drift report, an anchor scenario, and CI smoke coverage.

**Architecture:** Three small Python files under `tests/regression/` form the harness. `anchor.py` defines the pinned scenario; `baseline.py` is a CLI that snapshots a baseline JSON locally; `compare.py` is a CLI that diffs a current run against a baseline and prints per-summary-result drift. A pytest smoke test at `tests/test_regression.py` exercises both the anchor and the drift function. CI runs the existing pytest suite (which now includes the smoke + unit tests) plus a one-line invocation of `compare.py` in no-baseline mode to catch CLI bitrot. Baseline JSON files are local-only and gitignored.

**Tech Stack:** Python 3.13, pytest, hpvsim (currently v2.3.x on `v3.0-dev`), sciris, GitHub Actions.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `tests/regression/__init__.py` | Create | Empty; makes `tests/regression/` an importable package |
| `tests/regression/anchor.py` | Create | `make_sim()` + `run_and_summarize()` + `__main__` runner; the pinned scenario |
| `tests/regression/baseline.py` | Create | CLI: run anchor, write JSON to `tests/regression_baselines/anchor.json` |
| `tests/regression/compare.py` | Create | `compute_drift()` pure fn + CLI: run anchor, load baseline, print per-key drift table; supports `--baseline`; no-baseline mode exits without running anchor |
| `tests/regression/README.md` | Create | Full usage doc: anchor pars, generate-baseline / compare workflow, drift semantics, gate behavior |
| `tests/test_regression.py` | Create | pytest: smoke test for anchor + unit tests for `compute_drift` |
| `tests/README.md` | Modify | Append one paragraph pointing to `tests/regression/README.md` |
| `.github/workflows/tests.yaml` | Modify | Add a step running `python regression/compare.py` after the existing pytest step |
| `MIGRATION_PLAN.md` | Modify | §Implementation conventions item 2: replace preliminary "summary result" list with the pinned 9-key set; M0 sub-tasks: name the three scripts |

`tests/regression_baselines/` is already in `.gitignore`; it gets created on first `baseline.py` run and stays local.

---

## Task 1: Anchor scenario (TDD via smoke test)

**Files:**
- Create: `tests/regression/__init__.py`
- Create: `tests/regression/anchor.py`
- Create: `tests/test_regression.py` (smoke test only at this stage; unit tests come in Task 2)

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_regression.py` with this initial content:

```python
"""Tests for the v2 -> v3 regression harness.

Smoke test for the anchor sim (this file) plus unit tests for the drift
computation in tests/regression/compare.py (added in Task 2).
"""

import sys
from pathlib import Path

# tests/ is on sys.path when pytest is invoked from tests/, but be robust:
sys.path.insert(0, str(Path(__file__).parent))

from regression.anchor import run_and_summarize  # noqa: E402


def test_anchor_runs():
    short, total_pop = run_and_summarize()
    expected_keys = {
        'total HPV infections',
        'total cancers',
        'total cancer deaths',
        'mean HPV prevalence (%)',
        'mean cancer incidence (per 100k)',
        'mean age of infection (years)',
        'mean age of cancer (years)',
        'mean age of cancer death (years)',
    }
    missing = expected_keys - set(short.keys())
    assert not missing, f'short_summary missing keys: {missing}'
    assert total_pop > 0, f'total_pop should be positive, got {total_pop}'
```

- [ ] **Step 2: Run the test to verify it fails with ImportError**

Run from the repo root:

```bash
cd tests && pytest test_regression.py -v
```

Expected: collection error / `ModuleNotFoundError: No module named 'regression'`.

- [ ] **Step 3: Create the package init**

Create `tests/regression/__init__.py` as an empty file (zero bytes).

- [ ] **Step 4: Create `tests/regression/anchor.py`**

```python
"""Anchor scenario for the v2 -> v3 migration regression harness.

Vanilla 4-genotype HPV sim, Nigeria, fixed seed, no interventions, no analyzers.
This module is the scientific definition of the M0 anchor; tooling under
tests/regression/ (baseline.py, compare.py) imports run_and_summarize() and
make_sim() from here.

Run as a script to print the summary:
    python tests/regression/anchor.py
"""

import sciris as sc
import hpvsim as hpv

# Pinned anchor pars. Do not change without coordinating with regression baselines.
PARS = dict(
    n_agents   = 10e3,
    location   = 'nigeria',
    genotypes  = [16, 18, 'hi5', 'ohr'],
    start      = 1990,
    end        = 2060,
    dt         = 0.5,
    burnin     = 20,
    rand_seed  = 0,
    verbose    = 0,
)


def make_sim():
    """Build (but do not run) the anchor sim."""
    return hpv.Sim(sc.dcp(PARS))


def run_and_summarize():
    """Run the anchor sim and return (short_summary_dict, total_population_float)."""
    sim = make_sim()
    sim.run()
    short = dict(sim.short_summary)
    total_pop = float(sim.results['n_alive'][-1])
    return short, total_pop


if __name__ == '__main__':
    short, total_pop = run_and_summarize()
    print('Short summary:')
    for k, v in short.items():
        print(f'  {k:<40} {v:>12.4g}')
    print(f'  {"total population":<40} {total_pop:>12.4g}')
```

- [ ] **Step 5: Run the smoke test to verify it passes**

```bash
cd tests && pytest test_regression.py::test_anchor_runs -v
```

Expected: PASS. Wall-clock ~30–60s (the anchor runs a 70-year, 10k-agent, 4-genotype sim end-to-end).

If this is far slower than expected, do NOT change the anchor pars — escalate as a concern in the report.

- [ ] **Step 6: Sanity-check the `__main__` runner**

```bash
python tests/regression/anchor.py
```

Expected output: a printed table of the 8 short_summary keys plus `total population`. Numbers should be plausible (e.g., `mean HPV prevalence (%)` between 0 and 50; `total cancers` positive; `total population` near `10000` give-or-take demographic growth).

- [ ] **Step 7: Commit**

```bash
git add tests/regression/__init__.py tests/regression/anchor.py tests/test_regression.py
git commit -m "$(cat <<'EOF'
M0: add anchor scenario and smoke test

tests/regression/anchor.py defines the pinned anchor sim (Nigeria,
4 genotypes, seed 0, 1990-2060, no interventions). Exposes make_sim()
and run_and_summarize() for downstream tooling.

tests/test_regression.py exercises the anchor end-to-end as a CI
smoke check; later tasks add unit tests alongside.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Drift computation (TDD)

**Files:**
- Create: `tests/regression/compare.py` (drift function only at this stage; CLI lands in Task 3)
- Modify: `tests/test_regression.py` (add unit tests)

- [ ] **Step 1: Add failing unit tests for `compute_drift`**

Append the following to `tests/test_regression.py`:

```python


# --- Unit tests for tests/regression/compare.py:compute_drift -----------------

from regression.compare import compute_drift  # noqa: E402


def test_compute_drift_within_threshold():
    baseline = {'a': 100.0, 'b': 50.0}
    current  = {'a': 105.0, 'b': 49.0}
    rows = compute_drift(baseline, current, threshold=0.10)
    by_key = {r['key']: r for r in rows}
    assert by_key['a']['rel_diff'] == 0.05
    assert by_key['a']['over_threshold'] is False
    assert by_key['b']['rel_diff'] == -0.02
    assert by_key['b']['over_threshold'] is False


def test_compute_drift_over_threshold():
    baseline = {'a': 100.0}
    current  = {'a': 120.0}
    rows = compute_drift(baseline, current, threshold=0.10)
    assert rows[0]['rel_diff'] == 0.20
    assert rows[0]['over_threshold'] is True


def test_compute_drift_zero_baseline_flagged():
    baseline = {'a': 0.0}
    current  = {'a': 1.0}
    rows = compute_drift(baseline, current, threshold=0.10)
    assert rows[0]['rel_diff'] is None
    assert rows[0]['abs_diff'] == 1.0
    assert rows[0]['over_threshold'] is True


def test_compute_drift_skips_keys_missing_from_current():
    baseline = {'a': 1.0, 'b': 2.0}
    current  = {'a': 1.05}  # 'b' missing
    rows = compute_drift(baseline, current)
    keys = [r['key'] for r in rows]
    assert keys == ['a']
```

- [ ] **Step 2: Run the new tests to verify they fail with ImportError**

```bash
cd tests && pytest test_regression.py -v
```

Expected: collection error or `ModuleNotFoundError: No module named 'regression.compare'`. The previously-passing `test_anchor_runs` should also fail to collect because of the new top-level import.

- [ ] **Step 3: Create `tests/regression/compare.py` with the pure function only**

```python
"""Compare a current run of the anchor scenario against a stored v2 baseline.

This file is a stub at the end of Task 2 (only compute_drift is implemented).
The CLI lands in Task 3.
"""

THRESHOLD = 0.10  # 10% relative drift


def compute_drift(baseline_summary, current_summary, threshold=THRESHOLD):
    """Compute per-key drift records.

    Args:
        baseline_summary: dict of {key: number} — the stored v2 baseline.
        current_summary:  dict of {key: number} — the current run's summary.
        threshold:        relative-drift threshold (default 0.10 = 10%).

    Returns:
        List of dicts with keys: key, baseline, current, abs_diff, rel_diff,
        over_threshold. Keys present in baseline but missing from current are
        skipped (not reported in the returned list). If the baseline value is
        zero, rel_diff is None and over_threshold is True.
    """
    rows = []
    for k in baseline_summary.keys():
        if k not in current_summary:
            continue
        b = float(baseline_summary[k])
        c = float(current_summary[k])
        abs_diff = c - b
        if b == 0:
            rel_diff = None
            over = True
        else:
            rel_diff = abs_diff / b
            over = abs(rel_diff) > threshold
        rows.append({
            'key': k,
            'baseline': b,
            'current': c,
            'abs_diff': abs_diff,
            'rel_diff': rel_diff,
            'over_threshold': over,
        })
    return rows
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
cd tests && pytest test_regression.py -v
```

Expected: all 5 tests PASS (`test_anchor_runs` plus the 4 `test_compute_drift_*`).

- [ ] **Step 5: Commit**

```bash
git add tests/regression/compare.py tests/test_regression.py
git commit -m "$(cat <<'EOF'
M0: add drift computation with unit tests

compute_drift() returns per-key relative-drift records vs. a baseline
dict. Threshold default 0.10 (10%). Zero-baseline values are flagged
as over_threshold with rel_diff=None.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Comparison CLI

**Files:**
- Modify: `tests/regression/compare.py` (add CLI on top of `compute_drift`)

- [ ] **Step 1: Replace `tests/regression/compare.py` with the full version**

```python
"""Compare a current run of the anchor scenario against a stored v2 baseline.

Prints a per-summary-result drift table. The development gate is informational:
exit code is always 0 regardless of threshold breaches. If no baseline file is
present, prints a notice and exits clean without running the anchor (used by CI;
the anchor-runs check is the pytest smoke test's job).

Usage:
    python tests/regression/compare.py
    python tests/regression/compare.py --baseline path/to/file.json
    python tests/regression/compare.py --threshold 0.05
"""

import argparse
import json
import sys
from pathlib import Path

THRESHOLD = 0.10  # 10% relative drift

# Default baseline location: tests/regression_baselines/anchor.json (gitignored).
DEFAULT_BASELINE = Path(__file__).resolve().parent.parent / 'regression_baselines' / 'anchor.json'


def compute_drift(baseline_summary, current_summary, threshold=THRESHOLD):
    """Compute per-key drift records.

    Args:
        baseline_summary: dict of {key: number} — the stored v2 baseline.
        current_summary:  dict of {key: number} — the current run's summary.
        threshold:        relative-drift threshold (default 0.10 = 10%).

    Returns:
        List of dicts with keys: key, baseline, current, abs_diff, rel_diff,
        over_threshold. Keys present in baseline but missing from current are
        skipped. If the baseline value is zero, rel_diff is None and
        over_threshold is True.
    """
    rows = []
    for k in baseline_summary.keys():
        if k not in current_summary:
            continue
        b = float(baseline_summary[k])
        c = float(current_summary[k])
        abs_diff = c - b
        if b == 0:
            rel_diff = None
            over = True
        else:
            rel_diff = abs_diff / b
            over = abs(rel_diff) > threshold
        rows.append({
            'key': k,
            'baseline': b,
            'current': c,
            'abs_diff': abs_diff,
            'rel_diff': rel_diff,
            'over_threshold': over,
        })
    return rows


def format_table(rows, threshold=THRESHOLD):
    """Format drift rows as a printable table (str)."""
    out = []
    out.append(f'{"key":<40} {"baseline":>12} {"current":>12} {"abs_diff":>12} {"rel_diff":>10} {"over":>6}')
    out.append('-' * 96)
    for r in rows:
        rel = f'{r["rel_diff"]*100:+.2f}%' if r['rel_diff'] is not None else 'n/a'
        flag = 'YES' if r['over_threshold'] else ''
        out.append(
            f'{r["key"]:<40} {r["baseline"]:>12.4g} {r["current"]:>12.4g} '
            f'{r["abs_diff"]:>12.4g} {rel:>10} {flag:>6}'
        )
    n_over = sum(1 for r in rows if r['over_threshold'])
    out.append('')
    out.append(
        f'{n_over}/{len(rows)} keys exceed +/- {threshold*100:.0f}% relative drift '
        f'threshold (informational; exit 0 regardless).'
    )
    return '\n'.join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description='Compare anchor run against v2 baseline.')
    p.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE,
                   help=f'Baseline JSON path (default: {DEFAULT_BASELINE})')
    p.add_argument('--threshold', type=float, default=THRESHOLD,
                   help='Relative drift threshold (default 0.10).')
    args = p.parse_args(argv)

    if not args.baseline.exists():
        print(f'No baseline at {args.baseline}; skipping diff.')
        print('To generate a baseline, run: python tests/regression/baseline.py')
        return 0

    # Local import: only run the anchor when we actually need to.
    sys.path.insert(0, str(Path(__file__).parent))
    from anchor import run_and_summarize  # noqa: E402

    with open(args.baseline) as f:
        baseline = json.load(f)
    baseline_summary = baseline['summary']

    short, total_pop = run_and_summarize()
    current_summary = {**{k: float(v) for k, v in short.items()},
                       'total population': total_pop}

    rows = compute_drift(baseline_summary, current_summary, threshold=args.threshold)
    print(format_table(rows, threshold=args.threshold))
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 2: Run unit tests to confirm `compute_drift` still passes**

```bash
cd tests && pytest test_regression.py -v -k compute_drift
```

Expected: 4 PASS (the smoke test is filtered out).

- [ ] **Step 3: Run `compare.py` in no-baseline mode**

From the repo root, with `tests/regression_baselines/anchor.json` not yet existing:

```bash
python tests/regression/compare.py
```

Expected output (timing: < 1 second; does NOT run the anchor):

```
No baseline at C:\...\tests\regression_baselines\anchor.json; skipping diff.
To generate a baseline, run: python tests/regression/baseline.py
```

Exit code 0. If `tests/regression_baselines/anchor.json` already exists from prior local work, temporarily move it aside (`mv tests/regression_baselines/anchor.json /tmp/`) for this check, then move it back.

- [ ] **Step 4: Commit**

```bash
git add tests/regression/compare.py
git commit -m "$(cat <<'EOF'
M0: wire compare.py CLI on top of compute_drift

CLI loads a baseline JSON, runs the anchor, prints a drift table.
Always exits 0. No-baseline mode exits without running the anchor;
that lets CI smoke-check the CLI cheaply (the pytest smoke test
covers anchor execution).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Baseline generator

**Files:**
- Create: `tests/regression/baseline.py`

- [ ] **Step 1: Create `tests/regression/baseline.py`**

```python
"""Generate a regression baseline from the currently-installed hpvsim.

Runs the anchor scenario and saves a JSON file containing the per-key
short_summary plus total population, with metadata (hpvsim version, anchor pars).

The output directory is gitignored; baseline files stay local-only.

Usage:
    python tests/regression/baseline.py
    python tests/regression/baseline.py --out path/to/baseline.json
"""

import argparse
import json
import sys
from pathlib import Path

import hpvsim as hpv

# Allow running as `python tests/regression/baseline.py` from the repo root.
sys.path.insert(0, str(Path(__file__).parent))
from anchor import PARS, run_and_summarize  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent.parent / 'regression_baselines' / 'anchor.json'


def build_baseline():
    short, total_pop = run_and_summarize()
    return {
        'metadata': {
            'hpvsim_version': hpv.__version__,
            'pars': dict(PARS),  # JSON-serializable: ints, floats, str, list of mixed types
        },
        'summary': {
            **{k: float(v) for k, v in short.items()},
            'total population': total_pop,
        },
    }


def main(argv=None):
    p = argparse.ArgumentParser(description='Generate v2 regression baseline.')
    p.add_argument('--out', type=Path, default=DEFAULT_OUT,
                   help=f'Output JSON path (default: {DEFAULT_OUT}).')
    args = p.parse_args(argv)

    baseline = build_baseline()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(baseline, f, indent=2)
    print(f'Wrote baseline to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 2: Run baseline.py and verify output**

```bash
python tests/regression/baseline.py
```

Expected: ~30–60s wall clock, then `Wrote baseline to .../tests/regression_baselines/anchor.json`.

Verify the file was created:

```bash
ls -l tests/regression_baselines/anchor.json
```

Verify the JSON shape:

```bash
python -c "import json; d=json.load(open('tests/regression_baselines/anchor.json')); print(sorted(d.keys())); print(sorted(d['summary'].keys()))"
```

Expected output (something like):
```
['metadata', 'summary']
['mean HPV prevalence (%)', 'mean age of cancer (years)', 'mean age of cancer death (years)', 'mean age of infection (years)', 'mean cancer incidence (per 100k)', 'total HPV infections', 'total cancer deaths', 'total cancers', 'total population']
```

- [ ] **Step 3: Run `compare.py` against the just-generated baseline**

```bash
python tests/regression/compare.py
```

Expected: ~30–60s wall clock (re-runs the anchor), then a drift table where every row shows ~0% rel_diff (or very close — `seed=0` is fixed and the package hasn't changed between baseline generation and this run). All `over` flags should be empty. The footer line should read `0/9 keys exceed +/- 10% relative drift threshold (informational; exit 0 regardless).` Exit code 0.

If any row shows non-zero drift here, the anchor sim is non-deterministic at fixed seed and that is a real bug — escalate as a concern.

- [ ] **Step 4: Commit**

```bash
git add tests/regression/baseline.py
git commit -m "$(cat <<'EOF'
M0: add baseline generator CLI

baseline.py runs the anchor against the currently-installed hpvsim
and writes a JSON snapshot to tests/regression_baselines/anchor.json
(gitignored). Used by developers to capture a v2 reference baseline
before starting migration work.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: CI workflow update

**Files:**
- Modify: `.github/workflows/tests.yaml`

- [ ] **Step 1: Add a step to run `compare.py` after the existing pytest step**

Open `.github/workflows/tests.yaml`. After the existing `Run all tests` step (which runs `pytest test_*.py -n auto ...`), insert a new step before the `Publish Test Report` step. The relevant region after the change should look like this:

```yaml
      - name: Run all tests
        working-directory: ./tests
        run: pytest test_*.py -n auto --durations=0 --junitxml=test-results.xml # Run actual tests
      - name: Smoke-check regression compare CLI (no-baseline mode)
        working-directory: ./tests
        run: python regression/compare.py
      - name: Publish Test Report
        uses: mikepenz/action-junit-report@v3
        if: always() # always run even if the previous step fails
        with:
          report_paths: './tests/test-results.xml'
```

The exact `Edit` invocation:

`old_string`:
```
      - name: Run all tests
        working-directory: ./tests
        run: pytest test_*.py -n auto --durations=0 --junitxml=test-results.xml # Run actual tests
      - name: Publish Test Report
```

`new_string`:
```
      - name: Run all tests
        working-directory: ./tests
        run: pytest test_*.py -n auto --durations=0 --junitxml=test-results.xml # Run actual tests
      - name: Smoke-check regression compare CLI (no-baseline mode)
        working-directory: ./tests
        run: python regression/compare.py
      - name: Publish Test Report
```

- [ ] **Step 2: Validate the YAML parses**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/tests.yaml'))"
```

Expected: no output, exit 0. Any traceback means the YAML is malformed; fix indentation.

- [ ] **Step 3: Locally simulate the new step**

From the repo root:

```bash
cd tests && python regression/compare.py
```

Note: this is the same invocation CI will run. If a baseline exists locally from Task 4, this will run the anchor and print a drift table (~60s). If no baseline exists, it prints the no-baseline message and exits in <1s.

For CI's purposes, only the no-baseline path matters (CI won't have a baseline file). To simulate that explicitly, point at a non-existent path:

```bash
cd tests && python regression/compare.py --baseline /tmp/does_not_exist.json
```

Expected output:
```
No baseline at /tmp/does_not_exist.json; skipping diff.
To generate a baseline, run: python tests/regression/baseline.py
```

Exit code 0.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/tests.yaml
git commit -m "$(cat <<'EOF'
M0: CI smoke-checks regression compare CLI

Adds a workflow step that runs python regression/compare.py after
the pytest suite. In no-baseline mode (CI has no baseline file) it
exits in <1s and proves the CLI script itself doesn't bitrot. The
pytest suite already exercises the anchor sim end-to-end via
test_regression.py:test_anchor_runs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Documentation

**Files:**
- Create: `tests/regression/README.md`
- Modify: `tests/README.md`

- [ ] **Step 1: Create `tests/regression/README.md`**

```markdown
# Regression harness

This directory holds the v2 → v3 migration regression harness used during the
HPVsim v3.0 port. It runs *outside* the standard pytest flow: the harness's
job is to compare a current run of an anchor scenario against a stored v2
baseline and report per-summary-result drift to the developer.

The harness is deliberately small and informational — it is the **development
gate** described in `MIGRATION_PLAN.md` §Implementation conventions item 2.
It does **not** fail PRs; the **release gate** (overlapping uncertainty
intervals against the analysis-repo suite) is the scientific gate and lives
elsewhere.

## What's here

| File | Role |
|---|---|
| `anchor.py` | Pinned anchor scenario: vanilla 4-genotype HPV sim, Nigeria, seed 0, 1990–2060, no interventions. Exposes `make_sim()` and `run_and_summarize()`; runs as `__main__` for an ad-hoc summary print. |
| `baseline.py` | CLI: runs the anchor, writes a JSON baseline to `../regression_baselines/anchor.json` (gitignored). |
| `compare.py` | CLI: runs the anchor, loads a baseline, prints a per-key drift table. Exits 0 always. No-baseline mode exits without running the anchor. |
| `__init__.py` | Empty; makes this directory an importable package for the pytest smoke test in `tests/test_regression.py`. |

## Anchor scenario

Pinned in `anchor.py:PARS`:

| Par | Value |
|---|---|
| `n_agents` | `10e3` |
| `location` | `'nigeria'` |
| `genotypes` | `[16, 18, 'hi5', 'ohr']` |
| `start` | `1990` |
| `end` | `2060` |
| `dt` | `0.5` |
| `burnin` | `20` |
| `rand_seed` | `0` |
| `verbose` | `0` |

No interventions, no analyzers. Nigeria was chosen because the existing v2.x
multi-scenario script (`tests/generate_v2_baselines.py`) already uses Nigeria
as one of its three locations and Nigeria is well-represented in the
validation-repo suite.

## Generating a baseline

Baselines are local-only and gitignored. The recommended workflow:

1. Check out a clean v2.3.x environment (typically `git checkout rc2.3` and
   `pip install -e .`, or install `hpvsim==2.3.x` from PyPI in a separate venv).
2. Run:

   ```bash
   python tests/regression/baseline.py
   ```

3. The baseline lands at `tests/regression_baselines/anchor.json`. Keep that
   file as your migration target.

The script takes ~30–60s to run the anchor sim. Once the baseline is in place,
return to `v3.0-dev` and use it as the comparison reference.

## Running the comparison

```bash
python tests/regression/compare.py
```

Output: a table of per-key drift, e.g.

```
key                                      baseline      current     abs_diff   rel_diff   over
------------------------------------------------------------------------------------------------
total HPV infections                        12345        12350           5     +0.04%
mean HPV prevalence (%)                      8.2          8.2            0     +0.04%
...

0/9 keys exceed +/- 10% relative drift threshold (informational; exit 0 regardless).
```

Optional flags:

- `--baseline PATH` — diff against a different baseline file.
- `--threshold 0.05` — change the threshold (default 0.10).

## Drift semantics

- **Relative drift:** `(current - baseline) / baseline`. A row is flagged when
  `|rel_diff| > threshold` (default 10%).
- **Zero-baseline guard:** if the baseline value is zero (not expected for any
  pinned key in the anchor scenario, but guarded anyway), the row reports
  absolute drift only and is flagged.
- **The threshold is informational.** A flagged row signals that the developer
  should investigate. Either the change in this PR is the cause and is
  legitimately fixing or breaking equivalence; or the drift is expected
  feature-misalignment that requires a tracking issue per `MIGRATION_PLAN.md`
  §Implementation conventions item 2. The PR is not blocked by drift.

## When to refresh the baseline

- After a new patch release of v2.3 lands on `main` and is merged into
  `v3.0-dev`.
- After an explicit decision that drift introduced by a milestone is the new
  target (i.e., feature-misalignment that has been investigated and accepted).
- Otherwise: don't. Stable baseline = stable signal.

## CI

CI runs:

- The pytest smoke test (`tests/test_regression.py:test_anchor_runs`) which
  imports `anchor.run_and_summarize` and exercises the sim end-to-end.
- `python regression/compare.py` (no-baseline mode) which proves the CLI
  imports and parses arguments cleanly.

Neither step fails on drift. Drift is a developer-local concern.
```

- [ ] **Step 2: Append a pointer to `tests/README.md`**

Read the existing `tests/README.md`. Then append (preserving existing content) at the end of the file:

```markdown

## Regression harness

There is also a regression harness under `tests/regression/` used for the v2 → v3
migration. It runs outside the standard pytest flow and is documented in
[`tests/regression/README.md`](regression/README.md). The harness compares a
pinned anchor scenario run against a stored v2 baseline and reports per-key drift;
it is informational, not a CI gate.
```

The exact `Edit` operation: open `tests/README.md`, navigate to the end of the file, and append the section above. If the file already ends with `\n`, append starting with a blank line as shown.

- [ ] **Step 3: Commit**

```bash
git add tests/regression/README.md tests/README.md
git commit -m "$(cat <<'EOF'
M0: document the regression harness

tests/regression/README.md describes the harness end-to-end: anchor
scenario, baseline-generation workflow, comparison usage, drift
semantics, when to refresh the baseline, CI behavior.

tests/README.md gets a one-paragraph pointer so the harness is
discoverable from the standard tests entry point.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Migration plan touch-ups

**Files:**
- Modify: `MIGRATION_PLAN.md`

- [ ] **Step 1: Update §Implementation conventions item 2**

In `MIGRATION_PLAN.md`, locate the "Dual validation gates" item under "Implementation conventions" (currently item 2). The current text contains:

```
   - **Development gate (per PR).** An *anchor scenario* (vanilla natural history, no interventions, fixed seed) plus per-milestone *capability scenarios* are run against stored v2.x baselines. Target: ±10% per summary result, where "summary result" initially means total HPV prevalence, age-standardized CIN prevalence, age-standardized cancer incidence, and total population; the exact list is pinned in M0 alongside the comparison script. **On failure the gate is informational, not auto-blocking**: the PR carries either a fix, or an explicit note classifying the drift as expected feature-misalignment with a tracking issue for re-convergence.
```

Replace with:

```
   - **Development gate (per PR).** An *anchor scenario* (vanilla natural history, no interventions, fixed seed) plus per-milestone *capability scenarios* are run against locally-stored v2.x baselines. Target: ±10% relative drift per summary result. The pinned summary-result set, established in M0, is `sim.short_summary` (total HPV infections, total cancers, total cancer deaths, mean HPV prevalence, mean cancer incidence, mean ages of infection / cancer / cancer death) plus total population. **On failure the gate is informational, not auto-blocking**: the PR carries either a fix, or an explicit note classifying the drift as expected feature-misalignment with a tracking issue for re-convergence.
```

The exact `Edit` invocation has `old_string` set to the original paragraph and `new_string` to the replacement above.

- [ ] **Step 2: Update M0 sub-tasks to name the three scripts**

In `MIGRATION_PLAN.md`, locate the M0 Sub-tasks list. Currently:

```
**Sub-tasks:**
- Set up CI on `v3.0-dev` (adapted from rc2.3 CI).
- Commit a deterministic v2.x baseline-generation script for the regression harness. Generated baseline files stay local (gitignored), never committed.
- Write the anchor scenario script: vanilla 4-genotype HPV sim, one country, fixed seed, no interventions.
- Write the ±10% comparison script that diffs a v3 run against the locally-stored v2 baseline and emits per-summary-result drift.
- Document how to run the regression harness in a `CONTRIBUTING.md` (or equivalent) section.
```

Replace with:

```
**Sub-tasks:**
- Set up CI on `v3.0-dev` (adapted from rc2.3 CI); add a smoke-check step for the comparison CLI.
- Commit a deterministic v2.x baseline-generation script (`tests/regression/baseline.py`). Generated baseline files stay local (gitignored), never committed.
- Write the anchor scenario script (`tests/regression/anchor.py`): vanilla 4-genotype HPV sim, Nigeria, fixed seed (`0`), no interventions, 1990–2060.
- Write the ±10% comparison script (`tests/regression/compare.py`) that diffs a current run against the locally-stored v2 baseline and emits per-summary-result relative drift.
- Document how to run the regression harness in `tests/regression/README.md` and add a pointer from `tests/README.md`.
```

- [ ] **Step 3: Verify both edits land cleanly**

```bash
git diff MIGRATION_PLAN.md
```

Confirm only the two paragraphs above changed; no other parts of the file were edited.

- [ ] **Step 4: Commit**

```bash
git add MIGRATION_PLAN.md
git commit -m "$(cat <<'EOF'
M0: pin summary-result set in MIGRATION_PLAN

§Implementation conventions item 2: replace the preliminary list
('total HPV prevalence, age-standardized CIN prevalence, ...') with
the established v2 sim.short_summary set plus total population, now
that M0 has settled the choice.

M0 sub-tasks: name the three harness scripts and document the
README location.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: End-to-end verification

This task contains no commits; it is a manual verification pass before pushing.

- [ ] **Step 1: Confirm the working tree is clean**

```bash
git status --short
```

Expected: only the 7 untracked items present at the start of M0 work (the calibration `.db` files, `_seed_scan.py`, `pars_v2.3.0.json`, `uv.lock`, `.claude/`). No modified or staged files.

- [ ] **Step 2: Confirm all M0 commits landed**

```bash
git log --oneline v3.0-dev ^"$(git merge-base v3.0-dev rc2.3)" -- tests/regression tests/test_regression.py tests/README.md .github/workflows/tests.yaml MIGRATION_PLAN.md
```

Expected: 7 new commits from this plan (Tasks 1–7), in order.

- [ ] **Step 3: Run the full pytest suite**

```bash
cd tests && pytest test_*.py -n auto --durations=0
```

Expected: all tests pass, including the 5 new tests in `test_regression.py`. The smoke test (`test_anchor_runs`) takes ~30–60s; the 4 `test_compute_drift_*` tests are sub-second.

- [ ] **Step 4: Run the full developer workflow**

```bash
# Generate a baseline (~60s)
python tests/regression/baseline.py

# Run the comparison (~60s, expect 0/9 over threshold)
python tests/regression/compare.py
```

Expected: baseline writes to `tests/regression_baselines/anchor.json`. Compare prints a drift table where every `rel_diff` is exactly 0.00% (since the package hasn't changed between baseline-generation and compare). Footer reads `0/9 keys exceed +/- 10% relative drift threshold`.

- [ ] **Step 5: Run the CI smoke command path**

```bash
cd tests && python regression/compare.py --baseline /tmp/nope.json
```

Expected:
```
No baseline at /tmp/nope.json; skipping diff.
To generate a baseline, run: python tests/regression/baseline.py
```

Exit 0, < 1s wall-clock.

- [ ] **Step 6: Confirm `.gitignore` still excludes generated baselines**

```bash
git check-ignore tests/regression_baselines/anchor.json
```

Expected: `tests/regression_baselines/anchor.json` (path printed = file is ignored).

- [ ] **Step 7: Push to origin**

```bash
git push origin v3.0-dev
```

Expected: clean push. CI on the v3.0-dev branch should kick off (visible at `gh run list --branch v3.0-dev` or the GitHub UI).

- [ ] **Step 8: Confirm CI green**

Wait for the CI run to complete (typically 3–5 minutes for the v3.0-dev test job).

```bash
gh run list --branch v3.0-dev --limit 1
```

Expected: status `completed`, conclusion `success`. If the run is still in progress, recheck in a couple of minutes.

If CI fails, investigate the specific failing step in the run log and fix forward. Common pitfalls to check first:
- New `tests/test_regression.py` collected but anchor takes longer than expected → the workflow timeout is 5 minutes; verify locally how long the smoke test takes and consider whether n_agents needs adjustment (escalate as DONE_WITH_CONCERNS rather than silently changing the anchor).
- `python regression/compare.py` step exits non-zero → investigate `compare.py` import resolution; CI runs from `./tests` working directory.

---

## Self-review checklist

After completing all 8 tasks, verify against the spec at
`docs/superpowers/specs/2026-04-28-hpvsim-m0-foundation-design.md`:

| Spec requirement | Implementing task |
|---|---|
| `tests/regression/anchor.py` defines pinned anchor scenario | Task 1 |
| Anchor pars: nigeria, 4 genotypes, seed 0, 1990–2060, dt 0.5, burnin 20, n_agents 10e3 | Task 1 |
| `tests/regression/baseline.py` CLI generates baseline JSON | Task 4 |
| `tests/regression/compare.py` CLI runs anchor + diffs baseline | Tasks 2 & 3 |
| Pinned 9-key summary result set | Task 1 (anchor exposes them) + Task 4 (baseline writes them) |
| ±10% relative drift threshold | Task 2 (default in `compute_drift`) |
| Zero-baseline guard | Task 2 |
| No-baseline mode does not run the anchor | Task 3 |
| `tests/test_regression.py` smoke test for anchor | Task 1 |
| Unit tests for `compute_drift` | Task 2 |
| `tests/regression/README.md` documentation | Task 6 |
| `tests/README.md` pointer | Task 6 |
| CI step for `compare.py` no-baseline | Task 5 |
| MIGRATION_PLAN.md §Impl conventions item 2 updated | Task 7 |
| MIGRATION_PLAN.md M0 sub-tasks updated | Task 7 |
| Baseline files stay local-only (gitignored) | Pre-existing `.gitignore`; verified Task 8 step 6 |
| No migration code in `hpvsim/` | Out of scope; nothing in plan touches `hpvsim/` |

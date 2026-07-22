# M09 ledger-bound analyzers → grow engine — port design

**Date:** 2026-07-08
**Source branches:** `m09-analyzers` (engine + analyzers), `m09-plotting` (superset: adds `plotting.py` + `.plot()` methods)
**Target branch:** new port branch off `m08-rwanda-on-grow` (the grow multiscale engine)
**Scope:** port `age_causal_infection` and `dalys` from the abandoned ledger engine onto the grow engine. Single-scale (`ms_agent_ratio == 1`) behavior unchanged; `ms_agent_ratio > 1` behavior becomes engine-correct via real fine agents instead of scheduled ledger data.

---

## 0. Context

M09 was branched off `main` (v2.3 release) and carries the **ledger/overlay** multiscale engine — the approach abandoned in M07 for being intervention-blind. The grow engine (`m08-rwanda-on-grow`) replaced it: extra cancers are grown as **real fine agents** (`_grow_fine_agents`, `hpv.py:534`) rather than scheduled as data (`_multiscale_ledger`).

Interface gap confirmed:

| | `m09-analyzers` | `m08-rwanda-on-grow` (grow) |
|---|---|---|
| multiscale mechanism | `_multiscale_ledger` (scheduled data) | `_grow_fine_agents` (real agents) |
| `ledger` refs in `hpv.py` | 36 | 0 |
| `ledger` refs in `analyzers.py` | 17 | 0 |
| `_cancer_events` | present | **gone** |
| `plotting.py` | new (340 lines) | does not exist |
| `analyzers.py` classes | `snapshot`, `AgeResults`, `age_pyramid`, `age_causal_infection`, `dalys`, `results_by_genotype` | only `AgeResults` |

## 1. Key insight — the port is mostly deletion

M09's two analyzers are dual-path:
- a `step()` **live-agent** path for `ms_agent_ratio == 1` (reads `sim.people`), and
- a `finalize()` **ledger** path for ratio>1 (reads `m._cancer_events`), switched by `self._use_ledger`.

On grow, extra cancers are real agents in `sim.people` carrying `fine=True`, `scale=1/ratio`, and a full trajectory (`ti_infected/ti_first_infection/ti_cin/ti_cancerous/ti_dead_cancer`). The fine agent's `ti_infected` is set to the grow tick with a fresh natural-history trajectory. So the live-agent `step()` path **already sees every cancer at every ratio**.

Therefore: **delete the ledger branch and run the single live-agent path unconditionally.** Grow has no `age_causal`/`dalys` yet, so this is an additive port, not a two-way reconcile.

## 2. `age_causal_infection` changes

1. **Delete** `self._use_ledger`, the `if self._use_ledger: return` guard in `step()`, and the entire `finalize()` ledger branch iterating `m._cancer_events`. Keep the array-ification tail of `finalize()`.
2. **Weight by scale.** M09's step path hard-codes `weight = np.ones(len(new))` (valid only when all scales are 1). Replace with `people.scale.raw[new]` (fall back to ones if `scale` absent), matching `dalys`. This makes the age distribution population-correct at ratio>1: shrunk base cancers and fine cancers each contribute `1/ratio`.
3. **Docstring rewrite.** The M09 caveat ("ledger-path age_causal not identical to the ratio==1 agent path") is **removed** — on grow, fine agents run the same natural-history code path, so `age_causal` is self-consistent across ratios. Note that fine agents carry a fresh trajectory (`ti_infected = grow_tick`), so the back-trace `cur - (ti - ti_infected)*dt` yields the fine agent's own causal-infection age.

## 3. `dalys` changes

1. **Delete** the same `_use_ledger` machinery and the `finalize()` ledger branch. `finalize()` collapses to `self.dalys = self.yll + self.yld`.
2. **No weighting change** — the step path already reads `sim.people.scale.raw` and applies it in `_accumulate`. Fine agents (`scale=1/ratio`) and shrunk base agents flow through correctly.
3. **Death age already correct** — fine agents get `ti_dead_cancer` at grow time, so `death_age = cancer_age + (ti_dead - ti)*dt` works; YLL/YLD attribute at onset year (incidence-based).

## 4. Correctness deltas (grow is stricter, not looser)

- **Competing mortality is now real.** On grow, fine agents are subject to independent background/emigration hazard (`demographics.py:285 _emigrate_fine`, the AgeMigration fix from the v2-faithful work). A fine agent that dies before its scheduled cancer is correctly dropped by the existing `cancerous & alive` gate — whereas the ledger recorded scheduled cancers as realized. Grow's DALYs/age-causal counts may therefore sit slightly below the ledger's. This is the intended correction.
- **Same-tick competing death (~1%)** — the M09 gate (`ti_cancerous==ti & cancerous & alive`) already handles this; unchanged.

## 5. Consistency gate (the real acceptance test)

The ported analyzers must agree with the engine's own per-genotype results. On a ratio>1 grow run, assert (scale-weighted, within Monte-Carlo tolerance):
- `dalys` cancer-onset count ≈ `Σ_genotype new_cancers`;
- `age_causal_infection` event count (`len(age_cancer)`, scale-weighted) ≈ the same.

Divergence here means the analyzer and the engine's cancer accounting disagree — the bug the ledger path used to paper over.

## 6. Tests to migrate

- `tests/test_m09_analyzers.py` — rewrite ratio>1 assertions from "reads `_cancer_events`" to "grows fine agents, analyzer picks them up from `people`." Keep ratio==1 assertions.
- `tests/test_multiscale*.py` that reference `ledger` — audit; the engine-level ones already died with the ledger on grow.
- `results/m09_plots/COMPARISON.md` — **regenerate on grow.** Produced on the ledger engine; grow tracks a higher (unbiased) cancer level per the v2.3 multiscale-bias fix, so the absolute-count row and likely age/dwell means will shift.

## 7. Free riders (no ledger dependency — port as-is)

`snapshot`, `age_pyramid`, `results_by_genotype`, and all `.plot()` methods (on `m09-plotting`) read only `self.*` arrays or `m.results[key]` — engine-agnostic. `plotting.py` transfers wholesale.

## 8. Sequencing

1. Port `analyzers.py` additions from `m09-analyzers` onto a branch off `m08-rwanda-on-grow`.
2. Apply the two deletions + the age_causal scale-weight change.
3. Add the consistency gate test; migrate `test_m09_analyzers.py`.
4. Port `plotting.py` from `m09-plotting`.
5. Regenerate `COMPARISON.md` on grow; review shifted numbers.

Net: ~40–60 lines deleted, ~1 line changed (scale weight), docstrings rewritten, tests re-based, validation re-run.

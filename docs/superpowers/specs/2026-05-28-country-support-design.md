# Country support — expose all v2 demographic locations

**Date:** 2026-05-28
**Branch:** `country-support` (off `m07-multisim`; PR targets `v3.0-dev` once `m07-multisim` lands)
**Status:** Spec drafted; implementation pending.

---

## Goal

Match v2's shipped behavior for country support: any country present in
the bundled demographic data files (`hpvsim/data/files/populations.obj`,
`populations_by_sex.obj`, `birth_rates.obj`, `mx.obj` — 298 entries from
UN WPP 2024) is a valid `location=` argument to `hpv.Sim` and
`hpv.data.load_country`. Today only `'nigeria'` is accepted, gated by
`_KNOWN_LOCATIONS = ['nigeria']` in `hpvsim/data/country.py:31`.

## Context

This was an oversight in the original `MIGRATION_PLAN.md`. M02 ("Natural
history parity") completed `hpv.data.load_country()` and the country
adapter, but only one country (Nigeria) was wired up. Downstream
milestones implicitly need more: M04 ("Run a full calibration for
India"), M08 ("Reproduces `hpvsim_rwanda` outputs"), and M10 (full
analysis-repo validation suite including
`hpvsim_methods_manuscript`, `hpvsim_india`, `hpvsim_rwanda`,
`hpv_faster_kenya`) all assume non-Nigeria locations work.

Investigation surfaced that the underlying data and loaders **already
support all 298 v2 countries**:

- `hpvsim/data/loaders.py` is functionally identical to the
  v2-quarantined version (`hpvsim/_v2_legacy/data/loaders.py`).
- `hpvsim/data/files/populations.obj` and the sibling `.obj` files
  contain 298 entries each (countries plus regional aggregates).
- `get_country_aliases()` exists and is exported, providing fuzzy-match
  for common name variants (e.g. `"Cote d'Ivoire"` → `"Côte d'Ivoire"`).
- Bypassing the `_KNOWN_LOCATIONS` gate, `load_country('india', year=1990)`
  returns valid age/birth/death/pop data and network pars.

The only restriction is the one-line allowlist in `country.py`. Removing
it unblocks 297 additional locations.

## Scope

### In scope

- Remove the `_KNOWN_LOCATIONS = ['nigeria']` gate. Replace the
  pre-validation check in `load_country` with reliance on the downstream
  loader's existing error path (`_loaders.map_entries` already raises a
  `sc.suggest`-formatted `ValueError` for unknown locations).
- Update the `load_country` docstring to reflect what's actually
  supported ("any country in the bundled UN WPP 2024 data; see
  `hpv.data.get_country_aliases()` for accepted aliases").
- Update the comment in `_default_network_pars` to make the
  location-agnostic intent explicit, not a TODO. v2's shipped behavior
  was identical (network calibration was per-analysis, not per-country
  data).
- Amend `MIGRATION_PLAN.md`:
  - Add a sub-task under M02 ("Natural history parity") tagged
    "Post-merge sub-task added 2026-05-28 (oversight correction): expose
    all v2-shipped demographic locations."
  - Update the "Scope items not pinned to a milestone" table entry on
    location-name normalization to reflect that this work lands first.

### Out of scope

- **Per-country network calibration.** v2 did not ship per-country
  network calibration data; analysis scripts (e.g. upstream
  `hpvsim_methods_manuscript/plot_fig56.py`'s `make_network('india')`)
  define their own network pars per location. v3 follows the same
  pattern. Centralized per-country network presets are a separate,
  future decision.
- **New tests.** The user has explicitly opted out of country-level
  pytest coverage in favor of relying on existing M2 tests +
  downstream-use validation. The gate-removal change is small enough
  that smoke-checking via the methods-fig reproduction workflow (next
  task) is sufficient.
- **Adding new country data.** Everything we're enabling is already
  bundled.
- **Subnational regions / location-name normalization.** The
  `MIGRATION_PLAN.md`-listed M4 follow-up item stays as-is; this PR
  doesn't expand the fuzzy-match beyond what `get_country_aliases()`
  already provides.
- **Aggregates / non-country entries** (e.g., `'Africa'`, `'AUKUS'`):
  these load successfully through the demographic loaders, but their
  epidemiological interpretation is the caller's responsibility. Not
  filtered out; not endorsed.

## Implementation

### `hpvsim/data/country.py`

```python
# BEFORE
_KNOWN_LOCATIONS = ['nigeria']

def load_country(location, year=None):
    """..."""
    location = location.lower()
    if location not in _KNOWN_LOCATIONS:
        raise ValueError(
            f"Unknown location {location!r}. Supported locations: {_KNOWN_LOCATIONS}."
        )
    return dict(...)
```

```python
# AFTER
def load_country(location, year=None):
    """Return Starsim-shaped data for ``location``.

    Any country present in the bundled UN WPP 2024 demographic files is
    valid (see ``hpv.data.get_country_aliases()`` for accepted name
    variants). The underlying loaders raise a ``ValueError`` with
    suggestion-based diagnostics for unknown names.

    Network calibration (debut, partners, layer_probs, cross_layer,
    mixing) is intentionally location-agnostic; analysis scripts supply
    per-country network pars as needed.
    """
    location = location.lower()
    return dict(...)
```

The `_KNOWN_LOCATIONS` symbol is deleted, not preserved as `[]` or as
the full 298-entry list. Anything wanting to iterate "all known
countries" can read the `populations.obj` keys directly via
`hpv.data.loaders.load_file(hpv.data.loaders.files.age_dist).keys()`.

### `MIGRATION_PLAN.md`

Two edits:

1. Under **M02 Sub-tasks** (line ~127), add a new bullet:
   > **Post-merge sub-task (added 2026-05-28, oversight correction):**
   > Expose all v2-shipped demographic locations by removing the
   > `_KNOWN_LOCATIONS = ['nigeria']` gate. M02 shipped only Nigeria;
   > the bundled data files cover 298 entries from UN WPP 2024 and the
   > loaders work for all of them. See
   > `docs/superpowers/specs/2026-05-28-country-support-design.md`.

2. Update the **Scope items not pinned to a milestone** entry on
   "Location-name normalization + subnational regions" to mention that
   the all-locations exposure lands ahead of any further normalization
   work.

## Validation

Per user direction: no new pytest coverage. Smoke validation:

```python
python -c "import hpvsim as hpv; \
    sim = hpv.Sim(location='india', start=1990, stop=1991, n_agents=1000); \
    sim.run(); \
    print('OK:', sim.results.dt)"
```

Plus repeating for `'rwanda'` and one alias case (e.g. `'USA'` →
`United States of America`). Manual; not in CI.

## Risk / what could go wrong

1. **Edge-case country names** — some of the 298 entries have unusual
   characters or are aggregates rather than countries (e.g.,
   `'Africa'`, `'AUKUS'`). The demographic loaders work for these but
   the resulting sim may be epidemiologically nonsensical. Acceptable:
   caller's responsibility.

2. **Missing data years per country** — `_age_data(location, year)`
   currently passes `year` through; if a country has no data for the
   requested year the loader raises. Acceptable: existing error path.

3. **Network results unrealistic outside Nigeria-like contexts** — the
   network calibration is Nigeria-tuned. Sims on `'india'` with default
   network pars will not match `hpvsim_methods_manuscript` Fig 6
   numbers. Documented in the docstring; analysis scripts override as
   needed.

## Post-implementation deltas

(Filled in at PR time if any deviations from this spec emerge.)
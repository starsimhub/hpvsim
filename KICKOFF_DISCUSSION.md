# HPVsim v3.0 Migration — Kick-off Discussion Points

## 1. Definition of done

What does "feature parity" mean for the v3.0 release?

- **Option A: Full parity** — every v2.x feature is ported, including therapeutic vaccination, DALYs analyzer, Sweep class, etc.
- **Option B: Core parity** — the features needed to replicate published and in-progress analyses (see §3), with remaining features deferred to v3.1+.
- **Option C: Core parity + deprecation list** — explicitly drop features that are unused or superseded by Starsim/STIsim equivalents.

Sub-questions:
- Do we require identical numerical output, or epidemiologically equivalent output (same distributions, within stochastic noise)?
- Which v2.x features, if any, should we explicitly *not* port? Candidates:
  - Therapeutic vaccination (`txvx`) — is anyone using this?
  - `EventSchedule` — rarely used
  - Custom `settings.py` — could rely on `ss.options` instead
  - Waning immunity — implemented in v2.x but never used in any published analysis. Port it anyway for future flexibility?

## 2. Timeline and effort

The migration plan has 10 milestones with ~50 issues. Key questions:

- **Target release date?** Mid-2026 was mentioned — is that still realistic?
- **Who is available to work on this, and how much time?**
- **What can be parallelized?** Milestones 1–4 are sequential (each builds on the prior), but testing (M8) and documentation (M9) can run in parallel.
- **Are there external deadlines driving this?** E.g., conference submissions, partner deliverables, grant milestones.

Time estimates to be provided by Ryan after initial scoping.

## 3. Scientific validation

Proposed approach: replicate a minimum of 2–3 published/in-progress analyses end-to-end with v3.0 and confirm equivalent results.

### Candidate analyses (short list)

**Published papers:**
1. **Stuart et al. (2024)** — *HPVsim: An agent-based model of HPV transmission and cervical disease*, PLOS Comp Bio. The original methods paper; replicating this is a natural baseline.
2. **Stuart et al. (2024)** — *Inferring the natural history of HPV from global cancer registries*, Sci Rep. Multi-country calibration — good test of calibration + natural history.
3. **Stuart et al. (2026)** — *The role of HPV single-dose vaccination in expanding access in GAVI-supported countries*, Vaccine. Tests vaccination interventions at scale.
4. **Lahdhiri et al. (2025)** — *HPV DNA Screening and Vaccination Strategies in Tunisia*, Sci Rep. Tests screening + vaccination workflows.
5. **Sturman et al. (2025)** — *Can pruning improve agent-based models' calibration?*, J Theor Bio. Tests calibration infrastructure.

**In-progress / under review:**
6. **HPV Faster Kenya** — Ongoing work, tests real-world application of the migration.
7. **Quantium team's models** — IPG pilot; good choice since it exercises the model via an external team.
8. **HPV Nigeria infant vaccine model** — Under review; tests infant vaccination scenarios.
9. **HPV elimination in Rwanda** — Under review; active partners, IARC collaboration.

### Proposed minimum set for v3.0 release sign-off

| # | Analysis | Why |
|---|---|---|
| 1 | Multi-country calibration (Stuart 2024, Sci Rep) | Validates core engine + calibration across many settings |
| 2 | GAVI single-dose vaccination (Stuart 2026) | Validates vaccination interventions at scale |
| 3 | HPV Faster Kenya | Validates real-world ongoing use |
| 4 | Quantium team's models | External validation by non-core team (IPG pilot) |
| 5 | HPV elimination in Rwanda | Tests active partnerships, IARC collaboration buy-in |
| 6 | HPV Nigeria infant vaccine | Tests infant vaccination, under review timing aligns |

**Discussion:** Is 6 analyses too many? Too few? Which are must-haves vs nice-to-haves?

## 4. RACI

Proposed roles — to be filled in:

| Role | Person(s) |
|---|---|
| **Responsible** (doing the migration work) | ? |
| **Accountable** (owns the release decision) | ? |
| **Consulted** (domain expertise, review) | ? |
| **Informed** (stakeholders, downstream users) | ? |

Sub-questions:
- Who reviews PRs on the `rc3` branch?
- Who owns the scientific validation (running the replication analyses)?
- Who is the point of contact for external teams (Quantium, IARC)?
- Should external collaborators (Quantium, IARC) be involved in validation, or do we validate internally first?

## 5. Open architectural decisions

These should be resolved early as they affect the migration plan:

1. **Network**: Use STIsim's `StructuredSexual` as-is, or port HPVsim's custom network? (rc3 currently uses STIsim's.)
2. **HIV module**: Use STIsim's transmission-based HIV model, or port v2.x's simpler incidence-based approach?
3. **Waning immunity**: Port it (never used in practice) or defer?
4. **Therapeutic vaccination**: Port it or defer to v3.1+?
5. **Features to explicitly drop**: Anything else we should not port?

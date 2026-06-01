"""Throwaway: test whether an inert pre-allocated agent pool perturbs the
active population's epidemic+cancer trajectory. CRN-safety check for a future
multiscale pre-allocation design. DO NOT COMMIT."""
import numpy as np
import starsim as ss
import hpvsim as hpv

N_BASE = 8000
N_POOL = 10000
N_INERT = N_POOL - N_BASE  # 2000
STOP = 2040
SEEDS = range(6)

COMMON = dict(location='nigeria', genotypes=['hpv16'], start=1990,
              stop=STOP, dt=0.25, verbose=0)


def _state_arrays(mod):
    """Return the per-genotype state arrays we need to clobber."""
    return mod


class InertPool(ss.Analyzer):
    """On first step, picks the top-N_INERT uids and makes them inert; on every
    step (including the first), re-asserts inertness so nothing re-infects them."""

    def __init__(self, n_inert):
        super().__init__()
        self.n_inert = n_inert
        self.pool_uids = None
        # Track whether pool ever participated (sanity check).
        self.ever_infected = False
        self.max_scale_seen = 0.0

    def _inert(self, sim):
        uids = self.pool_uids
        mod = sim.diseases.hpv16
        # Network exclusion + never-resplit marker.
        mod.multiscale_fine[uids] = True
        # Clear disease state.
        mod.susceptible[uids] = False
        mod.infected[uids] = False
        mod.precin[uids] = False
        mod.cin[uids] = False
        mod.cancerous[uids] = False
        for ti_name in ('ti_infected', 'ti_cin', 'ti_cancerous',
                        'ti_dead_cancer', 'ti_clearance'):
            getattr(mod, ti_name)[uids] = np.nan
        sim.people.scale[uids] = 0.0

    def step(self):
        sim = self.sim
        if self.pool_uids is None:
            auids = np.asarray(sim.people.auids)
            order = np.argsort(auids)
            self.pool_uids = ss.uids(np.sort(auids[order[-self.n_inert:]]))
        else:
            # Sanity: record if any pool agent got infected / regained scale
            # BEFORE we re-clobber this step.
            mod = sim.diseases.hpv16
            if np.any(np.asarray(mod.infected[self.pool_uids])):
                self.ever_infected = True
            sc = np.asarray(sim.people.scale[self.pool_uids])
            self.max_scale_seen = max(self.max_scale_seen, float(np.nanmax(sc)) if sc.size else 0.0)
        self._inert(sim)


def run_baseline(sd):
    s = hpv.Sim(n_agents=N_BASE, ms_agent_ratio=1, rand_seed=sd, **COMMON)
    s.run()
    inf = np.asarray(s.results.hpv16.new_infections).copy()
    can = np.asarray(s.results.hpv16.new_cancers).copy()
    return inf, can


def run_pool(sd):
    ana = InertPool(N_INERT)
    s = hpv.Sim(n_agents=N_POOL, ms_agent_ratio=1, rand_seed=sd,
                analyzers=[ana], **COMMON)
    s.init()
    # Immediately inert the top-2000 uids before run. The sim auto-adds other
    # analyzers (e.g. HPVTotal), so find ours by type.
    ana_live = next(a for a in s.analyzers.values() if isinstance(a, InertPool))
    auids = np.asarray(s.people.auids)
    pool_uids = ss.uids(np.sort(np.sort(auids)[-N_INERT:]))
    ana_live.pool_uids = pool_uids
    ana_live._inert(s)
    s.run()
    inf = np.asarray(s.results.hpv16.new_infections).copy()
    can = np.asarray(s.results.hpv16.new_cancers).copy()
    return inf, can, ana_live, s, pool_uids


def main():
    print(f"stop={STOP}, seeds={list(SEEDS)}, n_base={N_BASE}, n_pool={N_POOL}, n_inert={N_INERT}")
    for sd in SEEDS:
        b_inf, b_can = run_baseline(sd)
        p_inf, p_can, ana, s, pool_uids = run_pool(sd)

        # Sanity check at end of run.
        mod = s.diseases.hpv16
        end_infected = int(np.sum(np.asarray(mod.infected[pool_uids])))
        # auids may have changed (deaths); restrict to those still alive.
        alive_pool = pool_uids[np.isin(np.asarray(pool_uids), np.asarray(s.people.auids))]
        end_scale = np.asarray(s.people.scale[alive_pool]) if len(alive_pool) else np.array([])
        max_end_scale = float(np.nanmax(end_scale)) if end_scale.size else 0.0

        di = b_inf - p_inf
        dc = b_can - p_can
        max_abs_i = float(np.max(np.abs(di)))
        max_abs_c = float(np.max(np.abs(dc)))
        denom_i = float(np.max(np.abs(b_inf))) or 1.0
        denom_c = float(np.max(np.abs(b_can))) or 1.0
        rel_i = max_abs_i / denom_i
        rel_c = max_abs_c / denom_c
        identical = np.array_equal(b_inf, p_inf) and np.array_equal(b_can, p_can)

        print(f"seed={sd}: identical={identical} | "
              f"inf max_abs={max_abs_i:.6g} rel={rel_i:.3g} | "
              f"can max_abs={max_abs_c:.6g} rel={rel_c:.3g} | "
              f"SANITY ever_infected={ana.ever_infected} end_infected={end_infected} "
              f"max_scale_seen={ana.max_scale_seen} max_end_scale={max_end_scale} "
              f"sum_base_inf={b_inf.sum():.1f} sum_base_can={b_can.sum():.3f}")


if __name__ == '__main__':
    main()

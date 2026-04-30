"""HPV genotype as a Starsim Infection.

M01: single-genotype, transmission-only with SIS clearance (no precin/CIN/cancer).
M02 will add natural-history states (precin, CIN, cancer) and override step_state.
M03 will instantiate one HPV per genotype and add a CrossImmunity connector.

Default values for ``beta`` and ``init_prev`` are taken from v2's
``parameters.py``: ``beta=0.25`` (per-sex-act probability before sex-direction
scaling, applied per-pair via Starsim's standard
``net_beta = 1 - (1-p)**acts``), and ``init_hpv_prev`` is age- and sex-
stratified (see :data:`_INIT_HPV_PREV_TABLE`).
"""

import numpy as np
import starsim as ss


_KNOWN_GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')


# v2 default initial HPV prevalence table from
# hpvsim/_v2_legacy/parameters.py make_pars: pars['init_hpv_prev'].
# Age brackets are inclusive lower bounds; the last bracket extends to age 150.
# For M01 (single genotype) we apply the full table to HPV16.
_INIT_HPV_PREV_AGE_BRACKETS = np.array([12, 17, 24, 34, 44, 64, 80, 150])
_INIT_HPV_PREV_M = np.array([0.0, 0.25, 0.60, 0.25, 0.05, 0.01, 0.0005, 0.0])
_INIT_HPV_PREV_F = np.array([0.0, 0.35, 0.70, 0.25, 0.05, 0.01, 0.0005, 0.0])


def _age_stratified_init_prev(module, sim, uids):
    """Per-uid initial-infection probability based on v2's age/sex table.

    Used as the ``p`` callable inside ``ss.bernoulli(p=...)`` so each agent
    gets a draw against an age- and sex-specific probability matching v2.
    Argument names follow Starsim's distribution-callable convention
    (module, sim, uids).
    """
    age = np.asarray(sim.people.age[uids])
    is_female = np.asarray(sim.people.female[uids])
    # Bin agent ages into the v2 brackets.
    bin_idx = np.searchsorted(_INIT_HPV_PREV_AGE_BRACKETS, age, side='right') - 1
    bin_idx = np.clip(bin_idx, 0, len(_INIT_HPV_PREV_AGE_BRACKETS) - 1)
    out = np.zeros(len(uids))
    out[is_female] = _INIT_HPV_PREV_F[bin_idx[is_female]]
    out[~is_female] = _INIT_HPV_PREV_M[bin_idx[~is_female]]
    return out


class HPV(ss.Infection):
    """Single-genotype HPV disease module.

    The ``genotype`` attribute identifies which strain this instance models
    and is the duck-type marker M03's ``hpv.CrossImmunity`` connector will
    use to discover HPV diseases (mirrors rotasim's
    ``hasattr(disease, 'G')`` pattern).
    """

    def __init__(self, genotype='hpv16', pars=None, **kwargs):
        if genotype not in _KNOWN_GENOTYPES:
            raise ValueError(
                f'Unknown genotype {genotype!r}. Known: {list(_KNOWN_GENOTYPES)}.'
            )
        self.genotype = genotype
        if 'name' not in kwargs:
            kwargs['name'] = genotype
        super().__init__()
        # Defaults sourced from v2 parameters.py:
        #  - beta = 0.25 per sex-act (a scalar, NOT a Rate; SexualNetwork's
        #    net_beta applies it per-act via 1 - (1-p)**acts).
        #  - init_prev = age- and sex-stratified per v2's init_hpv_prev table.
        #  - dur_inf placeholder; M02 replaces with v2's get_genotype_pars
        #    duration distribution for HPV16.
        self.define_pars(
            init_prev=ss.bernoulli(p=_age_stratified_init_prev),
            beta=0.25,
            dur_inf=ss.lognorm_ex(mean=ss.years(2.0)),
        )
        self.update_pars(pars=pars, **kwargs)
        # ss.Infection already provides: susceptible, infected, rel_sus,
        # rel_trans, ti_infected. We add ti_clearance for SIS dynamics.
        self.define_states(
            ss.FloatArr('ti_clearance', label='Time of natural clearance'),
        )

    def init_results(self):
        """Add per-timestep first-infection accumulators so the post-run mean
        age-of-(first-)infection can be computed (matches v2's per-agent
        ``date_infectious`` semantics; SIS re-infections are excluded).
        """
        super().init_results()
        self.define_results(
            ss.Result('new_first_infections_count',
                      dtype=int, scale=False,
                      label='New first-time infections this step'),
            ss.Result('new_first_infections_age_sum',
                      dtype=float, scale=False,
                      label='Sum of ages at first infection this step'),
        )

    def set_prognoses(self, uids, sources=None):
        """Mark uids as infected; schedule clearance per dur_inf;
        accumulate first-infection-age sum for the per-step mean.

        Counts first infections only (not SIS re-infections), to match
        v2's per-agent ``date_infectious`` semantics (in v2's natural-
        history model HPV16 confers immunity, so each agent has at most
        one infection event recorded; M01's SIS dynamics generate
        re-infections that we exclude from this metric).
        """
        super().set_prognoses(uids, sources)
        ti = self.ti
        # Determine first-time-infected agents BEFORE we overwrite ti_infected.
        first_time_mask = np.isnan(np.asarray(self.ti_infected[uids]))
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = ti
        self.ti_clearance[uids] = ti + self.pars.dur_inf.rvs(uids)
        # Accumulate per-step first-infection (count, age_sum) so a v2-
        # equivalent mean age of (first) infection can be computed in
        # post-processing.
        if first_time_mask.any():
            first_uids = uids[first_time_mask]
            ages = np.asarray(self.sim.people.age[first_uids])
            self.results.new_first_infections_count[ti] += int(len(first_uids))
            self.results.new_first_infections_age_sum[ti] += float(ages.sum())

    def step_state(self):
        """SIS: agents past ti_clearance return to susceptible."""
        clearing = (self.infected & (self.ti_clearance <= self.ti)).uids
        if len(clearing):
            self.infected[clearing] = False
            self.susceptible[clearing] = True
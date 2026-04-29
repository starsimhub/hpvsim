"""HPV genotype as a Starsim Infection.

M01: single-genotype, transmission-only with SIS clearance (no precin/CIN/cancer).
M02 will add natural-history states (precin, CIN, cancer) and override step_state.
M03 will instantiate one HPV per genotype and add a CrossImmunity connector.
"""

import starsim as ss


_KNOWN_GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')


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
        self.define_pars(
            init_prev=ss.bernoulli(p=0.05),
            beta=ss.peryear(0.5),
            dur_inf=ss.lognorm_ex(mean=ss.years(2.0)),
        )
        self.update_pars(pars=pars, **kwargs)
        # ss.Infection already provides: susceptible, infected, rel_sus,
        # rel_trans, ti_infected. We add ti_clearance for SIS dynamics.
        self.define_states(
            ss.FloatArr('ti_clearance', label='Time of natural clearance'),
        )

    def set_prognoses(self, uids, sources=None):
        """Mark uids as infected; schedule clearance per dur_inf."""
        super().set_prognoses(uids, sources)
        ti = self.ti
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = ti
        self.ti_clearance[uids] = ti + self.pars.dur_inf.rvs(uids)

    def step_state(self):
        """SIS: agents past ti_clearance return to susceptible."""
        clearing = (self.infected & (self.ti_clearance <= self.ti)).uids
        if len(clearing):
            self.infected[clearing] = False
            self.susceptible[clearing] = True
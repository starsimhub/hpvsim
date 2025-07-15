"""
Connector for HPVsim which unites results and attributes across genotypes
"""

import starsim as ss
import stisim as sti
import sciris as sc
import numpy as np
import hpvsim as hpv

__all__ = ["HPV", "hpv_hiv_connector"]


class HPV(ss.Connector, hpv.Genotype):

    def __init__(self, pars=None, genotypes=None, **kwargs):
        """
        Class to unite the immunity and infection status across all HPV genotypes.
        """
        super().__init__(name='hpv')  # This will call Genotype.__init__ due to MRO

        # Handle parameters
        self.pars = ss.Pars()  # Wipe the Genotype's pars, since this is a connector
        default_pars = hpv.ImmPars()
        self.define_pars(**default_pars)
        self.update_pars(pars, **kwargs)

        self.genotypes = genotypes

        # # Construct cross-immunity
        # if self.pars.cross_immunity is None:
        #     cross_immunity = self.get_cross_immunity()
        #     self.pars.cross_immunity = cross_immunity

        # Define the states that are shared across genotypes
        self.define_states(
            ss.State("susceptible", label="susceptible", default=True),
            ss.State("infected", label="infected"),
            ss.State("latent", label="latent"),
            ss.State("precin", label="precin"),
            ss.State("cin", label="CIN"),
            ss.State("cancerous", label="cancerous"),
            ss.FloatArr("nti_cancer", label="Number of timesteps spent with cancer"),
            ss.FloatArr("ti_cancer", label="Timestep of cancer"),
            ss.FloatArr("ti_cancer_death", label="Timestep of cancer death"),
        )
        return

    def init_results(self):
        """ Initialize results for the HPV connector. """
        hpv.Genotype.init_results(self)  # Call the parent class's init_results
        results = sc.autolist()
        for gname in self.genotypes.keys():
            results += ss.Result(f"cancer_share_{gname}", label=f"Cancer share {gname}", scale=False)
        self.define_results(*results)
        return

    def update_results(self):
        hpv.Genotype.update_results(self)
        return

    def reset_states(self, uids=None):
        if uids is None:
            uids = self.sim.people.alive
        self.susceptible[uids] = True
        self.infected[uids] = False
        self.latent[uids] = False
        self.precin[uids] = False
        self.cin[uids] = False
        return

    def step_genotype_states(self):
        """
        Update states prior to transmission
        """
        for genotype in self.genotypes.values():
            genotype._step_states()
        return

    def step_states(self):
        """
        Check agents' disease status across all genotypes and update their states accordingly.
        """
        self.reset_states()  # Clear states

        for genotype in self.genotypes.values():
            self.susceptible[:] &= genotype.susceptible[:]
            self.infected[:] |= genotype.infected[:]
            self.latent[:] |= genotype.latent[:]
            self.precin[:] |= genotype.precin[:]
            self.cin[:] |= genotype.cin[:]

            # For cancers, we take the minimum across genotypes. It's possible that
            # an individual has multiple genotypes, but we want to track the earliest
            # cancer diagnosis and the earliest cancer death time.
            # We will also need to wipte any later dates
            self.ti_cancer[:] = np.fmin(self.ti_cancer[:], genotype.ti_cancer[:])
            self.ti_cancer_death[:] = np.fmin(self.ti_cancer_death[:], genotype.ti_cancer_death[:])
            self.nti_cancer[:] = np.fmin(self.nti_cancer[:], genotype.nti_cancer[:])
            later_cancers = genotype.ti_cancer[:] > self.ti_cancer[:]
            self.ti_cancer[later_cancers] = np.nan  # Wipe later cancer dates
            self.ti_cancer_death[later_cancers] = np.nan  # Wipe later cancer death dates

            # For infections and CINs, we take the maximum across genotypes
            # This is because an individual can be infected with multiple genotypes, and we want to
            # track the most recent infection time
            self.ti_infected[:] = np.fmax(self.ti_infection[:], genotype.ti_infection[:])
            self.ti_precin[:] = np.fmax(self.ti_precin[:], genotype.ti_precin[:])
            self.ti_cin[:] = np.fmax(self.ti_cin[:], genotype.ti_cin[:])

        return

    def update_immunity(self):

        pass

    def step(self):
        """ Update the cross-immunity and relative susceptibility and severity """
        self.step_genotype_states()  # Update states for each genotype
        self.step_states()  # Update the connector states based on genotypes
        self.update_immunity()
        return

    def infect(self):
        """ Don't allow HPV infections through this connector """
        pass

class hpv_hiv_connector(ss.Connector):
    def __init__(self, hpv=None, hiv=None, pars=None, **kwargs):
        super().__init__()
        if not isinstance(hpv, ss.Connector):
            print("We are expecting the HPV superconnector here")
            self.hpv = hpv
        else:
            self.hpv = hpv
        self.hiv = hiv
        self.define_pars(
            rel_sus_hpv=lambda cd4: [2.2 if i < 200 else 1.5 for i in cd4],
            rel_sev_hpv=lambda cd4: [2.2 if i < 200 else 1.5 for i in cd4],
        )
        self.update_pars(pars, **kwargs)

    def step(self):
        hiv_inds = self.hiv.infected.true()
        self.hpv.rel_sev[hiv_inds] = self.pars.rel_sev_hpv(self.hiv.cd4[hiv_inds])
        self.hpv.rel_sus[hiv_inds] = self.pars.rel_sus_hpv(self.hiv.cd4[hiv_inds])
        return


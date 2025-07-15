"""
Connector for HPVsim which unites results and attributes across genotypes
"""

import starsim as ss
import stisim as sti
import sciris as sc
import numpy as np
import hpvsim as hpv

__all__ = ["HPV", "hpv_hiv_connector"]


class HPV(ss.Connector):

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
        hpv.Genotype.init_results(self)
        results = sc.autolist()
        for gname in self.sim.genotypes.keys():
            results += ss.Result(f"cancer_share_{gname}", label=f"Cancer share {gname}", scale=False)
        self.define_results(*results)

        # genotypes = self.sim.genotypes
        # base_genotype = genotypes[0] if genotypes else None
        # results = sc.autolist([sc.dcp(r) for r in base_genotype.results.values() if isinstance(r, ss.Result)])

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
        for genotype in self.sim.genotypes.values():
            self.susceptible[:] &= genotype.susceptible[:]
            self.infected[:] |= genotype.infected[:]
            self.latent[:] |= genotype.latent[:]
            self.precin[:] |= genotype.precin[:]
            self.cin[:] |= genotype.cin[:]
            self.ti_cancer[:] = np.minimum(self.ti_cancer[:], genotype.ti_cancer[:])
            self.ti_cancer_death[:] = np.minimum(self.ti_cancer_death[:], genotype.ti_cancer_death[:])
            self.nti_cancer[:] = np.minimum(self.nti_cancer[:], genotype.nti_cancer[:])
        return

    def update_immunity(self):
        """
        Update the relative susceptibility and severity of each genotype based on cross-immunity.
        TODO, refactor/remove
        """
        cross_immunity = self.pars.cross_immunity
        self.sus_imm[:] = 0
        self.sev_imm[:] = 0
        for i, genotype in enumerate(self.genotypes):
            for other_genotype in self.genotypes:
                self.sus_imm[:] += (
                    cross_immunity[genotype.name][other_genotype.name]
                    * self.genotypes[i].sus_imm[:]
                )
                self.sev_imm[:] += (
                    cross_immunity[genotype.name][other_genotype.name]
                    * self.genotypes[i].sev_imm[:]
                )
            self.sev_imm[:] *= self.rel_sev[:]
            self.sus_imm[:] *= self.rel_sus[:]
            self.genotypes[i].rel_sev[:] = 1 - np.minimum(
                self.sev_imm, np.ones_like(self.sev_imm)
            )
            self.genotypes[i].rel_sus[:] = 1 - np.minimum(
                self.sus_imm, np.ones_like(self.sus_imm)
            )

        ti = self.ti
        for gtype in self.genotypes:  # TODO, fix
            other_gtypes = [g for g in self.genotypes if g != gtype]
            # find women who became cancerous today
            cancerous_today = (gtype.ti_cancer == ti).uids
            if len(cancerous_today):
                for other_gtype in other_gtypes:
                    cancerous_future = (other_gtype.ti_cancer > ti).uids
                    remove_uids = cancerous_today.intersect(cancerous_future)
                    other_gtype.ti_cancer[remove_uids] = np.nan
        return

    def step(self):
        """ Update the cross-immunity and relative susceptibility and severity """
        self.step_genotype_states()
        self.step_states()
        self.update_states()
        # self.update_immunity()

        return


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


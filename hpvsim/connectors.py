"""
Connector for HPVsim which unites results and attributes across genotypes
"""

import starsim as ss
import sciris as sc
import numpy as np

__all__ = ["hpv", "hpv_hiv_connector"]


class hpv(ss.Connector):

    def __init__(self, genotypes, pars=None, **kwargs):
        super().__init__()
        self.genotypes = sc.promotetolist(genotypes)
        self.define_pars(
            cross_imm_med=0.3,
            cross_imm_high=0.5,
            cross_immunity=None,
        )
        self.update_pars(pars, **kwargs)

        if self.pars.cross_immunity is None:
            cross_immunity = self.get_cross_immunity()
            self.pars.cross_immunity = cross_immunity

        self.define_states(
            ss.FloatArr("sus_imm", default=0, label="immunity to infection"),
            ss.FloatArr("sev_imm", default=0, label="immunity to severe disease"),
            ss.FloatArr("rel_sev", default=1, label="relative severity"),
            ss.FloatArr("rel_sus", default=1, label="relative susceptibility"),
            ss.FloatArr("n_precin", default=0, label="number precin"),
            ss.FloatArr("n_cin", default=0, label="number cin"),
            ss.FloatArr("n_cancerous", default=0, label="number cancerous"),
            ss.State("precin", default=False, label="precin"),
            ss.State("cin", default=False, label="cin"),
            ss.State("cancerous", default=False, label="cancerous"),
        )

        return

    def init_results(self):
        super().init_results()

        results = [
            ss.Result("infections", label="HPV infections"),
            ss.Result("cins", label="CINs"),
            ss.Result("cancers", label="cancers"),
            ss.Result("cancer_incidence", label="Cancer incidence", scale=False),
            ss.Result("prevalence", label="Prevalence", scale=False),
            ss.Result("n_hpv_18_49", label="HPV 18-49", scale=True),
            ss.Result("n_pop_18_49", label="Population 18-49", scale=True),
            ss.Result(
                "hpv_prevalence_18_49", label="HPV prevalence 18-49", scale=False
            ),
            ss.Result("prevalence_15_24", label="Prevalence 15-24", scale=False),
            ss.Result("prevalence_25_34", label="Prevalence 25-34", scale=False),
            ss.Result("prevalence_35_44", label="Prevalence 35-44", scale=False),
            ss.Result("prevalence_45_54", label="Prevalence 45-54", scale=False),
            ss.Result("prevalence_55_64", label="Prevalence 55-64", scale=False),
            ss.Result("cancers_20_34", label="Cancers 20-34", scale=True),
            ss.Result("cancers_35_49", label="Cancers 35-49", scale=True),
            ss.Result("cancers_50_64", label="Cancers 50-64", scale=True),
            ss.Result("cancers_65_79", label="Cancers 65-79", scale=True),
            ss.Result("sus_20_34", label="Sus pop 20-34", scale=True),
            ss.Result("sus_35_49", label="Sus pop 35-49", scale=True),
            ss.Result("sus_50_64", label="Sus pop 50-64", scale=True),
            ss.Result("sus_65_79", label="Sus pop 65-79", scale=True),
        ]
        for genotype in self.genotypes:
            results.append(
                ss.Result(
                    f"cancer_share_{genotype}",
                    label=f"Cancer distribution {genotype}",
                    scale=False,
                )
            )
            results.append(
                ss.Result(
                    f"cin_share_{genotype}",
                    label=f"CIN share {genotype}",
                    scale=False,
                )
            )
            results.append(
                ss.Result(
                    f"precin_share_{genotype}",
                    label=f"Pre-CIN share {genotype}",
                    scale=False,
                )
            )

        self.define_results(*results)
        return

    def update_results(self):
        super().update_results()
        ti = self.ti
        women = self.sim.people.female.uids
        ages = [15, 25, 35, 45, 55]
        cancer_ages = [20, 35, 50, 65]

        new_cancers = []
        self.n_precin[:] = 0
        self.n_cin[:] = 0
        self.n_cancerous[:] = 0
        self.precin[:] = False
        self.cin[:] = False
        self.cancerous[:] = False

        for module in self.modules:
            new_cancers = np.array((module.ti_cancer == ti).uids).tolist()
            self.n_precin[:] += module.precin[:]
            self.n_cin[:] += module.cin[:]
            self.n_cancerous[:] += module.cancerous[:]

        # Calculate the share of each genotype in CIN and Cancer
        for i, module in enumerate(self.modules):
            self.results[f"precin_share_{self.genotypes[i]}"][ti] = sc.safedivide(
                module.precin.true().sum(), self.n_precin.sum()
            )

            self.results[f"cin_share_{self.genotypes[i]}"][ti] = sc.safedivide(
                module.cin.true().sum(), self.n_cin.sum()
            )
            self.results[f"cancer_share_{self.genotypes[i]}"][ti] = sc.safedivide(
                module.cancerous.true().sum(), self.n_cancerous.sum()
            )

        new_cancers = ss.uids(list(set(new_cancers)))
        precin_uids = self.get_precin_uids()
        cin_uids = self.get_cin_uids()
        cancerous_uids = self.get_cancerous_uids()
        self.precin[precin_uids] = True
        self.cin[cin_uids] = True
        self.cancerous[cancerous_uids] = True

        # Calculate cancer incidence
        scale_factor = 1e5
        sus_pop = np.setdiff1d(women, self.cancerous.true())
        denominator = len(sus_pop) / scale_factor
        self.results["cancer_incidence"][ti] = sc.safedivide(
            len(new_cancers), denominator
        )

        age_group = (
            (self.sim.people.female)
            & (self.sim.people.age >= 18)
            & (self.sim.people.age < 50)
        ).uids
        infectious_age = self.precin.true().intersect(age_group)
        self.results["n_hpv_18_49"][ti] = len(infectious_age)
        self.results["n_pop_18_49"][ti] = len(age_group)
        self.results[f"hpv_prevalence_18_49"][ti] = sc.safedivide(
            len(infectious_age), len(age_group)
        )

        for age in ages:
            age_group = (
                (self.sim.people.female)
                & (self.sim.people.age >= age)
                & (self.sim.people.age < age + 10)
            ).uids
            infectious_age = self.precin.true().intersect(age_group)
            self.results[f"prevalence_{age}_{age+9}"][ti] = sc.safedivide(
                len(infectious_age), len(age_group)
            )

        for age in cancer_ages:
            age_group = (
                (self.sim.people.female)
                & (self.sim.people.age >= age)
                & (self.sim.people.age < age + 15)
            ).uids
            cancer_ages = new_cancers.intersect(age_group)
            sus_pop_ages = sus_pop.intersect(age_group)
            self.results[f"cancers_{age}_{age+14}"][ti] = len(cancer_ages)
            self.results[f"sus_{age}_{age+14}"][ti] = len(sus_pop_ages)

        return

    def get_precin_uids(self):
        uids = self.n_precin.true()
        if len(uids):
            return uids
        else:
            return ss.uids([])

    def get_cin_uids(self):
        uids = self.n_cin.true()
        if len(uids):
            return uids
        else:
            return ss.uids([])

    def get_cancerous_uids(self):
        uids = self.n_cancerous.true()
        if len(uids):
            return uids
        else:
            return ss.uids([])

    def init_post(self):
        """Initialize the values of the states; the last step of initialization"""
        super().init_post()
        return

    def step(self):

        cross_immunity = self.pars.cross_immunity
        self.sus_imm[:] = 0
        self.sev_imm[:] = 0
        for i, genotype in enumerate(self.genotypes):
            for other_genotype in self.genotypes:
                self.sus_imm[:] += (
                    cross_immunity[genotype.name][other_genotype.name]
                    * self.modules[i].sus_imm[:]
                )
                self.sev_imm[:] += (
                    cross_immunity[genotype.name][other_genotype.name]
                    * self.modules[i].sev_imm[:]
                )
            self.sev_imm[:] *= self.rel_sev[:]
            self.sus_imm[:] *= self.rel_sus[:]
            self.modules[i].rel_sev[:] = 1 - np.minimum(
                self.sev_imm, np.ones_like(self.sev_imm)
            )
            self.modules[i].rel_sus[:] = 1 - np.minimum(
                self.sus_imm, np.ones_like(self.sus_imm)
            )

        ti = self.ti
        for module in self.modules:  # TODO, fix
            other_modules = [m for m in self.modules if m != module]
            # find women who became cancerous today
            cancerous_today = (module.ti_cancer == ti).uids
            if len(cancerous_today):
                for other_module in other_modules:
                    cancerous_future = (other_module.ti_cancer > ti).uids
                    remove_uids = cancerous_today.intersect(cancerous_future)
                    other_module.ti_cancer[remove_uids] = np.nan
        return

    def get_cross_immunity(self):
        """
        Get the cross immunity between each genotype in a sim
        """
        cross_imm_med = self.pars.cross_imm_med
        cross_imm_high = self.pars.cross_imm_high
        own_imm_hr = 0.9
        genotypes = self.genotypes

        default_pars = dict(
            # All values based roughly on https://academic.oup.com/jnci/article/112/10/1030/5753954 or assumptions
            hpv16=dict(
                hpv16=1.0,  # Default for own-immunity
                hpv18=cross_imm_high,
                hi5=cross_imm_med,
                ohr=cross_imm_med,
                hr=cross_imm_med,
                lr=cross_imm_med,
            ),
            hpv18=dict(
                hpv16=cross_imm_high,
                hpv18=1.0,  # Default for own-immunity
                hi5=cross_imm_med,
                ohr=cross_imm_med,
                hr=cross_imm_med,
                lr=cross_imm_med,
            ),
            hi5=dict(
                hpv16=cross_imm_med,
                hpv18=cross_imm_med,
                hi5=own_imm_hr,
                ohr=cross_imm_med,
                hr=cross_imm_med,
                lr=cross_imm_med,
            ),
            ohr=dict(
                hpv16=cross_imm_med,
                hpv18=cross_imm_med,
                hi5=cross_imm_med,
                ohr=own_imm_hr,
                hr=cross_imm_med,
                lr=cross_imm_med,
            ),
            lr=dict(
                hpv16=cross_imm_med,
                hpv18=cross_imm_med,
                hi5=cross_imm_med,
                ohr=cross_imm_med,
                hr=cross_imm_med,
                lr=own_imm_hr,
            ),
        )

        genotype_pars = dict()
        for genotype in genotypes:
            genotype_pars[genotype] = dict()
            for other_genotype in genotypes:
                genotype_pars[genotype][other_genotype] = default_pars[genotype.name][
                    other_genotype.name
                ]

        return genotype_pars


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


# Analyzer to compute results that will be used in calibration

import numpy as np
import starsim as ss
import sciris as sc

__all__ = ["hiv_hpv_results"]


class hiv_hpv_results(ss.Analyzer):

    def __init__(self, hiv=None, hpv=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hiv = hiv
        self.hpv = hpv
        return

    def init_results(self):
        """Initialize results"""
        super().init_results()
        results = [
            ss.Result(
                "hpv_prevalence_hiv_pos",
                label="HPV prevalence in HIV positive",
                scale=False,
            ),
            ss.Result(
                "hpv_prevalence_hiv_neg",
                label="HPV prevalence in HIV negative",
                scale=False,
            ),
            ss.Result(
                "hpv_prevalence_hiv_pos_15_24",
                label="HPV prevalence in HIV pos 15-24",
                scale=False,
            ),
            ss.Result(
                "hpv_prevalence_hiv_pos_25_34",
                label="HPV prevalence in HIV pos 25-34",
                scale=False,
            ),
            ss.Result(
                "hpv_prevalence_hiv_pos_35_44",
                label="HPV prevalence in HIV pos 35-44",
                scale=False,
            ),
            ss.Result(
                "hpv_prevalence_hiv_pos_45_54",
                label="HPV prevalence inHIV pos 45-54",
                scale=False,
            ),
            ss.Result(
                "hpv_prevalence_hiv_pos_55_64",
                label="HPV prevalence in HIV pos 55-64",
                scale=False,
            ),
            ss.Result(
                "hpv_prevalence_hiv_neg_15_24",
                label="HPV prevalence in HIV neg 15-24",
                scale=False,
            ),
            ss.Result(
                "hpv_prevalence_hiv_neg_25_34",
                label="HPV prevalence in HIV neg 25-34",
                scale=False,
            ),
            ss.Result(
                "hpv_prevalence_hiv_neg_35_44",
                label="HPV prevalence in HIV neg 35-44",
                scale=False,
            ),
            ss.Result(
                "hpv_prevalence_hiv_neg_45_54",
                label="HPV prevalence in HIV neg 45-54",
                scale=False,
            ),
            ss.Result(
                "hpv_prevalence_hiv_neg_55_64",
                label="HPV prevalence in HIV neg 55-64",
                scale=False,
            ),
        ]
        self.define_results(*results)
        return

    def update_results(self):
        """Update results"""
        ti = self.ti
        people = self.sim.people
        hiv = self.hiv
        hpv = self.hpv

        hiv_pos_women = hiv.infected & people.female
        hiv_neg_women = ~hiv.infected & people.female

        def cond_prob(num, denom):
            n_num = np.count_nonzero(num & denom)
            n_denom = np.count_nonzero(denom)
            return sc.safedivide(n_num, n_denom)

        self.results["hpv_prevalence_hiv_pos"][ti] = cond_prob(hpv.precin, hiv.infected)
        self.results["hpv_prevalence_hiv_neg"][ti] = cond_prob(
            hpv.precin, ~hiv.infected
        )
        hiv_pos_age_groups = {
            "hpv_prevalence_hiv_pos_15_24": (15, 24),
            "hpv_prevalence_hiv_pos_25_34": (25, 34),
            "hpv_prevalence_hiv_pos_35_44": (35, 44),
            "hpv_prevalence_hiv_pos_45_54": (45, 54),
            "hpv_prevalence_hiv_pos_55_64": (55, 64),
        }
        for key, (age_min, age_max) in hiv_pos_age_groups.items():
            group = hiv_pos_women & (people.age >= age_min) & (people.age <= age_max)
            self.results[key][ti] = cond_prob(hpv.precin, group)

        hiv_neg_age_groups = {
            "hpv_prevalence_hiv_neg_15_24": (15, 24),
            "hpv_prevalence_hiv_neg_25_34": (25, 34),
            "hpv_prevalence_hiv_neg_35_44": (35, 44),
            "hpv_prevalence_hiv_neg_45_54": (45, 54),
            "hpv_prevalence_hiv_neg_55_64": (55, 64),
        }
        for key, (age_min, age_max) in hiv_neg_age_groups.items():
            group = hiv_neg_women & (people.age >= age_min) & (people.age <= age_max)
            self.results[key][ti] = cond_prob(hpv.precin, group)

        return

    def step(self):
        return

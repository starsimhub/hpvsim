"""
Module for HPV
"""

import numpy as np
import starsim as ss
import sciris as sc
import stisim as sti
import hpvsim as hpv

__all__ = ["make_hpv", "Genotype", "get_genotype_choices"]


class Genotype(sti.BaseSTI):
    """
    Base class for a single genotype of HPV
    """

    def __init__(self, pars=None, genotype=None, **kwargs):
        super().__init__(name=genotype)

        # Handle parameters
        default_pars = hpv.HPVPars(genotype=genotype)
        self.define_pars(**default_pars)
        self.update_pars(pars, **kwargs)

        self.define_states(
            # States
            ss.State("latent", label="latent"),
            ss.State("precin", label="precin"),
            ss.State("cin", label="cin"),
            ss.State("cancerous", label="cancerous"),

            # Duration and timestep of states
            ss.FloatArr("dur_precin", label="Duration of infection without HSIL (years)"),
            ss.FloatArr("dur_cin", label="Duration of HSIL (years)"),
            ss.FloatArr("dur_cancer", label="Duration of cancer (years)"),
            ss.FloatArr("nti_precin", label="Number of timesteps spent with infection without HSIL"),
            ss.FloatArr("nti_cin", label="Number of timesteps spent with HSIL"),
            ss.FloatArr("nti_cancer", label="Number of timesteps spent with cancer"),
            ss.FloatArr("ti_cancer", label="Timestep of cancer"),
            ss.FloatArr("ti_cancer_death", label="Timestep of cancer death"),
            ss.FloatArr("ti_cin", label="Timestep of CIN"),
            ss.FloatArr("ti_clearance", label="Timestep of clearance"),

            # Immunity states
            ss.FloatArr("rel_sev", default=1, label="relative severity"),
            ss.FloatArr("sus_imm", default=0, label="immunity to infection"),
            ss.FloatArr("sev_imm", default=0, label="immunity to severe disease"),
        )

        return

    def init_results(self):
        super().init_results()
        results = [
            ss.Result("new_cins", label="CINs"),
            ss.Result("new_cancers", label="Cancers"),
            ss.Result("cancer_incidence", label="Cancer incidence", scale=False),
            ss.Result("cancer_deaths", label="Cancer deaths"),
        ]
        self.define_results(*results)
        return

    def step_state(self):
        """
        Update states prior to transmission
        """
        self.rel_sus[:] = 1
        self.rel_sev[:] = 1
        self.update_infection()
        return

    def set_prognoses(self, uids, sources=None):
        """
        Set the prognoses for people infected with HPV
        """
        self.wipe_dates(uids)  # Wipe any previous dates

        # First separate out men and women
        m_uids = uids & self.sim.people.male
        f_uids = uids & self.sim.people.female
        self.ti_infected[uids] = self.ti
        self.infected[uids] = True
        self.precin[f_uids] = True
        self.susceptible[uids] = False

        # Deal with men first
        timesteps_infected_m = self.pars.dur_infection_male.rvs(m_uids)
        self.ti_clearance[m_uids] = self.ti + timesteps_infected_m

        # Set the duration of precin and determine who will progress to CIN
        self.nti_precin[f_uids] = self.pars.dur_precin.rvs(f_uids)  # * self.sev_imm[uids]  # Duration in timesteps
        self.dur_precin[f_uids] = self.nti_precin[f_uids] * self.t.dt_year  # Duration of infection in years
        cin_probs = self.get_cin_prob(f_uids)  # Function determining CIN prob is based on duration in years

        self.pars.cin_prob.set(cin_probs)
        cin, no_cin = self.pars.cin_prob.split(f_uids)
        self.ti_cin[cin] = self.ti + self.nti_precin[cin]
        self.ti_clearance[no_cin] = self.ti + self.nti_precin[no_cin]

        # Set the duration of CIN and determine who will progress to cancer
        self.nti_cin[cin] = self.pars.dur_cin.rvs(cin)  # * self.sev_imm[cin]
        self.dur_cin[cin] = self.nti_cin[cin] * self.t.dt_year  # Duration of HSIL in years
        cancer_probs = self.get_cancer_prob(cin)
        self.pars.cancer_prob.set(cancer_probs)
        cancer, no_cancer = self.pars.cancer_prob.split(cin)
        self.ti_cancer[cancer] = self.ti + self.nti_cin[cancer]
        self.ti_clearance[no_cancer] = self.ti + self.nti_cin[no_cancer]

        # Set duration of cancer and time of cancer mortality
        self.nti_cancer[cancer] = self.pars.dur_cancer.rvs(cancer)
        self.dur_cancer[cancer] = self.nti_cancer[cancer] * self.t.dt_year  # Duration of cancer in years
        self.ti_cancer_death[cancer] = self.ti + self.nti_cancer[cancer]
        return

    def set_immunity(self, uids):
        """
        Set immunity levels for those who've just cleared
        """
        sero_converted = self.pars.sero_prob.filter(uids)
        init_imm = self.pars.init_imm.rvs(sero_converted)
        init_cell_imm = self.pars.init_cell_imm.rvs(sero_converted)
        self.sus_imm[sero_converted] = init_imm
        self.sev_imm[sero_converted] = init_cell_imm
        return

    def clear_infection(self, uids):
        self.susceptible[uids] = True
        self.infected[uids] = False
        self.latent[uids] = False
        self.precin[uids] = False
        self.cin[uids] = False
        self.ti_clearance[uids] = self.ti
        return

    def wipe_dates(self, uids):
        """
        Clear all previous dates, times, and durations, except for ti_infected.
        This is called when a person gets infected or treated.
        """
        self.ti_cin[uids] = np.nan
        self.ti_cancer[uids] = np.nan
        self.ti_clearance[uids] = np.nan
        self.nti_precin[uids] = np.nan
        self.nti_cin[uids] = np.nan
        self.nti_cancer[uids] = np.nan
        self.dur_precin[uids] = np.nan
        self.dur_cin[uids] = np.nan
        self.dur_cancer[uids] = np.nan
        return

    def update_infection(self):
        """
        Update infection states
        """
        ti = self.ti

        # Find men who clear infection
        new_clearance = self.infected & self.sim.people.male & (self.ti_clearance <= ti)
        if new_clearance.any():
            self.clear_infection(new_clearance)

        # Find women without HSIL who clear infection
        new_clearance = self.precin & (self.ti_clearance <= ti)
        if new_clearance.any():
            self.clear_infection(new_clearance)
            self.set_immunity(new_clearance)

        # Find those who progress to CIN
        new_progression = self.precin & (self.ti_cin <= ti)
        if new_progression.any():
            self.precin[new_progression] = False
            self.cin[new_progression] = True
            self.ti_cin[new_progression] = ti

        # Find those who clear CIN
        new_clearance = self.cin & (self.ti_clearance <= ti)
        if new_clearance.any():
            self.clear_infection(new_clearance)
            self.set_immunity(new_clearance)

        # Find those who progress to cancer
        new_cancers = self.cin & (self.ti_cancer <= ti)
        if new_cancers.any():
            self.cin[new_cancers] = False
            self.infected[new_cancers] = False
            self.cancerous[new_cancers] = True
            self.ti_cancer[new_cancers] = ti

        # Find those who die of cancer
        new_deaths = self.cancerous & (self.ti_cancer_death <= ti)
        if new_deaths.any():
            self.cancerous[new_deaths] = False
            self.ti_cancer_death[new_deaths] = ti
            self.sim.people.request_death(new_deaths)

        return

    def get_cancer_prob(self, uids):
        """
        Get the probability of progressing to cancer
        """
        dur_cin = self.dur_cin[uids]
        sev = hpv.compute_severity_integral(dur_cin, rel_sev=self.rel_sev[uids], pars=self.pars.cin_fn)
        tp = self.pars.cancer_fn.transform_prob
        cancer_probs = 1 - np.power(1 - tp, sev**2)
        return cancer_probs

    def get_cin_prob(self, uids):
        """
        Get the probability of progressing to CIN
        """
        return hpv.logf2(self.dur_precin[uids] * self.rel_sev[uids], **self.pars.cin_fn)

    def update_results(self):
        super().update_results()
        res = self.results
        ti = self.ti
        ppl = self.sim.people
        f = ppl.female

        # Incident CINs and cancers
        res.new_cins[ti] = np.count_nonzero(np.round(self.ti_cin) == ti)
        res.new_cancers[ti] = np.count_nonzero(np.round(self.ti_cancer) == ti)

        # Calculate cancer incidence
        scale_factor = 1e5
        sus_pop = (ppl.age >= 15) & f & (~self.cancerous)
        denominator = np.count_nonzero(sus_pop) / scale_factor
        res[ti] = sc.safedivide(res.new_cancers[ti], denominator)

        return


def get_genotype_choices():
    """
    Define valid genotype names
    """
    # List of choices available
    choices = {
        'hpv16':    ['hpv16', '16'],
        'hpv18':    ['hpv18', '18'],
        'hi5':      ['hi5hpv', 'hi5hpv', 'cross-protective'],
        'ohr':      ['ohrhpv', 'non-cross-protective'],
        'hr':       ['allhr', 'allhrhpv', 'hrhpv', 'oncogenic', 'hr10', 'hi10'],
        'lo':       ['lohpv'],
    }
    mapping = {name: key for key,synonyms in choices.items() for name in synonyms} # Flip from key:value to value:key
    return choices, mapping


def make_hpv(genotype=None, hpv_pars=None, **kwargs):
    """
    Factory function to create HPV modules based on provided genotypes and parameters.
    If genotypes are not provided, defaults to HPV16
    """
    if genotype is None: genotype = 16  # Assign default

    # Define the options for genotypes
    g_options, g_mapping = get_genotype_choices()
    if sc.isnumber(genotype): genotype = f'hpv{genotype}'  # Convert e.g. 16 to hpv16
    if sc.checktype(genotype, str):
        if genotype not in g_options.keys():
            errormsg = f'Genotype {genotype} is not one of the inbuilt options.'
            raise ValueError(errormsg)

        else:
            hpv_pars = sc.mergedicts(hpv.make_genotype_pars(genotype), hpv_pars)
            hpv_module = Genotype(genotype=genotype, pars=hpv_pars, name=genotype, **kwargs)

    return hpv_module

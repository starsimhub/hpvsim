"""
Module for HPV
"""

import numpy as np
import starsim as ss
import sciris as sc
import stisim as sti
import hpvsim as hpv

__all__ = ["make_hpv", "Genotype", "HPV", "get_genotype_choices"]


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
        self.add_states()
        return

    def add_states(self):
        states = [
            # Mutually exclusive & collectively exhaustive viral infection states
            # - Susceptible, added by the Infection base class
            # - Infectious, a derived state defined as infected & ~inactive
            # - Inactive, added below, includes latent infections and those with cancer
            ss.State("inactive", label="Inactive"),

            # Dysplasia states
            ss.State("precin", label="HPV without HSIL"),
            ss.State("cin", label="HPV with HSIL"),
            ss.State("cancerous", label="Cancerous"),
            ss.State("dead_cancer", label="Died of cancer"),

            # Duration and timestep of states
            ss.FloatArr("dur_precin", label="Duration of infection without HSIL (years)"),
            ss.FloatArr("dur_cin", label="Duration of HSIL (years)"),
            ss.FloatArr("dur_cancer", label="Duration of cancer (years)"),
            ss.FloatArr("nti_precin", label="Number of timesteps spent with infection without HSIL"),
            ss.FloatArr("nti_cin", label="Number of timesteps spent with HSIL"),
            ss.FloatArr("nti_cancer", label="Number of timesteps spent with cancer"),
            ss.FloatArr("ti_cin", label="Timestep of CIN"),
            ss.FloatArr("ti_cancer", label="Timestep of cancer"),
            ss.FloatArr("ti_cancer_death", label="Timestep of cancer death"),
            ss.FloatArr("ti_clearance", label="Timestep of clearance / control"),

            # Immunity states
            ss.FloatArr("rel_sev", default=1, label="relative severity"),
            ss.FloatArr("own_sus_imm", default=0, label="Self-immunity to infection"),
            ss.FloatArr("own_sev_imm", default=0, label="Self-immunity to severe disease"),
            ss.FloatArr("sus_imm", default=0, label="Immunity to infection"),
            ss.FloatArr("sev_imm", default=0, label="Immunity to severe disease"),
        ]
        self.define_states(*states)
        return

    # Derived states
    @property
    def infectious(self):
        """
        Inactive infections include people with cancer and people with latent infections
        """
        return self.infected & ~self.inactive

    @property
    def latent(self):
        """
        Latent infections are those that are inactive but not cleared
        """
        return self.inactive & ~self.cancerous

    @property
    def abnormal(self):
        """
        Boolean array of everyone with abnormal cells. Includes women with CIN and cancer
        """
        return self.cin | self.cancerous

    def init_results(self):
        """ Initialize results for the HPV genotype."""
        super().init_results()  # Adds all the age/sex results from BaseSTI
        results = [
            ss.Result("new_cins", label="CINs"),
            ss.Result("new_cancers", label="Cancers"),
            ss.Result("cancer_incidence", label="Cancer incidence", scale=False),
            ss.Result("cancer_deaths", label="Cancer deaths"),
        ]
        self.define_results(*results)
        return

    def set_prognoses(self, uids, sources=None, record_ti_infected=True):
        """
        Set the prognoses for people infected with HPV
        """
        self.wipe_dates(uids)  # Wipe any previous dates

        # First separate out men and women
        m_uids = uids & self.sim.people.male
        f_uids = uids & self.sim.people.female
        if record_ti_infected: self.ti_infected[uids] = self.ti
        self.set_infection(uids)  # Set the infection state for all uids

        # Deal with men first
        timesteps_infected_m = self.pars.dur_infection_male.rvs(m_uids)
        self.ti_clearance[m_uids] = self.ti + timesteps_infected_m

        # Set the duration of precin and determine who will progress to CIN
        self.nti_precin[f_uids] = self.pars.dur_precin.rvs(f_uids) * (1 - self.sev_imm[f_uids])  # Duration in timesteps
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
        Set immunity levels for those who've just cleared an infection or CIN.
        Everyone gets sev_imm, which is intended to represent T-cell immunity
        and shortens the duration of subsequent infections.
        Sero-converted individuals get sus_imm, which is intended to represent
        B-cell immunity and prevents re-infection.
        """
        sero_converted = self.pars.sero_prob.filter(uids)
        inf_imm = self.pars.inf_imm.rvs(sero_converted)
        cell_imm = self.pars.cell_imm.rvs(uids)
        self.own_sus_imm[sero_converted] = np.maximum(self.own_sus_imm[sero_converted], inf_imm)
        self.own_sev_imm[uids] = np.maximum(self.own_sev_imm[uids], cell_imm)
        return

    def set_infection(self, uids):
        """ Set infection states for new or reactivated infections """
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.inactive[uids] = False
        self.precin[uids] = True
        self.cin[uids] = False
        return

    def clear_infection(self, uids):
        """ Clear the infection for the given uids. """
        self.susceptible[uids] = True
        self.infected[uids] = False
        self.inactive[uids] = False
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

    def step_state(self):
        pass

    def _step_state(self):
        """
        Logic that would normally be called at each time step to update the states of the HPV module.
        Using a different method name because we don't want this to be called automatically by the simulation step.
        Instead, it gets called by the HPV connector
        """
        ti = self.ti

        # Find men who clear infection
        new_clearance = self.infected & self.sim.people.male & (self.ti_clearance <= ti)
        if new_clearance.any():
            self.clear_infection(new_clearance)

        # Find those who progress to CIN
        new_progression = self.precin & (self.ti_cin <= ti)
        if new_progression.any():
            self.precin[new_progression] = False
            self.cin[new_progression] = True
            self.ti_cin[new_progression] = ti

        # Find women who clear or control infection
        new_undetectables = self.infected & (self.ti_clearance <= ti)
        if new_undetectables.any():
            # Check if we're modeling latency
            if self.pars.p_control.pars.p > 0:
                controlled, cleared = self.pars.p_control.split(new_undetectables)
                self.susceptible[controlled] = False  # They are still infected
                self.inactive[controlled] = True  # They are not infectious
                self.latent[controlled] = True
                self.ti_clearance[controlled] = np.nan
            else:
                cleared = new_undetectables

            self.clear_infection(cleared)
            self.set_immunity(cleared)

        # Find those who progress to cancer
        new_cancers = self.cin & (self.ti_cancer <= ti)
        if new_cancers.any():
            self.cin[new_cancers] = False
            self.inactive[new_cancers] = True
            self.cancerous[new_cancers] = True
            self.ti_cancer[new_cancers] = ti

        # Check if we're modeling latency and need to capture reactivation
        if self.pars.p_control.pars.p > 0:
            reactivated = self.pars.p_reactivate.filter(self.latent)
            if reactivated.any():
                self.set_prognoses(reactivated, record_ti_infected=False)  # Reactivate the infection

        # Find those who die of cancer
        new_deaths = self.cancerous & (self.ti_cancer_death <= ti)
        if new_deaths.any():
            self.dead_cancer[new_deaths] = True
            self.cancerous[new_deaths] = False
            self.ti_cancer_death[new_deaths] = ti
            self.sim.people.request_death(new_deaths)

        return

    def get_cancer_prob(self, uids):
        """
        Get the probability of progressing to cancer
        """
        dur_cin = self.dur_cin[uids]
        if self.pars.cancer_fn.get("method") == "cin_integral":
            cancer_pars = sc.mergedicts(self.pars.cancer_fn, self.pars.cin_fn)
        else:
            cancer_pars = self.pars.cancer_fn
        cancer_probs = hpv.compute_cancer_prob(dur_cin, rel_sev=self.rel_sev[uids], pars=cancer_pars)
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
        res.cancer_incidence[ti] = sc.safedivide(res.new_cancers[ti], denominator)

        return


class HPV(ss.Connector, Genotype):

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

        # Genotypes - TODO, should there be some processing of genotypes?
        self.genotypes = ss.ndict(genotypes)
        self.gkeys = self.genotypes.keys() if self.genotypes else []
        self.n_genotypes = len(self.gkeys)

        return

    def add_states(self):
        states = [
            ss.State("susceptible", label="susceptible", default=True),
            ss.State("infected", label="infected"),
            ss.State("inactive", label="inactive"),
            ss.State("precin", label="precin"),
            ss.State("cin", label="CIN"),
            ss.State("cancerous", label="cancerous"),
            ss.State("dead_cancer", label="Died from cancer"),
            ss.FloatArr("nti_cancer", label="Number of timesteps spent with cancer"),
            ss.FloatArr("ti_cancer", label="Timestep of cancer"),
            ss.FloatArr("ti_cin", label="Timestep of CIN"),
            ss.FloatArr("ti_cancer_death", label="Timestep of cancer death"),
        ]
        self.define_states(*states)
        return

    def init_pre(self, sim):
        """
        Initialize the HPV connector prior to simulation.
        This will set up the genotypes and their states.
        """
        super().init_pre(sim)
        p = self.pars  # Short alias for parameters
        if p.sus_imm_matrix is None:
            p.sus_imm_matrix = hpv.make_immunity_matrix(self.gkeys, p.cross_imm_sus_med, p.cross_imm_sus_high)
        if p.sev_imm_matrix is None:
            p.sev_imm_matrix = hpv.make_immunity_matrix(self.gkeys, p.cross_imm_sev_med, p.cross_imm_sev_high)
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
        self.inactive[uids] = False
        self.precin[uids] = False
        self.cin[uids] = False
        self.cancerous[uids] = False
        return

    def step_genotype_states(self):
        """
        Update states prior to transmission
        """
        for genotype in self.genotypes.values():
            genotype._step_state()
        return

    def step_states(self):
        """
        Check agents' disease status across all genotypes and update their states accordingly.
        """
        self.reset_states()  # Clear states

        for genotype in self.genotypes.values():
            self.susceptible[:] &= genotype.susceptible[:]
            self.infected[:] |= genotype.infected[:]
            self.inactive[:] |= genotype.inactive[:]
            self.precin[:] |= genotype.precin[:]
            self.cin[:] |= genotype.cin[:]
            self.cancerous[:] |= genotype.cancerous[:]

            # For cancers, we take the minimum across genotypes. It's possible that
            # an individual has multiple genotypes, but we want to track the earliest
            # cancer diagnosis and the earliest cancer death time.
            # We will also need to wipte any later dates
            self.ti_cancer[:] = np.fmin(self.ti_cancer[:], genotype.ti_cancer[:])
            self.ti_cancer_death[:] = np.fmin(self.ti_cancer_death[:], genotype.ti_cancer_death[:])
            self.nti_cancer[:] = np.fmin(self.nti_cancer[:], genotype.nti_cancer[:])
            later_cancers = genotype.ti_cancer > self.ti_cancer  # Find later cancer dates
            self.ti_cancer[later_cancers] = np.nan  # Wipe later cancer dates
            self.ti_cancer_death[later_cancers] = np.nan  # Wipe later cancer death dates

            # For infections and CINs, we take the maximum across genotypes
            # This is because an individual can be infected with multiple genotypes, and we want to
            # track the most recent infection time
            self.ti_infected[:] = np.fmax(self.ti_infected[:], genotype.ti_infected[:])
            self.ti_cin[:] = np.fmax(self.ti_cin[:], genotype.ti_cin[:])

        return

    def step(self):
        """ Update the cross-immunity and relative susceptibility and severity """
        self.step_genotype_states()  # Update states for each genotype
        self.step_states()  # Update the connector states based on genotypes
        self.update_immunity()
        return

    def update_immunity(self):
        """
        Update overall sus_imm and sev_imm for each genotype by combining across all genotypes.
        """
        sus_imm_arr = np.array([genotype.own_sus_imm for genotype in self.genotypes.values()])
        sev_imm_arr = np.array([genotype.own_sev_imm for genotype in self.genotypes.values()])

        # Set the susceptibility and severity immunity based on the genotypes
        sus_imm = np.dot(self.pars.sus_imm_matrix, sus_imm_arr)
        sev_imm = np.dot(self.pars.sev_imm_matrix, sev_imm_arr)

        for gname, genotype in self.genotypes.items():
            # Set the susceptibility and severity immunity for each genotype
            gidx = self.gkeys.index(gname)
            # Clip array to be between existing value and 1
            genotype.sus_imm[:] = np.clip(sus_imm[gidx, :], genotype.sus_imm[:], 1.0)
            genotype.sev_imm[:] = np.clip(sev_imm[gidx, :], genotype.sev_imm[:], 1.0)

        return

    # Methods to skip
    def infect(self):
        """ Don't allow HPV infections through this connector """
        pass

    def validate_beta(self, run_checks=False):
        """ Skip this method for the HPV connector, as it does not have a beta parameter. """
        pass


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

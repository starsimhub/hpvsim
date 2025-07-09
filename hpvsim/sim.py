"""
Create a simulation for running the HPV module.
"""

import starsim as ss
import sciris as sc
import pandas as pd
import hpvsim as hpv
import stisim as sti
import numpy as np


__all__ = ["Sim"]


class Sim(ss.Sim):
    """Custom simulation class for HPV module, inheriting from starsim.Sim."""
    def __init__(self, pars=None, sim_pars=None, hpv_pars=None, nw_pars=None, imm_pars=None,
                 genotypes=None, datafolder=None, location=None,
                 label=None, people=None, demographics=None, diseases=None, networks=None,
                 interventions=None, analyzers=None, connectors=None, **kwargs):

        # Inputs and defaults
        self.location = location
        self.datafolder = datafolder
        self.hpv_pars = None    # Parameters for the HPV modules - processed later
        self.nw_pars = None     # Parameters for the networks - processed later
        self.imm_pars = None    # Parameters for cross-immunity, used in the HPV connector
        self.pars = None        # Parameters for the simulation - processed later

        # World Standard Population, used to calculate age-standardised rates (ASR) of incidence
        self.age_bin_edges = np.array([0,   5,  10,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60,  65,  70,  75, 80, 85, 100]),
        self.standard_pop_weights = np.array([.12, .10, .09, .09, .08, .08, .06, .06, .06, .06, .05, .04, .04, .03, .02, .01, 0.005, 0.005, 0]),
        self.standard_pop = np.array([self.age_bin_edges, self.standard_pop_weights])

        # Call the constructor of the parent class WITHOUT pars or module args
        super().__init__(pars=None, label=label)

        # Separate the parameters, storing sim pars now and saving module pars to process in init
        sim_kwargs = dict(label=label, people=people, demographics=demographics, diseases=diseases, networks=networks,
                    interventions=interventions, analyzers=analyzers, connectors=connectors)
        sim_kwargs = {key: val for key, val in sim_kwargs.items() if val is not None}
        self.pars = self.separate_pars(pars, sim_pars, hpv_pars, nw_pars, imm_pars, sim_kwargs, **kwargs)

        return

    def separate_pars(self, pars=None, sim_pars=None, hpv_pars=None, nw_pars=None, imm_pars=None, sim_kwargs=None, **kwargs):
        """
        Create a nested dict of parameters that get passed to Sim constructor and the component modules
        Prioritization:
            - If any key appears in both pars and *_pars, the value from *_pars will be used.
            - If any key appears in both pars and kwargs, the value from kwargs will be used.
        """
        # Marge in pars and kwargs
        all_pars = sc.mergedicts(pars, sim_pars, hpv_pars, nw_pars, imm_pars, sim_kwargs, kwargs)

        # Deal with sim pars
        default_sim_pars = hpv.make_sim_pars()  # Make default parameters using values from parameters.py
        user_sim_pars = {k: v for k, v in all_pars.items() if k in default_sim_pars.keys()}
        sim_pars = sc.mergedicts(default_sim_pars, user_sim_pars, sim_pars, _copy=True)

        # Deal with HPV pars
        default_hpv_pars = hpv.make_hpv_pars()
        user_hpv_pars = {k: v for k, v in all_pars.items() if k in default_hpv_pars.keys()}
        hpv_pars = sc.mergedicts(default_hpv_pars, user_hpv_pars, hpv_pars, _copy=True)

        # Deal with network pars
        default_nw_pars = hpv.make_network_pars()
        user_nw_pars = {k: v for k, v in all_pars.items() if k in default_nw_pars.keys()}
        nw_pars = sc.mergedicts(default_nw_pars, user_nw_pars, nw_pars, _copy=True)

        # Deal with immunity pars
        default_imm_pars = hpv.make_imm_pars()
        user_imm_pars = {k: v for k, v in all_pars.items() if k in default_imm_pars.keys()}
        imm_pars = sc.mergedicts(default_imm_pars, user_imm_pars, imm_pars, _copy=True)

        # Store the parameters for the modules - thse will be fed into the modules during init
        self.hpv_pars = hpv_pars    # Parameters for the HPV modules
        self.nw_pars = nw_pars      # Parameters for the networks
        self.imm_pars = imm_pars    # Parameters for cross-immunity, used in the HPV connector

        return sim_pars

    def init(self, force=False, **kwargs):
        """
        Perform all initializations for the sim
        """
        # Process the genotypes and HPV connector
        genotypes, hpv_connector = self.process_genotypes()
        self.pars['diseases'] += genotypes
        self.pars['connectors'] += hpv_connector

        super().init(force=force, **kwargs)  # Call the parent init method
        return self

    def process_genotypes(self):
        """
        Process the genotypes to create HPV modules.
        If genotypes are provided, they will be used; otherwise, default to HPV16 and HPV18.
        """
        # Genotypes may be provided in various forms; process them here
        self.pars['genotypes'] = sc.tolist(self.pars['genotypes'])  # Make shorter
        genotypes = sc.autolist()
        for gtype in self.pars['genotypes']:
            if sc.checktype(gtype, (str, int)):
                genotypes += hpv.make_hpv(genotype=gtype, hpv_pars=self.hpv_pars)
            elif isinstance(gtype, hpv.HPV):
                genotypes += gtype
            else:
                raise ValueError(f"Invalid genotype type: {type(gtype)}. Must be str, int, or hpv.HPV.")

        hpv_connector = hpv.hpv(genotypes=genotypes, imm_pars=self.imm_pars)
        return genotypes, hpv_connector

    def process_location(self):
        """ Process the location to create people and demographics if not provided. """

        # TODO: Do this better
        if self.location in ['kenya', 'india']:
            dflocation = self.location.replace(" ", "_")
            total_pop = {
                'kenya': {2020: 52.2e6}[self.pars.start],
                'india': {2020: 1.4e9}[self.pars.start]
            }[dflocation]
            ppl = ss.People(
                self.pars.n_agents,
                age_data=pd.read_csv(f"{self.datafolder}/{dflocation}_age.csv", index_col="age")["value"]
            )
            fertility_data = pd.read_csv(f"{self.datafolder}/{dflocation}_asfr.csv")
            pregnancy = ss.Pregnancy(unit='month', fertility_rate=fertility_data)
            death_data = pd.read_csv(f"{self.datafolder}/{dflocation}_deaths.csv")
            death = ss.Deaths(unit='year', death_rate=death_data, rate_units=1)
            self.pars['demographics'] = [pregnancy, death]
            self.pars['people'] = ppl

        else:
            raise ValueError(f"Location {self.location} not supported")

        return
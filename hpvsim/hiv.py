'''
Defines classes and methods for hiv natural history
'''

import numpy as np
import sciris as sc
import pandas as pd
from . import utils as hpu
from . import defaults as hpd
from . import base as hpb
from .data import loaders as hpdata
from scipy.stats import weibull_min


class HIVsim(hpb.ParsObj):
    '''
        A class based around performing operations on a self.pars dict.
        '''

    def __init__(self, sim, location, art_datafile, hiv_datafile, hiv_pars=None):
        pars = self.load_data(location=location, hiv_datafile=hiv_datafile, art_datafile=art_datafile)
        self.people = sim.people
        # Define default parameters, can be overwritten by hiv_pars
        pars['hiv_pars'] = {
            'rel_sus': 2.2,  # Increased risk of acquiring HPV
            'rel_hiv_sev_infl': 0.5,  # Speed up growth of disease severity
            'reactivation_prob': 3, # Unused for now, TODO: add in rel_reactivation to make functional
            'time_to_hiv_death_shape': 2, # based on https://royalsocietypublishing.org/action/downloadSupplement?doi=10.1098%2Frsif.2013.0613&file=rsif20130613supp1.pdf
            'time_to_hiv_death_scale': lambda a: 21.182 - 0.2717*a # based on https://royalsocietypublishing.org/action/downloadSupplement?doi=10.1098%2Frsif.2013.0613&file=rsif20130613supp1.pdf
        }

        self.init_states()
        if hiv_pars is not None:
            pars = sc.mergedicts(pars, hiv_pars)
        self.update_pars(pars, create=True)
        self.init_results(sim)

        return


    def init_states(self):
        hiv_states = [
            hpd.State('cd4', hpd.default_int, -1),
            hpd.State('vl', hpd.default_float, np.nan),
            hpd.State('art', bool, False),
            hpd.State('date_dead_hiv', hpd.default_float, np.nan),
            hpd.State('dead_hiv', bool, False),
        ]
        self.people.meta.all_states += hiv_states


    def init_results(self, sim):

        def init_res(*args, **kwargs):
            ''' Initialize a single result object '''
            output = hpb.Result(*args, **kwargs, npts=sim.res_npts)
            return output

        self.resfreq = sim.resfreq
        # Initialize storage
        results = sc.objdict()

        na = len(sim['age_bins']) - 1  # Number of age bins

        results['hiv_infections'] = init_res('Number HIV infections')
        results['hiv_infections_by_age'] = init_res('Number HIV infections by age', n_rows=na)
        results['n_hiv'] = init_res('Number living with HIV')
        results['n_hiv_by_age'] = init_res('Number living with HIV by age', n_rows=na)
        results['hiv_prevalence'] = init_res('HIV prevalence')
        results['hiv_prevalence_by_age'] = init_res('HIV prevalence by age', n_rows=na)
        results['hiv_incidence'] = init_res('HIV incidence')
        results['hiv_incidence_by_age'] = init_res('HIV incidence by age', n_rows=na)
        self.results = results
        return

    # %% HIV methods

    def set_hiv_prognoses(self, inds, year=None):
        ''' Set HIV outcomes (for now only ART) '''

        art_cov = self['art_adherence']  # Shorten

        # Extract index of current year
        all_years = np.array(list(art_cov.keys()))
        year_ind = sc.findnearest(all_years, year)
        nearest_year = all_years[year_ind]

        # Figure out which age bin people belong to
        age_bins = art_cov[nearest_year][0, :]
        age_inds = np.digitize(self.people.age[inds], age_bins)

        # Apply ART coverage by age to people
        art_covs = art_cov[nearest_year][1, :]
        art_adherence = art_covs[age_inds]
        self.people.art_adherence[inds] = art_adherence
        self.people.rel_sev_infl[inds] = (1 - art_adherence) * self['hiv_pars']['rel_hiv_sev_infl']
        self.people.rel_sus[inds] = (1 - art_adherence) * self['hiv_pars']['rel_sus']

        # Draw time to HIV mortality
        shape = self['hiv_pars']['time_to_hiv_death_shape']
        scale = self['hiv_pars']['time_to_hiv_death_scale'](self.people.age[inds])
        time_to_hiv_death = weibull_min.rvs(c=shape, scale=scale, size=len(inds))
        self.people.date_dead_hiv[inds] = self.people.t + sc.randround(time_to_hiv_death / self.people.dt)

        return

    def check_hiv_mortality(self):
        '''
        Check for new deaths from HIV
        '''
        filter_inds = self.people.true('hiv')
        inds = self.people.check_inds(self.people.dead_hiv, self.people.date_dead_hiv, filter_inds=filter_inds)
        self.people.remove_people(inds, cause='hiv')
        return

    def apply(self, year=None):
        '''
        Wrapper method that checks for new HIV infections, updates prognoses, etc.
        '''

        new_infection_inds = self.new_hiv_infections(year) # Newly acquired HIV infections
        if len(new_infection_inds):
            self.set_hiv_prognoses(new_infection_inds, year=year)  # Set ART adherence for those with HIV
            self.update_hpv_progs(new_infection_inds) # Update any HPV prognoses
        self.check_hiv_mortality()
        new_infections = self.people.scale_flows(new_infection_inds) # Return scaled number of infections
        return new_infections

    def new_hiv_infections(self, year=None):
        '''Apply HIV infection rates to population'''
        hiv_pars = self['infection_rates']
        all_years = np.array(list(hiv_pars.keys()))
        year_ind = sc.findnearest(all_years, year)
        nearest_year = all_years[year_ind]
        hiv_year = hiv_pars[nearest_year]
        dt = self.people.pars['dt']

        hiv_probs = np.zeros(len(self.people), dtype=hpd.default_float)
        for sk in ['f', 'm']:
            hiv_year_sex = hiv_year[sk]
            age_bins = hiv_year_sex[:, 0]
            hiv_rates = hiv_year_sex[:, 1] * dt
            mf_inds = self.people.is_female if sk == 'f' else self.people.is_male
            mf_inds *= self.people.alive  # Only include people alive
            age_inds = np.digitize(self.people.age[mf_inds], age_bins)
            hiv_probs[mf_inds] = hiv_rates[age_inds]
        hiv_probs[self.people.hiv] = 0  # not at risk if already infected

        # Get indices of people who acquire HIV
        hiv_inds = hpu.true(hpu.binomial_arr(hiv_probs))
        self.people.hiv[hiv_inds] = True
        self.people.date_hiv[hiv_inds] = self.people.t

        if self.people.t % self.resfreq == self.resfreq - 1:
            # Update stock and flows
            idx = int(self.people.t / self.resfreq)
            self.results['hiv_infections'][idx] = self.people.scale_flows(hiv_inds)
            self.results[f'hiv_infections_by_age'][:, idx] = np.histogram(self.people.age[hiv_inds], bins=self.people.age_bins, weights=self.people.scale[hiv_inds])[0]
            self.results['n_hiv'][idx] = self.people.count('hiv')
            hivinds = hpu.true(self.people['hiv'])
            self.results[f'n_hiv_by_age'][:, idx] = np.histogram(self.people.age[hivinds], bins=self.people.age_bins, weights=self.people.scale[hivinds])[0]

        return hiv_inds

    def update_hpv_progs(self, hiv_inds):
        dt = self.people.pars['dt']
        for g in range(self.people.pars['n_genotypes']):
            gpars = self.people.pars['genotype_pars'][self.people.pars['genotype_map'][g]]
            hpv_inds = hpu.itruei((self.people.is_female & self.people.episomal[g, :]), hiv_inds)  # Women with HIV who have episomal HPV
            if len(hpv_inds):  # Reevaluate these women's severity markers and determine whether they will develop cellular changes
                self.people.set_severity_pars(hpv_inds, g, gpars)
                self.people.set_severity(hpv_inds, g, gpars, dt)
        return

    def get_hiv_data(self, location=None, hiv_datafile=None, art_datafile=None, verbose=False):
        '''
        Load HIV incidence and art coverage data, if provided
        ART adherance calculations use life expectancy data to infer lifetime average coverage
        rates for people in different age buckets. To give an example, suppose that ART coverage
        over 2010-2020 is given by:
            art_coverage = [0.23,0.3,0.38,0.43,0.48,0.52,0.57,0.61,0.65,0.68,0.72]
        The average ART adherence in 2010 will be higher for younger cohorts than older ones.
        Someone expected to die within a year would be given an average lifetime ART adherence
        value of 0.23, whereas someone expected to survive >10 years would be given a value of 0.506.

        Args:
            location (str): must be provided if you want to run with HIV dynamics
            hiv_datafile (str):  must be provided if you want to run with HIV dynamics
            art_datafile (str):  must be provided if you want to run with HIV dynamics
            verbose (bool):  whether to print progress

        Returns:
            hiv_inc (dict): dictionary keyed by sex, storing arrays of HIV incidence over time by age
            art_cov (dict): dictionary keyed by sex, storing arrays of ART coverage over time by age
            life_expectancy (dict): dictionary storing life expectancy over time by age
        '''

        if hiv_datafile is None and art_datafile is None:
            hiv_incidence_rates, art_adherence = None, None

        else:

            # Load data
            life_exp = self.get_life_expectancy(location=location,
                                           verbose=verbose)  # Load the life expectancy data (needed for ART adherance calcs)
            df_inc = pd.read_csv(hiv_datafile)  # HIV incidence
            df_art = pd.read_csv(art_datafile)  # ART coverage

            # Process HIV and ART data
            sex_keys = ['Male', 'Female']
            sex_key_map = {'Male': 'm', 'Female': 'f'}

            ## Start with incidence file
            years = df_inc['Year'].unique()
            hiv_incidence_rates = dict()

            # Processing
            for year in years:
                hiv_incidence_rates[year] = dict()
                for sk in sex_keys:
                    sk_out = sex_key_map[sk]
                    hiv_incidence_rates[year][sk_out] = np.concatenate(
                        [
                            np.array(df_inc[(df_inc['Year'] == year) & (df_inc['Sex'] == sk_out)][['Age', 'Incidence']],
                                     dtype=hpd.default_float),
                            np.array([[150, 0]])  # Add another entry so that all older age groups are covered
                        ]
                    )

            # Now compute ART adherence over time/age
            art_adherence = dict()
            years = df_art['Year'].values
            for i, year in enumerate(years):

                # Use the incidence file to determine which age groups we want to calculate ART coverage for
                ages_inc = hiv_incidence_rates[year]['m'][:, 0]  # Read in the age groups we have HIV incidence data for
                ages_ex = life_exp[year]['m'][:, 0]  # Age groups available in life expectancy file
                ages = np.intersect1d(ages_inc, ages_ex)  # Age groups we want to calculate ART coverage for

                # Initialize age-specific ART coverage dict and start filling it in
                cov = np.zeros(len(ages), dtype=hpd.default_float)
                for j, age in enumerate(ages):
                    idx = np.where(life_exp[year]['f'][:, 0] == age)[0]  # Finding life expectancy for this age group/year
                    this_life_exp = life_exp[year]['f'][idx, 1]  # Pull out value
                    last_year = int(year + this_life_exp)  # Figure out the year in which this age cohort is expected to die
                    year_ind = sc.findnearest(years, last_year)  # Get as close to the above year as possible within the data
                    if year_ind > i:  # Either take the mean of ART coverage from now up until the year of death
                        cov[j] = np.mean(df_art[i:year_ind]['ART Coverage'].values)
                    else:  # Or, just use ART overage in this year
                        cov[j] = df_art.iloc[year_ind]['ART Coverage']

                art_adherence[year] = np.array([ages, cov])

        return hiv_incidence_rates, art_adherence

    def load_data(self, location=None, hiv_datafile=None, art_datafile=None):
        ''' Load any data files that are used to create additional parameters, if provided '''
        hiv_data = sc.objdict()
        hiv_data.infection_rates, hiv_data.art_adherence = self.get_hiv_data(location=location, hiv_datafile=hiv_datafile, art_datafile=art_datafile)
        return hiv_data

    def get_life_expectancy(self, location, verbose=False):
        '''
        Get life expectancy data by location
        life_expectancy (dict): dictionary storing life expectancy over time by age
        '''
        if location is not None:
            if verbose:
                print(f'Loading location-specific life expectancy data for "{location}" - needed for HIV runs')
            try:
                life_expectancy = hpdata.get_life_expectancy(location=location)
                return life_expectancy
            except ValueError as E:
                errormsg = f'Could not load HIV data for requested location "{location}" ({str(E)})'
                raise NotImplementedError(errormsg)
        else:
            raise NotImplementedError('Cannot load HIV data without a specified location')

    def finalize(self, sim):
        '''
        Compute prevalence, incidence.
        '''
        res = self.results
        simres = sim.results

        # Compute HIV incidence and prevalence
        def safedivide(num, denom):
            ''' Define a variation on sc.safedivide that respects shape of numerator '''
            answer = np.zeros_like(num)
            fill_inds = (denom != 0).nonzero()
            if len(num.shape) == len(denom.shape):
                answer[fill_inds] = num[fill_inds] / denom[fill_inds]
            else:
                answer[:, fill_inds] = num[:, fill_inds] / denom[fill_inds]
            return answer

        self.results['hiv_prevalence_by_age'][:] = safedivide(res['n_hiv_by_age'][:], simres['n_alive_by_age'][:])
        self.results['hiv_incidence'][:] = sc.safedivide(res['hiv_infections'][:], (simres['n_alive'][:] - res['n_hiv'][:]))
        self.results['hiv_incidence_by_age'][:] = sc.safedivide(res['hiv_infections_by_age'][:], (simres['n_alive_by_age'][:] - res['n_hiv_by_age'][:]))
        self.results['hiv_prevalence'][:] = sc.safedivide(res['n_hiv'][:], simres['n_alive'][:])
        return
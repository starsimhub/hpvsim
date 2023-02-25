'''
Defines classes and methods for HIV natural history
'''

import numpy as np
import sciris as sc
import pandas as pd
from scipy.stats import weibull_min
from . import utils as hpu
from . import defaults as hpd
from . import base as hpb


class HIVsim(hpb.ParsObj):
    '''
        A class based around performing operations on a self.pars dict.
        '''

    def __init__(self, sim, art_datafile, hiv_datafile, hiv_pars=None):
        pars = self.load_data(hiv_datafile=hiv_datafile, art_datafile=art_datafile)
        self.people = sim.people
        # Define default parameters, can be overwritten by hiv_pars
        pars['hiv_pars'] = {
            'rel_sus': 2.2,  # Increased risk of acquiring HPV
            'rel_hiv_sev_infl': {'cd4_200': 0.36, 'cd4_200_500': 0.76},  # Speed up growth of disease severity
            'rel_hiv_imm': {'cd4_200': 0.36, 'cd4_200_500': 0.76},  # Reduction in immunity acquired after infection/vaccination
            'reactivation_prob': 3, # Unused for now, TODO: add in rel_reactivation to make functional
            'time_to_hiv_death_shape': 2, # shape parameter for weibull distribution, based on https://royalsocietypublishing.org/action/downloadSupplement?doi=10.1098%2Frsif.2013.0613&file=rsif20130613supp1.pdf
            'time_to_hiv_death_scale': lambda a: 21.182 - 0.2717*a, # scale parameter for weibull distribution, based on https://royalsocietypublishing.org/action/downloadSupplement?doi=10.1098%2Frsif.2013.0613&file=rsif20130613supp1.pdf
            'cd4_start': dict(dist='normal', par1=594, par2=20),
            'cd4_trajectory': lambda f: (24.363 - 16.672*f)**2, # based on https://docs.idmod.org/projects/emod-hiv/en/latest/hiv-model-healthcare-systems.html?highlight=art#art-s-impact-on-cd4-count
            'cd4_reconstitution': lambda m: 15.584*m - 0.2113*m**2, # growth in CD4 count following ART initiation
            'art_failure_prob': 0.1, # Percentage of people on ART who will not suppress virus successfully
            'dt_art': 1.0 # Timestep (annually) at which ART updates are made
        }

        self.init_states()
        if hiv_pars is not None:
            pars['hiv_pars'] = sc.mergedicts(pars['hiv_pars'], hiv_pars)
        self.update_pars(pars, create=True)
        self.init_results(sim)

        y = np.linspace(0,1,101)
        cd4_decline = self['hiv_pars']['cd4_trajectory'](y)
        self.cd4_decline_diff = np.diff(cd4_decline)

        return


    def init_states(self):
        hiv_states = [
            hpd.State('cd4', hpd.default_float, np.nan),
            hpd.State('art', bool, False),
            hpd.State('date_art', hpd.default_float, np.nan),
            hpd.State('date_dead_hiv', hpd.default_float, np.nan),
            hpd.State('dead_hiv', bool, False),
            hpd.State('dur_hiv', hpd.default_float, np.nan),
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

    def set_hiv_prognoses(self, inds, year=None, incident=True):
        ''' Set HIV outcomes '''

        art_cov = self['art_adherence']  # Shorten
        shape = self['hiv_pars']['time_to_hiv_death_shape']

        # Extract index of current year
        all_years = np.array(list(art_cov.keys()))
        year_ind = sc.findnearest(all_years, year)
        nearest_year = all_years[year_ind]

        # Apply ART coverage by age to people
        art_covs = art_cov[nearest_year]#[1, :]

        art_probs = np.zeros(len(self.people), dtype=hpd.default_float)
        art_probs[inds] = art_covs

        # Get indices of people who are on ART
        art_bools = hpu.binomial_arr(art_probs)
        art_inds = hpu.true(art_bools)
        self.people.art[art_inds] = True
        self.people.date_art[art_inds] = self.people.t
        self.people.date_dead_hiv[art_inds] = np.nan
        self.people.dur_hiv[art_inds] = np.nan

        if incident:
            # Filter those who are not on ART and assign time to HIV death
            no_art_inds = np.setdiff1d(inds, art_inds)
            scale = self['hiv_pars']['time_to_hiv_death_scale'](self.people.age[no_art_inds])
            scale = np.maximum(scale, 0)
            time_to_hiv_death = weibull_min.rvs(c=shape, scale=scale, size=len(no_art_inds))
            self.people.dur_hiv[no_art_inds] = time_to_hiv_death
            self.people.date_dead_hiv[no_art_inds] = self.people.t + sc.randround(time_to_hiv_death / self.people.dt)

        # Find those on ART who will not be virologically suppressed and assign time to HIV death
        art_failure_prob = self['hiv_pars']['art_failure_prob']
        art_failure_probs = np.full(len(art_inds), fill_value=art_failure_prob, dtype=hpd.default_float)
        art_failure_bools = hpu.binomial_arr(art_failure_probs)
        art_failure_inds = art_inds[art_failure_bools]

        scale = self['hiv_pars']['time_to_hiv_death_scale'](self.people.age[art_failure_inds])
        scale = np.maximum(scale, 0)
        time_to_hiv_death = weibull_min.rvs(c=shape, scale=scale, size=len(art_failure_inds))
        self.people.dur_hiv[art_failure_inds] = time_to_hiv_death
        self.people.date_dead_hiv[art_failure_inds] = self.people.t + sc.randround(time_to_hiv_death / self.people.dt)

        return


    def check_hiv_mortality(self):
        '''
        Check for new deaths from HIV
        '''
        filter_inds = self.people.true('hiv')
        inds = self.people.check_inds(self.people.dead_hiv, self.people.date_dead_hiv, filter_inds=filter_inds)
        self.people.remove_people(inds, cause='hiv')
        return


    def check_cd4(self):
        '''
        Check for current cd4
        '''
        filter_inds = self.people.true('hiv')
        if len(filter_inds):
            art_inds = filter_inds[hpu.true(self.people.art[filter_inds])]
            not_art_inds = filter_inds[hpu.false(self.people.art[filter_inds])]

            # First take care of people not on ART
            frac_prognosis = 100*(self.people.t - self.people.date_hiv[not_art_inds])* self.people.dt/self.people.dur_hiv[not_art_inds]
            cd4_change = self.cd4_decline_diff[frac_prognosis.astype(np.int64)]
            self.people.cd4[not_art_inds] += cd4_change

            # Now take care of people on ART
            months_on_ART = (self.people.t - self.people.date_art[art_inds])*12
            cd4_change = self['hiv_pars']['cd4_reconstitution'](months_on_ART)
            self.people.cd4[art_inds] += cd4_change

            cd4_200_inds = sc.findinds(self.people.cd4 < 200)
            cd4_200_500_inds = sc.findinds((self.people.cd4 > 200) & (self.people.cd4 < 500))

            if len(cd4_200_inds):
                self.people.rel_sev_infl[cd4_200_inds] = self['hiv_pars']['rel_hiv_sev_infl']['cd4_200']
                self.people.rel_sus[cd4_200_inds] = self['hiv_pars']['rel_sus']
                self.people.rel_imm[cd4_200_inds] = self['hiv_pars']['rel_hiv_imm']['cd4_200']
                self.update_hpv_progs(cd4_200_inds)  # Update any HPV prognoses

            if len(cd4_200_500_inds):
                self.people.rel_sev_infl[cd4_200_500_inds] = self['hiv_pars']['rel_hiv_sev_infl']['cd4_200_500']
                self.people.rel_sus[cd4_200_500_inds] = self['hiv_pars']['rel_sus']
                self.people.rel_imm[cd4_200_500_inds] = self['hiv_pars']['rel_hiv_imm']['cd4_200']
                self.update_hpv_progs(cd4_200_500_inds)  # Update any HPV prognoses

        return


    def step(self, year=None):
        '''
        Wrapper method that checks for new HIV infections, updates prognoses, etc.
        '''
        # Pull out anyone with prevalent infection who is not on ART, check if they get on today
        t = self.people.t
        dt = self.people.dt

        update_freq = max(1, int(self['hiv_pars']['dt_art'] / dt)) # Ensure it's an integer not smaller than 1
        if t % update_freq == 0:
            hiv_inds = self.people.true('hiv')
            if len(hiv_inds):
                self.set_hiv_prognoses(hiv_inds, year=year, incident=False)

        new_infection_inds = self.new_hiv_infections(year) # Newly acquired HIV infections
        if len(new_infection_inds):
            self.set_hiv_prognoses(new_infection_inds, year=year)  # Set ART adherence for those with HIV

        self.check_hiv_mortality()
        self.check_cd4()
        self.update_hiv_results(new_infection_inds)
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
        self.people.cd4[hiv_inds] = hpu.sample(**self['hiv_pars']['cd4_start'], size=len(hiv_inds))
        self.people.date_hiv[hiv_inds] = self.people.t

        return hiv_inds


    def update_hpv_progs(self, hiv_inds):
        dt = self.people.pars['dt']
        for g in range(self.people.pars['n_genotypes']):
            gpars = self.people.pars['genotype_pars'][self.people.pars['genotype_map'][g]]
            hpv_inds = hpu.itruei((self.people.is_female & self.people.episomal[g, :]), hiv_inds)  # Women with HIV who have episomal HPV
            if len(hpv_inds):  # Reevaluate these women's severity markers and determine whether they will develop cellular changes
                self.people.set_severity_pars(hpv_inds, g, gpars)
                self.people.set_severity(hpv_inds, g, gpars, dt, set_sev=False)
        return

    def update_hiv_results(self, hiv_inds):
        if self.people.t % self.resfreq == self.resfreq - 1:
            # Update stock and flows
            idx = int(self.people.t / self.resfreq)
            self.results['hiv_infections'][idx] = self.people.scale_flows(hiv_inds)
            self.results['hiv_infections_by_age'][:, idx] = np.histogram(self.people.age[hiv_inds], bins=self.people.age_bins, weights=self.people.scale[hiv_inds])[0]
            self.results['n_hiv'][idx] = self.people.count('hiv')
            hivinds = hpu.true(self.people['hiv'])
            self.results['n_hiv_by_age'][:, idx] = np.histogram(self.people.age[hivinds], bins=self.people.age_bins, weights=self.people.scale[hivinds])[0]


    def get_hiv_data(self, hiv_datafile=None, art_datafile=None):
        '''
        Load HIV incidence and art coverage data, if provided

        Args:
            location (str): must be provided if you want to run with HIV dynamics
            hiv_datafile (str):  must be provided if you want to run with HIV dynamics
            art_datafile (str):  must be provided if you want to run with HIV dynamics
            verbose (bool):  whether to print progress

        Returns:
            hiv_inc (dict): dictionary keyed by sex, storing arrays of HIV incidence over time by age
            art_cov (dict): dictionary keyed by sex, storing arrays of ART coverage over time by age
        '''

        if hiv_datafile is None and art_datafile is None:
            hiv_incidence_rates, art_adherence = None, None

        else:

            # Load data
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
                art_adherence[year] = df_art.iloc[i]['ART Coverage']

        return hiv_incidence_rates, art_adherence


    def load_data(self, hiv_datafile=None, art_datafile=None):
        ''' Load any data files that are used to create additional parameters, if provided '''
        hiv_data = sc.objdict()
        hiv_data.infection_rates, hiv_data.art_adherence = self.get_hiv_data(hiv_datafile=hiv_datafile, art_datafile=art_datafile)
        return hiv_data


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
        sim.results = sc.mergedicts(simres, self.results)
        return
'''
Tests for single simulations
'''

#%% Imports and settings
import sciris as sc
import hpvsim as hpv
import numpy as np
import pandas as pd
import matplotlib.pylab as pl

do_plot = 0
do_save = 0
debug = 0

n_agents = [50e3,500][debug] # Swap between sizes
start = [1950,1990][debug]
ms_agent_ratio = [100,10][debug]
hiv_datafile = ['../test_data/hiv_incidence_south_africa.csv', '../test_data/south_africa_female_hiv_mortality.csv', '../test_data/south_africa_male_hiv_mortality.csv']
# art_datafile = ['../test_data/south_africa_art_coverage_by_age_males.csv', '../test_data/south_africa_art_coverage_by_age_females.csv']
art_datafile = ['../test_data/art_coverage_south_africa.csv']


#%% Define the tests

def test_hiv():
    ''' Basic test to show that it runs '''
    sc.heading('Testing hiv')

    fig_string = ''

    pars = {
        'n_agents': n_agents,
        'location': 'south africa',
        'model_hiv': True,
        'start': start,
        'end': 2030,
        'ms_agent_ratio': ms_agent_ratio,
        # 'hiv_pars': dict(rel_imm=dict(lt200=1, gt200=1),
        #                 hiv_death_adj=1.5)
    }


    sim = hpv.Sim(
        pars=pars,
        hiv_datafile=hiv_datafile,
        art_datafile=art_datafile,
    )
    sim.run()
    to_plot = {
        'HIV prevalence': [
            'hiv_prevalence_by_age',
            'female_hiv_prevalence_by_age',
            'male_hiv_prevalence_by_age'
        ],
        'ART coverage':[
            'art_coverage',
            'hiv_prevalence'
        ],
        'Age standardized cancer incidence (per 100,000 women)': [
            'asr_cancer_incidence',
            'cancer_incidence_with_hiv',
            'cancer_incidence_no_hiv',
        ],
        'Cancers (total and by HIV status)': [
            'cancers',
            'cancers_no_hiv',
            'cancers_with_hiv',
        ]
    }
    sim.plot(to_plot=to_plot)


    rsa_df = pd.read_csv('../test_data/RSA_data.csv').set_index('Unnamed: 0').T

    simres = sim.results
    years = simres['year']
    year_ind = sc.findinds(years, 1985)[0]

    fig, ax = pl.subplots()
    ax.plot(years[year_ind:], simres['art_coverage'][year_ind:], label='HPVsim')
    ax.scatter(years[year_ind:], rsa_df['art_coverage'], label='Thembisa')
    ax.set_title(f'ART coverage {fig_string}')
    ax.legend()
    fig.show()

    fig, ax = pl.subplots()
    ax.plot(years[year_ind:], simres['hiv_mortality'][year_ind:], label='HIV mortality')
    ax.plot(years[year_ind:], simres['excess_hiv_mortality'][year_ind:], label='Excess HIV mortality')
    ax.set_title(f'HIV mortality {fig_string}')
    ax.legend()
    # fig.show()

    # ## TEST SPACE ###
    hiv_mortality_df = simres['hiv_mortality_by_age'][:,year_ind:]
    fig, axes = pl.subplots(1, 2)
    for ia, age in enumerate(sim.pars['age_bin_edges'][:-1]):
        if ia < 10:
            ax = axes[0]
        else:
            ax = axes[1]
        ax.plot(years[year_ind:], hiv_mortality_df[ia,:], label=age)
    for ax in axes:
        ax.legend()
    fig.suptitle(f'HIV mortality by age {fig_string}')
    fig.tight_layout()
    # fig.show()

    fig, axes = pl.subplots(3, 1, figsize=(12, 12))
    ax = axes[0]
    ax.plot(years[year_ind:], simres['n_females_with_hiv_alive'][year_ind:], label='HPVsim, females 10+')
    ax.plot(years[year_ind:], simres['n_males_with_hiv_alive'][year_ind:], label='HPVsim, males 10+')
    ax.scatter(years[year_ind:], rsa_df['n_females_with_hiv_alive'], label='Thembisa, females 15+')
    ax.scatter(years[year_ind:], rsa_df['n_males_with_hiv_alive'], label='Thembisa, males 15+')
    ax.set_title(f'Prevalent HIV infections 10+ {fig_string}')

    ax = axes[1]
    female_denom = np.sum(simres['n_females_alive_by_age'][2:, year_ind:], axis=0)
    male_denom = np.sum(
        (simres['n_males_with_hiv_alive_by_age'][2:, year_ind:] + simres['n_males_no_hiv_alive_by_age'][2:, year_ind:]),
        axis=0)
    ax.plot(years[year_ind:-1], female_denom[:-1], label='Females alive 10+, HPVsim')
    ax.plot(years[year_ind:-1], male_denom[:-1], label='Males alive 10+, HPVsim')
    ax.scatter(years[year_ind:-1], rsa_df['female_pop'][:-1], label='Females alive 10+, Thembisa')
    ax.scatter(years[year_ind:-1], rsa_df['male_pop'][:-1], label='Males alive 10+, Thembisa')
    ax.set_title(f'Alive, 10+ {fig_string}')

    ax = axes[2]
    ax.plot(years[year_ind:], simres['female_hiv_prevalence'][year_ind:], label='Females, HPVsim')
    ax.plot(years[year_ind:], simres['male_hiv_prevalence'][year_ind:], label='Males, HPVsim')
    ax.scatter(years[year_ind:], rsa_df['female_hiv_prevalence'], label='Females, Thembisa')
    ax.scatter(years[year_ind:], rsa_df['male_hiv_prevalence'], label='Males, Thembisa')
    ax.set_title(f'HIV prevalence 10+ {fig_string}')
    ax.legend()
    fig.show()

    cancer_mortality_by_age = simres['cancer_deaths_by_age'][:,year_ind:] / simres['n_females_alive_by_age'][:,year_ind:]
    fig, axes = pl.subplots(1, 2)
    for ia, age in enumerate(sim.pars['age_bin_edges'][:-1]):
        if ia < 10:
            ax = axes[0]
        else:
            ax = axes[1]
        ax.plot(years[year_ind:], cancer_mortality_by_age[ia,:], label=age)
    for ax in axes:
        ax.legend()
    fig.suptitle(f'Cancer mortality by age {fig_string}')
    fig.tight_layout()
    fig.show()


    fig, ax = pl.subplots()
    ax.plot(years[year_ind:], simres['hiv_infections'][year_ind:], label='HPVsim')
    ax.scatter(years[year_ind:], rsa_df['hiv_infections'], label='Thembisa')
    ax.set_title(f'New HIV infections {fig_string}')
    ax.legend()
    fig.show()

    fig, ax = pl.subplots()
    ax.plot(years[year_ind:], simres['n_females_with_hiv_alive'][year_ind:], label='HPVsim, females 10+')
    ax.plot(years[year_ind:], simres['n_males_with_hiv_alive'][year_ind:], label='HPVsim, males 10+')
    ax.scatter(years[year_ind:], rsa_df['n_females_with_hiv_alive'], label='Thembisa, females 15+')
    ax.scatter(years[year_ind:], rsa_df['n_males_with_hiv_alive'], label='Thembisa, males 15+')
    ax.set_title(f'Prevalent HIV infections {fig_string}')
    ax.legend()
    fig.show()

    fig, ax = pl.subplots()
    ax.plot(years[year_ind:], simres['n_alive'][year_ind:], label='HPVsim')
    ax.scatter(years[year_ind:], rsa_df['n_alive'], label='Thembisa')
    ax.set_title(f'Total population {fig_string}')
    ax.legend()
    fig.show()



    import seaborn as sns

    # Prevalence by age
    fig, ax = pl.subplots()

    sim_data = pd.DataFrame(simres['hiv_prevalence_by_age'][:, year_ind:].T,
                            index=pd.Index(years[year_ind:], name='Year'), columns=sim.pars['age_bin_edges'][:-1])
    sdm = pd.melt(sim_data.reset_index(), id_vars=['Year'], var_name='Age', value_name='HIV Prevalence')
    sdm['AgeBin'] = pd.cut(sdm['Age'], bins=sim.pars['age_bin_edges'], include_lowest=True, right =False)
    sdm['Source'] = 'HPVsim'

    cols = ['Year', 'AgeBin', 'HIV Prevalence', 'Source']
    byage = pd.concat([sdm[cols]], ignore_index=True)

    sns.lineplot(data=byage, x='Year', y='HIV Prevalence', hue='AgeBin', palette='Set1', ax=ax)

    ax.set_title(f'HIV prevalence by age {fig_string}')
    ax.legend(bbox_to_anchor=(1.05, 1), ncol=1)
    fig.tight_layout()
    fig.show()

    fig, ax = pl.subplots()

    sim_data = pd.DataFrame(simres['female_hiv_prevalence_by_age'][:, year_ind:].T,
                            index=pd.Index(years[year_ind:], name='Year'), columns=sim.pars['age_bin_edges'][:-1])
    sdm = pd.melt(sim_data.reset_index(), id_vars=['Year'], var_name='Age', value_name='HIV Prevalence')
    sdm['AgeBin'] = pd.cut(sdm['Age'], bins=sim.pars['age_bin_edges'], include_lowest=True, right =False)
    sdm['Source'] = 'HPVsim'

    cols = ['Year', 'AgeBin', 'HIV Prevalence', 'Source']
    byage = pd.concat([sdm[cols]], ignore_index=True)

    sns.lineplot(data=byage, x='Year', y='HIV Prevalence', hue='AgeBin', palette='Set1', ax=ax)

    ax.set_title(f'Female HIV prevalence by age {fig_string}')
    ax.legend(bbox_to_anchor=(1.05, 1), ncol=1)
    fig.tight_layout()
    fig.show()


    # New infections by age and sex
    fig, axes = pl.subplots(1, 2, figsize=(12, 8))
    sdms = []
    for sex in ['female', 'male']:
        sim_data = pd.DataFrame(simres[f'{sex}_hiv_infections_by_age'][:, year_ind:].T,
                                index=pd.Index(years[year_ind:], name='Year'), columns=sim.pars['age_bin_edges'][:-1])
        sdm = pd.melt(sim_data.reset_index(), id_vars=['Year'], var_name='Age', value_name='HIV Infections')
        sdm['AgeBin'] = pd.cut(sdm['Age'], bins=sim.pars['age_bin_edges'], include_lowest=True, right=False)
        sdm['Sex'] = 'f' if sex == 'female' else 'm'
        sdm['Source'] = 'HPVsim'
        sdms.append(sdm)
    sdm = pd.concat(sdms)
    hiv = pd.read_csv(hiv_datafile[0])
    hiv['AgeBin'] = pd.cut(hiv['Age'], bins=sim.pars['age_bin_edges'], include_lowest=True, right=False)
    x = hiv.groupby(['Year', 'AgeBin', 'Sex'])['Incidence'].mean().reset_index().sort_values(
        by=['Sex', 'AgeBin']).reset_index()
    hiv_infections = []
    for sex in ['females', 'males']:
        pop_data = pd.DataFrame((simres[f'n_{sex}_no_hiv_alive_by_age'][:, year_ind:]).T, index=years[year_ind:],
                                columns=sim.pars['age_bin_edges'][:-1])
        pop_data = pop_data.melt(ignore_index=False)
        sex_label = 'f' if sex == 'females' else 'm'
        hiv_infections_this_sex = x[x['Sex'] == sex_label]['Incidence'].values * pop_data['value'].values
        hiv_infections += list(hiv_infections_this_sex)
    x['HIV Infections'] = hiv_infections
    x['Source'] = 'Thembisa'
    cols = ['Year', 'AgeBin', 'Sex', 'HIV Infections', 'Source']
    byage = pd.concat([sdm[cols], x[cols]], ignore_index=True)

    for i_s, sex in enumerate(['f', 'm']):
        ax = axes[i_s]
        byage_to_plot = byage[byage['Sex'] == sex]
        sns.lineplot(data=byage_to_plot, x='Year', y='HIV Infections', hue='AgeBin', style='Source', palette='Set1',
                     ax=ax)
        sex_label = 'Female' if sex == 'f' else 'Male'
        ax.set_title(f'{sex_label}')

    axes[0].get_legend().remove()
    axes[1].legend(bbox_to_anchor=(1.05, 1), ncol=1)
    fig.suptitle(f'HIV infections {fig_string}')
    fig.tight_layout()
    fig.show()

    # ## TEST SPACE ###
    # test_df = self.pop_age_trend[(self.pop_age_trend.year >= 2010) & (self.pop_age_trend.year <= 2020)]
    # import matplotlib.pylab as pl
    # fig, axes = pl.subplots(1, 2)
    # for year in np.unique(test_df['year'].values):
    #     subset_df = test_df[test_df['year'] == year]
    #     axes[0].plot(subset_df['age'], subset_df['male'], label=year)
    #     axes[1].plot(subset_df['age'], subset_df['female'])
    # axes[0].set_title('Male')
    # axes[1].set_title('Female')
    # axes[0].set_xlabel('Age')
    # axes[0].set_ylabel('Pop size')
    # axes[1].set_xlabel('Age')
    # axes[0].legend()
    # fig.tight_layout()
    # fig.show()
    return sim


def test_hiv_epi():
    ''' Run various epi tests for HIV '''

    # Define baseline parameters and initialize sim
    base_pars = dict(
        location='south africa',
        n_agents=5e3,
        n_years=30,
        dt=0.25,
        verbose=0,
        analyzers=hpv.age_causal_infection()
    )
    hiv_settings = dict(model_hiv=True, hiv_datafile=hiv_datafile, art_datafile=art_datafile)

    # Test 1: if HIV mortality is zero, then cancer incidence should be higher with HIV on
    s0 = hpv.Sim(pars=base_pars, label='No HIV').run()
    s1 = hpv.Sim(pars=base_pars, **hiv_settings, hiv_pars={'model_hiv_death':False,}, label='HIV without mortality').run()

    var = 'cancers'
    v0 = s0.results[var][:].sum()
    v1 = s1.results[var][:].sum()
    print(f'Checking {var:10s} with sim "{s0.label}" vs "{s1.label}"... ', end='')
    assert v0 <= v1, f'Expected {var} to be lower in sim "{s0.label}" than in sim "{s1.label}", but {v0} > {v1})'
    print(f'✓ ({v0} <= {v1})')

    # Test 2: with HIV on, the average age of cancer should be younger
    s2 = hpv.Sim(pars=base_pars, **hiv_settings, label='With HIV').run()
    age_cancer_0 = np.mean(s0.get_analyzer().age_cancer)
    age_cancer_2 = np.mean(s2.get_analyzer().age_cancer)
    print(f'Checking mean age of cancer with sim "{s0.label}" vs "{s2.label}', end='')
    assert age_cancer_0 >= age_cancer_2, f'Expected mean age of cancer to be older in sim "{s0.label}" than in sim "{s2.label}", but {age_cancer_2} > {age_cancer_0})'
    print(f'✓ ({age_cancer_0} >= {age_cancer_2})')

    # Test 3: there should be more cancers with HIV off compared to a counterfactual where HIV is on but has no impact on HPV
    hiv_pars = {
        'rel_sus': {'lt200': 1, 'gt200': 1},
        'rel_sev': {'lt200': 1, 'gt200': 1},
        'rel_imm': {'lt200': 1, 'gt200': 1},
        }
    s3 = hpv.Sim(pars=base_pars, **hiv_settings, hiv_pars=hiv_pars, label='HIV without HPV impact').run()

    var = 'cancers'
    v0 = s0.results[var][:].sum()
    v3 = s3.results[var][:].sum()
    print(f'Checking {var:10s} with sim "{s0.label}" vs "{s3.label}"... ', end='')
    assert v0 >= v3, f'Expected {var} to be lower in sim "{s0.label}" than in sim "{s3.label}", but {v3} > {v0})'
    print(f'✓ ({v0} >= {v3})')

    return


def test_impact_on_cancer():
    sc.heading('Testing hiv')

    pars = {
        'n_agents': n_agents,
        'location': 'south africa',
        'start': start,
        'end': 2030
    }

    base_sim = hpv.Sim(
        pars=pars,
        hiv_datafile=hiv_datafile,
        art_datafile=art_datafile
    )

    scenarios = {
        'no_hiv': {
            'name': 'No HIV',
            'pars': {
                'model_hiv': False
            }
        },
        'hiv_baseline': {
            'name': 'HIV',
            'pars': {
                'model_hiv': True
            }
        }
    }

    metapars = {'n_runs': 2}
    scens = hpv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(debug=debug)
    to_plot = {
        'HPV prevalence': [
            'hpv_prevalence',
        ],
        'Age standardized cancer incidence (per 100,000 women)': [
            'asr_cancer_incidence',
        ],

    }
    scens.plot(to_plot=to_plot)
    return scens


def test_calibration_hiv():
    sc.heading('Testing calibration with hiv pars')
    pars = {
        'n_agents': n_agents,
        'location': 'south africa',
        'model_hiv': True,
        'start': start,
        'end': 2020,
    }

    sim = hpv.Sim(
        pars=pars,
        hiv_datafile=hiv_datafile,
        art_datafile=art_datafile
    )

    calib_pars = dict(
        beta=[0.05, 0.010, 0.20],
    )
    genotype_pars = dict(
        hpv16=dict(
            cancer_fn=dict(ld50=[20, 15, 30, 0.5]),
        ),
        hpv18=dict(
            cancer_fn=dict(ld50=[20, 15, 30, 0.5]),
        )
    )

    hiv_pars = dict(
        rel_sus= dict(
            lt200=[3, 2,4]
        )
    )


    calib = hpv.Calibration(sim, calib_pars=calib_pars, genotype_pars=genotype_pars, hiv_pars=hiv_pars,
                            datafiles=[
                                '../test_data/south_africa_hpv_data.csv',
                                '../test_data/south_africa_cancer_data_2020.csv',
                                '../test_data/south_africa_cancer_data_hiv_2020.csv',
                                '../test_data/south_africa_cancer_incidence_by_age_with_hiv.csv',
                            ],
                            total_trials=3, n_workers=1)
    calib.calibrate(die=True)
    calib.plot(res_to_plot=4)
    return sim, calib


#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()
    sim0 = test_hiv()
    # test_hiv_epi()
    # scens0 = test_impact_on_cancer()
    # sim1, calib = test_calibration_hiv()
    sc.toc(T)
    print('Done.')

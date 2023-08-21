#%% from https://github.com/SALib/SALib/tree/0d6d1551243a6ad7c8cf46947728db6bcfebd7b1

"""Multi-output analysis and plotting example."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SALib.test_functions import lake_problem
from SALib.test_functions import Ishigami
from SALib import ProblemSpec, sample
from SALib.analyze import sobol

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from test_sampler import PredefinedSampler
from test_precalib_exploration import *
import optuna as op
import sciris as sc
import hpvsim as hpv


#%% Functions for sobol sampler and analyzer

def heatmap(sp, cmap='Blues', index='ST'):
    met_data = sp.analysis
    fig, ax = plt.subplots(1, 1, figsize=(15, 1+0.5*len(met_data)))
    s_data = np.vstack([met_data[key][index] for key in met_data.keys() if index in met_data[key]])
    
    sns.heatmap(s_data, cmap=cmap, ax=ax, annot=True, fmt=".1f")
    ax.xaxis.set_ticks(np.arange(0.5, len(sp['names']) + 0.5))
    ax.xaxis.set_ticklabels(sp['names'], rotation=90)
    ax.yaxis.set_ticks(np.arange(0.5, len(met_data) + 0.5))
    ax.yaxis.set_ticklabels(list(met_data.keys()))
    plt.show()
# %%
if __name__ == "__main__":

    custom_param_space = pd.read_csv(f'param_space_filled_nolatency.csv', index_col = 0)
    custom_param_space = custom_param_space[custom_param_space['Notes'].isin(['Assume'])][:10] # Filter parameters
    result_folder = '../../hpvsim_txvx_analyses/results'
    org_calib = sc.load(f'{result_folder}/india_calib_aug7_nolatency.obj')
    org_sim = org_calib.sim
    best_pars = org_calib.trial_pars_to_sim_pars()
    org_sim.update_pars(best_pars)
    org_pars = org_sim.pars
    location ='india'
    datafiles = [
        f'test_data/{location}_hpv_prevalence.csv',
        f'test_data/{location}_cancer_cases.csv',
        f'test_data/{location}_cin1_types.csv',
        f'test_data/{location}_cin3_types.csv',
        f'test_data/{location}_cancer_types.csv',
    ]
    extra_sim_result_keys = ['asr_cancer_incidence', 'n_precin_by_age', 'n_cin1_by_age', 'n_females_alive_by_age']
    extra_sim_analyzers = [hpv.age_causal_infection(start_year=2000)]  


    sp = ProblemSpec(
        {
            "names": [name.replace("genotype_pars/", "").replace("/", "_") for name in custom_param_space.param_name.values],
            "bounds": np.array([custom_param_space.lower_bound.values.astype(float), custom_param_space.upper_bound.values.astype(float)]).T,
        }
    )

    X = sample.saltelli.sample(sp, 2) # N *(2D+2) where N = number of sample, D = input dimension
    total_trials = len(X)
    n_workers = 8

    base_pars = sc.dcp(org_pars)
    base_pars['analyzers'] = []
    calib_list = get_calib_list(custom_param_space)
    calib_pars = sc.objdict()
    for i in range(len(calib_list)):
        calib_pars = sc.mergenested(calib_list[i], calib_pars)
    genotype_pars = None
    if 'genotype_pars' in calib_pars:
        genotype_pars = calib_pars.pop('genotype_pars')
    search_space = {key: value for key, value in zip(sp['names'], sp['bounds'])}

    sampler = PredefinedSampler(search_space, samp_list = X)
    (sim, calib)= precalib_SA(location, base_pars, calib_pars, genotype_pars, total_trials, n_workers, datafiles, extra_sim_result_keys, extra_sim_analyzers, sampler, result_folder)

    sc.saveobj('results/sobol.obj', calib)

# %% X and Y's number and order should be the same !!
calib = sc.load('results/sobol.obj')
result_df = calib.df
param_cols = list(set(sp["names"]) & set(result_df.columns))
result_df = result_df[['index','mismatch']+param_cols]
result_df = result_df.drop_duplicates(['mismatch']+param_cols)

matching_indices = []
for df_row in result_df[param_cols].values:
    matching_index = np.where((df_row == X).all(axis=1))[0]
    matching_indices.append(matching_index[0] if matching_index.size > 0 else None)
result_df['X_index'] = matching_indices
result_df = result_df.sort_values('X_index')

Y = np.array(result_df['mismatch'].values)
res = dict()
res['mismatch'] = sobol.analyze(sp, Y)
sp._analysis = res
heatmap(sp, index='ST')

# %%


#%% Imports and settings
import hpvsim as hpv
import os
import sciris as sc
import pandas as pd
import numpy as np
import math
import warnings
    
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')


#%% First, make a csv that lists all parameters
def get_all_param_space(sim):
    param_space = pd.DataFrame(columns=["param_name", "param_type", "base_value","lower_bound",  "upper_bound", "Notes"])

    def get_all_keys(dictionary, prefix=''):
        keys = []
        for key, value in dictionary.items():
            new_key = prefix + str(key) if prefix else str(key)
            if isinstance(value, dict):
                keys.extend(get_all_keys(value, new_key + '/'))
            else:
                param_name = new_key
                base_value = value
                param_type = "numeric" if isinstance(value, (int, float)) and not math.isnan(value) else "other"
                keys.append([param_name, param_type, base_value])
        return keys

    all_keys = get_all_keys(sim.pars)

    for key in all_keys:
        param_space = pd.concat([param_space, pd.DataFrame([key], columns=["param_name", "param_type", "base_value"])])

    param_space.reset_index(drop=True, inplace=True)
    return param_space


#%% Given custom_made param_space, functions to prepare for calibration

# Create calib_pars dictionary from csv file (string to nested dict)
def create_nested_dict(keys, base_value, lower_bound, upper_bound, n_test):
    nested_dict = {}
    current_dict = nested_dict
    key_list = keys.split('/')
    last_key = key_list[-1]
    for key in key_list[:-1]:
        current_dict[key] = {}
        current_dict = current_dict[key]
    current_dict[last_key] = [base_value, lower_bound, upper_bound, math.floor((upper_bound - lower_bound) / (n_test-1)*10e3)/10e3]
    return nested_dict

# Nested dict's keys to string
def get_nested_dict_keys(nested_dict, prefix=''):
    keys = []
    for key, value in nested_dict.items():
        new_key = prefix + '_' + key if prefix else key
        if isinstance(value, dict):
            nested_keys = get_nested_dict_keys(value, prefix=new_key)
            keys.extend(nested_keys)
        else:
            keys.append(new_key)
    return ''.join(keys)

def estimator(actual, predicted):
        actuals = []
        for i in actual:
            i_list = [idx for idx in i.split(',')]
            i_list[0] = float(i_list[0].replace('[', ''))
            i_list[1] = float(i_list[1].replace(']', ''))
            actuals.append(i_list)
        gofs = np.zeros(len(predicted))
        for iv, val in enumerate(predicted):
            if val> np.max(actuals[iv]):
                gofs[iv] = abs(np.max(actuals[iv])-val)
            elif val < np.min(actuals[iv]):
                gofs[iv] = abs(np.min(actuals[iv])-val)

        actual_max = np.array(actuals).max()
        if actual_max > 0:
            gofs /= actual_max

        gofs = np.mean(gofs)

        return gofs

# Get list of calibration parameters
def get_calib_list(custom_param_space, n_test):
    calib_list = []
    for _, row in custom_param_space.iterrows():
        keys = row['param_name']
        base_value = row['base_value']
        lower_bound = row['lower_bound']
        upper_bound = row['upper_bound']
        if(row['param_type']=='numeric'):
            calib_param = create_nested_dict(keys, float(base_value), float(lower_bound), float(upper_bound), n_test)
        calib_list.append(calib_param)
    return calib_list


def one_way_SA(base_pars, calib_pars, genotype_pars, n_test, datafiles, extra_sim_result_keys, extra_sim_analyzers):
    import optuna as op
    import multiprocessing as mp
    pars_name = ''
    if calib_pars is not None: pars_name = get_nested_dict_keys(calib_pars)
    elif genotype_pars is not None: pars_name = get_nested_dict_keys(genotype_pars)
    name = f'{location}_{pars_name}_calib'

    if f'{name}.obj' in os.listdir('results'):
        print(f"{name} already processed. Skipping.")
        return

    sim = hpv.Sim(pars=base_pars)

    calib = hpv.Calibration(sim, calib_pars=calib_pars, genotype_pars=genotype_pars,
                            name=name, estimator=estimator,
                            sampler = op.samplers.BruteForceSampler(),
                            datafiles=datafiles, extra_sim_result_keys=extra_sim_result_keys,
                            extra_sim_analyers=extra_sim_analyzers,
                            n_trials=1, n_workers=min(mp.cpu_count(),n_test))
    calib.calibrate(die=True)
    sc.saveobj(f'results/{name}.obj', calib)

    
# def plot_tornado_chart()


#%% Run as a script
if __name__ == '__main__':
    T = sc.tic()

    # Make full parameter space if you haven't done already
    # sim = sc.load('test_data/india_calib_july18.obj').sim
    # param_space = get_all_param_space(sim)
    # # Save the parameter space to a csv. If a parameter is numeric or can vary, fill the lower and upper bound at best. 
    # # Also, if you have the best parameter estimate, change the base values based on that.
    # # If a parameter is hard to test lower and upper bound, fill with NA 
    # param_space.to_csv('param_space.csv')

    # Read calibration files and decide settings
    org_calib = sc.load('test_data/india_calib_july18.obj')
    org_sim = org_calib.sim
    best_pars = org_calib.trial_pars_to_sim_pars()
    org_sim.update_pars()
    org_pars = org_sim.pars
    custom_param_space = pd.read_csv('param_space_filled.csv', index_col = 0)
    custom_param_space = custom_param_space[custom_param_space['Notes'].isin(['Assume'])]  # Filter parameters
    
    # Use your Own Calibration settings
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
    n_test = 8

    calib_list = get_calib_list(custom_param_space, n_test)

    for calib_to_pars in calib_list:
        print(calib_to_pars)
        base_pars = sc.dcp(org_pars)
        base_pars['analyzers'] = []
        calib_pars = None; genotype_pars = None 
        if 'genotype_pars' in calib_to_pars.keys():
            genotype_pars = calib_to_pars['genotype_pars']
        else:
            calib_pars = calib_to_pars
        one_way_SA(base_pars, calib_pars, genotype_pars, n_test, datafiles, extra_sim_result_keys, extra_sim_analyzers)

    # sc.toc(T)
    print('Done.')

# %% Observe results

# Read results
location = 'india'
import glob
pattern = f'results/{location}_*_calib.obj'
file_paths = glob.glob(pattern)

df_glob = pd.DataFrame()
for file_path in file_paths:
    calib = sc.load(file_path)
    calib_result = pd.DataFrame()
    param_name = (calib.initial_pars.keys()[0])
    param_name = file_path.split(str(location)+"_")[1].split("_calib")[0]
    calib_result['param_name'] = [param_name]*len(calib.df)
    calib_result['param_value'] = [calib.df.iloc[i,-1] for i in range(len(calib.df))]
    calib_result['asr_cancer_incidence'] = [calib.extra_sim_results[i]['asr_cancer_incidence'].values[-1] for i in range(len(calib.df))]
    calib_result['median_age_causal_infection_cancer'] = [np.median(calib.extra_analyzer_results[i]['age_causal_infection'].age_cancer) for i in range(len(calib.df))]
    calib_result['median_age_causal_infection_hpv'] = [np.median(calib.extra_analyzer_results[i]['age_causal_infection'].age_causal) for i in range(len(calib.df))]
    calib_result['mismatch'] = [calib.df.iloc[i].mismatch for i in range(len(calib.df))]
    df_glob = pd.concat([df_glob, calib_result], ignore_index=True)


# %% Plot results

import seaborn as sns
import matplotlib.pyplot as plt
param_list = list(df_glob.param_name.unique())
outcome_list = df_glob.columns.difference(['param_name', 'param_value', 'stand_outcome'])

def full_sensitivity_chart(df, outcome):
    param_list = df['param_name'].unique()
    y_range = [df[outcome].min()*0.8,df[outcome].max()*1.2]
    nrow = len(param_list)//5+1
    fig, axes = plt.subplots(nrows=nrow, ncols=5, figsize=(15, 3*nrow))
    axes = axes.flatten()
    for i, param in enumerate(param_list):
        df_param = df[df['param_name'] == param]
        ax = axes[i]
        sns.lineplot(data=df_param, x='param_value', y=outcome,ax=ax)
        ax.set_title(f"{param}")
        ax.set_xlabel('Range')
        ax.set_ylabel('')
        ax.set_ylim(y_range)
    plt.suptitle(f'Sensitivity to {outcome}', fontsize=15)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'plots/{location}_{outcome}_full_all.png')

def tornado_chart_sorted(df, outcome):
    param_list = df['param_name'].unique()
    values = [df[df['param_name'] == param][outcome].max() - (df[df['param_name'] == param][outcome]).min() for param in param_list]
    sorted_params, sorted_values = zip(*sorted(zip(param_list, values), key=lambda x: x[1], reverse=False))
    fig, ax = plt.subplots(figsize=(6, 2+0.2*len(param_list)))
    ax.barh(sorted_params, sorted_values, align='center', color='skyblue')
    ax.axvline(x=0, color='black', linestyle='--')
    ax.set_xlabel('Max - Min')
    ax.set_title('Max-Min Chart for \n'+outcome, fontsize=15)
    ax.tick_params(axis='y', which='both', length=0)
    plt.tight_layout()
    plt.savefig(f'plots/{location}_{outcome}_maxmin_all.png')

for outcome in outcome_list:
    df_remain = df_glob[~df_glob['param_name'].str.contains('episomal')]
    full_sensitivity_chart(df_remain, outcome)
    tornado_chart_sorted(df_remain, outcome)

# sexual_beh_param = ['age_act_pars', 'condoms', 'dur_pship',']

# sexual_df = df_glob[df_glob['param_name'].isin(sexual_beh_param)]

# for outcome in outcome_list:
#     full_sensitivity_chart(sexual_df, outcome,sexual_beh_param)
#     tornado_chart_sorted(sexual_df, outcome, sexual_beh_param)



# %%

# TODO: 
# Try CMAES - with restart
# Clustering with outcome variables?
# VM? Simulation time - similar processing time when n_agents is large even with multiple cpus (~5 min per param).
# modify calibration.py plot function to set rn by resname
# debug - nested dict.
# Write TxVx document

# DONE:
# address nested dicts in get_full_pars, trial_to_sim, and sim_to_sample_pars (did not touch genotypepars and hivpars)
# Use BruteForceSampler for exploration. If this sampler is used, your input for calib_pars should include [lower, upper, stepsize]. Optuna will run for each stepsize
# Changed extra_sim_analyers to extra_sim_analyzers

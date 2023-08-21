
#%% Imports and settings
import hpvsim as hpv
import os
import sciris as sc
import pandas as pd
import numpy as np
import math
import warnings
    
pd.set_option('display.max_columns', None)
# warnings.filterwarnings('ignore')

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
def create_nested_dict(keys, base_value, lower_bound, upper_bound, n_test=None):
    nested_dict = {}
    current_dict = nested_dict
    key_list = keys.split('/')
    last_key = key_list[-1]
    for key in key_list[:-1]:
        current_dict[key] = {}
        current_dict = current_dict[key]
    if n_test is not None:
        current_dict[last_key] = [base_value, lower_bound, upper_bound, math.floor((upper_bound - lower_bound) / (n_test-1)*1e10)/1e10]
    else:
        current_dict[last_key] = [base_value, lower_bound, upper_bound]
    return nested_dict

# Nested dict's keys to string. Supports only one parameter
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
def get_calib_list(custom_param_space, n_test=None):
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

def precalib_SA(location, base_pars, calib_pars, genotype_pars, total_trials, n_workers, datafiles, extra_sim_result_keys, extra_sim_analyzers, sampler, result_folder):
    import optuna as op
    import multiprocessing as mp
    pars_name = ''
    # if calib_pars is not None: pars_name = get_nested_dict_keys(calib_pars)
    # elif genotype_pars is not None: pars_name = get_nested_dict_keys(genotype_pars)
    # name = f'{location}_{pars_name}_calib_nolatent'
    name = 'test'
    if f'{name}.obj' in os.listdir(result_folder):
        print(f"{name} already processed. Skipping.")
        return

    sim = hpv.Sim(pars=base_pars)

    calib = hpv.Calibration(sim, calib_pars=calib_pars, genotype_pars=genotype_pars,
                            name=name, estimator=estimator,
                            sampler = sampler,
                            datafiles=datafiles, extra_sim_result_keys=extra_sim_result_keys,
                            extra_sim_analyzers=extra_sim_analyzers,
                            total_trials=total_trials, n_workers=n_workers)
    calib.calibrate(die=True)
    # calib.plot()
    # sc.saveobj(f'{result_folder}/{name}.obj', calib)
    return (sim, calib)


    
# def plot_tornado_chart()


#%% Run as a script
if __name__ == '__main__':
    T = sc.tic()

    # Make full parameter space if you haven't done already
    # sim = sc.load('test_data/india_calib_july18.obj').sim
    # param_space = get_all_param_space(sim)
    # param_space.to_csv('param_space.csv')
    # custom_param_space = pd.read_csv(f'param_space_filled_nolatency.csv', index_col = 0)
    # custom_param_space = custom_param_space[custom_param_space['Notes'].isin(['Assume'])]  # Filter parameters

    # Read calibration files and decide settings
    result_folder = '../../hpvsim_txvx_analyses/results'
    org_calib = sc.load(f'{result_folder}/india_calib_aug7_nolatency.obj')
    org_sim = org_calib.sim
    best_pars = org_calib.trial_pars_to_sim_pars()
    org_sim.update_pars(best_pars)
    org_pars = org_sim.pars
    # custom_param_space = pd.read_csv(f'param_space_filled_nolatency.csv', index_col = 0)
    # custom_param_space = custom_param_space[custom_param_space['Notes'].isin(['Assume','Provided'])]  # Filter parameters
    
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
    n_test = 1
    # org_pars['n_agents'] = 100

    #%% actual running
    import optuna as op

    # calib_list = get_calib_list(custom_param_space, n_test)
    calib_to_pars = [{'n_agents': [org_pars['n_agents'],org_pars['n_agents'],org_pars['n_agents']+10,1]}][0]
    # for calib_to_pars in calib_list:
        # print(calib_to_pars)
    base_pars = sc.dcp(org_pars)
    base_pars['analyzers'] = []
    calib_pars = None; genotype_pars = None 
    if 'genotype_pars' in calib_to_pars.keys():
        genotype_pars = calib_to_pars['genotype_pars']
    else:
        calib_pars = calib_to_pars
    
    (sim, calib)= precalib_SA(base_pars, calib_pars, genotype_pars, n_test, datafiles, extra_sim_result_keys, extra_sim_analyzers, op.samplers.BruteForceSampler() ,result_folder)

    sc.toc(T)
    print('Done.')

# 

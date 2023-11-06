#%% Imports and settings
import sciris as sc
import hpvsim as hpv
import test_sampler
import numpy as np
import pandas as pd
import math
from SALib.sample import latin, sobol

def get_all_keys(pars, prefix=''):
    ''' Sub-function to get parameter names, types, and base values from nested dict '''
    keys = []
    for key, value in pars.items():
        new_key = prefix + str(key) if prefix else str(key)
        if isinstance(value, dict):
            keys.extend(get_all_keys(value, new_key + '/'))
        else:
            param_name = new_key
            default_value = value
            param_type = 'other'
            if isinstance(value, (int, float)) and not math.isnan(value):
                param_type = "numeric"
            elif isinstance(value, (str)):
                param_type = "string"
            elif isinstance(value, (np.ndarray, list)):
                param_type = "matrix/list"
            keys.append([param_name, param_type, default_value])
    return keys

def get_all_param_space(pars):
    ''' Get parameter settings from a sim.pars to df'''
    param_space = pd.DataFrame(columns=["param_name", "param_type", "default_value", "lower_bound",  "upper_bound", "Notes"])
    all_keys = get_all_keys(pars)
    for key in all_keys:
        if 'death_rates' not in key[0]:
            param_space = pd.concat([param_space, pd.DataFrame([key], columns=["param_name", "param_type", "default_value"])])
    param_space.reset_index(drop=True, inplace=True)
    return param_space

def create_nested_dict_with_values(keys, default_value, lower_bound, upper_bound, n_test=None):
    ''' Subfunction that maps a single parameter with default, lower, and upper bound to nested dict'''
    nested_dict = {}
    current_dict = nested_dict
    key_list = keys.split('/')
    last_key = key_list[-1]
    for key in key_list[:-1]:
        current_dict[key] = {}
        current_dict = current_dict[key]
    if n_test is not None:
        current_dict[last_key] = [default_value, lower_bound, upper_bound, math.floor((upper_bound - lower_bound) / (n_test-1)*1e13)/1e13]
    else:
        current_dict[last_key] = [default_value, lower_bound, upper_bound]
    
    return nested_dict

def get_calib_parameters(param_df):    
    ''' Read csv file and modify calibration parameters to be ready for optuna and lhs sampling'''
    calib_pars = sc.objdict()
    genotype_pars = sc.objdict()  
    for _, row in param_df.iterrows():
        keys = row['param_name']
        default_value = row['default_value']
        lower_bound = row['lower_bound']
        upper_bound = row['upper_bound']
        calib_param = create_nested_dict_with_values(keys, float(default_value), float(lower_bound), float(upper_bound))
        calib_pars = sc.mergenested(calib_param, calib_pars)
    if 'genotype_pars' in calib_pars:
        genotype_pars =  calib_pars.pop('genotype_pars')

    calib_space = sc.objdict(names = [name.replace("genotype_pars/", "").replace("/", "_") for name in param_df.param_name.values], 
                        bounds = np.array([param_df.lower_bound.values.astype(float), param_df.upper_bound.values.astype(float)]).T,
                        default_value = param_df.default_value.values.astype(float))
    calib_space['num_vars'] = len(calib_space['names'])
    return (calib_pars, genotype_pars, calib_space)

def estimator(actual, predicted):
    ''' Custom estimator to use for bounded target data'''
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

def get_sample_one_way(total_trials, calib_space):
    import copy
    N_per_param = round(total_trials/calib_space['num_vars'])
    samples = [copy.deepcopy(calib_space['default_value']) for _ in range(N_per_param * calib_space['num_vars'])]
    count = 0
    for param_name, param_bounds, default_value in zip(calib_space.names, calib_space.bounds, calib_space.default_value):
        param_samples = np.linspace(param_bounds[0], param_bounds[1], N_per_param)
        for samp in param_samples:
            samples[count][count // N_per_param] = samp
            count += 1
    return np.array(samples)

def get_sample(sample_method, sample_seed, calib_space, total_trials):
    try:
        if sample_method == 'lhs':
            return latin.sample(calib_space, total_trials, sample_seed)
        elif sample_method == 'one-way':
            sample_list = get_sample_one_way(total_trials, calib_space)
            if len(sample_list) < 1:
                raise ValueError(f"Total trials should be larger than {(calib_space['num_vars'])} to generate one-way sampled list")
            return sample_list
        elif sample_method == 'sobol':
            N = round(total_trials / (2 * (1 + calib_space['num_vars'])))
            if N < 1:
                raise ValueError(f"Total trials should be larger than {(2 * (1 + calib_space['num_vars']))} to generate one-way sampled list")
            return sobol.sample(problem=calib_space, N=N, seed=sample_seed)
    except ValueError as e:
      print(f"Simulation stopped due to a ValueError: {str(e)}")


def run_parameter_exploration(location, datafiles, default_pars, calib_pars, genotype_pars, calib_space, total_trials, n_workers, name, save_results, 
                              sample_method='sobol', sample_seed=1234):
    ''' Run parameter exploration'''
    # Use a custom sampler that has full list of sample sets from lhs
    search_space = {key: value for key, value in zip(calib_space['names'], calib_space['bounds'])}
    X = get_sample(sample_method, sample_seed, calib_space, total_trials)
    sampler = test_sampler.PredefinedSampler(search_space, samp_list = X) 

    # Create Calibration object
    # Due to issues in parallel computing, the sampler sometimes samples the same space. We suggest adding more total trials so that all sampled points are evaluated
    sim = hpv.Sim(default_pars)
    calib = hpv.Calibration(sim, calib_pars=calib_pars, genotype_pars=genotype_pars,
                            name=name,
                            sampler = sampler, verbose = False,
                            datafiles=datafiles, 
                            total_trials=total_trials+3*n_workers, n_workers=n_workers) 
    
    # Finally, calibrate the model
    calib.calibrate(die=False)
    if save_results:
        import os
        if not os.path.exists('results'):
            os.makedirs('results')
        sc.saveobj(f'results/{name}.obj', calib)
    return calib

def expand_array_column(df, array_column_name, sub_column_name):
    ''' Expand list types of outcomes '''
    array_data = df[array_column_name]
    df[array_column_name] = df[array_column_name].apply(lambda x: sum(x**2) if 'dist' in array_column_name else sum(x))
    max_length = max(len(row) for row in array_data)
    column_names = [f'{array_column_name}_{sub_column_name[i]}' for i in range(max_length)]
    array_df = pd.DataFrame(array_data.tolist(), columns=column_names)
    return pd.concat([df, array_df], axis=1)

def organize_results(calib, calib_space):
    ''' Organize results'''
    result_df = calib.df
    param_cols = calib_space.names
    result_df = result_df[['index','mismatch']+param_cols]
    result_df = result_df.sort_values('index').reset_index(drop=True)

    for sim_key in calib.sim_results_keys:
        result_df[sim_key] = [result[sim_key] for result in calib.sim_results]
        if 'genotype_dist' in sim_key:
            subcols = calib.glabels
        else: subcols = [i for i in range(len(result_df[sim_key,0]))]
        result_df = expand_array_column(result_df, sim_key, subcols)
    
    for analyzer_key in calib.age_results_keys:
        year = [key for key in calib.analyzer_results[0][analyzer_key].keys() if key != 'bins'][0]
        result_df[analyzer_key] = [result[analyzer_key][year] for result in calib.analyzer_results]
        age_bins = calib.analyzer_results[0][analyzer_key]['bins']
        subcols = [f'age_{int(age_bins[i])}-{int(age_bins[i+1]-1)}' if i < len(age_bins)-1 else f'age{int(age_bins[-1])}+' for i in range(len(age_bins))]
        result_df = expand_array_column(result_df, analyzer_key, subcols)

    return (result_df)

def fit_model(model_name, X, Y):
    ''' Fitting a model to identify important parameters'''
    from sklearn.preprocessing import StandardScaler
    from sklearn.inspection import permutation_importance
    
    if model_name=='LinearRegression':
        from sklearn.linear_model import LinearRegression 
        reg_model = LinearRegression()
    elif model_name == 'RandomForest':
        from sklearn.ensemble import RandomForestRegressor
        reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == 'XGBoost':
        from xgboost import XGBRegressor
        reg_model = XGBRegressor(n_estimators=100)

    param_importance = pd.DataFrame(columns = Y.columns, index = X.columns)
    scaler = StandardScaler()
    norm_X = scaler.fit_transform(X)
    for i in range(Y.shape[1]):  # Assuming Y is a 2D array with shape (n_samples, n_targets)
        Y_target = Y.iloc[:,i]  # Select the i-th target variable
        reg_model.fit(norm_X, Y_target)
        perm_importances = permutation_importance(reg_model, norm_X, Y_target)
        param_importance[Y.columns[i]] = perm_importances.importances_mean
    return param_importance

def get_interest_outcome(calib, Y, outcome_level):
    outcomes = [[] for _ in range(3)]
    outcomes[0] = ['mismatch']
    outcomes[1] = sorted(['mismatch']+calib.sim_results_keys+calib.age_results_keys)
    outcomes[2] = sorted(Y.columns)
    return (outcomes[outcome_level])

def heatmap(param_importance, outcomes, save_plot=False, sort_by=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    df = pd.DataFrame(StandardScaler().fit_transform(param_importance), index=param_importance.index, columns = param_importance.columns)[outcomes]
    if sort_by is not None:
        df = df.sort_values(by=sort_by, ascending = False)
    plt.figure(figsize=(0.2*df.shape[0], 0.2*df.shape[1]))
    heatmap = sns.heatmap(np.abs(df.T), cmap="Blues", linewidths=0.5, linecolor="white")
    heatmap.set_xticks([0.5 + i for i in range(len(df.index))]) 
    heatmap.set_yticks([0.5 + i for i in range(len(df.columns))])
    heatmap.set_xticklabels(df.index, rotation=90, fontsize=7) 
    heatmap.set_yticklabels(df.columns, rotation=0, fontsize=7)
    plt.show()
    if save_plot:
        plt.savefig(f'results/pre_calib.png')

# %%

#%% Imports and settings
import sciris as sc
import hpvsim as hpv
import optuna as op
from tests.test_sampler import PredefinedSampler
import numpy as np
import pandas as pd
import math

# sub-function to get parameter names, types, and base values from nested dict
def get_all_keys(dictionary, prefix=''):
    keys = []
    for key, value in dictionary.items():
        new_key = prefix + str(key) if prefix else str(key)
        if isinstance(value, dict):
            keys.extend(get_all_keys(value, new_key + '/'))
        else:
            param_name = new_key
            base_value = value
            param_type = 'other'
            if isinstance(value, (int, float)) and not math.isnan(value):
                param_type = "numeric"
            elif isinstance(value, (str)):
                param_type = "string"
            elif isinstance(value, (np.ndarray, list)):
                param_type = "matrix/list"
            keys.append([param_name, param_type, base_value])
    return keys

def get_all_param_space(pars):
    # get parameter settings from a sim to df
    param_space = pd.DataFrame(columns=["param_name", "param_type", "base_value", "lower_bound",  "upper_bound", "Notes"])
    all_keys = get_all_keys(pars)
    for key in all_keys:
        if 'death_rates' not in key[0]:
            param_space = pd.concat([param_space, pd.DataFrame([key], columns=["param_name", "param_type", "base_value"])])
    param_space.reset_index(drop=True, inplace=True)
    return param_space

def sample_lhs(total_trials, calib_space):
    from pyDOE import lhs
    lb = calib_space.bounds[:,0]
    ub  = calib_space.bounds[:,1]
    sample = lhs(len(calib_space.names), total_trials)
    final_sample = lb + sample*(ub - lb)
    return final_sample

# subfunction that maps string param to nested dict
def create_nested_dict(keys, base_value, lower_bound, upper_bound, n_test=None):
    nested_dict = {}
    current_dict = nested_dict
    key_list = keys.split('/')
    last_key = key_list[-1]
    for key in key_list[:-1]:
        current_dict[key] = {}
        current_dict = current_dict[key]
    if n_test is not None:
        current_dict[last_key] = [base_value, lower_bound, upper_bound, math.floor((upper_bound - lower_bound) / (n_test-1)*1e13)/1e13]
    else:
        current_dict[last_key] = [base_value, lower_bound, upper_bound]
    return nested_dict

# Get list of calibration parameters
def get_calib_list(custom_param_space, n_test=None):
    calib_list = []
    for _, row in custom_param_space.iterrows():
        keys = row['param_name']
        base_value = row['base_value']
        lower_bound = row['lower_bound']
        upper_bound = row['upper_bound']
        calib_param = create_nested_dict(keys, float(base_value), float(lower_bound), float(upper_bound), n_test)
        calib_list.append(calib_param)
    return calib_list


# Read user-defined param space and define a calibration space. 
def get_calib_space(calib_file_name, filter_list):
    custom_param_space = pd.read_csv(calib_file_name, index_col = 0)
    custom_param_space = custom_param_space[custom_param_space['Notes'].isin(filter_list)]  # Filter parameters
    calib_space = sc.objdict(names = [name.replace("genotype_pars/", "").replace("/", "_") for name in custom_param_space.param_name.values], 
                            bounds = np.array([custom_param_space.lower_bound.values.astype(float), custom_param_space.upper_bound.values.astype(float)]).T)
    return custom_param_space, calib_space

# Custom estimator to use for bounded target data
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

def run_precalib_exploration(location, datafiles, default_pars, custom_param_space, calib_space, total_trials, n_workers, name, save_results):
    # Prepare for calibration
    X = sample_lhs(total_trials, calib_space)
    calib_list = get_calib_list(custom_param_space)
    calib_pars = sc.objdict()
    for i in range(len(calib_list)):
        calib_pars = sc.mergenested(calib_list[i], calib_pars)
    genotype_pars = None
    if 'genotype_pars' in calib_pars:
        genotype_pars = calib_pars.pop('genotype_pars')

    # Use custom sampler with predefined sample sets
    search_space = {key: value for key, value in zip(calib_space['names'], calib_space['bounds'])}
    sampler = PredefinedSampler(search_space, samp_list = X) 

    # Finally, run the model
    # Due to issues in parallel computing, the sampler sometimes samples the same space. We suggest adding more total trials so that all sampled points are evaluated
       
    sim = hpv.Sim(default_pars)
    calib = hpv.Calibration(sim, calib_pars=calib_pars, genotype_pars=genotype_pars,
                            name=name, estimator=estimator, 
                            sampler = sampler, 
                            datafiles=datafiles, 
                            total_trials=total_trials+3*n_workers, n_workers=n_workers) 
    calib.calibrate(die=False)
    if save_results:
        sc.saveobj(f'results/{name}.obj', calib)

# expand list types of outcomes
def expand_array_column(df, array_column_name):
    array_data = df[array_column_name]
    df[array_column_name] = df[array_column_name].apply(lambda x: sum(x**2) if 'dist' in array_column_name else sum(x))
    max_length = max(len(row) for row in array_data)
    column_names = [f'{array_column_name}_col{i+1}' for i in range(max_length)]
    array_df = pd.DataFrame(array_data.tolist(), columns=column_names)
    return pd.concat([df, array_df], axis=1)

# Read results
def organize_results(calib, calib_space):
    result_df = calib.df
    param_cols = [param for param in calib_space["names"] if param in result_df.columns]
    result_df = result_df[['index','mismatch']+param_cols]
    result_df = result_df.sort_values('index').reset_index(drop=True)
    result_df['hpv_prevalence'] = [result['lsil_prevalence'][2020] for result in calib.analyzer_results]
    result_df['cancers'] = [result['cancers'][2020] for result in calib.analyzer_results]
    result_df['cin1_genotype_dist'] = [result['cin1_genotype_dist'] for result in calib.sim_results]
    result_df['cin3_genotype_dist'] = [result['cin3_genotype_dist'] for result in calib.sim_results]
    result_df['cancerous_genotype_dist'] = [result['cancerous_genotype_dist'] for result in calib.sim_results]
    result_df = expand_array_column(result_df, 'hpv_prevalence')
    result_df = expand_array_column(result_df, 'cancers')
    result_df = expand_array_column(result_df, 'cin1_genotype_dist')
    result_df = expand_array_column(result_df, 'cin3_genotype_dist')
    result_df = expand_array_column(result_df, 'cancerous_genotype_dist')
    return (result_df)


# Fitting a model to identify important parameters
def fit_model(model_name, X, Y):
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

def get_interest_outcome(Y, outcome_level):
    # Custom sorting method to organize results
    def custom_sort_key(col):
        match = col.split('_col')
        prefix = match[0]
        suffix = match[1] if len(match) > 1 else None
        return (outcomes[1].index(prefix), int(suffix) if suffix else -1)
    outcomes = [[] for _ in range(3)]
    outcomes[0] = ['mismatch']
    outcomes[1] = ['mismatch','hpv_prevalence','cancers','cin1_genotype_dist','cin3_genotype_dist','cancerous_genotype_dist']
    outcomes[2] = sorted(Y.columns, key=custom_sort_key)
    return (outcomes[outcome_level])

def heatmap(param_importance, outcomes, save_plot):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    df = pd.DataFrame(StandardScaler().fit_transform(param_importance), index=param_importance.index, columns = param_importance.columns)[outcomes]
    plt.figure(figsize=(0.2*df.shape[0], 0.2*df.shape[1]))
    heatmap = sns.heatmap(df.T, cmap="Blues", linewidths=0.5, linecolor="white")
    heatmap.set_xticks([0.5 + i for i in range(len(df.index))]) 
    heatmap.set_yticks([0.5 + i for i in range(len(df.columns))])
    heatmap.set_xticklabels(df.index, rotation=90, fontsize=7) 
    heatmap.set_yticklabels(df.columns, rotation=0, fontsize=7)
    plt.show()
    if save_plot:
        plt.savefig('pre_calib.png')

#%% Configure a simulation with some parameters. If you already have .obj file, you could read sim from there
# location ='india'
# pars = dict(n_agents=10e3, start=1980, end=2020, n_years=40, location=location, verbose=0)
# sim = hpv.Sim(pars)
# sim.run()
# default_pars = sim.pars

# # Read all input parameters from sim and save to csv. Custom fill parameters' lower and upper bounds.
# param_space = get_all_param_space(default_pars)
# param_space.to_csv('param_space.csv')

# #%% When the lower and upper bound 
# # Read user-defined param space and define a calibration space. 
# custom_param_space, calib_space = get_calib_space('param_space_filled.csv', ['Assume'])

# # Set calibration settings
# total_trials = 1000 #number of total lhs samples
# n_workers = 40 # number of CPUs
# name = f'precalib_{location}'
# save_results = True

# # Finally, run calibration
# run_precalib_exploration(location, default_pars, custom_param_space, calib_space, total_trials, n_workers, save_results)

# #%% Analyze the results

# # Read calibration results and organize the results
# calib = sc.load(f'results/{name}.obj')
# custom_param_space, calib_space  = get_calib_space('param_space_filled.csv', ['Assume'])
# result_df = organize_results(calib, calib_space)

# # Now analyze the parameter importance using machine learning. Current version supports LinearRegression, RandomForest, and XGBoost
# param_cols = calib_space['names']
# X = result_df[param_cols]
# Y = result_df[result_df.columns.difference(param_cols+['index'])]
# param_importance = fit_model('XGBoost', X, Y)

# # %%Plot results. Users can specify the outcome level 0 to 2
# outcomes = get_interest_outcome(Y, outcome_level=1)
# outcomes = ['hpv_prevalence']
# heatmap(param_importance, outcomes)

# %%

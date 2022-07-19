'''
Tests for single simulations
'''

#%% Imports and settings
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
 
# Prognoses to fit to
prognoses = dict(
        duration_cutoffs  = np.array([0,       1,          2,          3,          4,          5,          10]),      # Duration cutoffs (lower limits)
        seroconvert_probs = np.array([0.25,    0.5,        0.95,       1.0,        1.0,        1.0,        1.0]),    # Probability of seroconverting given duration of infection
        cin1_probs        = np.array([0.015,   0.3655,     0.86800,    1.0,        1.0,        1.0,        1.0]),   # Conditional probability of developing CIN1 given HPV infection
        cin2_probs        = np.array([0.020,   0.0287,     0.0305,     0.06427,    0.1659,     0.3011,     0.4483]),   # Conditional probability of developing CIN2 given CIN1, derived from Harvard model calibration
        cin3_probs        = np.array([0.007,   0.0097,     0.0102,     0.0219,     0.0586,     0.112,      0.1779]),   # Conditional probability of developing CIN3 given CIN2, derived from Harvard model calibration
        cancer_probs      = np.array([0.002,   0.003,      0.0564,     0.1569,     0.2908,     0.3111,     0.5586]),   # Conditional probability of developing cancer given CIN3, derived from Harvard model calibration
        )

def f2(x, k, x0):
    ''' 2 parameter logistic function '''
    return 1 / (1. + np.exp(-k * (x - x0)))


def f3(x, k, x0, b):
    ''' 3 parameter logistic function '''
    return 1 / (1. + np.exp(-k * (x - x0))) + b


def f4(x, k, x0, a, b):
    ''' 4 parameter logistic function '''
    return a / (1. + np.exp(-k * (x - x0))) + b


#%% Run as a script
if __name__ == '__main__':

    x = prognoses['duration_cutoffs']
    xfull = np.linspace(0,10,101)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    ax = axes.flatten()
    opt_pars = {}

    for ai,var in enumerate(['cin1_probs']):#, 'cin2_probs', 'cin3_probs', 'cancer_probs']):

        y = prognoses[var]
        f = f2
        popt, pcov = opt.curve_fit(f, x, y, method='trf')
        opt_pars[var] = popt
        y_fit = f(x, *popt)
        yfull = f(xfull, *popt)
        ax[ai].plot(x, y, 'ro', label="Values from Jane's model")
        ax[ai].plot(x, y_fit, 'bv', label="Values from logistic function")
        ax[ai].plot(xfull, yfull, 'k-')
        ax[ai].set_title(var)
        ax[ai].set_xlabel("Duration")
        ax[ai].set_ylabel("Probability of progressing")
        if ai==0:
            ax[ai].legend()

    plt.show()

    print('Done.')
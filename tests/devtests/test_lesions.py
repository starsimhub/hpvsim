"""
Experiments with dysplasia models
"""

import numpy as np
import sciris as sc
import os
import sys


if __name__ == '__main__':

    plot_weibull=1
    plot_gompertz_samples=1


    if plot_weibull:
        from scipy.stats import weibull_min
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)
        c = 6.25 # Fixed
        lam_range = np.linspace(4,10,7) # Time to peak
        cols = plt.cm.Oranges(lam_range/10)
        max_x = 15 # For plotting
        x = np.linspace(0, max_x, 100) # x axis

        # Plot dysplastic cell growth for different durations
        for i,lam in enumerate(lam_range):
            rv = weibull_min(c, loc=0, scale=lam)
            sf = (lam/c)**3 # Scale factor to adjust peak height
            ax.plot(x, sf*rv.pdf(x), color=cols[i], lw=2,
                    label=f'Time to peak = {lam} years')

        ax.axhline(y=0.33, ls=':', c='k')
        ax.axhline(y=0.67, ls=':', c='k')
        ax.legend(loc='best', frameon=False)
        ax.set_ylabel('Cell dysplasia proportion')
        ax.set_xlabel('Duration of infection')
        ax.set_ylim([0, 1])
        plt.show()


    if plot_gompertz_samples:
        from scipy.stats import gompertz
        import matplotlib.pyplot as plt
        import sciris as sc

        fig, ax = plt.subplots(1, 1)
        c = 0.01 # Shape parameter - fixed for now
        times_to_peak = np.linspace(1,10,4) # Durations of infection
        n_peak_samples = 10
        cols = sc.gridcolors(4)
        max_x = 15 # For plotting
        x = np.linspace(0, max_x, 100) # x axis

        def mean_peak(x, k=1/3, x0=5, max_x=0.8):
            ''' Logistic relationship between duration and mean peak dysplasia '''
            return max_x/(1+np.exp(-k*(x-x0)))

        def lognormal(par1, par2=0.1, size=1):
            ''' Lognormal distribution for peak dysplasia '''
            mean = np.log(par1**2 / np.sqrt(par2**2 + par1**2))  # Computes the mean of the underlying normal distribution
            sigma = np.sqrt(np.log(par2**2 / par1**2 + 1))  # Computes sigma for the underlying normal distribution
            samples = np.random.lognormal(mean=mean, sigma=sigma, size=size)
            return samples

        # Plot cell dysplasia for different durations
        for i,time in enumerate(times_to_peak):
            scale = time / np.log(1 / c)
            rv = gompertz(c, loc=0, scale=scale)
            default_peak = rv.pdf(time)
            this_mean_peak = mean_peak(time)
            mean_sf = this_mean_peak / default_peak
            ax.plot(x, mean_sf * rv.pdf(x), color=cols[i], lw=2, label=f'Time to peak = {time} years')
            for ps in range(n_peak_samples):
                sf = lognormal(this_mean_peak)/default_peak
                ax.plot(x, sf*rv.pdf(x), color=cols[i], lw=1, alpha=0.5)

        ax.axhline(y=0.33, ls=':', c='k')
        ax.axhline(y=0.67, ls=':', c='k')
        ax.legend(loc='best', frameon=False)
        ax.set_ylabel('Cell dysplasia proportion')
        ax.set_xlabel('Duration of infection')
        ax.set_ylim([0, 1])
        plt.show()

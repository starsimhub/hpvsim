"""
Validate the HIV data files
"""

import numpy as np
import sciris as sc
import hpvsim as hpv
import pylab as pl

T = sc.timer()

location = 'South Africa'
hiv_start = 1985
art_start = 2000
end = 2030
age_min = 0
age_max = 90
sex_ratio = 0.5 # Don't have data on it
hiv_years = sc.inclusiverange(hiv_start, end)
art_years = sc.inclusiverange(art_start, end)
ages = sc.inclusiverange(age_min, age_max, dtype=int)

fn = sc.objdict()
fn.inci = '../test_data/hiv_incidence_south_africa.csv'
fn.data  = '../test_data/RSA_data.csv'
fn.art_m     = '../test_data/south_africa_art_coverage_by_age_males.csv'
fn.art_f     = '../test_data/south_africa_art_coverage_by_age_females.csv'


sc.heading('Loading data...')

# Load age distribution
age_dists = dict()
for year in art_years:
    age_dists[year] = hpv.data.get_age_distribution(location, year=year)

# Load incidence and overall data file
d = sc.objdict()
d.inci = sc.dataframe.read_csv(fn.inci)
d.data = sc.dataframe.read_csv(fn.data).set_index('Unnamed: 0').T

# Load ART
d.art = sc.objdict()
for sex in ['m','f']:
    d.art[sex] = sc.dataframe.read_csv(fn[f'art_{sex}'])
    
    
sc.heading('Computing...')

# Compute ART
res = sc.objdict()
cov = np.zeros(len(art_years))
for y,year in enumerate(art_years):
    total_pop = age_dists[year][:,2].sum()
    age_dist = age_dists[year][ages,2]*sex_ratio
    n_art = 0
    for sex in ['m','f']:
        n_art += (d.art[sex][str(int(year))]*age_dist).sum()
    cov[y] = n_art/total_pop
res.art = sc.dataframe(years=art_years, coverage=cov)

# Compute incidence


sc.heading('Plotting...')

sc.options(dpi=150)
pl.figure(figsize=(18,6))

pl.subplot(1,3,1)
pl.plot(res.art.years.values, res.art.coverage.values, label='Computed by age')
pl.plot(d.data.index.astype(int).values, d.data.art_coverage.values, label='Overall "data"')
pl.legend()
pl.title('ART coverage')

pl.subplot(1,3,1)
        
        

sc.figlayout()
pl.show()


T.toc()
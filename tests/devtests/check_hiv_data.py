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
hiv_years = sc.inclusiverange(hiv_start, end, dtype=int)
art_years = sc.inclusiverange(art_start, end, dtype=int)
ages = sc.inclusiverange(age_min, age_max, dtype=int)

fn = sc.objdict()
fn.inci = '../test_data/hiv_incidence_south_africa.csv'
fn.data  = '../test_data/RSA_data.csv'
fn.art_m     = '../test_data/south_africa_art_coverage_by_age_males.csv'
fn.art_f     = '../test_data/south_africa_art_coverage_by_age_females.csv'


sc.heading('Loading data...')

# Load age distribution
age_dists = dict()
for year in hiv_years:
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
res = sc.objdict()

# Compute ART
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
inc = np.zeros(len(hiv_years))
for y,year in enumerate(hiv_years):
    print(f'Working on {year}...')
    n_inc = 0
    count = 0
    for sex in ['m','f']:
        for age in ages:
            age_match  = np.isclose(d.inci.Age.values, age)
            year_match = np.isclose(d.inci.Year, year)
            sex_match = d.inci.Sex == sex
            matches = age_match * year_match * sex_match
            count += matches.sum()
            this_inci = d.inci[matches].Incidence.sum()
            this_pop = age_dists[year][age,2]*sex_ratio
            n_inc += this_inci*this_pop
    assert count
    inc[y] = n_inc
res.inc = sc.dataframe(years=hiv_years, incidence=inc)

# Compute prevalence
mort = 0.055 # Mortality in the absence of ART (estimate)
prev_zero_mort = np.zeros(len(hiv_years))
prev_with_mort = np.zeros(len(hiv_years))
prev_with_art  = np.zeros(len(hiv_years))
cum_inc = np.cumsum(res.inc.incidence.values)
inci_with_mort = res.inc.incidence.values.copy()
inci_with_art  = res.inc.incidence.values.copy()
for y,year in enumerate(hiv_years):
    total_pop = age_dists[year][:,2].sum()
    prev_zero_mort[y] = cum_inc[y]/total_pop
    
    inci_with_mort[:y] *= 1-mort
    prev_with_mort[y] = inci_with_mort[:y+1].sum()/total_pop
    
    if year < 2000:
        art_cov = 0
    else:
        art_cov = cov[y-(art_start-hiv_start)]
    inci_with_art[:y] *= 1-(mort*(1-art_cov))
    prev_with_art[y] = inci_with_art[:y+1].sum()/total_pop
    
res.prev = sc.dataframe(
    years=hiv_years,
    prev_zero_mort=prev_zero_mort,
    prev_with_mort=prev_with_mort,
    prev_with_art=prev_with_art,
)


#%%
sc.heading('Plotting...')

sc.options(dpi=150)
pl.figure(figsize=(18,6))
kw = dict(lw=3, alpha=0.7)
data_years = d.data.index.astype(int).values

pl.subplot(1,3,1)
pl.plot(res.art.years.values, res.art.coverage.values, label='Computed by age', **kw)
pl.plot(data_years, d.data.art_coverage.values, label='Overall "data"', **kw)
pl.legend()
pl.title('ART coverage')

pl.subplot(1,3,2)
pl.plot(res.inc.years.values, res.inc.incidence.values, label='Computed by age', **kw)
pl.plot(data_years, d.data.hiv_infections.values, label='Overall "data"', **kw)
pl.legend()
pl.title('HIV incidence')

pl.subplot(1,3,3)
pl.plot(res.prev.years.values, res.prev.prev_zero_mort.values, label='Computed by age (with zero mortality)', **kw)
pl.plot(res.prev.years.values, res.prev.prev_with_mort.values, label=f'Computed by age (with {mort*100}% mortality)', **kw)
pl.plot(res.prev.years.values, res.prev.prev_with_art.values, label=f'Computed by age (with ART-adjusted mortality)', **kw)
pl.plot(data_years, d.data.hiv_prevalence.values, label='Overall "data"', **kw)
pl.legend()
pl.title('HIV prevalence')
        
        

sc.figlayout()
pl.show()


T.toc()
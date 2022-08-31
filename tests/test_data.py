'''
Tests for data
'''

#%% Imports and settings
import sciris as sc
import hpvsim as hpv

do_plot = 1
do_save = 0


#%% Define the tests

def test_total_pop():
    sc.heading('Testing total population')

    loc_map = dict(kenya='kenya')
    r = sc.objdict()

    for key,locs in loc_map.items():
        r[key] = hpv.data.get_total_pop(locs)

    return r



#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    r1 = test_total_pop()

    sc.toc(T)
    print('Done.')
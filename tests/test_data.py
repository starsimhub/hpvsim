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

    locs = ['kenya']
    r = sc.objdict()
    s = sc.objdict()

    for loc in locs:
        tp = hpv.data.get_total_pop(loc)
        n_agents = round(tp.pop_size[0]/10e3)
        start = tp.year[0]
        end = 2030
        r[loc] = tp
        s[loc] = hpv.Sim(location=loc, n_agents=n_agents, start=start, end=end).run()

    return r,s



#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    r,s = test_total_pop()

    sc.toc(T)
    print('Done.')
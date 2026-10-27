'''
Profile a typical HPVsim run (hpv.demo(): Nigeria, four genotypes): where the time of each step goes

Pass follow= to sim.profile() to profile a function line by line, e.g. follow=hpv.HPV.set_prognoses.
'''
import hpvsim as hpv

if __name__ == '__main__':
    sim = hpv.demo(run=False, plot=False, n_agents=20e3, ms_agent_ratio=100)
    sim.profile()

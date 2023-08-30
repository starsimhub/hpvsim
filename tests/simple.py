import hpvsim as hpv

sim = hpv.Sim(location='india', start=1960, dt=0.25, end=2020).run()
sim.plot()
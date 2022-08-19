import hpvsim as hpv
import sciris as sc


do_plot = False
storage = "mysql://hpvsim_user@localhost/hpvsim_db"
name = "hpvsim-example-calibration"


def test_calibration():

    sc.heading('Testing calibration')

    pars = dict(n_agents=5e3, start=1980, end=2020, dt=0.5, location='south africa',
                init_hpv_dist=dict(
                    hpv16=0.9,
                    hpv18=0.1
                ))
    sim = hpv.Sim(pars, genotypes=[16,18])
    calib_pars = dict(
        beta=[0.05, 0.010, 0.20],
        hpv_control_prob=[.9, 0.1, 1],
    )
    genotype_pars = dict(
        hpv16=dict(
            dysp_rate=[0.5, 0.2, 1.0],
            prog_rate=[0.5, 0.2, 1.0],
            dur_none = dict(par1=[1.0, 0.5, 2.5])
        ),
        hpv18=dict(
            dysp_rate=[0.5, 0.2, 1.0],
            prog_rate=[0.5, 0.2, 1.0],
            dur_none=dict(par1=[1.0, 0.5, 2.5])
        )
    )

    calib = hpv.Calibration(sim, calib_pars=calib_pars, genotype_pars=genotype_pars,
                            datafiles=[
                                '../test_data/south_africa_hpv_data.csv',
                                '../test_data/south_africa_cancer_data_2020.csv',
                                # 'test_data/south_africa_type_distribution_cancer.csv'
                            ],
                            storage=storage,
                            name=name,
                            total_trials=2, n_workers=1)
    calib.calibrate()
    if do_plot:
        calib.plot(top_results=4)
    return sim, calib



if __name__ == "__main__":
    sim,calib = test_calibration()
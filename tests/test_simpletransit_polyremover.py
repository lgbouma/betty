"""
Fit Gaia DR2 1833519030401513984, aka TIC 117689799, with a simpletransit
model, after first removing local polynomials from each transit window.
"""

import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import pickle, os, corner, pytest
from collections import OrderedDict

from os.path import join
from importlib.machinery import SourceFileLoader

from astrobase.lcmath import find_lc_timegroups

from betty.helpers import (
    get_tic117689799_lightcurve, _subset_cut, _quicklcplot
)
from betty.posterior_table import make_posterior_table
from betty.modelfitter import ModelFitter
from betty.paths import TESTDATADIR, TESTRESULTSDIR, BETTYDIR

EPHEMDICT = {
    'TIC_117689799': {'t0': 2458684.712181, 'per': 2.157913, 'tdur':2/24},
}

@pytest.mark.skip(reason="PyMC3 sampling too cray for Github Actions.")
def test_localpolytransit(starid='TIC_117689799', N_samples=1000):

    modelid = 'simpletransit'

    # load + visualize
    if starid == 'TIC_117689799':
        time, flux, flux_err, tess_texp = get_tic117689799_lightcurve()
    else:
        raise NotImplementedError

    from cdips.lcproc.detrend import transit_window_polynomial_remover
    outpath = join(TESTRESULTSDIR, f'{starid}.png')
    d = transit_window_polynomial_remover(
        time, flux, flux_err, EPHEMDICT[starid]['t0'],
        EPHEMDICT[starid]['per'], EPHEMDICT[starid]['tdur'], n_tdurs=5.,
        method='poly_2', plot_outpath=outpath
    )

    datasets = OrderedDict()

    time = np.hstack([d[f'time_{ix}'] for ix in range(d['ngroups'])])
    flux = np.hstack([d[f'flat_flux_{ix}'] for ix in range(d['ngroups'])])
    flux_err = np.hstack([d[f'flux_err_{ix}'] for ix in range(d['ngroups'])])
    texp = np.nanmedian(np.diff(time))

    datasets['tess'] = [time, flux, flux_err, tess_texp]

    priorpath = join(
        TESTDATADIR,
        'gaiatwo0001833519030401513984-0014-cam1-ccd4_tess_v01_llc_simpletransit_priors.py'
    )
    if not os.path.exists(priorpath):
        raise FileNotFoundError(f'need to create {priorpath}')
    priormod = SourceFileLoader('prior', priorpath).load_module()
    priordict = priormod.priordict

    pklpath = join(BETTYDIR, f'test_{starid}_{modelid}.pkl')

    m = ModelFitter(modelid, datasets, priordict, plotdir=TESTRESULTSDIR,
                    pklpath=pklpath, overwrite=0, N_samples=N_samples,
                    N_cores=os.cpu_count())

    print(pm.summary(m.trace, var_names=list(priordict)))

    summdf = pm.summary(m.trace, var_names=list(priordict), round_to=10,
                        kind='stats', stat_funcs={'median':np.nanmedian},
                        extend=True)

    fitindiv = 1
    phaseplot = 1
    cornerplot = 1
    posttable = 1
    localpolyindivpanels = 1

    PLOTDIR = TESTRESULTSDIR

    import betty.plotting as bp

    if localpolyindivpanels:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_localpolyindivpanels.png')
        bp.plot_localpolyindivpanels(d, m, summdf, outpath, modelid=modelid)

    if phaseplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_phaseplot.png')
        bp.plot_phasefold(m, summdf, outpath, modelid=modelid, inppt=1,
                          binsize_minutes=20)

    if posttable:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_posteriortable.tex')
        make_posterior_table(pklpath, priordict, outpath, modelid, makepdf=1)

    if fitindiv:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_fitindiv.png')
        bp.plot_fitindiv(m, summdf, outpath, modelid=modelid)

    if cornerplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_cornerplot.png')
        bp.plot_cornerplot(list(priordict), m, outpath)


if __name__ == "__main__":
    test_localpolytransit(starid='TIC_117689799', N_samples=1000)

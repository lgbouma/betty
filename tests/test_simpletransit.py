"""
In this test, the TESS WASP-4 data are fitted simultaneously for {"period",
"t0", "r", "b", "u0", "u1"}.
"""

import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import pickle, os, corner, pytest
from collections import OrderedDict
from pymc3.backends.tracetab import trace_to_dataframe
import exoplanet as xo

from os.path import join
from importlib.machinery import SourceFileLoader

import betty.plotting as bp

from betty.helpers import get_wasp4_lightcurve, _subset_cut
from betty.modelfitter import ModelFitter

from betty.paths import TESTDATADIR, TESTRESULTSDIR, BETTYDIR

@pytest.mark.skip(reason="PyMC3 sampling too cray for Github Actions.")
def test_simpletransit():

    starid = 'WASP_4'
    modelid = 'simpletransit'

    datasets = OrderedDict()
    time, flux, flux_err, tess_texp = get_wasp4_lightcurve()
    time, flux, flux_err = _subset_cut(time, flux, flux_err, n=2.0)
    datasets['tess'] = [time, flux, flux_err, tess_texp]

    priorpath = join(TESTDATADIR, f'{starid}_priors.py')
    priormod = SourceFileLoader('prior', priorpath).load_module()
    priordict = priormod.priordict

    pklpath = join(BETTYDIR, f'test_{starid}_{modelid}.pkl')

    m = ModelFitter(modelid, datasets, priordict, plotdir=TESTRESULTSDIR,
                    pklpath=pklpath, overwrite=0, N_samples=1000,
                    N_cores=os.cpu_count(), target_accept=0.8)

    print(pm.summary(m.trace, var_names=list(priordict)))

    summdf = pm.summary(m.trace, var_names=list(priordict), round_to=10,
                        kind='stats', stat_funcs={'median':np.nanmedian},
                        extend=True)

    # require: precise
    params = ['t0', 'period']
    for _p in params:
        assert summdf.T[_p].loc['sd'] * 10 < priordict[_p][2]

    # require: priors were accurate
    for _p in params:
        absdiff = np.abs(summdf.T[_p].loc['mean'] - priordict[_p][1])
        priorwidth = priordict[_p][2]
        assert absdiff < priorwidth

    fitindiv = 0
    phaseplot = 0
    cornerplot = 1

    PLOTDIR = TESTRESULTSDIR
    if phaseplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_phaseplot.png')
        bp.plot_phasefold(m, summdf, outpath, modelid=modelid, inppt=1)

    if fitindiv:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_fitindiv.png')
        bp.plot_fitindiv(m, summdf, outpath, modelid=modelid)

    if cornerplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_cornerplot.png')
        bp.plot_cornerplot(list(priordict), m, outpath)


if __name__ == "__main__":
    test_simpletransit()

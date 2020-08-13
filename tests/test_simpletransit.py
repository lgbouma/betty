"""
In this test, the TESS WASP-4 data are fitted simultaneously for {"period",
"t0", "r", "b", "u0", "u1"}.
"""

import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import pickle, os, corner
from collections import OrderedDict
from pymc3.backends.tracetab import trace_to_dataframe
import exoplanet as xo

from os import join

from betty.helpers import get_wasp4_lightcurve
from betty.modelfitter import ModelFitter

from betty.paths import TESTDATADIR, TESTRESULTSDIR, BETTYDIR

@pytest.mark.skip(reason="[WIP]")
def test_simpletransit():

    time, flux, flux_err, tess_texp = get_wasp4_lightcurve()

    datasets = OrderedDict()
    datasets['tess'] = [time, flux, flux_err, tess_texp]

    # dict with teff, logg, and uncertainties on each. keys matter.
    stardict = pd.read_csv(join(TESTDATADIR, 'wasp4_starinfo.csv')).to_dict()
    starid = stardict['starid']

    modelid = 'simpletransit'

    pklpath = join(BETTYDIR, f'test_{starid}_{modelid}.pkl')

    m = ModelFitter(modelid, datasets, stardict, plotdir=TESTRESULTSDIR,
                    pklpath=pklpath, overwrite=0, N_samples=1000,
                    target_accept=0.8)

    import IPython; IPython.embed()

    #FIXME: prior_d
    summdf = pm.summary(m.trace, var_names=list(prior_d.keys()), round_to=10,
                        kind='stats', stat_funcs={'median':np.nanmedian},
                        extend=True)

    outpath = join(PLOTDIR, f'{REALID}_{modelid}_fitindiv.png')
    tp.plot_fitindiv(m, summdf, outpath, modelid=modelid)

    outpath = join(PLOTDIR, f'{REALID}_{modelid}_phaseplot.png')
    # NOTE: need to figure out parameter management
    tp.plot_phasefold(m, summdf, outpath, modelid=modelid, inppt=1)

    outpath = join(PLOTDIR, f'{REALID}_{modelid}_subsetcorner.png')
    tp.plot_subsetcorner(m, outpath)

    outpath = join(PLOTDIR, f'{REALID}_{modelid}_cornerplot.png')
    # NOTE: need to figure out parameter management
    tp.plot_cornerplot(m, outpath)


if __name__ == "__main__":
    test_simpletransit()

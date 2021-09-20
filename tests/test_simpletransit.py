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

try:
    import betty.plotting as bp
except ModuleNotFoundError as e:
    print(f'WRN! {e}')
    pass

from betty.helpers import (
    get_wasp4_lightcurve, _subset_cut, retrieve_tess_lcdata
)
from betty.posterior_table import make_posterior_table
from betty.modelfitter import ModelFitter

from betty.paths import TESTDATADIR, TESTRESULTSDIR, BETTYDIR

from astrobase.services.identifiers import (
    simbad_to_tic
)
from astrobase.services.tesslightcurves import (
    get_two_minute_spoc_lightcurves
)

EPHEMDICT = {
    'WASP_4': {'t0': 1355.1845, 'per': 1.338231466, 'tdur':2.5/24},
    'HAT-P-14': {'t0': 1984.6530, 'per': 4.62787, 'tdur':2.5/24}
}


@pytest.mark.skip(reason="PyMC3 sampling too cray for Github Actions.")
def test_simpletransit(starid='WASP_4', N_samples=1000):

    modelid = 'simpletransit'

    datasets = OrderedDict()
    if starid == 'WASP_4':
        time, flux, flux_err, tess_texp = get_wasp4_lightcurve()
    else:
        ticid = simbad_to_tic(starid)
        lcfiles = (
            get_two_minute_spoc_lightcurves(ticid, download_dir=TESTDATADIR)
        )
        d = retrieve_tess_lcdata(
            lcfiles, provenance='spoc', merge_sectors=1, simple_clean=1
        )
        time, flux, flux_err, _, tess_texp = (
            d['time'], d['flux'], d['flux_err'], d['qual'], d['texp']
        )

    time, flux, flux_err = _subset_cut(
        time, flux, flux_err, n=2.0, t0=EPHEMDICT[starid]['t0'],
        per=EPHEMDICT[starid]['per'], tdur=EPHEMDICT[starid]['tdur']
    )

    datasets['tess'] = [time, flux, flux_err, tess_texp]

    priorpath = join(TESTDATADIR, f'{starid}_priors.py')
    if not os.path.exists(priorpath):
        # TODO: auto-get star info. follow
        # tessttvfinder.src.measure_transit_times_from_lightcurve by
        # querying nasa exoplanet archive.
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

    # require: precise
    params = ['t0', 'period']
    for _p in params:
        assert summdf.T[_p].loc['sd'] * 10 < priordict[_p][2]

    # require: priors were accurate
    for _p in params:
        absdiff = np.abs(summdf.T[_p].loc['mean'] - priordict[_p][1])
        priorwidth = priordict[_p][2]
        assert absdiff < priorwidth

    fitindiv = 1
    phaseplot = 1
    cornerplot = 1
    posttable = 1

    PLOTDIR = TESTRESULTSDIR

    if posttable:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_posteriortable.tex')
        make_posterior_table(pklpath, priordict, outpath, modelid, makepdf=1)

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
    test_simpletransit(starid='WASP_4', N_samples=1000)
    #test_simpletransit(starid='HAT-P-14', N_samples=5000)

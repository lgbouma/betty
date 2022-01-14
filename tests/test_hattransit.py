"""
In this test, the HAT TOI-2337 data are fitted simultaneously for {"period",
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
    _subset_cut, retrieve_tess_lcdata
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
    'TOI_2337': {'t0': 56197.75437, 'per': 2.995240, 'tdur':5/24}
}


@pytest.mark.skip(reason="PyMC3 sampling too cray for Github Actions.")
def test_simpletransit(starid='TOI_2337', N_samples=1000):

    modelid = 'simpletransit'

    datasets = OrderedDict()
    if starid == 'TOI_2337':
        csvpath = os.path.join(TESTDATADIR, 'toi-2337-hatlc.csv')
        df = pd.read_csv(csvpath)
        time, mag, mag_err = (
            np.array(df['time']), np.array(df['tfamag']), np.array(df['err'])
        )
        from betty.helpers import _given_mag_get_flux
        flux, flux_err = _given_mag_get_flux(mag, err_mag=mag_err)
        texp = np.nanmedian(np.diff(time))

    datasets['hatnet'] = [time, flux, flux_err, texp]

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

    fitindiv = 1
    phaseplot = 1
    cornerplot = 1
    posttable = 0

    PLOTDIR = TESTRESULTSDIR

    if posttable:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_posteriortable.tex')
        make_posterior_table(pklpath, priordict, outpath, modelid, makepdf=1)

    if phaseplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_phaseplot.png')
        bp.plot_phasefold(m, summdf, outpath, modelid=modelid, inppt=1,
                          ylim=(-4,4))

    if fitindiv:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_fitindiv.png')
        bp.plot_fitindiv(m, summdf, outpath, modelid=modelid)

    if cornerplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_cornerplot.png')
        bp.plot_cornerplot(list(priordict), m, outpath)


if __name__ == "__main__":
    test_simpletransit(starid='TOI_2337', N_samples=3000)

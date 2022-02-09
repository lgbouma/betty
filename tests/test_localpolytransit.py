"""
Fit Gaia DR2 1833519030401513984, aka TIC 117689799, for a local polynomial +
transit model.
"""

import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import pickle, os, corner, pytest
from collections import OrderedDict
from pymc3.backends.tracetab import trace_to_dataframe
import exoplanet as xo

from copy import deepcopy
from os.path import join
from importlib.machinery import SourceFileLoader

from astrobase.lcmath import find_lc_timegroups

from betty.helpers import (
    get_tic117689799_lightcurve, _subset_cut
)
from betty.posterior_table import make_posterior_table
from betty.modelfitter import ModelFitter
import betty.plotting as bp
from betty.paths import TESTDATADIR, TESTRESULTSDIR, BETTYDIR

EPHEMDICT = {
    'WASP_4': {'t0': 1355.1845, 'per': 1.338231466, 'tdur':2.5/24},
    'HAT-P-14': {'t0': 1984.6530, 'per': 4.62787, 'tdur':2.5/24},
    'TIC_117689799': {'t0': 2458684.712181, 'per': 2.157913, 'tdur':1/24},
}

def _quicklcplot(time, flux, flux_err, outpath):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.errorbar(time, flux, yerr=flux_err, fmt='none', ecolor='k',
                elinewidth=0.5, capsize=2, mew=0.5)
    fig.savefig(outpath, dpi=400, bbox_inches='tight')
    print(f"Wrote {outpath}")


@pytest.mark.skip(reason="PyMC3 sampling too cray for Github Actions.")
def test_localpolytransit(starid='TIC_117689799', N_samples=1000):

    modelid = 'localpolytransit'

    datasets = OrderedDict()

    # load + visualize
    if starid == 'TIC_117689799':
        time, flux, flux_err, tess_texp = get_tic117689799_lightcurve()
    else:
        raise NotImplementedError
    outpath = join(TESTRESULTSDIR, f'{starid}_rawlc.png')
    _quicklcplot(time, flux, flux_err, outpath)

    # trim
    n_tdurs = 5.0
    time, flux, flux_err = _subset_cut(
        time, flux, flux_err, n=n_tdurs, t0=EPHEMDICT[starid]['t0'],
        per=EPHEMDICT[starid]['per'], tdur=EPHEMDICT[starid]['tdur']
    )
    outpath = join(TESTRESULTSDIR, f'{starid}_rawtrimlc.png')
    _quicklcplot(time, flux, flux_err, outpath)

    # given n*tdur omitted on each side of either transit, there is P-2*ntdur
    # space between each time group.
    mingap = EPHEMDICT[starid]['per'] - 3*n_tdurs*EPHEMDICT[starid]['tdur']
    assert mingap > 0
    ngroups, groupinds = find_lc_timegroups(time, mingap=mingap)

    for ix, g in enumerate(groupinds):
        tess_texp = np.nanmedian(np.diff(time[g]))
        datasets[f'tess_{ix}'] = [time[g], flux[g], flux_err[g], tess_texp]

    # load + append to prior
    priorpath = join(
        TESTDATADIR,
        'gaiatwo0001833519030401513984-0014-cam1-ccd4_tess_v01_llc_localpolytransit_priors.py'
    )
    if not os.path.exists(priorpath):
        raise FileNotFoundError(f'need to create {priorpath}')
    priormod = SourceFileLoader('prior', priorpath).load_module()
    _init_priordict = priormod.priordict
    priordict = deepcopy(_init_priordict)

    if modelid == 'localpolytransit' and 'tess_0_mean' not in priordict.keys():
        for n, (name, (x, y, yerr, texp)) in enumerate(datasets.items()):
            # mean + a1*(time-midtime) + a2*(time-midtime)^2.
            priordict[f'{name}_mean'] = ('Normal', 1, 0.1)
            priordict[f'{name}_a1'] = ('Uniform', -0.1, 0.1)
            priordict[f'{name}_a2'] = ('Uniform', -0.1, 0.1)

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

    PLOTDIR = TESTRESULTSDIR

    if posttable:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_trimmed_posteriortable.tex')
        make_posterior_table(pklpath, _init_priordict, outpath, modelid, makepdf=1)

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
        outpath = join(PLOTDIR, f'{starid}_{modelid}_trimmed_cornerplot.png')
        bp.plot_cornerplot(list(_init_priordict), m, outpath)



if __name__ == "__main__":
    test_localpolytransit(starid='TIC_117689799', N_samples=1000)

import os
import numpy as np, pandas as pd

from astropy.io import fits

from betty.paths import TESTDATADIR

from astrobase.services.identifiers import simbad_to_tic
from astrobase.services.tesslightcurves import get_two_minute_spoc_lightcurves


def get_model_transit(paramd, time_eval, t_exp=2/(60*24)):
    """
    you know the paramters, and just want to evaluate the median lightcurve.
    """
    import exoplanet as xo

    period = paramd['period']
    t0 = paramd['t0']
    try:
        r = paramd['r']
    except KeyError:
        r = np.exp(paramd['log_r'])

    b = paramd['b']
    u0 = paramd['u[0]']
    u1 = paramd['u[1]']

    r_star = paramd['r_star']
    logg_star = paramd['logg_star']

    try:
        mean = paramd['mean']
    except KeyError:
        mean_key = [k for k in list(paramd.keys()) if 'mean' in k]
        assert len(mean_key) == 1
        mean_key = mean_key[0]
        mean = paramd[mean_key]

    # factor * 10**logg / r_star = rho
    factor = 5.141596357654149e-05

    rho_star = factor*10**logg_star / r_star

    orbit = xo.orbits.KeplerianOrbit(
        period=period, t0=t0, b=b, rho_star=rho_star
    )

    u = [u0, u1]

    mu_transit = xo.LimbDarkLightCurve(u).get_light_curve(
            orbit=orbit, r=r, t=time_eval, texp=t_exp
    ).T.flatten()

    return mu_transit.eval() + mean


def _get_fitted_data_dict(m, summdf):

    instr = 'tess'
    _m = m.data[instr]

    d = {
        'x_obs': _m[0],
        'y_obs': _m[1],
        'y_orb': None, # NOTE: "detrended" beforehand
        'y_resid': None, # for now.
        'y_mod': None, # for now [could be MAP, if MAP were good]
        'y_err': _m[2]
    }

    params = ['period', 't0', 'log_r', 'b', 'u[0]', 'u[1]', f'{instr}_mean',
              'r_star', 'logg_star']

    paramd = {k:summdf.loc[k, 'median'] for k in params}
    y_mod_median = get_model_transit(paramd, d['x_obs'])
    d['y_mod'] = y_mod_median
    d['y_resid'] = d['y_obs']-y_mod_median

    return d, params, paramd


def _subset_cut(x_obs, y_obs, y_err, n=12, onlyodd=False, onlyeven=False):
    """
    n: [ t0 - n*tdur, t + n*tdur ]
    """

    t0 = 1355.1845
    per = 1.338231466
    tdur = 2.5/24 # roughly
    epochs = np.arange(-100,100,1)
    mid_times = t0 + per*epochs

    sel = np.zeros_like(x_obs).astype(bool)
    for tra_ind, mid_time in zip(epochs, mid_times):
        if onlyeven:
            if tra_ind % 2 != 0:
                continue
        if onlyodd:
            if tra_ind % 2 != 1:
                continue

        start_time = mid_time - n*tdur
        end_time = mid_time + n*tdur
        s = (x_obs > start_time) & (x_obs < end_time)
        sel |= s

    print(42*'#')
    print(f'Before subset cut: {len(x_obs)} observations.')
    print(f'After subset cut: {len(x_obs[sel])} observations.')
    print(42*'#')

    x_obs = x_obs[sel]
    y_obs = y_obs[sel]
    y_err = y_err[sel]

    return x_obs, y_obs, y_err




def get_wasp4_lightcurve():

    lcfile = os.path.join(TESTDATADIR,
                          'mastDownload/TESS/tess2018234235059-s0002-0000000402026209-0121-s',
                          'tess2018234235059-s0002-0000000402026209-0121-s_lc.fits')

    if not os.path.exists(lcfile):
        ticid = simbad_to_tic('WASP 4')
        lcfiles = get_two_minute_spoc_lightcurves(ticid, download_dir=TESTDATADIR)
        lcfile = lcfiles[0]

    hdul = fits.open(lcfile)
    d = hdul[1].data

    yval = 'PDCSAP_FLUX'
    time = d['TIME']
    _f, _f_err = d[yval], d[yval+'_ERR']
    flux = _f/np.nanmedian(_f)
    flux_err = _f_err/np.nanmedian(_f)
    qual = d['QUALITY']

    sel = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)

    tess_texp = np.nanmedian(np.diff(time[sel]))

    return (
        time[sel].astype(np.float64),
        flux[sel].astype(np.float64),
        flux_err[sel].astype(np.float64),
        tess_texp
    )

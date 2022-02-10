"""
Contents:

    flatten: a function to flatten lists of lists.
    retrieve_tess_lcdata: given lcfiles (spoc/cdips) get dict of time/flux/err.

    _subset_cut: slice/trim LC to transit windows.

    get_model_transit: given parameters, evaluate LC model.
    _get_fitted_data_dict_simpletransit: get params and model LCs from
        ModelFitter object.
    _get_fitted_data_dict_localpolytransit: ditto.
    _get_fitted_data_dict_alltransit: ditto, for "alltransit" model.

    _estimate_mode: get mode of unimodal pdf given samples, via gaussian KDE.

    get_wasp4_lightcurve: used for module testing.
    get_tic117689799_lightcurve: ditto

    _given_mag_get_flux
"""
import os, collections
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import OrderedDict

from astropy.io import fits

from betty.paths import TESTDATADIR

from astrobase.services.identifiers import simbad_to_tic
from astrobase.services.tesslightcurves import get_two_minute_spoc_lightcurves

import exoplanet as xo

# sensible default keys to use for light curve retrieval
LCKEYDICT = {
    'spoc': {'time': 'TIME',
             'flux': 'PDCSAP_FLUX',
             'flux_err': 'PDCSAP_FLUX_ERR',
             'qual': 'QUALITY'},
    'cdips': {'time': 'TMID_BJD',
              'flux': 'IRM1',
              'flux_err': 'IRE1',
              'qual': 'IRQ1'}
}

def flatten(l):
    for el in l:
        if (
            isinstance(el, collections.Iterable) and
            not isinstance(el, (str, bytes))
        ):
            yield from flatten(el)
        else:
            yield el


def retrieve_tess_lcdata(lcfiles, provenance=None, merge_sectors=1,
                         simple_clean=1):
    """
    retrieve_tess_lcdata: Given a list of files pointing to TESS light-curves
    (single-sector, or multi-sector), collect them, and return requested data
    keys.

    Kwargs:
        provenance: string. Can be 'spoc' or 'cdips'.

        merge_sectors: bool/int. Whether or not to merge multisectors. If false
        and multiple lcfiles are passed, raises error.

        simple_clean: whether to apply quick-hack cleaning diagnostics (e.g.,
        "all non-zero quality flags removed").

    Returns:

        lcdict: has keys
            `d['time'], d['flux'], d['flux_err'], d['qual'], d['texp']`
    """

    allowed_provenances = ['spoc', 'cdips']
    if provenance not in allowed_provenances:
        raise NotImplementedError

    if len(lcfiles) > 1 and not merge_sectors:
        raise NotImplementedError

    # retrieve time, flux, flux_err, and quality flags from fits files
    getkeydict = LCKEYDICT[provenance]
    getkeys = list(getkeydict.values())

    d = {}
    for l in lcfiles:
        d[l] = {}
        hdul = fits.open(l)
        for k,v in getkeydict.items():
            d[l][k] = hdul[1].data[v]
        hdul.close()

    # merge across sectors
    _d = {}

    for k,v in getkeydict.items():

        vec = np.hstack([
            d[l][k] for l in lcfiles
        ])

        if k == 'flux' or k == 'flux_err':
            # stitch absolute flux levels by median-dividing (before cleaning,
            # which will remove nans anyway)
            vec = np.hstack([
                d[l][k]/np.nanmedian(d[l]['flux']) for l in lcfiles
            ])

        _d[k] = vec

    if not simple_clean:
        raise NotImplementedError(
            'you probably want to apply simple_clean '
            'else you will have NaNs and bad quality flags.'
        )

    sel = (
        np.isfinite(_d['time'])
        &
        np.isfinite(_d['flux'])
        &
        np.isfinite(_d['flux_err'])
        &
        np.isfinite(_d['qual'])
    )

    if provenance == 'spoc':
        sel &= (_d['qual'] == 0)

    elif provenance == 'cdips':
        raise NotImplementedError('need to apply cdips quality flags')

    _f = _d['flux'][sel]
    outd = {
        'time': _d['time'][sel],
        'flux': _f/np.nanmedian(_f),
        'flux_err': _d['flux_err'][sel]/np.nanmedian(_f),
        'qual': _d['qual'][sel],
        'texp': np.nanmedian(np.diff(_d['time'][sel]))
    }

    return outd


def get_model_transit(paramd, time_eval, t_exp=2/(60*24)):
    """
    you know the paramters, and just want to evaluate the median lightcurve.
    """
    period = paramd['period']
    t0 = paramd['t0']

    r_star = paramd['r_star']
    logg_star = paramd['logg_star']

    try:
        ror = paramd['ror']
        r_pl = ror# * r_star
    except KeyError:
        log_ror = paramd['log_ror']
        r_pl = np.exp(log_ror)# * r_star

    b = paramd['b']
    ukey = 'u' if 'u[0]' in paramd.keys() else 'u_star'
    u0 = paramd[f'{ukey}[0]']
    u1 = paramd[f'{ukey}[1]']

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
            orbit=orbit, r=r_pl, t=time_eval, texp=t_exp
    ).T.flatten()

    return mu_transit.eval() + mean


def get_model_transit_quad(paramd, time_eval, _tmid, t_exp=2/(60*24),
                           includemean=1):
    """
    Same as get_model_transit, but for a transit + quadratic trend. The midtime
    of the trend must be the same as used in betty.modelfitter for the a1 and
    a2 coefficients to be correctly defined.
    """
    period = paramd['period']
    t0 = paramd['t0']

    r_star = paramd['r_star']
    logg_star = paramd['logg_star']

    try:
        ror = paramd['ror']
        r_pl = ror# * r_star
    except KeyError:
        ror = np.exp(paramd['log_ror'])
        r_pl = ror# * r_star

    b = paramd['b']
    u0 = paramd['u_star[0]']
    u1 = paramd['u_star[1]']

    try:
        mean = paramd['mean']
    except KeyError:
        mean_key = [k for k in list(paramd.keys()) if 'mean' in k]
        assert len(mean_key) == 1
        mean_key = mean_key[0]
        mean = paramd[mean_key]

    a1 = paramd[ [k for k in list(paramd.keys()) if '_a1' in k][0] ]
    a2 = paramd[ [k for k in list(paramd.keys()) if '_a2' in k][0] ]

    # factor * 10**logg / r_star = rho
    factor = 5.141596357654149e-05

    rho_star = factor*10**logg_star / r_star

    orbit = xo.orbits.KeplerianOrbit(
        period=period, t0=t0, b=b, rho_star=rho_star
    )

    u = [u0, u1]

    mu_transit = xo.LimbDarkLightCurve(u).get_light_curve(
            orbit=orbit, r=r_pl, t=time_eval, texp=t_exp
    ).T.flatten()

    mu_model = (
        mean +
        a1*(time_eval-_tmid) +
        a2*(time_eval-_tmid)**2 +
        mu_transit.eval()
    )

    if includemean:
        mu_trend = (
            mean +
            a1*(time_eval-_tmid) +
            a2*(time_eval-_tmid)**2
        )
    else:
        mu_trend = (
            a1*(time_eval-_tmid) +
            a2*(time_eval-_tmid)**2
        )

    return mu_model, mu_trend


def _quicklcplot(time, flux, outpath):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.scatter(time, flux, c='k', s=1)
    fig.savefig(outpath, dpi=400, bbox_inches='tight')
    print(f"Wrote {outpath}")



def _get_fitted_data_dict_simpletransit(m, summdf, N_model_times=int(1e3)):

    instrkeys = [k for k in m.priordict.keys() if '_mean' in k]
    if len(instrkeys) > 1:
        msg = 'Expected 1 instrument for this fit.'
        raise NotImplementedError(msg)
    instr = instrkeys[0].split('_')[0]
    _m = m.data[instr]

    d = {
        'x_obs': _m[0],
        'y_obs': _m[1],
        'y_orb': None, # NOTE: "detrended" beforehand
        'y_resid': None, # for now.
        'y_mod': None, # for now [could be MAP, if MAP were good]
        'y_err': _m[2]
    }

    ukey = 'u' if 'u[0]' in summdf else 'u_star'
    params = ['period', 't0', 'log_ror', 'b', f'{ukey}[0]', f'{ukey}[1]',
              f'{instr}_mean', 'r_star', 'logg_star']

    paramd = {k:summdf.loc[k, 'median'] for k in params}

    x_mod = np.linspace(
        d['x_obs'].min(), d['x_obs'].max(), N_model_times
    )
    d['x_mod'] = x_mod

    y_mod_median = get_model_transit(
        paramd, d['x_mod'], t_exp=np.nanmedian(np.diff(d['x_obs']))
    )
    #_quicklcplot(d['x_mod'], y_mod_median, 'temp.png')

    y_mod_median_at_xobs = get_model_transit(
        paramd, d['x_obs'], t_exp=np.nanmedian(np.diff(d['x_obs']))
    )
    d['y_mod'] = y_mod_median
    d['y_resid'] = d['y_obs']-y_mod_median_at_xobs

    assert len(d['x_mod']) == len(d['y_mod'])

    return d, params, paramd


def _get_fitted_data_dict_localpolytransit(
    m, summdf, bestfitmeans='median', singleinstrument='tess',
    N_model_times=int(1e3)
):
    """
    args:
        bestfitmeans: "map", "median", "mean, "mode"; depending on which you
        think will produce the better fitting model.

        singleinstrument: string 'kepler', 'tess', 'k2'
    """

    d = OrderedDict()

    instname = list(m.data.keys())[0].split('_')[0]

    for name in m.data.keys():

        d[name] = {}
        d[name]['x_obs'] = m.data[name][0]
        d[name]['y_obs'] = m.data[name][1]
        d[name]['y_err'] = m.data[name][2]

        params = ['period', 't0', 'log_ror', 'b', 'u_star[0]', 'u_star[1]',
                  'r_star', 'logg_star', f'{name}_mean', f'{name}_a1',
                  f'{name}_a2']

        _tmid = np.nanmedian(m.data[name][0])
        t_exp = np.nanmedian(np.diff(m.data[name][0]))

        if bestfitmeans == 'mode':
            paramd = {k: _estimate_mode(m.trace[k]) for k in params}
        elif bestfitmeans == 'median':
            paramd = {k : summdf.loc[k, 'median'] for k in params}
        elif bestfitmeans == 'mean':
            paramd = {k : summdf.loc[k, 'mean'] for k in params}
        elif bestfitmeans == 'map':
            paramd = {}
            for k in params:
                if 'u_star' not in k:
                    paramd[k] = m.map_estimate[k]
                else:
                    paramd[k] = (
                        m.map_estimate['u_star'][0] if k=='u_star[0]' else
                        m.map_estimate['u_star'][1]
                    )
        else:
            raise NotImplementedError

        x_mod = np.linspace(
            d[name]['x_obs'].min(), d[name]['x_obs'].max(), N_model_times
        )

        y_mod_median, y_mod_median_trend = (
            get_model_transit_quad(paramd, x_mod, _tmid,
                                   t_exp=t_exp, includemean=1)
        )
        y_mod_median_at_xobs, y_mod_median_trend_at_xobs = (
            get_model_transit_quad(paramd, d[name]['x_obs'], _tmid,
                                   t_exp=t_exp, includemean=1)
        )

        d[name]['x_mod'] = x_mod

        # this is used for phase-folded data, with the local trend removed.
        d[name]['y_mod'] = y_mod_median
        d[name]['y_mod_notrend'] = y_mod_median - y_mod_median_trend

        # NOTE: for this case, the "residual" of the observation minus the
        # quadratic trend is actually the "observation" (containing only the
        # transit, with the rotation signal removed)
        d[name]['y_obs_notrend'] = m.data[name][1] - y_mod_median_trend_at_xobs

        # the "full residual" is of the observation minus the quadratic trend
        # minus the transit model minus the transit
        d[name]['y_resid'] = m.data[name][1] - y_mod_median_at_xobs

        d[name]['params'] = params

        assert len(d[name]['x_mod']) == len(d[name]['y_mod'])

    # merge all the singleinstrument transits:
    # access time via e.g., d['tess']['x_obs']
    n_singleinstrument = len([k for k in d.keys() if singleinstrument in k])
    d[singleinstrument] = {}
    d['all'] = {}
    _p = ['x_obs', 'y_obs', 'y_err', 'y_resid', 'y_mod', 'x_mod',
          'y_obs_notrend', 'y_mod_notrend']
    for p in _p:
        d[singleinstrument][p] = np.hstack(
            [d[f'{singleinstrument}_{ix}'][p] for ix in
             range(n_singleinstrument)]
        )
    for p in _p:
        d['all'][p] = np.hstack(
            [d[f'{k}'][p] for k in d.keys() if '_' in k]
        )

    return d


def _get_fitted_data_dict_alltransit(m, summdf):

    d = OrderedDict()

    for name in m.data.keys():

        d[name] = {}
        d[name]['x_obs'] = m.data[name][0]
        d[name]['y_obs'] = m.data[name][1]
        d[name]['y_err'] = m.data[name][2]

        params = ['period', 't0', 'log_ror', 'b', 'u[0]', 'u[1]', 'r_star',
                  'logg_star', f'{name}_mean']

        paramd = {k : summdf.loc[k, 'median'] for k in params}
        y_mod_median = get_model_transit(
            paramd, d[name]['x_obs'],
            t_exp=np.nanmedian(np.diff(d[name]['x_obs']))
        )

        d[name]['y_mod'] = y_mod_median
        d[name]['y_resid'] = d[name]['y_obs'] - y_mod_median
        d[name]['params'] = params

    # merge all the available LC data
    d['all'] = {}
    _p = ['x_obs', 'y_obs', 'y_err', 'y_mod']
    for p in _p:
        d['all'][p] = np.hstack([d[f'{k}'][p] for k in d.keys()
                                 if '_' in k or 'tess' in k])

    return d


def _get_fitted_data_dict_allindivtransit(m, summdf, bestfitmeans='median'):
    """
    args:
        bestfitmeans: "map", "median", "mean, "mode"; depending on which you
        think will produce the better fitting model.
    """

    d = OrderedDict()

    for name in m.data.keys():

        d[name] = {}
        d[name]['x_obs'] = m.data[name][0]
        # d[name]['y_obs'] = m.data[name][1]
        d[name]['y_err'] = m.data[name][2]

        params = ['period', 't0', 'log_ror', 'b', 'u[0]', 'u[1]', 'r_star',
                  'logg_star', f'{name}_mean', f'{name}_a1', f'{name}_a2']

        _tmid = np.nanmedian(m.data[name][0])
        t_exp = np.nanmedian(np.diff(m.data[name][0]))

        if bestfitmeans == 'mode':
            paramd = {}
            for k in params:
                print(name, k)
                paramd[k] = _estimate_mode(m.trace[k])
        elif bestfitmeans == 'median':
            paramd = {k : summdf.loc[k, 'median'] for k in params}
        elif bestfitmeans == 'mean':
            paramd = {k : summdf.loc[k, 'mean'] for k in params}
        elif bestfitmeans == 'map':
            paramd = {k : m.map_estimate[k] for k in params}
        else:
            raise NotImplementedError

        y_mod_median, y_mod_median_trend = (
            get_model_transit_quad(paramd, d[name]['x_obs'], _tmid,
                                   t_exp=t_exp, includemean=1)
        )

        # this is used for phase-folded data, with the local trend removed.
        d[name]['y_mod'] = y_mod_median - y_mod_median_trend

        # NOTE: for this case, the "residual" of the observation minus the
        # quadratic trend is actually the "observation". this is b/c the
        # observation includes the rotation signal.
        d[name]['y_obs'] = m.data[name][1] - y_mod_median_trend

        d[name]['params'] = params

    # merge all the tess transits
    n_tess = len([k for k in d.keys() if 'tess' in k])
    d['tess'] = {}
    d['all'] = {}
    _p = ['x_obs', 'y_obs', 'y_err', 'y_mod']
    for p in _p:
        d['tess'][p] = np.hstack([d[f'tess_{ix}'][p] for ix in range(n_tess)])
    for p in _p:
        d['all'][p] = np.hstack([d[f'{k}'][p] for k in d.keys() if '_' in k])

    return d


def _subset_cut(x_obs, y_obs, y_err, n=12, t0=None, per=None, tdur=None,
                onlyodd=False, onlyeven=False):
    """
    Slice a time/flux/flux_err timeseries centered on transit windows, such
    that:

        n: [ t0 - n*tdur, t + n*tdur ]

    Args:

        t0: midtime epoch for window

        per: period [d]

        tdur: rough transit duration (T_14), used as above for window slicing

        onlyodd / onlyeven: boolean, for only analyzing certain subsets of the
            data.
    """

    epochs = np.arange(-200,200,1)
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


def _get_flux_err_as_stdev(x_obs, y_obs, t0=None, per=None, tdur=None):
    """
    Given time and flux from _subset_cut above (i.e., centered on a transit),
    return an array of the same length, with values set to be the standard
    deviation of the flux.
    """

    epochs = np.arange(-200,200,1)
    mid_times = t0 + per*epochs

    intra = np.zeros_like(x_obs).astype(bool)
    for tra_ind, mid_time in zip(epochs, mid_times):

        prefactor = 0.7 # set to let you omit all in transit points
        start_time = mid_time - prefactor*tdur
        end_time = mid_time + prefactor*tdur
        s = (x_obs > start_time) & (x_obs < end_time)
        intra |= s

    print(42*'#')
    print(f'Number of in-transit points: {len(x_obs[intra])}.')
    print(f'Omitting them for the estimation of time-series stdev...')
    print(42*'#')

    y_err = np.nanstd(y_obs[~intra])

    return np.ones_like(y_obs)*y_err


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


def p2p_rms(flux):
    """
    e.g., section 2.6 of Nardiello+2020:
        The point-to-point RMS is not sensitive to intrinsic stellar
        variability.  It's obtained by calculating the 84th-50th percentile of
        the distribution of the sorted residuals from the median value of
        δF_j = F_{j} - F_{j+1}, where j is an epoch index.
    """

    dflux = np.diff(flux)

    med_dflux = np.nanmedian(dflux)

    up_p2p = (
        np.nanpercentile( np.sort(dflux-med_dflux), 84 )
        -
        np.nanpercentile( np.sort(dflux-med_dflux), 50 )
    )
    lo_p2p = (
        np.nanpercentile( np.sort(dflux-med_dflux), 50 )
        -
        np.nanpercentile( np.sort(dflux-med_dflux), 16 )
    )

    p2p = np.mean([up_p2p, lo_p2p])

    return p2p


def get_tic117689799_lightcurve():

    lcfile = os.path.join(
        TESTDATADIR,
        'gaiatwo0001833519030401513984-0014-cam1-ccd4_tess_v01_llc_simpletransit_rawlc.csv'
    )

    assert os.path.exists(lcfile), f'expected to find {lcfile}'

    df = pd.read_csv(lcfile)

    time = np.array(df.time)
    _f = np.array(df.flux_PCA1)
    flux = _f/np.nanmedian(_f)

    # estimate flux error as p2p rms scatter
    flux_err = p2p_rms(flux)*np.ones_like(flux)

    sel = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)

    tess_texp = np.nanmedian(np.diff(time[sel]))

    return (
        time[sel].astype(np.float64),
        flux[sel].astype(np.float64),
        flux_err[sel].astype(np.float64),
        tess_texp
    )


def _estimate_mode(samples, N=1000):
    """
    Estimates the "mode" (really, maximum) of a unimodal probability
    distribution given samples from that distribution. Do it by approximating
    the distribution using a gaussian KDE, with an auto-tuned bandwidth that
    uses Scott's rule of thumb.
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html>

    args:

        samples: 1d numpy array of sampled values

        N: number of points at which to evalute the KDE. higher improves
        precision of the estimate.

    returns:

        Peak of the distribution. (Assuming it is unimodal, which should be
        checked.)
    """

    kde = gaussian_kde(samples, bw_method='scott')

    x = np.linspace(min(samples), max(samples), N)

    probs = kde.evaluate(x)

    peak = x[np.argmax(probs)]

    return peak


def _given_mag_get_flux(mag, err_mag=None):
    """
    Given a time-series of magnitudes, convert it to relative fluxes.
    """

    mag_0, f_0 = 12, 1e4
    flux = f_0 * 10**( -0.4 * (mag - mag_0) )
    fluxmedian = np.nanmedian(flux)
    flux /= fluxmedian

    if err_mag is None:
        return flux

    else:

        #
        # sigma_flux = dg/d(mag) * sigma_mag, for g=f0 * 10**(-0.4*(mag-mag0)).
        #
        err_flux = np.abs(
            -0.4 * np.log(10) * f_0 * 10**(-0.4*(mag-mag_0)) * err_mag
        )
        err_flux /= fluxmedian

        return flux, err_flux

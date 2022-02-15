"""
Posterior plots:
    plot_fitindiv
    plot_fitindivpanels
    plot_localpolyindivpanels
    plot_phasefold
    plot_phased_light_curve_gptransit
    plot_cornerplot
    plot_1d_posterior
    plot_grounddepth

MAP plots:
    plot_light_curve
    plot_multicolorlight_curve
    plot_phased_subsets

Helper functions:
    map_integer_to_character
    given_N_axes_get_3col_mosaic_subplots
    doublemedian, doublemean, doublepctile, get_ylimguess
"""
#############
## LOGGING ##
#############

import logging
from astrobase import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception

#############
## IMPORTS ##
#############

import os, corner, pickle
from datetime import datetime
from copy import deepcopy
from glob import glob
import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import matplotlib as mpl
from numpy import array as nparr

from aesthetic.plot import savefig, format_ax, set_style

from astrobase.lcmath import (
    phase_magseries, phase_bin_magseries, sigclip_magseries,
    find_lc_timegroups, phase_magseries_with_errs, time_bin_magseries
)

from betty.helpers import (
    _get_fitted_data_dict_simpletransit,
    _get_fitted_data_dict_localpolytransit, _get_fitted_data_dict_alltransit,
    _get_fitted_data_dict_allindivtransit, get_model_transit
)

from astrobase import periodbase
from astrobase.plotbase import skyview_stamp

from astropy.stats import LombScargle
from astropy import units as u, constants as const
from astropy.io import fits
from astropy.time import Time

import matplotlib.transforms as transforms
from matplotlib.ticker import FormatStrFormatter

def plot_fitindiv(m, summdf, outpath, overwrite=1, modelid=None,
                  singleinstrument='tess'):
    """
    Plot of flux timeseries, with individual transit windows and fits
    underneath.
    """

    set_style()

    if modelid not in ['simpletransit', 'allindivtransit', 'localpolytransit']:
        raise NotImplementedError

    if os.path.exists(outpath) and not overwrite:
        LOGINFO(f'found {outpath} and no overwrite')
        return

    if modelid == 'simpletransit':
        d, params, paramd = _get_fitted_data_dict_simpletransit(
            m, summdf
        )
        _d = d

    elif modelid == 'localpolytransit':
        raise NotImplementedError('not the right plot for this model')
        d = _get_fitted_data_dict_localpolytransit(
            m, summdf, bestfitmeans='map'
        )
        _d = d[singleinstrument]

    instrkeys = [k for k in m.priordict.keys() if '_mean' in k]
    if len(instrkeys) > 1:
        msg = 'Expected 1 instrument for this fit.'
        raise NotImplementedError(msg)
    instr = instrkeys[0].split('_')[0]

    time, flux, flux_err = (
        _d['x_obs'],
        _d['y_obs'],
        _d['y_err']
    )

    t_offset = np.nanmin(time)
    time -= t_offset

    t0 = summdf.loc['t0', 'median'] - t_offset
    per = summdf.loc['period', 'median']
    epochs = np.arange(-100,100,1)
    tra_times = t0 + per*epochs

    plt.close('all')

    ##########################################

    fig = plt.figure(figsize=(4, 3))
    fig, ax = plt.subplots(figsize=(4,3))

    # plot data
    yval = (flux - np.nanmedian(flux))*1e3
    ax.scatter(time, yval, c='k', zorder=3, s=0.5, rasterized=True,
               linewidths=0)

    # plot model
    from betty.helpers import get_model_transit
    modtime = np.linspace(time.min(), time.max(), int(1e4))
    ukey = 'u' if 'u[0]' in summdf else 'u_star'
    params = ['period', 't0', 'log_ror', 'b', f'{ukey}[0]', f'{ukey}[1]',
              f'{instr}_mean', 'r_star', 'logg_star']

    paramd = {k:summdf.loc[k, 'median'] for k in params}
    modflux = (
        get_model_transit(paramd, modtime + t_offset)
    )
    ax.plot(modtime, 1e3*(modflux-np.nanmedian(flux)),
            color='darkgray', alpha=1, rasterized=False, lw=0.7, zorder=1)

    ymin, ymax = ax.get_ylim()
    ax.vlines(
        tra_times, ymin, ymax, colors='darkgray', alpha=0.5,
        linestyles='--', zorder=-2, linewidths=0.2
    )
    ax.set_ylim((ymin, ymax))
    ax.set_xlim((np.nanmin(time)-1, np.nanmax(time)+1))
    ax.set_xlabel('Days from start')

    ax.set_ylabel('Relative flux [ppt]')

    fig.tight_layout(h_pad=0.5, w_pad=0.2)
    savefig(fig, outpath, writepdf=0, dpi=300)


def plot_fitindivpanels(m, summdf, outpath, overwrite=1, modelid=None,
                        singleinstrument='tess'):
    """
    Plot of flux timeseries, with individual transit windows and fits
    underneath.
    """

    set_style()

    if modelid not in ['localpolytransit']:
        raise NotImplementedError

    if os.path.exists(outpath) and not overwrite:
        LOGINFO(f'found {outpath} and no overwrite')
        return

    if modelid == 'localpolytransit':
        d = _get_fitted_data_dict_localpolytransit(
            m, summdf, bestfitmeans='map'
        )
        _d = d[singleinstrument]

    t_dur = np.nanmedian(m.trace.posterior.T_14)

    t0 = summdf.loc['t0', 'median']
    per = summdf.loc['period', 'median']
    epochs = np.arange(-1000, 1000, 1)
    tra_times = t0 + per*epochs

    # 2 extra keys are "tess" (singleinstrument key) and "all".
    N_transit_windows = len(d.keys()) - 2

    N_axes = N_transit_windows

    fig, axd = given_N_axes_get_3col_mosaic_subplots(N_axes)

    for ix in range(N_axes):

        ax = axd[map_integer_to_character(ix)]

        # _d contains the following keys:
        # ['x_obs', 'y_err', 'x_mod', 'y_mod', 'y_obs', 'y_resid', 'params']
        _d = d[f'{singleinstrument}_{ix}']

        # plot data
        #ax.scatter(_d['x_obs'], _d['y_obs'], c='k', zorder=3, s=0.5, rasterized=True,
        #           linewidths=0)

        x0 = 2457000
        y0 = np.nanmedian(_d['y_obs'])

        ax.errorbar(_d['x_obs']-x0, 1e3*(_d['y_obs']-y0), yerr=1e3*_d['y_err'], fmt='none',
                    ecolor='k', elinewidth=0.5, capsize=2, mew=0.5, zorder=2)

        ax.plot(_d['x_mod']-x0, 1e3*(_d['y_mod']-y0), color='darkgray', alpha=1,
                rasterized=False, lw=0.7, zorder=1)

        # midpoint of model is a decent mid-transit time guess
        t_mid = np.nanmedian(_d['x_mod']-x0)

        N_tdur = 4
        xmin, xmax = t_mid-N_tdur*t_dur/24, t_mid+N_tdur*t_dur/24
        ax.set_xlim((xmin, xmax))

        # set ylim and transit time line
        ymin, ymax = ax.get_ylim()
        sel = (tra_times-x0 > xmin) & (tra_times-x0 < xmax)
        ax.vlines(
            tra_times[sel]-x0, ymin, ymax, colors='darkgray', alpha=0.5,
            linestyles='--', zorder=-2, linewidths=0.2
        )
        ax.set_ylim((ymin, ymax))

        # get x axis to be :.1f
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.tick_params(axis='both', labelsize='small')

    fig.text(-0.01,0.5, 'Relative flux [ppt]', va='center', rotation=90,
             fontsize='large')
    fig.text(0.5,-0.01, 'Time [BTJD]', va='center', ha='center',
             fontsize='large')

    fig.tight_layout(h_pad=0.5, w_pad=0.2)
    savefig(fig, outpath, writepdf=1, dpi=300)


def plot_localpolyindivpanels(d, m, summdf, outpath, overwrite=1, modelid=None,
                              singleinstrument='tess'):
    """
    You have fitted a "simpletransit" model to a light curve, after first
    removing local trends with a Nth order polynomial.  Was this OK to do?
    How do the results look, for each window?
    """

    set_style()

    if modelid not in ['simpletransit']:
        raise NotImplementedError

    if os.path.exists(outpath) and not overwrite:
        LOGINFO(f'found {outpath} and no overwrite')
        return

    if modelid == 'simpletransit':
        __d, params, paramd = _get_fitted_data_dict_simpletransit(
            m, summdf
        )
        _d = __d

    t_dur = np.nanmedian(m.trace.posterior.T_14)
    t0 = summdf.loc['t0', 'median']
    per = summdf.loc['period', 'median']
    epochs = np.arange(-1000, 1000, 1)
    tra_times = t0 + per*epochs

    # 2 extra keys are "tess" (singleinstrument key) and "all".
    N_transit_windows = d['ngroups']

    N_axes = N_transit_windows

    fig, axd = given_N_axes_get_3col_mosaic_subplots(N_axes)

    for ix in range(N_axes):

        ax = axd[map_integer_to_character(ix)]

        # plot data
        x0 = 2457000
        y0 = np.nanmedian(d[f'flux_{ix}'])

        ax.errorbar(d[f'time_{ix}']-x0, 1e3*(d[f'flux_{ix}']-y0),
                    yerr=1e3*d[f'flux_err_{ix}'], fmt='none', ecolor='k',
                    elinewidth=0.5, capsize=2, mew=0.5, zorder=2)

        # plot model (local chi-squared polynomial, + median transit)

        y_polynomial = d[f'mod_flux_{ix}']

        y_transit = get_model_transit(
            paramd, d[f'mod_time_{ix}'],
            t_exp=np.nanmedian(np.diff(d[f'mod_time_{ix}'])), include_mean=0
        )

        ax.plot(d[f'mod_time_{ix}']-x0, 1e3*(y_polynomial + y_transit - y0),
                color='darkgray', alpha=1, rasterized=False, lw=0.7, zorder=1)

        # center on midpoint of model  (which might be off-transit, if the
        # model is too!)
        t_mid = np.nanmedian(d[f'mod_time_{ix}']-x0)

        N_tdur = 5
        xmin, xmax = t_mid-N_tdur*t_dur/24, t_mid+N_tdur*t_dur/24
        ax.set_xlim((xmin, xmax))

        # set ylim and transit time line
        ymin, ymax = ax.get_ylim()
        sel = (tra_times-x0 > xmin) & (tra_times-x0 < xmax)
        ax.vlines(
            tra_times[sel]-x0, ymin, ymax, colors='darkgray', alpha=0.5,
            linestyles='--', zorder=-2, linewidths=0.2
        )
        ax.set_ylim((ymin, ymax))

        # get x axis to be :.1f
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.tick_params(axis='both', labelsize='x-small')

    fig.text(-0.01,0.5, 'Relative flux [ppt]', va='center', rotation=90,
             fontsize='large')
    fig.text(0.5,-0.01, 'Time [BTJD]', va='center', ha='center',
             fontsize='large')

    fig.tight_layout(h_pad=0.5, w_pad=0.2)
    savefig(fig, outpath, writepdf=1, dpi=300)



def plot_phasefold(m, summdf, outpath, overwrite=0, modelid=None, inppt=0,
                   showerror=1, xlim=None, ylim=None, binsize_minutes=10,
                   savepdf=0, singleinstrument='tess'):
    """
    Options:
        inppt: Whether to median subtract and give flux in units of 1e-3.
        xlim: can be tuple. (Units: hours)
        binsize_minutes: bin to this many minutes in phase
        savepdf: whether to also save a pdf vesion
    """

    set_style()

    if modelid == 'simpletransit':
        d, params, paramd = _get_fitted_data_dict_simpletransit(
            m, summdf, N_model_times=int(2e4)
        )
        _d = d
    elif modelid == 'localpolytransit':
        d = _get_fitted_data_dict_localpolytransit(
            m, summdf, bestfitmeans='map'
        )
        _d = d[singleinstrument]
    else:
        raise NotImplementedError

    P_orb = summdf.loc['period', 'median']
    t0_orb = summdf.loc['t0', 'median']

    if modelid == 'simpletransit':
        ymodkey = 'y_mod'
        yobskey = 'y_obs'
    elif modelid == 'localpolytransit':
        ymodkey = 'y_mod_notrend'
        yobskey = 'y_obs_notrend'

    # phase and bin them.
    binsize_phase = (binsize_minutes / (60*24))/P_orb # prev: 5e-4
    orb_d = phase_magseries(
        _d['x_obs'], _d[yobskey], P_orb, t0_orb, wrap=True, sort=True
    )
    orb_bd = phase_bin_magseries(
        orb_d['phase'], orb_d['mags'], binsize=binsize_phase, minbinelems=3
    )
    mod_d = phase_magseries(
        _d['x_mod'], _d[ymodkey], P_orb, t0_orb, wrap=True, sort=True
    )
    resid_d = phase_magseries(
        _d['x_obs'], _d['y_resid'], P_orb, t0_orb, wrap=True, sort=True
    )
    resid_bd = phase_bin_magseries(
        resid_d['phase'], resid_d['mags'], binsize=binsize_phase,
        minbinelems=3
    )

    # make tha plot
    plt.close('all')

    fig, (a0, a1) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                 figsize=(4, 3), gridspec_kw=
                                 {'height_ratios':[3, 2]})

    if not inppt:

        a0.scatter(orb_d['phase']*P_orb*24, orb_d['mags'], color='gray', s=2,
                   alpha=0.8, zorder=4, linewidths=0, rasterized=True)
        a0.scatter(orb_bd['binnedphases']*P_orb*24, orb_bd['binnedmags'],
                   color='black', s=8, alpha=1, zorder=5, linewidths=0)
        a0.plot(mod_d['phase']*P_orb*24, mod_d['mags'], color='darkgray',
                alpha=0.8, rasterized=False, lw=1, zorder=1)

        a1.scatter(orb_d['phase']*P_orb*24, orb_d['mags']-mod_d['mags'],
                   color='gray', s=2, alpha=0.8, zorder=4, linewidths=0,
                   rasterized=True)
        a1.scatter(resid_bd['binnedphases']*P_orb*24, resid_bd['binnedmags'],
                   color='black', s=8, alpha=1, zorder=5, linewidths=0)
        a1.plot(mod_d['phase']*P_orb*24, mod_d['mags']-mod_d['mags'],
                color='darkgray', alpha=0.8, rasterized=False, lw=1, zorder=1)

    else:

        ydiff = 1 if modelid == 'simpletransit' else 0

        a0.scatter(orb_d['phase']*P_orb*24, 1e3*(orb_d['mags']-ydiff),
                   color='darkgray', s=7, alpha=0.35, zorder=3, linewidths=0,
                   rasterized=True)
        a0.scatter(orb_bd['binnedphases']*P_orb*24,
                   1e3*(orb_bd['binnedmags']-ydiff), color='black', s=18, alpha=1,
                   zorder=5, linewidths=0)
        a0.plot(mod_d['phase']*P_orb*24, 1e3*(mod_d['mags']-ydiff),
                color='gray', alpha=0.8, rasterized=False, lw=1, zorder=4)

        a1.scatter(resid_d['phase']*P_orb*24, 1e3*(resid_d['mags']),
                   color='darkgray', s=7, alpha=0.5, zorder=3, linewidths=0,
                   rasterized=True)
        a1.scatter(resid_bd['binnedphases']*P_orb*24,
                   1e3*resid_bd['binnedmags'], color='black', s=18, alpha=1,
                   zorder=5, linewidths=0)
        a1.plot(mod_d['phase']*P_orb*24, 1e3*(mod_d['mags']-mod_d['mags']),
                color='gray', alpha=0.8, rasterized=False, lw=1, zorder=4)

    if not inppt:
        a0.set_ylabel('Relative flux', fontsize='small')
    else:
        a0.set_ylabel('Relative flux [ppt]', fontsize='small')
    a1.set_ylabel('Residual [ppt]', fontsize='small')
    a1.set_xlabel('Hours from mid-transit', fontsize='small')

    yv = resid_d['mags']
    if inppt:
        yv = 1e3*(resid_d['mags'])
    a1.set_ylim((np.nanmedian(yv)-3.2*np.nanstd(yv),
                 np.nanmedian(yv)+3.2*np.nanstd(yv) ))

    for a in (a0, a1):
        a.set_xlim((-5, 5)) # hours
        if isinstance(xlim, tuple):
            a.set_xlim(xlim) # hours
        if isinstance(ylim, tuple):
            a.set_ylim(ylim)

    if showerror:
        trans = transforms.blended_transform_factory(
                a0.transAxes, a0.transData)
        if inppt:
            _e = 1e3*np.median(_d['y_err'])

            sampletime = np.nanmedian(np.diff(_d['x_obs']))*24*60 # minutes
            if binsize_minutes > sampletime:
                errorfactor = (sampletime/binsize_minutes)**(1/2)
            else:
                errorfactor = (binsize_minutes/sampletime)**(1/2)

            ydiff = np.abs(a0.get_ylim()[1] - a0.get_ylim()[0])
            _y = a0.get_ylim()[0] + 0.1*ydiff

            a0.errorbar(
                0.85, _y, yerr=errorfactor*_e,
                fmt='none', ecolor='black', alpha=1, elinewidth=1, capsize=2,
                transform=trans
            )

            LOGINFO(f'Median error [ppt]: {_e:.2f}, errorfactor: {errorfactor*_e:.2f}')

        else:
            raise NotImplementedError

    fig.tight_layout()

    savefig(fig, outpath, writepdf=savepdf, dpi=300)


def plot_cornerplot(var_names, m, outpath, overwrite=1):

    if os.path.exists(outpath) and not overwrite:
        return

    # corner plot of posterior samples
    plt.close('all')

    from corner import __version__
    import arviz as az

    if float(__version__[:3]) < 2.2:

        if isinstance(m.trace, dict):
            trace_df = pd.DataFrame(m.trace)
        elif isinstance(m.trace, az.data.inference_data.InferenceData):
            trace_df = m.trace.posterior.to_dataframe()
        else:
            trace_df = pm.trace_to_dataframe(m.trace, varnames=var_names)
        fig = corner.corner(trace_df, quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, title_kwargs={"fontsize": 12},
                            title_fmt='.2g')
        savefig(fig, outpath, writepdf=0, dpi=100)

    else:

        assert isinstance(m.trace, az.data.inference_data.InferenceData)

        fig = corner.corner(
            m.trace.posterior[var_names], quantiles=[0.16, 0.5, 0.84],
            show_titles=True, title_kwargs={"fontsize": 12}, title_fmt='.2g',
            divergences=True
        )
        savefig(fig, outpath, writepdf=0, dpi=100)



def plot_1d_posterior(samples, outpath, truth=None, xlabel=None):

    set_style()
    plt.close('all')

    f, ax = plt.subplots(figsize=(4,3))

    ax.hist(samples, bins=20, density=True, zorder=-1, color='k',
            histtype='step', alpha=0.7)

    if isinstance(xlabel, str):
        ax.set_xlabel(xlabel)

    if truth is not None:
        ymin, ymax = ax.get_ylim()
        ax.vlines(
            truth, ymin, ymax, colors='C0', alpha=0.5,
            linestyles='--', zorder=-2, linewidths=0.5
        )
        ax.set_ylim((ymin, ymax))

    ax.set_ylabel('Posterior probability')

    savefig(f, outpath, writepdf=0)


def plot_grounddepth(m, summdf, outpath, overwrite=1, modelid=None,
                     showerror=1, talkaspect=0, groundkeytobandpass=None,
                     tdur=2/24., xlim=None, ylim=None):
    """
    groundkeytobandpass (dict): e.g.,

        groundkeytobandpass = {'ground_0': 'i$_\mathrm{LCOGT}$',
                               'ground_1': 'i$_\mathrm{LCOGT}$',
                               'ground_2': 'i$_\mathrm{LCOGT}$',
                               'ground_3': 'g$_\mathrm{LCOGT}$',
                               'ground_4': 'z$_\mathrm{LCOGT}$'}

    xlim, ylim: optional tuples.
    """

    set_style()

    if os.path.exists(outpath) and not overwrite:
        LOGINFO('found {} and no overwrite'.format(outpath))
        return

    if modelid == 'simpletransit':
        raise NotImplementedError
    elif 'alltransit' in modelid:
        d = _get_fitted_data_dict_alltransit(m, summdf)
    elif modelid in ['allindivtransit', 'tessindivtransit']:
        raise NotImplementedError
        d = _get_fitted_data_dict_allindivtransit(m, summdf)
    else:
        raise NotImplementedError

    n_groundtra = len([k for k in list(d.keys()) if 'ground' in k])

    t0 = summdf.T['t0'].loc['mean'] + 2457000
    period = summdf.T['period'].loc['mean']

    tra_dict = {}

    from cdips.utils import astropytime_to_YYYYMMDD

    for k in list(d.keys()):
        if 'ground' in k:

            start_time_bjdtdb = 2457000 + np.nanmin(d[k]['x_obs'])
            med_time_bjdtdb = 2457000 + np.nanmedian(d[k]['x_obs'])
            start_time = Time(start_time_bjdtdb, format='jd')

            dstr = astropytime_to_YYYYMMDD(start_time)
            tstr = (
                astropytime_to_YYYYMMDD(start_time, sep='.') + ' ' +
                groundkeytobandpass[k]
            )

            tra_dict[k] = {
                'start_time': start_time,
                'tra_ix': int( np.round((med_time_bjdtdb - t0)/period, 0) ),
                'dstr': dstr,
                'tstr': tstr
            }

    tra_df = pd.DataFrame(tra_dict).T

    # plot the transits in true time order
    tra_df = tra_df.sort_values(by='start_time')

    ##########################################

    plt.close('all')

    if not talkaspect:
        fig, ax = plt.subplots(figsize=(3.3,5))
    else:
        fig, ax = plt.subplots(figsize=(4,4))

    inds = range(n_groundtra)

    t0 = summdf.loc['t0', 'median']
    per = summdf.loc['period', 'median']
    b = summdf.loc['b', 'median']
    epochs = np.arange(tra_df.tra_ix.min(), tra_df.tra_ix.max(), 1)
    tra_times = t0 + per*epochs

    ##########################################
    # 2*transit depth in ppt, between each mean.
    delta_y = 2 * 1e3 * (np.exp(summdf.loc['log_ror', 'median']))**2
    shift = 0
    for ind, r in tra_df.iterrows():

        tra_ix = r['tra_ix']
        tstr = r['tstr']
        dstr = r['dstr']
        name = r.name

        ##########################################
        # get quantities to be plotted

        gtime = d[ind]['x_obs']
        gflux = d[ind]['y_obs']
        gflux_err = d[ind]['y_err']

        gmodtime = np.linspace(np.nanmin(gtime)-1, np.nanmax(gtime)+1, int(1e4))

        params = d[ind]['params']
        paramd = {k:summdf.loc[k, 'median'] for k in params}

        if modelid not in ['alltransit_quad', 'allindivtransit']:
            # model is for the cadence of the observation, not of the model.
            gmodflux = get_model_transit(
                paramd, gmodtime, t_exp=np.nanmedian(np.diff(gtime))
            )
        else:
            raise NotImplementedError('verify the below works')
            _tmid = np.nanmedian(gtime)
            gmodflux, gmodtrend = (
                get_model_transit_quad(paramd, gmodtime, _tmid)
            )
            gmodflux -= gmodtrend

            # remove the trends before plotting
            _, gtrend = get_model_transit_quad(paramd, gtime, _tmid)
            gflux -= gtrend

        # bin too, by a factor of 3 in whatever the sampling rate is
        bintime = 3*np.nanmedian(np.diff(gtime))*24*60*60
        bd = time_bin_magseries(gtime, gflux, binsize=bintime, minbinelems=2)
        gbintime, gbinflux = bd['binnedtimes'], bd['binnedmags']

        mid_time = t0 + per*tra_ix
        tdur = tdur # roughly, in units of days
        n = 1.55 # sets window width
        start_time = mid_time - n*tdur
        end_time = mid_time + n*tdur

        s = (gtime > start_time) & (gtime < end_time)
        bs = (gbintime > start_time) & (gbintime < end_time)
        gs = (gmodtime > start_time) & (gmodtime < end_time)

        ax.scatter((gtime[s]-mid_time)*24,
                   (gflux[s] - np.max(gmodflux[gs]))*1e3 - shift,
                   c='darkgray', zorder=3, s=7, rasterized=False,
                   linewidths=0, alpha=0.5)

        ax.scatter((gbintime[bs]-mid_time)*24,
                   (gbinflux[bs] - np.max(gmodflux[gs]))*1e3 - shift,
                   c='black', zorder=4, s=18, rasterized=False,
                   linewidths=0)

        if modelid in ['alltransit']:
            l0 = (
                'All-transit fit'
            )

        ax.plot((gmodtime[gs]-mid_time)*24,
                (gmodflux[gs] - np.max(gmodflux[gs]))*1e3 - shift,
                color='gray', alpha=0.8, rasterized=False, lw=1, zorder=1,
                label=l0)

        props = dict(boxstyle='square', facecolor='white', alpha=0.7, pad=0.15,
                     linewidth=0)
        ax.text(np.nanpercentile(24*(gmodtime[gs]-mid_time), 97), 2.5 - shift,
                tstr, ha='right', va='bottom', bbox=props, zorder=6,
                fontsize='x-small')

        if showerror:
            raise NotImplementedError('below is deprecated, and from TOI837')
            _e = 1e3*np.median(gflux_err)

            # bin to roughly 5e-4 * 8.3 * 24 * 60 ~= 6 minute intervals
            sampletime = np.nanmedian(np.diff(gtime))*24*60*60 # seconds
            errorfactor = (sampletime/bintime)**(1/2)

            ax.errorbar(
                -2.35, -7 - shift, yerr=errorfactor*_e,
                fmt='none', ecolor='black', alpha=1, elinewidth=1, capsize=2,
            )

            LOGINFO(f'{_e:.2f}, {errorfactor*_e:.2f}')

        shift += delta_y

    if isinstance(ylim, tuple):
        ax.set_ylim(ylim)

    if isinstance(xlim, tuple):
        ax.set_xlim(xlim)

    format_ax(ax)

    fig.text(0.5,-0.01, 'Hours from mid-transit', ha='center',
             fontsize='medium')
    fig.text(-0.02,0.5, 'Relative flux [ppt]', va='center',
             rotation=90, fontsize='medium')

    fig.tight_layout(h_pad=0.2, w_pad=0.2)
    savefig(fig, outpath, writepdf=1, dpi=300)


def plot_light_curve(data, soln, outpath, mask=None):

    assert len(data.keys()) == 1
    name = list(data.keys())[0]
    x,y,yerr,texp = data[name]
    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    plt.close('all')
    set_style()
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    ax = axes[0]

    if len(x[mask]) > int(2e4):
        # see https://github.com/matplotlib/matplotlib/issues/5907
        mpl.rcParams['agg.path.chunksize'] = 10000

    ax.scatter(x[mask], y[mask], c="k", s=0.5, rasterized=True,
               label="data", linewidths=0, zorder=42)
    gp_mod = soln["gp_pred"] + soln["mean"]
    ax.plot(x[mask], gp_mod, color="C2", label="MAP gp model",
            zorder=41, lw=0.5)
    ax.legend(fontsize=10)
    ax.set_ylabel("$f$")

    ax = axes[1]
    ax.plot(x[mask], y[mask] - gp_mod, "k", label="data - MAPgp")
    for i, l in enumerate("b"):
        mod = soln["light_curves"][:, i]
        ax.plot(x[mask], mod, label="planet {0}".format(l))
    ax.legend(fontsize=10, loc=3)
    ax.set_ylabel("$f_\mathrm{dtr}$")

    ax = axes[2]
    ax.plot(x[mask], y[mask] - gp_mod, "k", label="data - MAPgp")
    for i, l in enumerate("b"):
        mod = soln["light_curves"][:, i]
        ax.plot(x[mask], mod, label="planet {0}".format(l))
    ax.legend(fontsize=10, loc=3)
    ax.set_ylabel("$f_\mathrm{dtr}$ [zoom]")
    ymin = np.min(mod)-0.05*abs(np.min(mod))
    ymax = abs(ymin)
    ax.set_ylim([ymin, ymax])

    ax = axes[3]
    mod = gp_mod + np.sum(soln["light_curves"], axis=-1)
    ax.plot(x[mask], y[mask] - mod, "k")
    ax.axhline(0, color="#aaaaaa", lw=1)
    ax.set_ylabel("residuals")
    ax.set_xlim(x[mask].min(), x[mask].max())
    ax.set_xlabel("time [days]")

    fig.tight_layout()

    savefig(fig, outpath, dpi=350)


BPCOLORDICT = {
    'g': 'blue',
    'r': 'green',
    'i': 'orange',
    'z': 'red'
}

def plot_multicolorlight_curve(data, soln, outpath, mask=None):

    plt.close('all')
    set_style()
    fig, ax = plt.subplots(1, 1, figsize=(4,3))

    shift = 0
    delta = 0.007 # might need to tune

    for n, (name, (x, y, yerr, texp)) in enumerate(data.items()):

        if 'muscat3' in name:
            bp = name.split("_")[-1]
            c = BPCOLORDICT[bp]
        else:
            raise NotImplementedError('How are line colors given?')

        _time, _flux = nparr(x), nparr(y)

        bintime = 600
        bd = time_bin_magseries(_time, _flux, binsize=bintime, minbinelems=2)
        _bintime, _binflux = bd['binnedtimes'], bd['binnedmags']

        ax.scatter(_time,
                   _flux - shift,
                   c='darkgray', zorder=3, s=7, rasterized=False,
                   linewidths=0, alpha=0.5)

        ax.scatter(_bintime,
                   _binflux - shift,
                   c=c, zorder=4, s=18, rasterized=False,
                   linewidths=0)

        mod = soln[f"{name}_mu_transit"]
        ax.plot(x, mod - shift, color=c, zorder=41, lw=0.5)

        props = dict(boxstyle='square', facecolor='white', alpha=0.7, pad=0.15,
                     linewidth=0)
        txt = f'{bp}-band'
        ax.text(np.nanpercentile(_time, 1), 2e-3 + np.nanmedian(_flux) - shift,
                txt, ha='left', va='top', bbox=props, zorder=6, color=c,
                fontsize='x-small')

        shift += delta

    ax.set_ylabel(f'Relative flux')
    ax.set_xlabel(f'BJDTDB')

    fig.tight_layout()

    savefig(fig, outpath, dpi=350)


def doublemedian(x):
    return np.median(np.median(x, axis=0), axis=0)

def doublemean(x):
    return np.nanmean(np.nanmean(x, axis=0), axis=0)

def doublepctile(x, SIGMA=[2.5,97.5]):
    # [16, 84] for 1-sigma
    # flatten/merge cores and chains. then percentile over both.
    return np.percentile(
        np.reshape(
            np.array(x), (x.shape[0]*x.shape[1], x.shape[2])
        ),
        SIGMA, axis=0
    )

def get_ylimguess(y):

    ylow = np.nanpercentile(y, 0.1)
    yhigh = np.nanpercentile(y, 99.9)
    ydiff = (yhigh-ylow)
    ymin = ylow - 0.35*ydiff
    ymax = yhigh + 0.35*ydiff
    return [ymin,ymax]


def plot_phased_light_curve_gptransit(
    data, soln, outpath, mask=None, from_trace=False,
    ylimd=None, binsize_minutes=20, map_estimate=None, fullxlim=False, BINMS=3,
    do_hacky_reprerror=False, alpha=1
    ):
    """
    Args:

        data (OrderedDict): data['tess'] = (time, flux, flux_err, t_exp)

        soln (az.data.inference_data.InferenceData): can be MAP solution from
        PyMC3. can also be the posterior's trace itself (m.trace.posterior).
        If the posterior is passed, bands showing the 2-sigma uncertainty
        interval will be drawn.

        outpath (str): where to save the output.

        from_trace: True is using m.trace.posterior

    Optional:

        map_estimate: if passed, uses this as the "best fit" line. Otherwise,
        the nanmean is used (nanmedian was also considered).

    """

    if not fullxlim:
        scale_x = lambda x: x*24
    else:
        scale_x = lambda x: x

    assert len(data.keys()) == 1
    name = list(data.keys())[0]
    x,y,yerr,texp = data[name]

    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    plt.close('all')
    set_style()
    fig = plt.figure(figsize=(0.66*5,0.66*6))
    axd = fig.subplot_mosaic(
        """
        A
        B
        """,
        gridspec_kw={
            "height_ratios": [1,1]
        }
    )

    if from_trace==True:
        _t0 = np.nanmean(soln["t0"])
        _per = np.nanmean(soln["period"])

        if len(soln["gp_pred"].shape)==3:
            # (4, 500, 46055), ncores X nchains X time
            medfunc = doublemean
            pctfunc = doublepctile
        elif len(soln["gp_pred"].shape)==2:
            medfunc = lambda x: np.mean(x, axis=0)
            pctfunc = lambda x: np.percentile(x, [2.5,97.5], axis=0)
        else:
            raise NotImplementedError
        gp_mod = (
            medfunc(soln["gp_pred"]) +
            medfunc(soln["mean"])
        )
        lc_mod = (
            medfunc(np.sum(soln["light_curves"], axis=-1))
        )
        lc_mod_band = (
            pctfunc(np.sum(soln["light_curves"], axis=-1))
        )

        _yerr = (
            np.sqrt(yerr[mask] ** 2 +
                    np.exp(2 * medfunc(soln["log_jitter"])))
        )

        med_error = np.nanmedian(yerr[mask])
        med_jitter = np.nanmedian(np.exp(medfunc(soln["log_jitter"])))

        LOGINFO(42*'-')
        LOGINFO(f'WRN! Median σ_f = {med_error:.2e}.  Median jitter = {med_jitter:.2e}')
        LOGINFO(42*'-')

    if (from_trace == False) or (map_estimate is not None):
        if map_estimate is not None:
            # If map_estimate is given, over-ride the mean/median estimate above,
            # take the MAP.
            LOGINFO('WRN! Overriding mean/median estimate with MAP.')
            soln = deepcopy(map_estimate)
        _t0 = soln["t0"]
        _per = soln["period"]
        gp_mod = soln["gp_pred"] + soln["mean"]
        lc_mod = soln["light_curves"][:, 0]
        _yerr = (
            np.sqrt(yerr[mask] ** 2 + np.exp(2 * soln["log_jitter"]))
        )

    x_fold = (x - _t0 + 0.5 * _per) % _per - 0.5 * _per

    #For plotting
    lc_modx = x_fold[mask]
    lc_mody = lc_mod[np.argsort(lc_modx)]
    if from_trace==True:
        lc_mod_lo = lc_mod_band[0][np.argsort(lc_modx)]
        lc_mod_hi = lc_mod_band[1][np.argsort(lc_modx)]
    lc_modx = np.sort(lc_modx)

    if len(x_fold[mask]) > int(2e4):
        # see https://github.com/matplotlib/matplotlib/issues/5907
        mpl.rcParams['agg.path.chunksize'] = 10000

    #
    # begin the plot!
    #
    ax = axd['A']

    y0 = (y[mask]-gp_mod) - np.nanmedian(y[mask]-gp_mod)
    ax.errorbar(scale_x(x_fold[mask]), 1e3*(y0), yerr=1e3*_yerr,
                color="darkgray", label="data", fmt='.', elinewidth=0.2,
                capsize=0, markersize=1, rasterized=True, zorder=-1,
                alpha=alpha)

    binsize_days = (binsize_minutes / (60*24))
    orb_bd = phase_bin_magseries(
        x_fold[mask], y0, binsize=binsize_days, minbinelems=3
    )
    ax.scatter(
        scale_x(orb_bd['binnedphases']), 1e3*(orb_bd['binnedmags']), color='k',
        s=BINMS,
        alpha=1, zorder=1002, rasterized=True#, linewidths=0.2, edgecolors='white'
    )

    ax.plot(scale_x(lc_modx), 1e3*lc_mody, color="C4", label="transit model",
            lw=1, zorder=1001, alpha=1)

    if from_trace==True:
        art = ax.fill_between(
            scale_x(lc_modx), 1e3*lc_mod_lo, 1e3*lc_mod_hi, color="C4",
            alpha=0.5, zorder=1000
        )
        art.set_edgecolor("none")

    ax.set_xticklabels([])

    # residual axis
    ax = axd['B']

    y1 = (y[mask]-gp_mod-lc_mod) - np.nanmedian(y[mask]-gp_mod-lc_mod)

    binsize_days = (binsize_minutes / (60*24))
    orb_bd = phase_bin_magseries(
        x_fold[mask], y1, binsize=binsize_days, minbinelems=3
    )
    ax.scatter(
        scale_x(orb_bd['binnedphases']), 1e3*(orb_bd['binnedmags']), color='k',
        s=BINMS, alpha=1, zorder=1002, rasterized=True#, linewidths=0.2, edgecolors='white'
    )
    ax.axhline(0, color="C4", lw=1, ls='-', zorder=1000)

    if from_trace==True:
        from scipy.ndimage import gaussian_filter
        sigma = 30
        LOGINFO(f'WRN! Smoothing plotted by by sigma={sigma}')
        _g =  lambda a: gaussian_filter(a, sigma=sigma)
        art = ax.fill_between(
            scale_x(lc_modx), 1e3*_g(lc_mod_hi-lc_mody), 1e3*_g(lc_mod_lo-lc_mody),
            color="C4", alpha=0.5, zorder=1000
        )
        art.set_edgecolor("none")

    ax.set_xlabel("Hours from mid-transit")
    if fullxlim:
        ax.set_xlabel("Days from mid-transit")

    fig.text(-0.01,0.5, 'Relative flux [ppt]', va='center',
             rotation=90)

    for k,a in axd.items():
        if not fullxlim:
            a.set_xlim(-0.4*24,0.4*24)
        else:
            a.set_xlim(-_per/2,_per/2)
        if isinstance(ylimd, dict):
            a.set_ylim(ylimd[k])
        else:
            # sensible default guesses
            _y = 1e3*(y[mask]-gp_mod)
            axd['A'].set_ylim(get_ylimguess(_y))
            _y = 1e3*(y[mask] - gp_mod - lc_mod)
            axd['B'].set_ylim(get_ylimguess(_y))

        format_ax(a)

    # NOTE: alt approach: override it as the stddev of the residuals. This is
    # dangerous, b/c if the errors are totally wrong, you might not know.
    if do_hacky_reprerror:
        sel = np.abs(orb_bd['binnedphases']*24)>3 # at least 3 hours from mid-transit
        binned_err = 1e3*np.nanstd((orb_bd['binnedmags'][sel]))
        LOGINFO(f'WRN! Overriding binned unc as the residuals. Binned_err = {binned_err:.4f} ppt')

    _x,_y = 0.8*max(axd['A'].get_xlim()), 0.7*min(axd['A'].get_ylim())
    axd['A'].errorbar(
        _x, _y, yerr=binned_err,
        fmt='none', ecolor='black', alpha=1, elinewidth=0.5, capsize=2,
        markeredgewidth=0.5
    )
    _x,_y = 0.8*max(axd['B'].get_xlim()), 0.6*min(axd['B'].get_ylim())
    axd['B'].errorbar(
        _x, _y, yerr=binned_err,
        fmt='none', ecolor='black', alpha=1, elinewidth=0.5, capsize=2,
        markeredgewidth=0.5
    )

    fig.tight_layout()

    savefig(fig, outpath, dpi=350)
    plt.close('all')


def plot_phased_subsets(
    data, soln, outpath, timesubsets, mask=None, from_trace=False,
    ylimd=None, binsize_minutes=20, map_estimate=None, fullxlim=False, BINMS=3,
    do_hacky_reprerror=False, yoffsetNsigma=4, inch_per_subset=1
    ):
    """
    Args:

        data (OrderedDict): data['tess'] = (time, flux, flux_err, t_exp)

        soln (az.data.inference_data.InferenceData): can MAP solution from
        PyMC3. can also be the posterior's trace itself (m.trace.posterior).
        If the posterior is passed, bands showing the 2-sigma uncertainty
        interval will be drawn.

        outpath (str): where to save the output.

        timesubsets (list): list of tuples of times to plot.

        from_trace: True is using m.trace.posterior

    Optional:

        map_estimate: if passed, uses this as the "best fit" line. Otherwise,
        the nanmean is used (nanmedian was also considered).

    """

    assert len(data.keys()) == 1
    name = list(data.keys())[0]
    x,y,yerr,texp = data[name]

    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    assert isinstance(timesubsets, list)
    N_subsets = len(timesubsets)

    plt.close('all')
    set_style()
    fig,axs = plt.subplots(ncols=2, figsize=(4,inch_per_subset*N_subsets), sharey=True)

    if from_trace==True:
        _t0 = np.nanmean(soln["t0"])
        _per = np.nanmean(soln["period"])

        if len(soln["gp_pred"].shape)==3:
            # (4, 500, 46055), ncores X nchains X time
            medfunc = doublemean
            pctfunc = doublepctile
        elif len(soln["gp_pred"].shape)==2:
            medfunc = lambda x: np.mean(x, axis=0)
            pctfunc = lambda x: np.percentile(x, [2.5,97.5], axis=0)
        else:
            raise NotImplementedError
        gp_mod = (
            medfunc(soln["gp_pred"]) +
            medfunc(soln["mean"])
        )
        lc_mod = (
            medfunc(np.sum(soln["light_curves"], axis=-1))
        )
        lc_mod_band = (
            pctfunc(np.sum(soln["light_curves"], axis=-1))
        )

        _yerr = (
            np.sqrt(yerr[mask] ** 2 +
                    np.exp(2 * medfunc(soln["log_jitter"])))
        )

    if (from_trace == False) or (map_estimate is not None):
        if map_estimate is not None:
            # If map_estimate is given, over-ride the mean/median estimate above,
            # take the MAP.
            LOGINFO('WRN! Overriding mean/median estimate with MAP.')
            soln = deepcopy(map_estimate)
        _t0 = soln["t0"]
        _per = soln["period"]
        gp_mod = soln["gp_pred"] + soln["mean"]
        lc_mod = soln["light_curves"][:, 0]
        _yerr = (
            np.sqrt(yerr[mask] ** 2 + np.exp(2 * soln["log_jitter"]))
        )

    x_fold = (x - _t0 + 0.5 * _per) % _per - 0.5 * _per

    if len(x_fold) > int(2e4):
        # see https://github.com/matplotlib/matplotlib/issues/5907
        mpl.rcParams['agg.path.chunksize'] = 10000

    # iterate over the passed time subsets to show binned models in each
    for time_ix, tsub in enumerate(timesubsets):

        t0,t1 = tsub[0], tsub[1]
        if len(tsub) == 3:
            label = tsub[2]
        else:
            label = None
        sel = (x>=t0) & (x<=t1)

        if time_ix == 0:
            # NB. the *residual* stdev should determine the spacing.
            c_yoffset = yoffsetNsigma*np.std(1e3*(y[mask][sel]-gp_mod[sel]))
            c_yoffset_resid = yoffsetNsigma*np.std(1e3*(y[mask][sel]-gp_mod[sel]-lc_mod[sel]))
        y_offset = -time_ix*c_yoffset_resid
        y_offset_resid = -time_ix*c_yoffset_resid

        #For plotting
        lc_modx = x_fold[mask][sel]
        lc_mody = lc_mod[sel][np.argsort(lc_modx)]
        if from_trace==True:
            lc_mod_lo = lc_mod_band[0][sel][np.argsort(lc_modx)]
            lc_mod_hi = lc_mod_band[1][sel][np.argsort(lc_modx)]
        lc_modx = np.sort(lc_modx)

        #
        # begin the plot!
        #
        ax = axs[0]

        binsize_days = (binsize_minutes / (60*24))
        orb_bd = phase_bin_magseries(
            x_fold[mask][sel], y[mask][sel]-gp_mod[sel], binsize=binsize_days, minbinelems=3
        )
        ax.scatter(
            orb_bd['binnedphases']*24, 1e3*(orb_bd['binnedmags'])+y_offset, color='k',
            s=BINMS,
            alpha=1, zorder=1002, rasterized=True#, linewidths=0.2, edgecolors='white'
        )

        if label is not None:
            props = dict(boxstyle='square', facecolor='white', alpha=0.95, pad=0.15,
                         linewidth=0)
            trans = transforms.blended_transform_factory(
                    ax.transAxes, ax.transData)
            ax.text(0.97, np.nanpercentile(1e3*(orb_bd['binnedmags'])+y_offset,95),
                    label, ha='right', va='center', bbox=props, zorder=9001,
                    transform=trans,
                    fontsize='x-small')


        ax.plot(24*lc_modx, 1e3*lc_mody+y_offset, color="C4", label="transit model",
                lw=1, zorder=1001, alpha=1)

        if from_trace==True:
            art = ax.fill_between(
                24*lc_modx, 1e3*lc_mod_lo+y_offset, 1e3*lc_mod_hi+y_offset, color="C4", alpha=0.5,
                zorder=1000
            )
            art.set_edgecolor("none")

        ax.set_ylabel("Relative flux [ppt]")

        # residual axis
        ax = axs[1]

        binsize_days = (binsize_minutes / (60*24))
        orb_bd = phase_bin_magseries(
            x_fold[mask][sel], y[mask][sel]-gp_mod[sel]-lc_mod[sel],
            binsize=binsize_days, minbinelems=3
        )
        ax.scatter(
            orb_bd['binnedphases']*24, 1e3*(orb_bd['binnedmags'])+y_offset_resid, color='k',
            s=BINMS, alpha=1, zorder=1002, rasterized=True#, linewidths=0.2, edgecolors='white'
        )
        ax.axhline(0+y_offset_resid, color="C4", lw=1, ls='-', zorder=1000)

        if label is not None:
            props = dict(boxstyle='square', facecolor='white', alpha=0.95, pad=0.15,
                         linewidth=0)
            trans = transforms.blended_transform_factory(
                    ax.transAxes, ax.transData)
            ax.text(0.97,
                    np.nanpercentile(1e3*(orb_bd['binnedmags'])+y_offset_resid,95),
                    label, ha='right', va='center', bbox=props, zorder=9001,
                    transform=trans,
                    fontsize='x-small')

        if from_trace==True:
            art = ax.fill_between(24*lc_modx,
                                  1e3*(lc_mod_hi-lc_mody)+y_offset_resid,
                                  1e3*(lc_mod_lo-lc_mody)+y_offset_resid,
                color="C4", alpha=0.5, zorder=1000
            )
            art.set_edgecolor("none")

    if isinstance(ylimd, dict):
        for k,v in ylimd.items():
            axs[int(k)].set_ylim(v)

    fig.text(0.5,-0.01, 'Hours from mid-transit', ha='center',
             fontsize='medium')

    for a in axs:
        if not fullxlim:
            a.set_xlim(-0.2*24,0.2*24)

    fig.tight_layout(h_pad=0)

    savefig(fig, outpath, dpi=350)
    plt.close('all')


def plot_phased_light_curve_samples(
    data, soln, outpath, mask=None, from_trace=False,
    ylimd=None, binsize_minutes=20, map_estimate=None, fullxlim=False, BINMS=3,
    do_hacky_reprerror=False, alpha=0.1, n_samples=50
    ):
    """
    Args:

        data (OrderedDict): data['tess'] = (time, flux, flux_err, t_exp)

        soln (az.data.inference_data.InferenceData): can be MAP solution from
        PyMC3. can also be the posterior's trace itself (m.trace.posterior).
        If the posterior is passed, bands showing the 2-sigma uncertainty
        interval will be drawn.

        outpath (str): where to save the output.

        from_trace: True is using m.trace.posterior

    Optional:

        map_estimate: if passed, uses this as the "best fit" line. Otherwise,
        the nanmean is used (nanmedian was also considered).

    """

    if not fullxlim:
        scale_x = lambda x: x*24
    else:
        scale_x = lambda x: x

    assert len(data.keys()) == 1
    name = list(data.keys())[0]
    x,y,yerr,texp = data[name]

    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    plt.close('all')
    set_style()
    fig = plt.figure(figsize=(0.66*5,0.66*6))
    axd = fig.subplot_mosaic(
        """
        A
        B
        """,
        gridspec_kw={
            "height_ratios": [1,1]
        }
    )

    assert from_trace

    n_cores, n_chains, n_time = soln["gp_pred"].shape

    np.random.seed(42)

    ii = np.random.randint(low=0, high=n_cores, size=n_samples)
    jj = np.random.randint(low=0, high=n_chains, size=n_samples)

    LOGINFO(f"Random index: {ii},{jj}")

    _t0 = soln["t0"][ii,jj].data.diagonal()
    _per = soln["period"][ii,jj].data.diagonal()

    # n_times x n_samples
    gp_mod = (
        soln["gp_pred"][ii,jj,:] +
        soln["mean"][ii,jj]
    ).data.diagonal()

    lc_mod = (
        np.sum(soln["light_curves"], axis=-1)[ii,jj,:]
    ).data.diagonal()

    _yerr = (
        np.sqrt( (yerr[mask] ** 2)[:,None] +
                (np.exp(2 * soln["log_jitter"][ii,jj].data.diagonal()))[None,:]
               )
    )

    med_error = np.nanmedian(yerr[mask])
    med_jitter = np.nanmedian(np.exp(soln["log_jitter"][ii,jj].data.diagonal()))

    LOGINFO(42*'-')
    LOGINFO(f'WRN! Median σ_f = {med_error:.2e}.  Median jitter = {med_jitter:.2e}')
    LOGINFO(42*'-')

    for nn in range(n_samples):

        LOGINFO(f"{nn}/{n_samples}...")

        x_fold = (x - _t0[nn] + 0.5 * _per[nn]) % _per[nn] - 0.5 * _per[nn]

        #For plotting
        lc_modx = x_fold[mask]
        lc_mody = lc_mod[:,nn][np.argsort(lc_modx)]
        lc_modx = np.sort(lc_modx)

        if len(x_fold[mask]) > int(2e4):
            # see https://github.com/matplotlib/matplotlib/issues/5907
            mpl.rcParams['agg.path.chunksize'] = 10000

        #
        # begin the plot!
        #
        ax = axd['A']

        y0 = (y[mask]-gp_mod[:,nn]) - np.nanmedian(y[mask]-gp_mod[:,nn])

        binsize_days = (binsize_minutes / (60*24))
        orb_bd = phase_bin_magseries(
            x_fold[mask], y0, binsize=binsize_days, minbinelems=3
        )
        ax.scatter(
            scale_x(orb_bd['binnedphases']), 1e3*(orb_bd['binnedmags']), color='k',
            s=BINMS, linewidths=0, marker='.',
            alpha=alpha, zorder=1002, rasterized=True#, linewidths=0.2, edgecolors='white'
        )

        ax.plot(scale_x(lc_modx), 1e3*lc_mody, color="C4", label="transit model",
                lw=1, zorder=1001, alpha=alpha)

        ax.set_xticklabels([])

        # residual axis
        ax = axd['B']

        y1 = (y[mask]-gp_mod[:,nn]-lc_mod[:,nn]) - np.nanmedian(y[mask]-gp_mod[:,nn]-lc_mod[:,nn])

        binsize_days = (binsize_minutes / (60*24))
        orb_bd = phase_bin_magseries(
            x_fold[mask], y1, binsize=binsize_days, minbinelems=3
        )
        ax.scatter(
            scale_x(orb_bd['binnedphases']), 1e3*(orb_bd['binnedmags']), color='k',
            linewidths=0, marker='.',
            s=BINMS, alpha=alpha, zorder=1002, rasterized=True#, linewidths=0.2, edgecolors='white'
        )

        if nn == 0:
            ax.axhline(0, color="C4", lw=1, ls='-', zorder=1000)

    ax.set_xlabel("Hours from mid-transit")
    if fullxlim:
        ax.set_xlabel("Days from mid-transit")

    fig.text(-0.01,0.5, 'Relative flux [ppt]', va='center',
             rotation=90)

    for k,a in axd.items():
        if not fullxlim:
            a.set_xlim(-0.4*24,0.4*24)
        else:
            a.set_xlim(-_per/2,_per/2)
        if isinstance(ylimd, dict):
            a.set_ylim(ylimd[k])
        else:
            # sensible default guesses
            _y = 1e3*(y[mask]-gp_mod)
            axd['A'].set_ylim(get_ylimguess(_y))
            _y = 1e3*(y[mask] - gp_mod - lc_mod)
            axd['B'].set_ylim(get_ylimguess(_y))

        format_ax(a)

    # NOTE: alt approach: override it as the stddev of the residuals. This is
    # dangerous, b/c if the errors are totally wrong, you might not know.
    if do_hacky_reprerror:
        sel = np.abs(orb_bd['binnedphases']*24)>3 # at least 3 hours from mid-transit
        binned_err = 1e3*np.nanstd((orb_bd['binnedmags'][sel]))
        LOGINFO(f'WRN! Overriding binned unc as the residuals. Binned_err = {binned_err:.4f} ppt')

    _x,_y = 0.8*max(axd['A'].get_xlim()), 0.7*min(axd['A'].get_ylim())
    axd['A'].errorbar(
        _x, _y, yerr=binned_err,
        fmt='none', ecolor='black', alpha=1, elinewidth=0.5, capsize=2,
        markeredgewidth=0.5
    )
    _x,_y = 0.8*max(axd['B'].get_xlim()), 0.6*min(axd['B'].get_ylim())
    axd['B'].errorbar(
        _x, _y, yerr=binned_err,
        fmt='none', ecolor='black', alpha=1, elinewidth=0.5, capsize=2,
        markeredgewidth=0.5
    )

    fig.tight_layout()

    savefig(fig, outpath, dpi=350)
    plt.close('all')


def map_integer_to_character(integer):
    """
    For an integer between 0 and 1000, get a unique unicode character from it.
    """

    assert isinstance(integer, int)
    assert integer >= 0 and integer <= 1000

    offset = 161

    return chr(integer + offset)


def given_N_axes_get_3col_mosaic_subplots(N_axes, return_axstr=0):
    """
    Given the number of axes required, generate a figure and axd subplot
    instance, with 3 columns, and however many rows are needed.
    """

    assert isinstance(N_axes, int)

    N_cols = 3

    N_rows = int(np.ceil(N_axes / N_cols))
    N_hanging = N_axes % N_cols  # hanging = number "floating" in the final row

    fig = plt.figure(figsize=(6,3*N_rows))

    axstr = ''

    if N_hanging == 0:
        # e.g., '\n012\n345\n'
        for i in range(N_axes):
            c = map_integer_to_character(i)
            if i % N_cols == 0:
                axstr += f'\n{c}'
            else:
                axstr += f'{c}'
        axstr += '\n'
    elif N_hanging == 1:
        # e.g., '\n012\n.3.\n'
        for i in range(N_axes):
            c = map_integer_to_character(i)
            if i % N_cols == 0 and int(i / N_cols) < N_rows-1:
                axstr += f'\n{c}'
            elif i % N_cols == 0 and  int(i / N_cols) == N_rows-1:
                axstr += f'\n.{c}.\n'
            else:
                axstr += f'{c}'
    elif N_hanging == 2:
        # e.g., '\n001122\n.3344.\n'
        for i in range(N_axes):
            c = map_integer_to_character(i)
            cp1 = map_integer_to_character(i+1)
            if i % N_cols == 0 and int(i / N_cols) < N_rows-1:
                axstr += f'\n{c}{c}'
            elif i % N_cols == 0 and  int(i / N_cols) == N_rows-1:
                axstr += f'\n.{c}{c}{cp1}{cp1}.\n'
                break
            else:
                axstr += f'{c}{c}'

    if return_axstr:
        return axstr

    axd = fig.subplot_mosaic(
        axstr
    )

    return fig, axd

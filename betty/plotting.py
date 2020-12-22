"""
plot_fitindiv
plot_phasefold
plot_cornerplot
plot_1d_posterior
"""
import os, corner, pickle
from datetime import datetime
from glob import glob
import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
from numpy import array as nparr

from aesthetic.plot import savefig, format_ax
from aesthetic.plot import set_style

from astrobase.lcmath import (
    phase_magseries, phase_bin_magseries, sigclip_magseries,
    find_lc_timegroups, phase_magseries_with_errs, time_bin_magseries
)

from betty.helpers import (
    _get_fitted_data_dict, _get_fitted_data_dict_alltransit,
    _get_fitted_data_dict_allindivtransit
)

from astrobase import periodbase
from astrobase.plotbase import skyview_stamp

from astropy.stats import LombScargle
from astropy import units as u, constants as const
from astropy.io import fits
from astropy.time import Time

import matplotlib.transforms as transforms

def plot_fitindiv(m, summdf, outpath, overwrite=1, modelid=None):
    """
    Plot of flux timeseries, with individual transit windows and fits
    underneath.
    """

    set_style()

    if modelid not in ['simpletransit', 'allindivtransit']:
        raise NotImplementedError

    if os.path.exists(outpath) and not overwrite:
        print('found {} and no overwrite'.format(outpath))
        return

    # NOTE: may need to generalize this
    instr = 'tess'
    time, flux, flux_err = (
        m.data[instr][0],
        m.data[instr][1],
        m.data[instr][2]
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
    params = ['period', 't0', 'log_r', 'b', 'u[0]', 'u[1]',
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


def plot_phasefold(m, summdf, outpath, overwrite=0, show_samples=0,
                   modelid=None, inppt=0, showerror=1, xlim=None,
                   binsize_minutes=10, savepdf=0
                  ):
    """
    Options:
        inppt: Whether to median subtract and give flux in units of 1e-3.
        xlim: can be tuple. (Units: hours)
        binsize_minutes: bin to this many minutes in phase
        savepdf: whether to also save a pdf vesion
    """

    set_style()

    if modelid == 'simpletransit':
        d, params, paramd = _get_fitted_data_dict(m, summdf)
        _d = d

    elif 'alltransit' in modelid:
        d = _get_fitted_data_dict_alltransit(m, summdf)
        _d = d['tess']

    elif modelid in ['allindivtransit', 'tessindivtransit']:
        raise NotImplementedError
        d = _get_fitted_data_dict_allindivtransit(m, summdf)
        _d = d['tess']

    else:
        raise NotImplementedError

    P_orb = summdf.loc['period', 'median']
    t0_orb = summdf.loc['t0', 'median']

    # phase and bin them.
    binsize_phase = (binsize_minutes / (60*24))/P_orb # prev: 5e-4
    orb_d = phase_magseries(
        _d['x_obs'], _d['y_obs'], P_orb, t0_orb, wrap=True, sort=True
    )
    orb_bd = phase_bin_magseries(
        orb_d['phase'], orb_d['mags'], binsize=binsize_phase, minbinelems=3
    )
    mod_d = phase_magseries(
        _d['x_obs'], _d['y_mod'], P_orb, t0_orb, wrap=True, sort=True
    )
    resid_bd = phase_bin_magseries(
        mod_d['phase'], orb_d['mags'] - mod_d['mags'], binsize=binsize_phase,
        minbinelems=3
    )

    # get the samples. shape: N_samples x N_time
    if show_samples:
        np.random.seed(42)
        N_samples = 20

        sample_df = pm.trace_to_dataframe(m.trace, var_names=params)
        sample_params = sample_df.sample(n=N_samples, replace=False)

        y_mod_samples = []
        for ix, p in sample_params.iterrows():
            print(ix)
            paramd = dict(p)
            y_mod_samples.append(get_model_transit(paramd, d['x_obs']))

        y_mod_samples = np.vstack(y_mod_samples)

        mod_ds = {}
        for i in range(N_samples):
            mod_ds[i] = phase_magseries(
                d['x_obs'], y_mod_samples[i, :], P_orb, t0_orb, wrap=True,
                sort=True
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

        ydiff = 1

        a0.scatter(orb_d['phase']*P_orb*24, 1e3*(orb_d['mags']-ydiff),
                   color='darkgray', s=7, alpha=0.35, zorder=3, linewidths=0,
                   rasterized=True)
        a0.scatter(orb_bd['binnedphases']*P_orb*24,
                   1e3*(orb_bd['binnedmags']-ydiff), color='black', s=18, alpha=1,
                   zorder=5, linewidths=0)
        a0.plot(mod_d['phase']*P_orb*24, 1e3*(mod_d['mags']-ydiff),
                color='gray', alpha=0.8, rasterized=False, lw=1, zorder=4)

        a1.scatter(orb_d['phase']*P_orb*24, 1e3*(orb_d['mags']-mod_d['mags']),
                   color='darkgray', s=7, alpha=0.5, zorder=3, linewidths=0,
                   rasterized=True)
        a1.scatter(resid_bd['binnedphases']*P_orb*24,
                   1e3*resid_bd['binnedmags'], color='black', s=18, alpha=1,
                   zorder=5, linewidths=0)
        a1.plot(mod_d['phase']*P_orb*24, 1e3*(mod_d['mags']-mod_d['mags']),
                color='gray', alpha=0.8, rasterized=False, lw=1, zorder=4)


    if show_samples:
        # NOTE: this comes out looking "bad" because if you phase up a model
        # with a different period to the data, it will produce odd
        # aliases/spikes.

        xvals, yvals = [], []
        for i in range(N_samples):
            xvals.append(mod_ds[i]['phase']*P_orb*24)
            yvals.append(mod_ds[i]['mags'])
            a0.plot(mod_ds[i]['phase']*P_orb*24, mod_ds[i]['mags'], color='C1',
                    alpha=0.2, rasterized=True, lw=0.2, zorder=-2)
            a1.plot(mod_ds[i]['phase']*P_orb*24,
                    mod_ds[i]['mags']-mod_d['mags'], color='C1', alpha=0.2,
                    rasterized=True, lw=0.2, zorder=-2)

        # # N_samples x N_times
        # from scipy.ndimage import gaussian_filter1d
        # xvals, yvals = nparr(xvals), nparr(yvals)
        # model_phase = xvals.mean(axis=0)
        # g_std = 100
        # n_std = 2
        # mean = gaussian_filter1d(yvals.mean(axis=0), g_std)
        # diff = gaussian_filter1d(n_std*yvals.std(axis=0), g_std)
        # model_flux_lower = mean - diff
        # model_flux_upper = mean + diff

        # ax.plot(model_phase, model_flux_lower, color='C1',
        #         alpha=0.8, lw=0.5, zorder=3)
        # ax.plot(model_phase, model_flux_upper, color='C1', alpha=0.8,
        #         lw=0.5, zorder=3)
        # ax.fill_between(model_phase, model_flux_lower, model_flux_upper,
        #                 color='C1', alpha=0.5, zorder=3, linewidth=0)

    if not inppt:
        a0.set_ylabel('Relative flux', fontsize='small')
    else:
        a0.set_ylabel('Relative flux [ppt]', fontsize='small')
    a1.set_ylabel('Residual [ppt]', fontsize='small')
    a1.set_xlabel('Hours from mid-transit', fontsize='small')

    yv = orb_d['mags']-mod_d['mags']
    if inppt:
        yv = 1e3*(orb_d['mags']-mod_d['mags'])
    a1.set_ylim((np.nanmedian(yv)-3.2*np.nanstd(yv),
                 np.nanmedian(yv)+3.2*np.nanstd(yv) ))

    for a in (a0, a1):
        a.set_xlim((-5, 5)) # hours
        if isinstance(xlim, tuple):
            a.set_xlim(xlim) # hours

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

            print(f'Median error [ppt]: {_e:.2f}, errorfactor: {errorfactor*_e:.2f}')

        else:
            raise NotImplementedError

    fig.tight_layout()

    savefig(fig, outpath, writepdf=savepdf, dpi=300)



def plot_cornerplot(var_names, m, outpath, overwrite=1):

    if os.path.exists(outpath) and not overwrite:
        return

    # corner plot of posterior samples
    plt.close('all')
    if isinstance(m.trace, dict):
        trace_df = pd.DataFrame(m.trace)
    else:
        trace_df = pm.trace_to_dataframe(m.trace, varnames=var_names)
    fig = corner.corner(trace_df, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        title_fmt='.2g')
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

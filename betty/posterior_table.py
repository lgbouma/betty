"""
Given the output chains from betty.modelfitter, turn them into a LaTeX
table that you can publish.
"""

import os, re, pickle
from copy import deepcopy
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pymc3 as pm

##########################################
# helper functions for string generation
##########################################
def normal_str(mu, sd, fmtstr=None):
    # fmtstr: e.g., "({:.5f}; {:.2f})"
    if fmtstr is None:
        return '$\\mathcal{N}'+'({}; {})$'.format(mu, sd)
    else:
        return '$\\mathcal{N}'+'{}$'.format(fmtstr).format(mu, sd)


def lognormal_str(mu, sd, fmtstr=None):
    # fmtstr: e.g., "({:.5f}; {:.2f})"
    if fmtstr is None:
        return '$\log\\mathcal{N}'+'({}; {})$'.format(mu, sd)
    else:
        return '$\log\\mathcal{N}'+'{}$'.format(fmtstr).format(mu, sd)


def truncnormal_str(mu, sd, fmtstr=None):
    # fmtstr: e.g., "({:.5f}; {:.2f})"
    if fmtstr is None:
        return '$\\mathcal{T}'+'({}; {})$'.format(mu, sd)
    else:
        return '$\\mathcal{T}'+'{}$'.format(fmtstr).format(mu, sd)


def uniform_str(lower, upper, fmtstr=None):
    # fmtstr: e.g., "({:.5f}; {:.2f})"
    if fmtstr is None:
        return '$\\mathcal{U}'+'({}; {})$'.format(lower, upper)
    else:
        return '$\\mathcal{U}'+'{}$'.format(fmtstr).format(lower, upper)


def loguniform_str(lower, upper, fmtstr=None):
    # fmtstr: e.g., "({:.5f}; {:.2f})"
    if fmtstr is None:
        return '$\log\\mathcal{U}'+'({}; {})$'.format(lower, upper)
    else:
        return '$\log\\mathcal{U}'+'{}$'.format(fmtstr).format(lower, upper)


##########################################
# main driver function
##########################################

def make_posterior_table(pklpath, priordict, outpath, modelid, makepdf=1,
                         overwrite=1):
    """
    Given the output chains from betty.modelfitter, turn them into a LaTeX
    table that you can publish.

    (Kw)args:

        pklpath: path to pickle containing PyMC3 model, samples, traces, etc.
        priordict: dictionary of priors that were used
        outpath: .tex file table will be written to
        modelid: string model identifier. used for a few path internals.
        makepdf: whether or not to compile the tex into a .pdf file to read
    """

    assert outpath.endswith('.tex')

    summarypath = outpath.replace('.tex', '_raw.csv')

    scols = ['median', 'mean', 'sd', 'hpd_3%', 'hpd_97%', 'ess_mean',
             'r_hat_minus1']

    if os.path.exists(summarypath) and not overwrite:
        df = pd.read_csv(summarypath, index_col=0)

    else:
        with open(pklpath, 'rb') as f:
            d = pickle.load(f)

        model = d['model']
        trace = d['trace']
        map_estimate = d['map_estimate']

        # stat_funcsdict = A list of functions or a dict of functions with
        # function names as keys used to calculate statistics. By default, the
        # mean, standard deviation, simulation standard error, and highest
        # posterior density intervals are included.
        stat_funcsdict = {
            'median': np.nanmedian
        }

        df = pm.summary(
            trace,
            round_to=10, kind='all',
            stat_funcs=stat_funcsdict,
            extend=True
        )

        df['r_hat_minus1'] = df['r_hat'] - 1

        outdf = df[scols]

        outdf.to_csv(summarypath, index=True)

    fitted_params = list(priordict.keys())

    n_fitted = len(fitted_params)

    derived_params = [
        'r', 'rho_star', 'r_planet', 'a_Rs', 'cosi', 'T_14', 'T_13'
    ]
    n_derived = len(derived_params)

    srows = []
    for f in fitted_params:
        srows.append(f)
    for d in derived_params:
        srows.append(d)

    df = df.loc[srows]

    df = df[scols]

    print(df)

    if modelid not in ['simpletransit', 'alltransit']:
        raise NotImplementedError('generalize priordict using the 0th key '
                                  'from the priordict tuple here...')

    pr = {
        'period': normal_str(
            mu=priordict['period'][1], sd=priordict['period'][2],
            fmtstr='({:.5f}; {:.5f})'
        ),
        't0': normal_str(
            mu=priordict['t0'][1], sd=priordict['t0'][2],
            fmtstr='({:.5f}; {:.5f})'
        ),
        'log_r': uniform_str(
            lower=priordict['log_r'][1], upper=priordict['log_r'][2],
            fmtstr='({:.3f}; {:.3f})'
        ),
        'b': r'$\mathcal{U}(0; 1+R_{\mathrm{p}}/R_\star)$',
        'u[0]': uniform_str(priordict['u[0]'][1], priordict['u[0]'][2],
                            fmtstr='({:.3f}; {:.3f})') + '$^{(2)}$',
        'u[1]': uniform_str(priordict['u[1]'][1], priordict['u[1]'][2],
                            fmtstr='({:.3f}; {:.3f})') + '$^{(2)}$',
        'r_star': truncnormal_str(
            mu=priordict['r_star'][1], sd=priordict['r_star'][2],
            fmtstr='({:.3f}; {:.3f})'
        ),
        'logg_star': normal_str(
            mu=priordict['logg_star'][1], sd=priordict['logg_star'][2],
            fmtstr='({:.3f}; {:.3f})'
        )
    }
    ufmt = '({:.2f}; {:.2f})'

    pr[f'tess_mean'] = normal_str(mu=priordict[f'tess_mean'][1],
                                  sd=priordict[f'tess_mean'][2], fmtstr=ufmt)


    for d in derived_params:
        pr[d] = '--'

    # round everything. requires a double transpose because df.round
    # operates column-wise
    if modelid in ['simpletransit', 'alltransit']:
        round_precision = [7, 7, 5, 4, 3, 3, 3, 3]
        n_rp = len(round_precision)
        for i in range(n_fitted - n_rp):
            round_precision.append(4)
    else:
        raise NotImplementedError
    for d in derived_params:
        round_precision.append(2)

    df = df.T.round(
        decimals=dict(
            zip(df.index, round_precision)
        )
    ).T

    df['priors'] = list(pr.values())

    # units
    ud = {
        'period': 'd',
        't0': 'd',
        'log_r': '--',
        'b': '--',
        'u[0]': '--',
        'u[1]': '--',
        'r_star': r'$R_\odot$',
        'logg_star': 'cgs'
    }
    ud[f'tess_mean'] = '--'

    ud['r'] = '--'
    ud['rho_star'] = 'g$\ $cm$^{-3}$'
    ud['r_planet'] = '$R_{\mathrm{Jup}}$'
    ud['a_Rs'] = '--'
    ud['cosi'] = '--'
    ud['T_14'] = 'hr'
    ud['T_13'] = 'hr'

    df['units'] = list(ud.values())

    df = df[
        ['units', 'priors', 'median', 'mean', 'sd', 'hpd_3%', 'hpd_97%',
         'ess_mean', 'r_hat_minus1']
    ]

    latexparams = [
        #useful
        r"$P$",
        r"$t_0^{(1)}$",
        r"$\log R_{\rm p}/R_\star$",
        "$b$",
        "$u_1$",
        "$u_2$",
        "$R_\star$",
        "$\log g$"
    ]
    latexparams.append(r'$\langle f \rangle$')

    from betty.helpers import flatten
    dlatexparams = [
        r"$R_{\rm p}/R_\star$",
        r"$\rho_\star$",
        r"$R_{\rm p}$",
        "$a/R_\star$",
        '$\cos i$',
        '$T_{14}$',
        '$T_{13}$'
    ]
    latexparams = flatten([latexparams, dlatexparams])
    df.index = latexparams

    _outpath = outpath.replace('.tex', '_clean_table.csv')
    df.to_csv(_outpath, float_format='%.12f', na_rep='NaN')
    print(f'made {_outpath}')

    # df.to_latex is dumb with float formatting.
    df.to_csv(outpath, sep=',', line_terminator=' \\\\\n',
              float_format='%.12f', na_rep='NaN')

    with open(outpath, 'r') as f:
        lines = f.readlines()

    for ix, l in enumerate(lines):

        # replace commas with latex ampersands
        thisline = deepcopy(l.replace(',', ' & '))

        # replace quotes with nada
        thisline = thisline.replace('"', '')

        # replace }0 with },0
        thisline = thisline.replace('}0', '},0')
        thisline = thisline.replace('}1', '},1')
        thisline = thisline.replace('}2', '},2')

        if ix == 0:
            lines[ix] = thisline
            continue

        # iteratively replace trailing zeros with whitespace
        while re.search("0{2,10}\ ", thisline) is not None:
            r = re.search("0{2,10}\ ", thisline)
            thisline = thisline.replace(
                thisline[r.start():r.end()],
                ' '
            )

        lines[ix] = thisline

    with open(outpath, 'w') as f:
        f.writelines(lines)

    print(f'made {outpath}')




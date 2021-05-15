"""
Given the output chains from betty.modelfitter, turn them into a LaTeX
table that you can publish.
"""

import os, re, pickle
from copy import deepcopy
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pymc3 as pm
from betty.helpers import flatten

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


def invgamma_str(a, b, fmtstr=None):
    # fmtstr: e.g., "({:.5f}; {:.2f})"
    if fmtstr is None:
        return '$\mathrm{InvGamma}'+'({}; {})$'.format(a, b)
    else:
        return '$\mathrm{InvGamma}'+'{}$'.format(fmtstr).format(a, b)





##########################################
# main driver function
##########################################

def make_posterior_table(pklpath, priordict, outpath, modelid, makepdf=1,
                         overwrite=1, var_names=None):
    """
    Given the output chains from betty.modelfitter, turn them into a LaTeX
    table that you can publish.

    (Kw)args:

        pklpath: path to pickle containing PyMC3 model, samples, traces, etc.
        priordict: dictionary of priors that were used
        outpath: .tex file table will be written to
        modelid: string model identifier. used for a few path internals.
        makepdf: whether or not to compile the tex into a .pdf file to read
        var_names: variable names for pm.summary (if None, default to
            everything, which can be a bit too much)
    """

    assert outpath.endswith('.tex')

    summarypath = outpath.replace('.tex', '_raw.csv')

    scols = ['median', 'mean', 'sd', 'hdi_3%', 'hdi_97%', 'ess_bulk',
             'r_hat_minus1']

    if os.path.exists(summarypath) and not overwrite:
        df = pd.read_csv(summarypath, index_col=0)
        print(f'Reading {summarypath}')

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
            trace.posterior,
            round_to=10, kind='all',
            stat_funcs=stat_funcsdict,
            extend=True,
            var_names=var_names
        )

        df['r_hat_minus1'] = df['r_hat'] - 1

        outdf = df[scols]

        outdf.to_csv(summarypath, index=True)
        print(f'Wrote {summarypath}')

    # cleaning step...
    newindex = []
    for i in df.index:
        if i!='u[0]':
            newindex.append(i.replace('[0]',''))
        else:
            newindex.append(i)
    df.index = newindex

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

    fndict = {
        'Normal': normal_str,
        'Uniform': uniform_str,
        'LogUniform': loguniform_str,
        'InverseGamma': invgamma_str,
        'TruncatedNormal': truncnormal_str,
        'ImpactParameter': r'$\mathcal{U}(0; 1+R_{\mathrm{p}}/R_\star)$',
        'EccentricityVanEylen19': '\citet{vaneylen19}',
    }

    # fmtstring, precision, unit, latexrepr
    PARAMFMTDICT = {
        'period':('({:.5f}; {:.5f})', 7, 'd', r"$P$"),
        't0':('({:.5f}; {:.5f})', 7, 'd', r"$t_0^{(1)}$"),
        'log_r':('({:.3f}; {:.3f})', 5, '--', r"$\log R_{\rm p}/R_\star$"),
        'b':(None, 4, '--', '$b$'),
        'u[0]':('({:.3f}; {:.3f})', 3, '--', "$u_1$"),
        'u[1]':('({:.3f}; {:.3f})', 3, '--', "$u_2$"),
        'r_star':('({:.3f}; {:.3f})', 3, r'$R_\odot$', "$R_\star$"),
        'logg_star':('({:.3f}; {:.3f})', 3, 'cgs', "$\log g$"),
        'mean':('({:.3f}; {:.3f})', 4, '--', r"$\langle f \rangle$"),
        'ecc':(None, 3, '--', r"$e^{(2)}$"),
        'omega':('({:.3f}; {:.3f})', 3, 'rad', r"$\omega$"),
        'log_jitter':('({:s}; {:.3f})', 3, '--', r"$\log \sigma_f$"),
        'rho':('({:.3f}; {:.3f})', 3, 'd', r"$\rho$"),
        'sigma':('({:.3f}; {:.3f})', 3, 'd$^{-1}$', r"$\sigma$"),
        'sigma_rot':('({:.3f}; {:.3f})', 3, 'd$^{-1}$', r"$\sigma_{\mathrm{rot}}$"),
        'log_prot':('({:.3f}; {:.3f})', 3, '$\log (\mathrm{d})$', r"$\log P_{\mathrm{rot}}$"),
        'log_Q0':('({:.3f}; {:.3f})', 3, '--', r"$\log Q_0$"),
        'log_dQ':('({:.3f}; {:.3f})', 3, '--', r"$\log \mathrm{d}Q$"),
        'f':('({:.3f}; {:.3f})', 3, '--', r"$f$"),
        'r':('--', 3, '--', r"$R_{\rm p}/R_\star$"),
        'rho_star':('--', 3, 'g$\ $cm$^{-3}$', r"$\rho_\star$"),
        'r_planet':('--', 3, '$R_{\mathrm{Jup}}$', r"$R_{\rm p}$"),
        'a_Rs':('--', 3, '--', "$a/R_\star$"),
        'cosi':('--', 3, '--', '$\cos i$'),
        'T_14':('--', 3, 'hr', '$T_{14}$'),
        'T_13':('--', 3, 'hr', '$T_{13}$'),
    }

    # make a dictionary, `pr`, with keys parameter name, and values latex
    # strings to be printed to the table, e.g., '$\\mathcal{N}(7.20281;
    # 0.01000)$'. for derived parameters, entry is "--".
    pr = {}
    round_precisions, units, latexs = [], [], []
    for p in fitted_params:
        fmts = PARAMFMTDICT[p][0]
        round_precisions.append(PARAMFMTDICT[p][1])
        units.append(PARAMFMTDICT[p][2])
        latexs.append(PARAMFMTDICT[p][3])
        if fmts is not None:
            pr[p] = fndict[priordict[p][0]](
                priordict[p][1], priordict[p][2], fmts
            )
        else:
            key = priordict[p][0]
            if isinstance(key, str) and len(key)>1:
                pr[p] = fndict[key]
            elif isinstance(key, str) and len(key)==1:
                key = priordict[p]
                pr[p] = fndict[key]
            else:
                raise NotImplementedError

    for p in derived_params:
        pr[p] = PARAMFMTDICT[p][0]
        round_precisions.append(PARAMFMTDICT[p][1])
        units.append(PARAMFMTDICT[p][2])
        latexs.append(PARAMFMTDICT[p][3])

    df = df.T.round(
        decimals=dict(
            zip(df.index, round_precisions)
        )
    ).T

    df['priors'] = list(pr.values())

    df['units'] = units

    selcols = ['units', 'priors', 'median', 'mean', 'sd', 'hdi_3%', 'hdi_97%',
               'ess_bulk', 'r_hat_minus1']

    df = df[selcols]
    df.index = latexs

    _outpath = outpath.replace('.tex', '_clean_table.csv')
    df.to_csv(_outpath, float_format='%.12f', na_rep='NaN')
    print(f'made {_outpath}')

    #
    # df.to_latex is dumb with float formatting. clean it.
    #
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

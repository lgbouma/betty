"""
Given the output chains from betty.modelfitter, turn them into a LaTeX
table that you can publish.

Main driver function:
    make_posterior_table
    table_tex_to_pdf
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
logging.getLogger("filelock").setLevel(logging.ERROR)

#############
## IMPORTS ##
#############
import os, re, pickle, subprocess
from os.path import join
from copy import deepcopy
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pymc3 as pm
from betty.helpers import flatten
from astropy import units as u
from numpy import array as nparr
from shutil import copyfile

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
        LOGINFO(f'Reading {summarypath}')

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
        LOGINFO(f'Wrote {summarypath}')

    # determine whether parametrization == 'log_depth_and_b' or 'log_ror_and_b'
    if 'log_ror' in list(outdf.index) and 'b' in list(outdf.index):
        parametrization = 'log_ror_and_b'
    elif 'log_depth' in list(outdf.index) and 'b' in list(outdf.index):
        parametrization = 'log_depth_and_b'
    else:
        raise NotImplementedError

    # cleaning step: pm.summary produces a few parameters with indices that end
    # with [0] even with they should be singletons
    cleanstrs = [
        'ecc[0]', 'omega[0]', 'cosi[0]', 'sini[0]', 'T_14[0]', 'T_13[0]'
    ]
    newindex = []
    for i in df.index:
        if i in cleanstrs:
            newindex.append(i.replace('[0]',''))
        else:
            newindex.append(i)
    df.index = newindex

    # make list of fitted parameters.  Kipping2013 "u_star" parameter gets
    # replaced with "u_star[0]" and "u_star[1]".
    _fitted_params = list(priordict.keys())
    fitted_params = []
    for f in _fitted_params:
        if 'u_star' in f:
            fitted_params.append('u_star[0]')
            fitted_params.append('u_star[1]')
        else:
            fitted_params.append(f)

    n_fitted = len(fitted_params)

    if parametrization == 'log_depth_and_b':
        derived_params = [
            'depth', 'ror', 'rho_star', 'r_planet', 'a_Rs', 'cosi', 'T_14', 'T_13'
        ]
    elif parametrization == 'log_ror_and_b':
        derived_params = [
            'ror', 'rho_star', 'r_planet', 'a_Rs', 'cosi', 'T_14', 'T_13'
        ]

    n_derived = len(derived_params)

    srows = []
    for f in fitted_params:
        srows.append(f)
    for d in derived_params:
        srows.append(d)

    df = df.loc[srows]

    df = df[scols]

    LOGINFO(df)

    fndict = {
        'Normal': normal_str,
        'Uniform': uniform_str,
        'LogUniform': loguniform_str,
        'InverseGamma': invgamma_str,
        'TruncatedNormal': truncnormal_str,
        'ImpactParameter': r'$\mathcal{U}(0; 1+R_{\mathrm{p}}/R_\star)$',
        'EccentricityVanEylen19': '\citet{vaneylen19}',
        'QuadLimbDark': '\citet{exoplanet:kipping13}',
    }

    # fmtstring, precision, unit, latexrepr
    PARAMFMTDICT = {
        'period':('({:.5f}; {:.5f})', 7, 'd', r"$P$"),
        't0':('({:.5f}; {:.5f})', 7, 'd', r"$t_0^{(1)}$"),
        'log_r':('({:.3f}; {:.3f})', 5, '--', r"$\log R_{\rm p}/R_\star$"),
        'log_ror':('({:.3f}; {:.3f})', 5, '--', r"$\log R_{\rm p}/R_\star$"),
        'log_depth':('({:.4f}; {:.4f})', 4, '--', r"$\log \delta$"),
        'b':('({:.3f}; {:.3f})', 4, '--', '$b^{(2)}$') if parametrization=='log_depth_and_b' else (None, 4, '--', '$b$'),
        'u_star[0]':(None, 3, '--', "$u_1$"),
        'u_star[1]':(None, 3, '--', "$u_2$"),
        'u[0]':('({:.3f}; {:.3f})', 3, '--', "$u_1$"),
        'u[1]':('({:.3f}; {:.3f})', 3, '--', "$u_2$"),
        'r_star':('({:.3f}; {:.3f})', 3, r'$R_\odot$', "$R_\star$"),
        'logg_star':('({:.3f}; {:.3f})', 3, 'cgs', "$\log g$"),
        'ecc':(None, 3, '--', r"$e^{(3)}$"),
        'omega':('({:.3f}; {:.3f})', 3, 'rad', r"$\omega$"),
        'log_jitter':('({:s}; {:.3f})', 3, '--', r"$\log \sigma_f$"),
        'rho':('({:.3f}; {:.3f})', 3, 'd', r"$\rho$"),
        'sigma':('({:.3f}; {:.3f})', 3, 'd$^{-1}$', r"$\sigma$"),
        'sigma_rot':('({:.3f}; {:.3f})', 3, 'd$^{-1}$', r"$\sigma_{\mathrm{rot}}$"),
        'log_prot':('({:.3f}; {:.3f})', 3, '$\log (\mathrm{d})$', r"$\log P_{\mathrm{rot}}$"),
        'log_Q0':('({:.3f}; {:.3f})', 3, '--', r"$\log Q_0$"),
        'log_dQ':('({:.3f}; {:.3f})', 3, '--', r"$\log \mathrm{d}Q$"),
        'f':('({:.3f}; {:.3f})', 5, '--', r"$f$"),
        'log_f':('({:.3f}; {:.3f})', 5, '--', r"$\log f$"),
        'depth':('--', 6, '--', r"$\delta$"),
        'r':('--', 3, '--', r"$R_{\rm p}/R_\star$"),
        'ror':('--', 3, '--', r"$R_{\rm p}/R_\star$"),
        'rho_star':('--', 3, 'g$\ $cm$^{-3}$', r"$\rho_\star$"),
        'r_planet':('--', 3, '$R_{\mathrm{Jup}}$', r"$R_{\rm p}$"),
        'a_Rs':('--', 3, '--', "$a/R_\star$"),
        'cosi':('--', 3, '--', '$\cos i$'),
        'T_14':('--', 3, 'hr', '$T_{14}$'),
        'T_13':('--', 3, 'hr', '$T_{13}$'),
        'mean':('({:.3f}; {:.3f})', 4, '--', r"$\langle f \rangle$"),
        'tess_mean':('({:.3f}; {:.3f})', 4, '--', r"$\langle f \rangle$"),
        'kepler_mean':('({:.3f}; {:.3f})', 4, '--', r"$\langle f \rangle$"),
    }
    # localpolytransit, or comparable fitters, use window-specific parameters.
    n_tra_max = 500
    for i in range(n_tra_max):
        # fmtstring, precision, unit, latexrepr
        PARAMFMTDICT[f'tess_{i}_mean'] = ('({:.3f}; {:.3f})', 4, '--', r"$a_{"+str(i)+",0;TESS"+"}$")
        PARAMFMTDICT[f'kepler_{i}_mean'] = ('({:.3f}; {:.3f})', 4, '--', r"$a_{"+str(i)+",0;Kepler"+"}$")
        PARAMFMTDICT[f'tess_{i}_a1'] = ('({:.3f}; {:.3f})', 4, '--', r"$a_{"+str(i)+",1;TESS"+"}$")
        PARAMFMTDICT[f'kepler_{i}_a1'] = ('({:.3f}; {:.3f})', 4, '--', r"$a_{"+str(i)+",1;Kepler"+"}$")
        PARAMFMTDICT[f'tess_{i}_a2'] = ('({:.3f}; {:.3f})', 4, '--', r"$a_{"+str(i)+",2;TESS"+"}$")
        PARAMFMTDICT[f'kepler_{i}_a2'] = ('({:.3f}; {:.3f})', 4, '--', r"$a_{"+str(i)+",2;Kepler"+"}$")

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
            if 'u_star' in p:
                # key for 'u_star[0]' and 'u_star[1]' parameters are both
                # mapped to same prior.
                priortuple = ('QuadLimbDark',)
            else:
                priortuple = priordict[p]
            key = priortuple[0]
            if isinstance(key, str) and len(key)>1:
                pr[p] = fndict[key]
            else:
                raise NotImplementedError

    for p in derived_params:
        pr[p] = PARAMFMTDICT[p][0]
        round_precisions.append(PARAMFMTDICT[p][1])
        units.append(PARAMFMTDICT[p][2])
        latexs.append(PARAMFMTDICT[p][3])

    olddf = deepcopy(df)

    selcols0 = ['units', 'priors', 'median', 'mean', 'sd', 'hdi_3%',
               'hdi_97%', 'ess_bulk', 'r_hat_minus1']
    selcols1 = ['median', 'mean', 'sd', 'hdi_3%', 'hdi_97%']

    #
    # NOTE: using this approach was an initial big mistake. You don't actually want to do it
    # this way, because pandas dataframes suck at dealing with mixed types. The
    # thing to do here is iterate over selcols1, and for each one do a lambda
    # function like for r_hat_minus1.
    #
    # df = df[selcols1].T.round(
    #     decimals=dict(
    #         zip(df.index, round_precisions)
    #     )
    # ).T

    df = pd.DataFrame({})
    df['units'] = units
    df['priors'] = list(pr.values())
    df['ess_bulk'] = nparr(olddf.ess_bulk.apply(lambda x: f"{x:.0f}"))
    df['r_hat_minus1'] = nparr(olddf.r_hat_minus1.apply(lambda x: f"{x:.1e}"))

    for col in selcols1:
        series = []
        for param, precision in zip(olddf.index, round_precisions):
            formatter = lambda x: "{x:.{precision}f}".format(x=x, precision=precision)
            entry = formatter(olddf.loc[param, col])
            series.append(entry)
        df[col] = series

    df.index = latexs

    # fix column ordering
    df = df[selcols0]

    _outpath = outpath.replace('.tex', '_clean_table.csv')
    df.to_csv(_outpath, float_format='%.7f', na_rep='NaN')
    LOGINFO(f'made {_outpath}')

    #
    # df.to_latex is dumb with float formatting. clean it.
    #

    # add R_earth by hardcoding the insert
    rp_ind = df.index.get_loc(r'$R_{\rm p}$')
    df0 = df.iloc[ : rp_ind+1 ]
    df2 = df.iloc[ rp_ind+1 : ]

    _params = 'median,mean,sd,hdi_3%,hdi_97%'.split(',')
    df1 = deepcopy(df.iloc[rp_ind])
    for _p in _params:
        df1[_p] = np.round((float(df1[_p])*u.Rjup).to(u.Rearth).value,3)
    df1['units'] = '$R_{\mathrm{Earth}}$'

    df = pd.concat([df0, pd.DataFrame(df1).T, df2])

    df.to_csv(outpath, sep=',', line_terminator=' \\\\\n',
              float_format='%.7f', na_rep='NaN')

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

        lines[ix] = thisline

    with open(outpath, 'w') as f:
        f.writelines(lines)

    LOGINFO(f'made {outpath}')


def table_tex_to_pdf(tex_table_path, pdf_path):
    """
    Given the LaTeX file made by make_posterior_table, compile it into a PDF
    file using pdflatex.
    """
    from betty.paths import TEXDATADIR

    headpath = join(TEXDATADIR, 'table_header.tex')
    tailpath = join(TEXDATADIR, 'table_tail.tex')
    with open(headpath, 'r') as f:
        headlines = f.readlines()
    with open(tailpath, 'r') as f:
        taillines = f.readlines()

    with open(tex_table_path, 'r') as f:
        midlines = f.readlines()

    # drop the header
    midlines = midlines[1:]
    # fix the citation since the output tex file does not have a bibliography
    midlines = [m.replace('\citet{exoplanet:kipping13}', 'Kipping 2013')
                for m in midlines]

    # python's list concatenation works with +
    outlines = headlines + midlines + taillines

    outdir = os.path.dirname(pdf_path)
    temp_texpath = join(
        outdir, 'source_'+os.path.basename(pdf_path).replace('.pdf','.tex')
    )
    with open(temp_texpath, 'w') as f:
        f.writelines(outlines)
    LOGINFO(f"Wrote {temp_texpath}")

    src_clspath = join(TEXDATADIR, 'aastex63.cls')
    dst_clspath = join(os.getcwd(), 'aastex63.cls')
    copyfile(src_clspath, dst_clspath)

    import socket
    if 'phtess' in socket.gethostname():
        # NOTE: better would be to test whether revtex4-1 is accessible on the
        # system.  However this is fine for now given that I am the only user!
        LOGWARNING(
            f'{socket.gethostname()} has a janky latex installation. ' +
            'Copying extra revtex4-1 files temporarily...'
        )
        fnames = [
            "aip4-1.rtx", "aps10pt4-1.rtx", "aps11pt4-1.rtx", "aps12pt4-1.rtx",
            "aps4-1.rtx", "apsrmp4-1.rtx", "ltxdocext.sty", "ltxfront.sty",
            "ltxgrid.sty", "ltxutil.sty", "revsymb4-1.sty", "revtex4-1.cls"
        ]
        for f in fnames:
            src_path = join(TEXDATADIR, f)
            dst_path = join(os.getcwd(), f)
            copyfile(src_path, dst_path)

    bash_command = f"pdflatex -output-directory {outdir} {temp_texpath}"
    LOGINFO(f"Running {bash_command}")
    proc = subprocess.run(
        bash_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    if 'phtess' in socket.gethostname():
        for f in fnames:
            dst_path = join(os.getcwd(), f)
            os.remove(dst_path)

    cwd = os.getcwd()
    bash_command = f"rm {outdir}/*aux {outdir}/*log {outdir}/*out {cwd}/aastex63.cls"
    LOGINFO(f"Running {bash_command}")
    proc = subprocess.run(
        bash_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        shell=True
    )

    _pdfpath = join(
        outdir, 'source_'+os.path.basename(pdf_path)
    )

    if os.path.exists(_pdfpath):
        os.rename(_pdfpath, pdf_path)
        LOGINFO(f"Made {pdf_path}.")

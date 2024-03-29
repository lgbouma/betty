"""
Full MAP and sampling recipes. Contents include:

ModelParser
ModelFitter
    run_simpletransit_inference
    run_localpolytransit_inference
    run_RotStochGPtransit_inference
    run_RotGPtransit_inference
    run_QuadMulticolorTransit_inference
    run_QuadMulticolorVaryDepthTransit_inference
    run_rvspotorbit_inference

Not yet implemented here (but see /timmy/):
    run_alltransit_inference
    run_rvorbit_inference
    run_allindivtransit_inference
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
import numpy as np, matplotlib.pyplot as plt, pandas as pd
import pymc3 as pm
import pymc3_ext as pmx
import pickle, os
from astropy import units as units, constants as const
from numpy import array as nparr
from functools import partial
from collections import OrderedDict

import exoplanet as xo
from celerite2.theano import terms, GaussianProcess
import aesara_theano_fallback.tensor as tt

from betty.constants import factor

class ModelParser:

    def __init__(self, modelid):
        self.initialize_model(modelid)

    def initialize_model(self, modelid):
        self.modelid = modelid
        self.modelcomponents = modelid.split('_')
        self.verify_modelcomponents()

    def verify_modelcomponents(self):

        validcomponents = ['simpletransit', 'localpolytransit', 'rvspotorbit',
                           'RotStochGPtransit', 'RotGPtransit',
                           'allindivtransit', 'alltransit',
                           'QuadMulticolorTransit',
                           'QuadMulticolorVaryDepthTransit']

        assert len(self.modelcomponents) >= 1

        for modelcomponent in self.modelcomponents:
            if modelcomponent not in validcomponents:
                errmsg = (
                    f'Got modelcomponent {modelcomponent}. '+
                    f'validcomponents include {validcomponents}.'
                )
                raise ValueError(errmsg)


class ModelFitter(ModelParser):
    """
    Given a modelid of the form "*transit", "rv_*" and a dataframe containing
    (time and flux), or (time and rv), run the inference.
    """

    def __init__(self, modelid, data_df, priordict, N_samples=2000, N_cores=16,
                 N_chains=4, plotdir=None, pklpath=None,
                 overwrite=1, map_optimization_method=None):

        self.N_samples = N_samples
        self.N_cores = N_cores
        self.N_chains = N_chains
        self.PLOTDIR = plotdir
        self.OVERWRITE = overwrite

        implemented_models = [
            'simpletransit', 'localpolytransit', 'allindivtransit',
            'oddindivtransit', 'evenindivtransit', 'rvorbit', 'rvspotorbit',
            'alltransit', 'RotStochGPtransit', 'RotGPtransit',
            'QuadMulticolorTransit', 'QuadMulticolorVaryDepthTransit'
        ]

        if modelid in implemented_models:
            assert isinstance(data_df, OrderedDict)
            self.data = data_df
            self.priordict = priordict

        if 'rv' in modelid:

            if modelid == 'rvorbit':
                raise NotImplementedError(
                    "i haven't implented a normal keplerian orbit yet"
                )

            elif modelid == 'rvspotorbit':
                pass

        self.initialize_model(modelid)

        if modelid not in implemented_models:
            raise NotImplementedError

        # hard-code against thread safety. (PyMC3 + matplotlib).
        make_threadsafe = False

        updatestr = (
            f'Beginning inference run for {modelid} '
            f'(N_samples={N_samples})...'
        )
        LOGINFO(updatestr)
        if modelid == 'simpletransit':
            self.run_simpletransit_inference(
                pklpath, make_threadsafe=make_threadsafe
            )

        elif modelid == 'QuadMulticolorTransit':
            self.run_QuadMulticolorTransit_inference(
                pklpath, make_threadsafe=make_threadsafe
            )

        elif modelid == 'QuadMulticolorVaryDepthTransit':
            self.run_QuadMulticolorVaryDepthTransit_inference(
                pklpath, make_threadsafe=make_threadsafe
            )

        elif modelid == 'RotStochGPtransit':
            self.run_RotStochGPtransit_inference(
                pklpath, make_threadsafe=make_threadsafe
            )

        elif modelid == 'RotGPtransit':
            self.run_RotGPtransit_inference(
                pklpath, make_threadsafe=make_threadsafe,
                map_optimization_method=map_optimization_method
            )

        elif modelid == 'localpolytransit':
            self.run_localpolytransit_inference(
                pklpath, make_threadsafe=make_threadsafe,
            )

        elif modelid == 'rvorbit':
            self.run_rvorbit_inference(
                pklpath, make_threadsafe=make_threadsafe
            )

        elif modelid == 'rvspotorbit':
            self.run_rvspotorbit_inference(
                pklpath, make_threadsafe=make_threadsafe
            )

        elif modelid in ['alltransit', 'alltransit_quad',
                         'alltransit_quaddepthvar']:
            self.run_alltransit_inference(
                pklpath, make_threadsafe=make_threadsafe
            )

        elif modelid in ['allindivtransit', 'oddindivtransit',
                         'evenindivtransit']:
            self.run_allindivtransit_inference(
                pklpath, make_threadsafe=make_threadsafe
            )

        LOGINFO(f'Finished PyMC3 for {modelid}...')


    def run_simpletransit_inference(self, pklpath, make_threadsafe=True):
        """
        Fit transit data for an Agol+19 transit. (Ignores any stellar
        variability).  Free parameters are {"period", "t0", "log_ror", "b",
        "u0", "u1", "log_jitter", "logg_star", "r_star"}.
        """

        p = self.priordict

        # if the model has already been run, pull the result from the
        # pickle. otherwise, run it.
        if os.path.exists(pklpath) and not self.OVERWRITE:
            LOGINFO(f'Found {pklpath}, loading from cache.')
            d = pickle.load(open(pklpath, 'rb'))
            self.model = d['model']
            self.trace = d['trace']
            self.map_estimate = d['map_estimate']
            return 1
        elif os.path.exists(pklpath) and self.OVERWRITE:
            LOGINFO(f'Found {pklpath}, and OVERWRITE is True. '
                    'Deleting and proceeding.')
            os.remove(pklpath)

        # assuming single instrument, get the data, and the instrument name
        assert len(self.data.keys()) == 1
        name = list(self.data.keys())[0]
        x,y,yerr,texp = self.data[name]

        with pm.Model() as model:

            # Shared parameters

            # Stellar parameters. (Following tess.world notebooks).
            logg_star = pm.Normal(
                "logg_star", mu=p['logg_star'][1], sd=p['logg_star'][2]
            )

            r_star = pm.Bound(pm.Normal, lower=0.0)(
                "r_star", mu=p['r_star'][1], sd=p['r_star'][2]
            )
            rho_star = pm.Deterministic(
                "rho_star", factor*10**logg_star / r_star
            )

            # A jitter term describing excess white noise
            log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(yerr)),
                                   sd=p['log_jitter'][2])

            # fix Rp/Rs across bandpasses
            if p['log_ror'][0] == 'Uniform':
                log_ror = pm.Uniform('log_ror', lower=p['log_ror'][1],
                                   upper=p['log_ror'][2], testval=p['log_ror'][3])
            else:
                raise NotImplementedError
            ror = pm.Deterministic('ror', tt.exp(log_ror))
            # planet radius in the same units as r_star
            r_pl = pm.Deterministic(
                "r_pl", ror*r_star
            )

            # Some orbital parameters
            t0 = pm.Normal(
                "t0", mu=p['t0'][1], sd=p['t0'][2], testval=p['t0'][1]
            )
            period = pm.Normal(
                'period', mu=p['period'][1], sd=p['period'][2],
                testval=p['period'][1]
            )
            b = xo.distributions.ImpactParameter(
                "b", ror=ror, testval=p['b'][1]
            )

            orbit = xo.orbits.KeplerianOrbit(
                period=period, t0=t0, b=b, rho_star=rho_star, r_star=r_star
            )

            # limb-darkening
            if 'u[0]' in p.keys() and 'u[1]' in p.keys():
                u0 = pm.Uniform(
                    'u[0]', lower=p['u[0]'][1],
                    upper=p['u[0]'][2],
                    testval=p['u[0]'][3]
                )
                u1 = pm.Uniform(
                    'u[1]', lower=p['u[1]'][1],
                    upper=p['u[1]'][2],
                    testval=p['u[1]'][3]
                )
                u_star = [u0, u1]

            else:
                assert 'u_star' in p.keys()
                assert p['u_star'][0] == 'QuadLimbDark'
                u_star = xo.QuadLimbDark("u_star")

            star = xo.LimbDarkLightCurve(u_star)

            # Loop over "instruments" (TESS, then each ground-based lightcurve)
            parameters = dict()
            lc_models = dict()
            roughdepths = dict()

            for n, (name, (x, y, yerr, texp)) in enumerate(self.data.items()):

                # Define per-instrument parameters in a submodel, to not need
                # to prefix the names. Yields e.g., "TESS_mean",
                # "elsauce_0_mean", "elsauce_2_a2"
                with pm.Model(name=name, model=model):

                    # Transit parameters.
                    mean = pm.Normal(
                        "mean", mu=p[f'{name}_mean'][1],
                        sd=p[f'{name}_mean'][2], testval=p[f'{name}_mean'][1]
                    )


                if self.modelid == 'simpletransit':
                    transit_lc = star.get_light_curve(
                        orbit=orbit, r=r_pl, t=x, texp=texp
                    ).T.flatten()

                    lc_models[name] = pm.Deterministic(
                        f'{name}_mu_transit',
                        mean +
                        transit_lc
                    )

                    roughdepths[name] = pm.Deterministic(
                        f'{name}_roughdepth',
                        pm.math.abs_(transit_lc).max()
                    )

                else:
                    raise NotImplementedError

                likelihood = pm.Normal(
                    f'{name}_obs',
                    mu=lc_models[name],
                    sigma=pm.math.sqrt(
                        yerr**2 + tt.exp(2 * log_jitter)
                    ),
                    observed=y
                )

            #
            # Derived parameters
            #

            # planet radius in jupiter radii
            r_planet = pm.Deterministic(
                "r_planet", (ror*r_star)*( 1*units.Rsun/(1*units.Rjup) ).cgs.value
            )

            #
            # eq 30 of winn+2010, ignoring planet density.
            #
            a_Rs = pm.Deterministic(
                "a_Rs",
                (rho_star * period**2)**(1/3)
                *
                (( (1*units.gram/(1*units.cm)**3) * (1*units.day**2)
                  * const.G / (3*np.pi)
                )**(1/3)).cgs.value
            )

            #
            # cosi. assumes e=0 (e.g., Winn+2010 eq 7)
            #
            cosi = pm.Deterministic("cosi", b / a_Rs)

            # probably safer than tt.arccos(cosi)
            sini = pm.Deterministic("sini", pm.math.sqrt( 1 - cosi**2 ))

            #
            # transit durations (T_14, T_13) for circular orbits. Winn+2010 Eq 14, 15.
            # units: hours.
            #
            T_14 = pm.Deterministic(
                'T_14',
                (period/np.pi)*
                tt.arcsin(
                    (1/a_Rs) * pm.math.sqrt( (1+ror)**2 - b**2 )
                    * (1/sini)
                )*24
            )

            T_13 =  pm.Deterministic(
                'T_13',
                (period/np.pi)*
                tt.arcsin(
                    (1/a_Rs) * pm.math.sqrt( (1-ror)**2 - b**2 )
                    * (1/sini)
                )*24
            )

            # Optimizing
            map_estimate = pm.find_MAP(model=model)

            if make_threadsafe:
                pass
            else:
                # NOTE: would usually plot MAP estimate here, but really
                # there's not a huge need.
                LOGINFO(map_estimate)
                pass

            LOGINFO('Got MAP estimate. Beginning sampling...')
            # sample from the posterior defined by this model.
            trace = pmx.sample(
                tune=self.N_samples, draws=self.N_samples,
                start=map_estimate, cores=self.N_cores,
                chains=self.N_chains,
                return_inferencedata=True,
                target_accept=0.9
            )

        with open(pklpath, 'wb') as buff:
            pickle.dump({'model': model, 'trace': trace,
                         'map_estimate': map_estimate}, buff)

        self.model = model
        self.trace = trace
        self.map_estimate = map_estimate


    def run_RotStochGPtransit_inference(self, pklpath, make_threadsafe=True):
        """
        Fit light curve for an Agol+19 transit, + a GP for the stellar
        variability, simultaneously.  Assumes single instrument.  The GP is a
        RotationTerm with a stochastic (SHOTerm) component.

        The stochastic term of the GP kernel has Q fixed to 1/3 for
        magical reasons not explained in the docs.

        Fits for:

            [f, log_dQ, log_Q0, log_prot, sigma_rot, rho, sigma, log_jitter,
            ecs, b, period, t0, log_ror, u[1], u[0], r_star, logg_star, mean]
        """

        p = self.priordict

        # if the model has already been run, pull the result from the
        # pickle. otherwise, run it.
        if os.path.exists(pklpath):
            LOGINFO(f'Found {pklpath}, loading from cache.')
            d = pickle.load(open(pklpath, 'rb'))
            self.model = d['model']
            self.trace = d['trace']
            self.map_estimate = d['map_estimate']
            return 1

        def build_model(mask=None, start=None):

            # assuming single instrument
            assert len(self.data.keys()) == 1
            name = list(self.data.keys())[0]
            x,y,yerr,texp = self.data[name]

            nplanets = 1

            if mask is None:
                mask = np.ones(len(x), dtype=bool)

            with pm.Model() as model:

                # Shared parameters
                mean = pm.Normal(
                    "mean", mu=p[f'mean'][1], sd=p[f'mean'][2],
                    testval=p[f'mean'][1]
                )

                # Stellar parameters.
                logg_star = pm.Normal(
                    "logg_star", mu=p['logg_star'][1], sd=p['logg_star'][2]
                )

                r_star = pm.Bound(pm.Normal, lower=0.0)(
                    "r_star", mu=p['r_star'][1], sd=p['r_star'][2]
                )
                rho_star = pm.Deterministic(
                    "rho_star", factor*10**logg_star / r_star
                )

                # limb-darkening. adopt uniform, rather than Kipping 2013 -- i.e., take
                # an "informed prior" approach.
                u_star = xo.QuadLimbDark("u_star")

                # fix Rp/Rs across bandpasses
                if p['log_ror'][0] == 'Uniform':
                    log_ror = pm.Uniform('log_ror', lower=p['log_ror'][1],
                                       upper=p['log_ror'][2], testval=p['log_ror'][3])
                else:
                    raise NotImplementedError
                ror = pm.Deterministic('ror', tt.exp(log_ror))
                # planet radius in the same units as r_star
                r_pl = pm.Deterministic(
                    "r_pl", ror*r_star
                )

                # orbital parameters for planet (single)
                t0 = pm.Normal(
                    "t0", mu=p['t0'][1], sd=p['t0'][2], testval=p['t0'][1]
                )
                period = pm.Normal(
                    'period', mu=p['period'][1], sd=p['period'][2],
                    testval=p['period'][1]
                )
                b = xo.distributions.ImpactParameter(
                    "b", ror=ror, testval=p['b'][1]
                )

                # eccentricity
                ecs = pmx.UnitDisk(
                    "ecs", shape=(2, nplanets),
                    testval=0.01 * np.ones((2, nplanets))
                )
                ecc = pm.Deterministic(
                    "ecc",
                    tt.sum(ecs ** 2, axis=0)
                )
                omega = pm.Deterministic(
                    "omega", tt.arctan2(ecs[1], ecs[0])
                )
                xo.eccentricity.vaneylen19(
                    "ecc_prior",
                    multi=False, shape=nplanets, fixed=True, observed=ecc
                )

                # orbit model
                orbit = xo.orbits.KeplerianOrbit(
                    period=period,
                    t0=t0,
                    b=b,
                    rho_star=rho_star,
                    r_star=r_star,
                    ecc=ecc,
                    omega=omega
                )

                star = xo.LimbDarkLightCurve(u_star)

                # NOTE: could loop over instruments here... (e.g., TESS, keplerllc,
                # keplersc, ground-based instruments...). Instead, opt for simpler
                # single-instrument approach.

                # Compute the model light curve
                light_curves = pm.Deterministic(
                    "light_curves",
                    star.get_light_curve(
                        orbit=orbit, r=r_pl, t=x[mask], texp=texp
                    )
                )

                # Line that adds the transit models of different planets in the system,
                # if relevant
                light_curve = pm.math.sum(light_curves, axis=-1) + mean
                resid = y[mask] - light_curve

                # Use the GP model from the stellar variability tutorial
                # https://gallery.exoplanet.codes/en/latest/tutorials/stellar-variability/
                # Literally the same prior parameters too.

                # A jitter term describing excess white noise
                log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(yerr)),
                                       sd=p['log_jitter'][2])

                #A term to describe the non-periodic variability
                sigma = pm.InverseGamma(
                    "sigma",
                    **pmx.estimate_inverse_gamma_parameters(
                        p['sigma'][1], p['sigma'][2]
                    )
                )

                # NOTE: default recommended by DFM is InvGamma(0.5, 2). But in
                # our model of [non-periodic SHO term] X [periodic SHO terms at
                # Prot and 0.5*Prot] this can produce tension against the
                # RotationTerm, leading to overfitting -- preferring to explain
                # the transits away as noise rather than signal. So, take a
                # uniform prior, U[1,10].  This roughly corresponds to the
                # "stochastically-driven, damped harmonic oscillator" in the
                # non-period SHO term

                rho = pm.Uniform("rho", lower=p['rho'][1], upper=p['rho'][2])
                #rho = pm.InverseGamma(
                #    "rho", **pmx.estimate_inverse_gamma_parameters(
                #        p['rho'][1], p['rho'][2]
                #    )
                #)

                # The parameters of the RotationTerm kernel
                sigma_rot = pm.InverseGamma(
                    "sigma_rot", **pmx.estimate_inverse_gamma_parameters(
                        p['sigma_rot'][1], p['sigma_rot'][2]
                    )
                )
                log_prot = pm.Normal("log_prot", mu=p['log_prot'][1],
                                     sd=p['log_prot'][2])
                prot = pm.Deterministic("prot", tt.exp(log_prot))
                log_Q0 = pm.Normal(
                    "log_Q0", mu=p['log_Q0'][1], sd=p['log_Q0'][2]
                )
                log_dQ = pm.Normal(
                    "log_dQ", mu=p['log_dQ'][1], sd=p['log_dQ'][2]
                )
                f = pm.Uniform(
                    "f", lower=p['f'][1], upper=p['f'][2]
                )

                # Set up the Gaussian Process model. See
                # https://celerite2.readthedocs.io/en/latest/tutorials/first/
                # for intro.
                # Non-periodic term
                kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=1/3.0)
                # Quasiperiodic term
                kernel += terms.RotationTerm(
                    sigma=sigma_rot,
                    period=prot,
                    Q0=tt.exp(log_Q0),
                    dQ=tt.exp(log_dQ),
                    f=f,
                )
                gp = GaussianProcess(
                    kernel,
                    t=x[mask],
                    diag=yerr[mask]**2 + tt.exp(2 * log_jitter),
                    mean=mean,
                    quiet=True,
                )

                # Compute the Gaussian Process likelihood and add it into the
                # the PyMC3 model as a "potential"
                gp.marginal("transit_obs", observed=resid)

                # Compute the mean model prediction for plotting purposes
                pm.Deterministic("gp_pred", gp.predict(resid))

                #
                # Begin: derived parameters
                #

                # planet radius in jupiter radii
                r_planet = pm.Deterministic(
                    "r_planet", (ror*r_star)*( 1*units.Rsun/(1*units.Rjup) ).cgs.value
                )

                #
                # eq 30 of winn+2010, ignoring planet density.
                #
                a_Rs = pm.Deterministic(
                    "a_Rs",
                    (rho_star * period**2)**(1/3)
                    *
                    (( (1*units.gram/(1*units.cm)**3) * (1*units.day**2)
                      * const.G / (3*np.pi)
                    )**(1/3)).cgs.value
                )

                #
                # cosi. assumes e!=0 (e.g., Winn+2010 eq 7)
                #
                cosi = pm.Deterministic(
                    "cosi", (b / a_Rs)/( (1-ecc**2)/(1+ecc*pm.math.sin(omega)) )
                )

                # safer than tt.arccos(cosi)
                sini = pm.Deterministic("sini", pm.math.sqrt( 1 - cosi**2 ))

                #
                # transit durations (T_14, T_13) for circular orbits. Winn+2010 Eq 14, 15.
                # units: hours.
                #
                T_14 = pm.Deterministic(
                    'T_14',
                    (period/np.pi)*
                    tt.arcsin(
                        (1/a_Rs) * pm.math.sqrt( (1+ror)**2 - b**2 )
                        * (1/sini)
                    )*24
                    *(
                        pm.math.sqrt(1-ecc**2)/(1+ecc*pm.math.sin(omega))
                    )
                )

                T_13 =  pm.Deterministic(
                    'T_13',
                    (period/np.pi)*
                    tt.arcsin(
                        (1/a_Rs) * pm.math.sqrt( (1-ror)**2 - b**2 )
                        * (1/sini)
                    )*24
                    *(
                        pm.math.sqrt(1-ecc**2)/(1+ecc*pm.math.sin(omega))
                    )
                )

                #
                # End: derived parameters
                #

                # Fit for the maximum a posteriori parameters. This worked best
                # (for "rudolf" project) when doing the GP parameters first,
                # then the transit parameters. "All at once"
                # (`pmx.optimize(start=start)`) did not work very well; a few
                # other combinations of parameters did not work very well
                # either. Using the plotting diagnostics below helped this.

                # NOTE: Method 1: don't do full optimization.
                np.random.seed(42)
                if start is None:
                    start = model.test_point

                map_soln = start

                # Transit: [log_r, b, t0, period, r_star, logg_star, ecs,
                # u_star]...]
                map_soln = pmx.optimize(
                    start=map_soln,
                    vars=[log_ror, b, ecc, omega, t0, period, r_star, logg_star,
                          u_star]
                )

                # All GP terms + jitter and mean.
                map_soln = pmx.optimize(
                    start=map_soln,
                    vars=[sigma_rot, prot, log_Q0, log_dQ, f, sigma, rho,
                          log_jitter, mean]
                )

                # # Full optimization
                map_soln = pmx.optimize(start=map_soln)


            return model, map_soln

        model, map_estimate = build_model()

        if make_threadsafe:
            pass
        else:
            # NOTE: would usually plot MAP estimate here, but really
            # there's not a huge need.
            LOGINFO(map_estimate)
            pass

        from betty import plotting as bp

        outpath = os.path.join(
            self.PLOTDIR,
            'flux_vs_time_map_estimate_RotStochGPtransit.png'
        )
        bp.plot_light_curve(self.data, map_estimate, outpath)

        outpath = os.path.join(
            self.PLOTDIR,
            'flux_vs_phase_map_estimate_RotStochGPtransit.png'
        )
        bp.plot_phased_light_curve_gptransit(
            self.data, map_estimate, outpath
        )

        LOGINFO('Got MAP estimate. Beginning sampling...')
        # sample from the posterior defined by this model.

        #
        # NOTE: some of Dan's tutorials do sigma-clipping at this point. If the
        # MAP estimate looks good enough... no need!
        #

        with model:
            trace = pmx.sample(
                tune=self.N_samples, draws=self.N_samples,
                start=map_estimate, cores=self.N_cores,
                chains=self.N_chains,
                return_inferencedata=True,
                initial_accept=0.8,
                target_accept=0.95
            )

        with open(pklpath, 'wb') as buff:
            pickle.dump({'model': model, 'trace': trace,
                         'map_estimate': map_estimate}, buff)

        self.model = model
        self.trace = trace
        self.map_estimate = map_estimate


    def run_RotGPtransit_inference(self, pklpath, make_threadsafe=True,
                                   map_optimization_method=None):
        """
        Fit light curve for an Agol+19 transit, + a GP for the stellar
        variability, simultaneously.  Assumes single instrument.  The GP is a
        RotationTerm (Prot and 0.5xProt) only.

        map_optimization_method is by default None, which optimizes everything
        simultaneously.  It can also be any "_"-separated combination of the
        strings "transit", "gpJitterMean", "RotTerm", "LimbDark", "JitterMean",
        "FullOptimize", and "bonus" ("SHOterm" is not in this model).  For
        instance, "transit_gpJitterMean_FullOptimize" will attempt to optimize
        the transit parameters, then all the RotTerm GP parameters + jitter and
        mean, then all parameters simultaneously.  For a few tested cases,
        'RotationTerm_transit_FullOptimize' was preferred.

        Fits for:

            [f, log_dQ, log_Q0, log_prot, sigma_rot, log_jitter, ecs, b,
            period, t0, log_ror (or log_depth), u[1], u[0], r_star, logg_star,
            mean]

            NOTE: log_ror can be substituted with log_depth in the prior for the
            "alternative" parametrization.  In this case, the impact parameter
            prior must be uniform (from 0 to 1).
        """

        p = self.priordict

        # At most one of these parametrizations can be used.
        var0 = 'log_depth' in p.keys()
        var1 = 'log_ror' in p.keys()
        assert int(var0) + int(var1) == 1

        # if the model has already been run, pull the result from the
        # pickle. otherwise, run it.
        if os.path.exists(pklpath):
            LOGINFO(f'Found {pklpath}, loading from cache.')
            d = pickle.load(open(pklpath, 'rb'))
            self.model = d['model']
            self.trace = d['trace']
            self.map_estimate = d['map_estimate']
            return 1

        def build_model(mask=None, start=None):

            # assuming single instrument
            assert len(self.data.keys()) == 1
            name = list(self.data.keys())[0]
            x,y,yerr,texp = self.data[name]

            nplanets = 1

            if mask is None:
                mask = np.ones(len(x), dtype=bool)

            with pm.Model() as model:

                # Shared parameters
                mean = pm.Normal(
                    "mean", mu=p[f'mean'][1], sd=p[f'mean'][2],
                    testval=p[f'mean'][1]
                )

                # Stellar parameters.
                logg_star = pm.Normal(
                    "logg_star", mu=p['logg_star'][1], sd=p['logg_star'][2]
                )

                r_star = pm.Bound(pm.Normal, lower=0.0)(
                    "r_star", mu=p['r_star'][1], sd=p['r_star'][2]
                )
                rho_star = pm.Deterministic(
                    "rho_star", factor*10**logg_star / r_star
                )

                # limb-darkening: adopt Kipping 2013.
                u_star = xo.QuadLimbDark("u_star")
                star = xo.LimbDarkLightCurve(u_star)

                if 'log_ror' in p.keys():

                    # "Classical parametrization": fit log(Rp/R*), and allow the
                    # correct impact parameter prior from 0 to 1+Rp/R*.
                    # This fixes Rp/Rs across bandpasses.
                    if p['log_ror'][0] == 'Uniform':
                        log_ror = pm.Uniform('log_ror', lower=p['log_ror'][1],
                                             upper=p['log_ror'][2],
                                             testval=p['log_ror'][3])
                    else:
                        raise NotImplementedError
                    ror = pm.Deterministic('ror', tt.exp(log_ror))

                    assert p['b'][0] == 'ImpactParameter'
                    b = xo.distributions.ImpactParameter(
                        "b", ror=ror, testval=p['b'][1]
                    )

                elif 'log_depth' in p.keys():

                    # Alternative parametrization: fit log(depth).  Requires
                    # impact parameter prior from 0 to 1, because otherwise
                    # there's a sqrt(1-b^2) that doesn't make sense, per
                    # https://github.com/exoplanet-dev/exoplanet/blob/e99d1bd68654f21efbbf8400a83889a470d2baf7/src/exoplanet/light_curves/limb_dark.py#L73

                    assert p['b'][0] == 'Uniform'
                    b = pm.Uniform("b", lower=0, upper=1)

                    log_depth = pm.Normal(
                        "log_depth",
                        mu=p['log_depth'][1],
                        sigma=p['log_depth'][2]
                    )
                    depth = pm.Deterministic(
                        'depth', tt.exp(log_depth)
                    )
                    ror = pm.Deterministic(
                        "ror",
                        star.get_ror_from_approx_transit_depth(
                            depth, b
                        ),
                    )

                else:
                    raise NotImplementedError

                # planet radius in the same units as r_star
                r_pl = pm.Deterministic(
                    "r_pl", ror*r_star
                )

                # orbital parameters for planet (single)
                t0 = pm.Normal(
                    "t0", mu=p['t0'][1], sd=p['t0'][2], testval=p['t0'][1]
                )
                period = pm.Normal(
                    'period', mu=p['period'][1], sd=p['period'][2],
                    testval=p['period'][1]
                )

                # eccentricity
                ecs = pmx.UnitDisk(
                    "ecs", shape=(2, nplanets),
                    testval=0.01 * np.ones((2, nplanets))
                )
                ecc = pm.Deterministic(
                    "ecc",
                    tt.sum(ecs ** 2, axis=0)
                )
                omega = pm.Deterministic(
                    "omega", tt.arctan2(ecs[1], ecs[0])
                )
                xo.eccentricity.vaneylen19(
                    "ecc_prior",
                    multi=False, shape=nplanets, fixed=True, observed=ecc
                )

                # orbit model
                orbit = xo.orbits.KeplerianOrbit(
                    period=period,
                    t0=t0,
                    b=b,
                    rho_star=rho_star,
                    r_star=r_star,
                    ecc=ecc,
                    omega=omega
                )

                light_curve = mean + tt.sum(
                    star.get_light_curve(orbit=orbit, r=r_pl, t=x[mask], texp=texp), axis=-1
                )
                resid = y[mask] - light_curve

                light_curves = pm.Deterministic(
                    "light_curves",
                    star.get_light_curve(
                        orbit=orbit, r=r_pl, t=x[mask], texp=texp
                    )
                )

                # A jitter term describing excess white noise
                log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(yerr)),
                                       sd=p['log_jitter'][2])

                # The parameters of the RotationTerm kernel
                sigma_rot = pm.InverseGamma(
                    "sigma_rot", **pmx.estimate_inverse_gamma_parameters(
                        p['sigma_rot'][1], p['sigma_rot'][2]
                    )
                )
                log_prot = pm.Normal(
                    "log_prot", mu=p['log_prot'][1], sd=p['log_prot'][2]
                )
                prot = pm.Deterministic("prot", tt.exp(log_prot))
                log_Q0 = pm.Normal(
                    "log_Q0", mu=p['log_Q0'][1], sd=p['log_Q0'][2]
                )
                log_dQ = pm.Normal(
                    "log_dQ", mu=p['log_dQ'][1], sd=p['log_dQ'][2]
                )
                fvarname = 'log_f' if 'log_f' in p.keys() else 'f'
                if fvarname == 'log_f':
                    log_f = pm.Uniform(
                        "log_f", lower=p['log_f'][1], upper=p['log_f'][2],
                        testval=p['log_f'][3]
                    )
                    f = pm.Deterministic("f", tt.exp(log_f))
                elif fvarname == 'f':
                    f = pm.Uniform(
                        "f", lower=p['f'][1], upper=p['f'][2]
                    )

                # Set up the Gaussian Process model. See
                # https://celerite2.readthedocs.io/en/latest/tutorials/first/
                # for intro.
                # Non-periodic term
                #kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=1/3.0)
                # Quasiperiodic term
                kernel = terms.RotationTerm(
                    sigma=sigma_rot,
                    period=prot,
                    Q0=tt.exp(log_Q0),
                    dQ=tt.exp(log_dQ),
                    f=f,
                )
                #
                # note mean of the GP is ~zero, since it only predicts the
                # stellar variability (which is then added in).  but not
                # exactly.  Total = median+transit+GP.  But if you let the GP
                # mean below be 0, you'd be saying "Total = mean+transit+GP",
                # which would be incorrect.  So, you subtract the mean, and add
                # the median.
                #
                gp = GaussianProcess(
                    kernel,
                    t=x[mask],
                    diag=yerr[mask]**2 + tt.exp(2 * log_jitter),
                    quiet=True,
                )

                # Compute the Gaussian Process likelihood and add it into the
                # the PyMC3 model as a "potential"
                gp.marginal("transit_obs", observed=resid)

                # Compute the mean model prediction for plotting purposes
                pm.Deterministic(
                    "gp_pred", gp.predict(resid)
                )

                #
                # Begin: derived parameters
                #

                # planet radius in jupiter radii
                r_planet = pm.Deterministic(
                    "r_planet", (ror*r_star)*( 1*units.Rsun/(1*units.Rjup) ).cgs.value
                )

                #
                # eq 30 of winn+2010, ignoring planet density.
                #
                a_Rs = pm.Deterministic(
                    "a_Rs",
                    (rho_star * period**2)**(1/3)
                    *
                    (( (1*units.gram/(1*units.cm)**3) * (1*units.day**2)
                      * const.G / (3*np.pi)
                    )**(1/3)).cgs.value
                )

                #
                # cosi. assumes e!=0 (e.g., Winn+2010 eq 7)
                #
                cosi = pm.Deterministic(
                    "cosi", (b / a_Rs)/( (1-ecc**2)/(1+ecc*pm.math.sin(omega)) )
                )

                # safer than tt.arccos(cosi)
                sini = pm.Deterministic("sini", pm.math.sqrt( 1 - cosi**2 ))

                #
                # transit durations (T_14, T_13) for circular orbits. Winn+2010
                # Eq 14, 15.  units: hours.
                #
                T_14 = pm.Deterministic(
                    'T_14',
                    (period/np.pi)*
                    tt.arcsin(
                        (1/a_Rs) * pm.math.sqrt( (1+ror)**2 - b**2 )
                        * (1/sini)
                    )*24
                    *(
                        pm.math.sqrt(1-ecc**2)/(1+ecc*pm.math.sin(omega))
                    )
                )

                T_13 =  pm.Deterministic(
                    'T_13',
                    (period/np.pi)*
                    tt.arcsin(
                        (1/a_Rs) * pm.math.sqrt( (1-ror)**2 - b**2 )
                        * (1/sini)
                    )*24
                    *(
                        pm.math.sqrt(1-ecc**2)/(1+ecc*pm.math.sin(omega))
                    )
                )

                #
                # End: derived parameters
                #

                # Fit for the maximum a posteriori parameters. This worked best
                # (for "rudolf" project) when doing the GP parameters first,
                # then the transit parameters. "All at once"
                # (`pmx.optimize(start=start)`) did not work very well; a few
                # other combinations of parameters did not work very well
                # either. Using the plotting diagnostics below helped this.

                # NOTE: Method 1: don't do full optimization.
                np.random.seed(42)

                LOGINFO(model.check_test_point())

                if start is None:
                    start = model.test_point

                map_soln = start

                if map_optimization_method is not None:

                    steps = map_optimization_method.split("_")
                    steps = [s for s in steps if 'then' not in s]

                    # Iterate over steps in the model. E.g., if
                    # map_optimization_method is "RotationTerm_transit", then
                    # optimization the RotationTerm first, then the transit.
                    for s in steps:

                        if s == 'transit':
                            # Transit: [log_r, b, t0, period, r_star,
                            # logg_star, ecs, u_star]...]
                            if 'log_ror' in p.keys():
                                map_soln = pmx.optimize(
                                    start=map_soln,
                                    vars=[log_ror, b, ecc, omega, t0, period, r_star,
                                          logg_star, u_star, mean]
                                )
                            elif 'log_depth' in p.keys():
                                map_soln = pmx.optimize(
                                    start=map_soln,
                                    vars=[log_depth, b, ecc, omega, t0, period, r_star,
                                          logg_star, u_star, mean]
                                )

                        if s == 'gpJitterMean':
                            # All GP terms + jitter and mean.
                            map_soln = pmx.optimize(
                                start=map_soln,
                                vars=[sigma_rot, prot, log_Q0, log_dQ, f,
                                      log_jitter, mean]
                            )

                        if s == 'RotationTerm':
                            # RotationTerm: sigma_rot, prot, log_Q0, log_dQ, f
                            map_soln = pmx.optimize(
                                start=map_soln,
                                vars=[sigma_rot, f, prot, log_Q0, log_dQ]
                            )

                        if s == 'SHOterm':
                            raise NotImplementedError('No SHOterm in this model')
                            # SHO term: sigma, rho. Describes non-periodic variability.
                            map_soln = pmx.optimize(
                                start=start,
                                vars=[sigma, rho]
                            )

                        if s == 'bonus':
                            # bonus
                            map_soln = pmx.optimize(
                                start=map_soln,
                                vars=[log_ror, b, log_Q0, log_dQ]
                            )

                        if s == 'LimbDark':
                            # Limb-darkening
                            map_soln = pmx.optimize(
                                start=map_soln,
                                vars=[u0, u1]
                            )

                        if s == 'JitterMean':
                            # Jitter and mean:
                            map_soln = pmx.optimize(
                                start=map_soln,
                                vars=[log_jitter, mean]
                            )

                        if s == 'FullOptimize':
                            # Full optimization
                            map_soln = pmx.optimize(start=map_soln)

                else:
                    # By default, do full optimization
                    map_soln = pmx.optimize(start=map_soln)

            return model, map_soln

        model, map_estimate = build_model()

        if make_threadsafe:
            pass
        else:
            # NOTE: would usually plot MAP estimate here, but really
            # there's not a huge need.
            LOGINFO(map_estimate)
            pass

        from betty import plotting as bp

        outpath = os.path.join(
            self.PLOTDIR,
            'flux_vs_time_map_estimate_RotGPtransit.png'
        )
        bp.plot_light_curve(self.data, map_estimate, outpath)

        outpath = os.path.join(
            self.PLOTDIR,
            'flux_vs_phase_map_estimate_RotGPtransit.png'
        )
        bp.plot_phased_light_curve_gptransit(
            self.data, map_estimate, outpath, do_hacky_reprerror=True
        )

        # sample from the posterior defined by this model.

        #
        # NOTE: some of Dan's tutorials do sigma-clipping at this point. If the
        # MAP estimate looks good enough... no need!
        #

        with model:
            trace = pmx.sample(
                tune=self.N_samples, draws=self.N_samples,
                start=map_estimate, cores=self.N_cores,
                chains=self.N_chains,
                return_inferencedata=True,
                initial_accept=0.8,
                target_accept=0.95
            )

        with open(pklpath, 'wb') as buff:
            pickle.dump({'model': model, 'trace': trace,
                         'map_estimate': map_estimate}, buff)

        self.model = model
        self.trace = trace
        self.map_estimate = map_estimate


    def run_QuadMulticolorTransit_inference(self, pklpath,
                                            make_threadsafe=True,
                                            map_optimization_method=None):
        """
        Fit a single transit + quadratic trend with multiple simultaneous bandpasses
        (presumably a ground-based transit).

        Fits for:
            [ {instrument}_mean, {instrument}_a1, {instrument}_a2,
            {instrument}_log_jitter, {instrument}_u_star,
            b, period, t0, log_ror, r_star, logg_star]

        where {instrument} iterates over whatever different instruments are in
        `self.data` (e.g., "muscat_i", "muscat_z", etc.).
        """

        p = self.priordict

        # if the model has already been run, pull the result from the
        # pickle. otherwise, run it.
        if os.path.exists(pklpath):
            LOGINFO(f'Found {pklpath}, loading from cache.')
            d = pickle.load(open(pklpath, 'rb'))
            self.model = d['model']
            self.trace = d['trace']
            self.map_estimate = d['map_estimate']
            return 1

        def build_model(mask=None, start=None):

            # allowing multiple instruments
            names = list(self.data.keys())
            nplanets = 1

            with pm.Model() as model:

                # Shared parameters

                # Stellar parameters.
                logg_star = pm.Normal(
                    "logg_star", mu=p['logg_star'][1], sd=p['logg_star'][2]
                )

                r_star = pm.Bound(pm.Normal, lower=0.0)(
                    "r_star", mu=p['r_star'][1], sd=p['r_star'][2]
                )
                rho_star = pm.Deterministic(
                    "rho_star", factor*10**logg_star / r_star
                )

                # fix Rp/Rs across bandpasses
                log_ror = pm.Uniform('log_ror', lower=p['log_ror'][1],
                                   upper=p['log_ror'][2], testval=p['log_ror'][3])
                ror = pm.Deterministic('ror', tt.exp(log_ror))
                # planet radius in the same units as r_star
                r_pl = pm.Deterministic(
                    "r_pl", ror*r_star
                )

                # orbital parameters for planet (single)
                t0 = pm.Normal(
                    "t0", mu=p['t0'][1], sd=p['t0'][2], testval=p['t0'][1]
                )
                period = pm.Normal(
                    'period', mu=p['period'][1], sd=p['period'][2],
                    testval=p['period'][1]
                )
                b = xo.distributions.ImpactParameter(
                    "b", ror=ror, testval=p['b'][1]
                )

                log_dur = pm.Normal(
                    "log_dur", mu=np.log(2.8/24), sigma=5
                )
                dur = pm.Deterministic("dur", tt.exp(log_dur))

                # orbit model.  assume zero eccentricity orbit.
                orbit = xo.orbits.KeplerianOrbit(
                    period=period,
                    t0=t0,
                    b=b,
                    duration=dur,
                    ror=ror,
                    #rho_star=rho_star,
                    r_star=r_star
                )

                # NOTE: this would be if you fixed LD over all
                # bandpasses
                # u_star = xo.QuadLimbDark("u_star")
                # star = xo.LimbDarkLightCurve(u_star)

                # Loop over "instruments" (TESS, then each ground-based lightcurve)
                lc_models = dict()
                roughdepths = dict()

                for n, (name, (x, y, yerr, texp)) in enumerate(self.data.items()):

                    #if mask is None:
                    #    mask = np.ones(len(x), dtype=bool)

                    # Define per-instrument parameters in a submodel, to not
                    # need to prefix the names. Yields e.g., "TESS_mean",
                    # "elsauce_0_mean", "elsauce_2_a2", "muscat3_i_a1", etc.
                    with pm.Model(name=name, model=model):

                        # Each bandpass gets its own LD.
                        u_star = xo.QuadLimbDark(f"{name}_u_star")
                        star = xo.LimbDarkLightCurve(u_star)

                        # Transit parameters.
                        mean = pm.Normal(
                            "mean", mu=p[f'{name}_mean'][1],
                            sd=p[f'{name}_mean'][2], testval=p[f'{name}_mean'][1]
                        )

                        # units: rel flux per day.
                        a1 = pm.Uniform(
                            "a1",
                            lower=p[f'{name}_a1'][1],
                            upper=p[f'{name}_a1'][2],
                            testval=p[f'{name}_a1'][3]
                        )
                        # units: rel flux per day^2.
                        a2 = pm.Uniform(
                            "a2",
                            lower=p[f'{name}_a2'][1],
                            upper=p[f'{name}_a2'][2],
                            testval=p[f'{name}_a2'][3]
                        )

                        # A jitter term describing excess white noise
                        log_jitter = pm.Normal(
                            f"{name}_log_jitter", mu=np.log(np.mean(yerr)),
                            sd=p[f'{name}_log_jitter'][2]
                        )

                        # midpoint for this definition of the quadratic trend
                        _tmid = np.nanmedian(x)
                        lc_models[name] = pm.Deterministic(
                            f'{name}_mu_transit',
                            mean +
                            a1*(x-_tmid) +
                            a2*(x-_tmid)**2 +
                            star.get_light_curve(
                                orbit=orbit, r=r_pl, t=x, texp=texp
                            ).T.flatten()
                        )

                    likelihood = pm.Normal(
                        f'{name}_obs', mu=lc_models[name],
                        sigma=pm.math.sqrt(
                            yerr**2 + tt.exp(2 * log_jitter)
                        ),
                        observed=y
                    )

                #
                # Derived parameters
                #

                # planet radius in jupiter radii
                r_planet = pm.Deterministic(
                    "r_planet", (ror*r_star)*( 1*units.Rsun/(1*units.Rjup) ).cgs.value
                )

                #
                # eq 30 of winn+2010, ignoring planet density.
                #
                _a_Rs, _ = xo.orbits.keplerian.get_aor_from_transit_duration(
                    dur, period, b, ror=ror
                )
                a_Rs = pm.Deterministic(
                    "a_Rs", _a_Rs
                )

                #
                # cosi. assumes e=0 (e.g., Winn+2010 eq 7)
                #
                cosi = pm.Deterministic("cosi", b / a_Rs)

                # probably safer than tt.arccos(cosi)
                sini = pm.Deterministic("sini", pm.math.sqrt( 1 - cosi**2 ))

                #
                # transit durations (T_14, T_13) for circular orbits. Winn+2010 Eq 14, 15.
                # units: hours.
                #
                T_14 = pm.Deterministic(
                    'T_14',
                    (period/np.pi)*
                    tt.arcsin(
                        (1/a_Rs) * pm.math.sqrt( (1+ror)**2 - b**2 )
                        * (1/sini)
                    )*24
                )

                T_13 =  pm.Deterministic(
                    'T_13',
                    (period/np.pi)*
                    tt.arcsin(
                        (1/a_Rs) * pm.math.sqrt( (1-ror)**2 - b**2 )
                        * (1/sini)
                    )*24
                )

                # Optimizing
                np.random.seed(42)
                if start is None:
                    start = model.test_point

                map_soln = start

                if map_optimization_method is not None:

                    steps = map_optimization_method.split("_")
                    steps = [s for s in steps if 'then' not in s]

                    raise NotImplementedError('Need to tune this model')

                else:
                    # By default, do full optimization
                    map_soln = pmx.optimize(start=map_soln)

            return model, map_soln

        model, map_estimate = build_model()

        if make_threadsafe:
            pass
        else:
            LOGINFO(map_estimate)
            pass

        from betty import plotting as bp

        outpath = os.path.join(
            self.PLOTDIR,
            'multicolorflux_vs_time_map_estimate_QuadMulticolorTransit.png'
        )
        bp.plot_multicolorlight_curve(self.data, map_estimate, outpath)

        with model:
            trace = pmx.sample(
                tune=self.N_samples, draws=self.N_samples,
                start=map_estimate, cores=self.N_cores,
                chains=self.N_chains,
                return_inferencedata=True,
                initial_accept=0.8,
                target_accept=0.95
            )

        with open(pklpath, 'wb') as buff:
            pickle.dump({'model': model, 'trace': trace,
                         'map_estimate': map_estimate}, buff)

        self.model = model
        self.trace = trace
        self.map_estimate = map_estimate


    def run_QuadMulticolorVaryDepthTransit_inference(self, pklpath,
                                                     make_threadsafe=True,
                                                     map_optimization_method=None):
        """
        Fit a single transit + quadratic trend with multiple
        simultaneous bandpasses.  Let transit depths vary.

        Fits for:
            [ {instrument}_mean, {instrument}_a1, {instrument}_a2,
            {instrument}_log_jitter, {instrument}_log_ror,
            {instrument}_u_star, b, period, t0, r_star, logg_star]

        where {instrument} iterates over whatever different instruments are in
        `self.data` (e.g., "muscat_i", "muscat_z", etc.).
        """

        p = self.priordict

        # if the model has already been run, pull the result from the
        # pickle. otherwise, run it.
        if os.path.exists(pklpath):
            LOGINFO(f'Found {pklpath}, loading from cache.')
            d = pickle.load(open(pklpath, 'rb'))
            self.model = d['model']
            self.trace = d['trace']
            self.map_estimate = d['map_estimate']
            return 1

        def build_model(mask=None, start=None):

            # allowing multiple instruments
            names = list(self.data.keys())
            nplanets = 1

            with pm.Model() as model:

                # Shared parameters

                # Stellar parameters.
                logg_star = pm.Normal(
                    "logg_star", mu=p['logg_star'][1], sd=p['logg_star'][2]
                )

                r_star = pm.Bound(pm.Normal, lower=0.0)(
                    "r_star", mu=p['r_star'][1], sd=p['r_star'][2]
                )
                rho_star = pm.Deterministic(
                    "rho_star", factor*10**logg_star / r_star
                )

                # orbital parameters for planet (single)
                t0 = pm.Normal(
                    "t0", mu=p['t0'][1], sd=p['t0'][2], testval=p['t0'][1]
                )
                period = pm.Normal(
                    'period', mu=p['period'][1], sd=p['period'][2],
                    testval=p['period'][1]
                )

                # NOTE: this is technically incorrect, U[0,1+Rp/Rs] is
                # better.  However in this model Rp is defined per
                # bandpass, which makes the usual correct impact
                # parameter distribution not usable. Here, we use the
                # fact that we already know the planet is not grazing.
                b = pm.Uniform('b', lower=p['b'][1], upper=p['b'][2],
                               testval=p['b'][3])

                # orbit model.  assume zero eccentricity orbit.
                orbit = xo.orbits.KeplerianOrbit(
                    period=period,
                    t0=t0,
                    b=b,
                    rho_star=rho_star,
                    r_star=r_star
                )

                # Loop over "instruments" (TESS, then each ground-based lightcurve)
                lc_models = dict()
                roughdepths = dict()

                for n, (name, (x, y, yerr, texp)) in enumerate(self.data.items()):

                    #if mask is None:
                    #    mask = np.ones(len(x), dtype=bool)

                    # Define per-instrument parameters in a submodel, to not
                    # need to prefix the names. Yields e.g., "TESS_mean",
                    # "elsauce_0_mean", "elsauce_2_a2", "muscat3_i_a1", etc.
                    with pm.Model(name=name, model=model):

                        u_star = xo.QuadLimbDark(f"{name}_u_star")
                        star = xo.LimbDarkLightCurve(u_star)

                        # let Rp/Rs float across bandpasses
                        log_ror = pm.Uniform(f'{name}_log_ror',
                                           lower=p[f'{name}_log_ror'][1],
                                           upper=p[f'{name}_log_ror'][2],
                                           testval=p[f'{name}_log_ror'][3])
                        ror = pm.Deterministic(f'{name}_ror', tt.exp(log_ror))
                        # planet radius in the same units as r_star
                        r_pl = pm.Deterministic(
                            f"{name}_r_pl", ror*r_star
                        )
                        # planet radius in jupiter radii
                        r_planet = pm.Deterministic(
                            f"{name}_r_planet", (ror*r_star)*( 1*units.Rsun/(1*units.Rjup) ).cgs.value
                        )

                        # Transit parameters.
                        mean = pm.Normal(
                            "mean", mu=p[f'{name}_mean'][1],
                            sd=p[f'{name}_mean'][2], testval=p[f'{name}_mean'][1]
                        )

                        # units: rel flux per day.
                        a1 = pm.Uniform(
                            "a1",
                            lower=p[f'{name}_a1'][1],
                            upper=p[f'{name}_a1'][2],
                            testval=p[f'{name}_a1'][3]
                        )
                        # units: rel flux per day^2.
                        a2 = pm.Uniform(
                            "a2",
                            lower=p[f'{name}_a2'][1],
                            upper=p[f'{name}_a2'][2],
                            testval=p[f'{name}_a2'][3]
                        )

                        # A jitter term describing excess white noise
                        log_jitter = pm.Normal(
                            f"{name}_log_jitter", mu=np.log(np.mean(yerr)),
                            sd=p[f'{name}_log_jitter'][2]
                        )

                        # midpoint for this definition of the quadratic trend
                        _tmid = np.nanmedian(x)
                        lc_models[name] = pm.Deterministic(
                            f'{name}_mu_transit',
                            mean +
                            a1*(x-_tmid) +
                            a2*(x-_tmid)**2 +
                            star.get_light_curve(
                                orbit=orbit, r=r_pl, t=x, texp=texp
                            ).T.flatten()
                        )

                    likelihood = pm.Normal(
                        f'{name}_obs', mu=lc_models[name],
                        sigma=pm.math.sqrt(
                            yerr**2 + tt.exp(2 * log_jitter)
                        ),
                        observed=y
                    )

                #
                # Derived parameters
                #

                #
                # eq 30 of winn+2010, ignoring planet density.
                #
                a_Rs = pm.Deterministic(
                    "a_Rs",
                    (rho_star * period**2)**(1/3)
                    *
                    (( (1*units.gram/(1*units.cm)**3) * (1*units.day**2)
                      * const.G / (3*np.pi)
                    )**(1/3)).cgs.value
                )

                #
                # cosi. assumes e=0 (e.g., Winn+2010 eq 7)
                #
                cosi = pm.Deterministic("cosi", b / a_Rs)

                # probably safer than tt.arccos(cosi)
                sini = pm.Deterministic("sini", pm.math.sqrt( 1 - cosi**2 ))

                #
                # transit durations (T_14, T_13) for circular orbits. Winn+2010 Eq 14, 15.
                # units: hours.
                #
                T_14 = pm.Deterministic(
                    'T_14',
                    (period/np.pi)*
                    tt.arcsin(
                        (1/a_Rs) * pm.math.sqrt( (1+ror)**2 - b**2 )
                        * (1/sini)
                    )*24
                )

                T_13 =  pm.Deterministic(
                    'T_13',
                    (period/np.pi)*
                    tt.arcsin(
                        (1/a_Rs) * pm.math.sqrt( (1-ror)**2 - b**2 )
                        * (1/sini)
                    )*24
                )

                # Optimizing
                np.random.seed(42)
                if start is None:
                    start = model.test_point

                map_soln = start

                if map_optimization_method is not None:

                    steps = map_optimization_method.split("_")
                    steps = [s for s in steps if 'then' not in s]

                    raise NotImplementedError('Need to tune this model')

                else:
                    # By default, do full optimization
                    map_soln = pmx.optimize(start=map_soln)

            return model, map_soln

        model, map_estimate = build_model()

        if make_threadsafe:
            pass
        else:
            LOGINFO(map_estimate)
            pass

        from betty import plotting as bp

        outpath = os.path.join(
            self.PLOTDIR,
            'multicolorflux_vs_time_map_estimate_QuadMulticolorVaryDepthTransit.png'
        )
        bp.plot_multicolorlight_curve(self.data, map_estimate, outpath)

        with model:
            trace = pmx.sample(
                tune=self.N_samples, draws=self.N_samples,
                start=map_estimate, cores=self.N_cores,
                chains=self.N_chains,
                return_inferencedata=True,
                initial_accept=0.8,
                target_accept=0.95
            )

        with open(pklpath, 'wb') as buff:
            pickle.dump({'model': model, 'trace': trace,
                         'map_estimate': map_estimate}, buff)

        self.model = model
        self.trace = trace
        self.map_estimate = map_estimate


    def run_localpolytransit_inference(self, pklpath, make_threadsafe=True):
        """
        Each transit window gets a local second-order polynomial, plus a
        Agol+19 transit.

        Free parameters for the transit are {"period", "t0", "log_ror", "b",
        "u0", "u1", "log_jitter"}.

        The limb-darkening coefficients are shared across transit windows.  A
        single instrument (e.g., TESS, Kepler, K2) is assumed.
        """

        p = self.priordict

        # if the model has already been run, pull the result from the
        # pickle. otherwise, run it.
        if os.path.exists(pklpath) and not self.OVERWRITE:
            LOGINFO(f'Found {pklpath}, loading from cache.')
            d = pickle.load(open(pklpath, 'rb'))
            self.model = d['model']
            self.trace = d['trace']
            self.map_estimate = d['map_estimate']
            return 1
        elif os.path.exists(pklpath) and self.OVERWRITE:
            LOGINFO(f'Found {pklpath}, and OVERWRITE is True. '
                    'Deleting and proceeding.')
            os.remove(pklpath)

        def build_model(start=None):

            with pm.Model() as model:

                # Shared parameters

                # Stellar parameters. (Following tess.world notebooks).
                logg_star = pm.Normal(
                    "logg_star", mu=p['logg_star'][1], sd=p['logg_star'][2]
                )

                r_star = pm.Bound(pm.Normal, lower=0.0)(
                    "r_star", mu=p['r_star'][1], sd=p['r_star'][2]
                )
                rho_star = pm.Deterministic(
                    "rho_star", factor*10**logg_star / r_star
                )

                # A jitter term describing excess white noise
                yerrs = np.concatenate(
                    [_yerr for (_, (_, _, _yerr, _)) in self.data.items()]
                ).ravel()
                log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(yerrs)),
                                       sd=p['log_jitter'][2])

                # fix Rp/Rs across bandpasses
                if p['log_ror'][0] == 'Uniform':
                    log_ror = pm.Uniform('log_ror', lower=p['log_ror'][1],
                                       upper=p['log_ror'][2], testval=p['log_ror'][3])
                else:
                    raise NotImplementedError
                ror = pm.Deterministic('ror', tt.exp(log_ror))
                # planet radius in the same units as r_star
                r_pl = pm.Deterministic(
                    "r_pl", ror*r_star
                )

                # Some orbital parameters
                t0 = pm.Normal(
                    "t0", mu=p['t0'][1], sd=p['t0'][2], testval=p['t0'][1]
                )
                period = pm.Normal(
                    'period', mu=p['period'][1], sd=p['period'][2],
                    testval=p['period'][1]
                )
                b = xo.distributions.ImpactParameter(
                    "b", ror=ror, testval=p['b'][1]
                )

                orbit = xo.orbits.KeplerianOrbit(
                    period=period, t0=t0, b=b, rho_star=rho_star, r_star=r_star
                )

                # limb-darkening
                if 'u[0]' in p.keys() and 'u[1]' in p.keys():
                    u0 = pm.Uniform(
                        'u[0]', lower=p['u[0]'][1],
                        upper=p['u[0]'][2],
                        testval=p['u[0]'][3]
                    )
                    u1 = pm.Uniform(
                        'u[1]', lower=p['u[1]'][1],
                        upper=p['u[1]'][2],
                        testval=p['u[1]'][3]
                    )
                    u_star = [u0, u1]

                else:
                    assert 'u_star' in p.keys()
                    assert p['u_star'][0] == 'QuadLimbDark'
                    u_star = xo.QuadLimbDark("u_star")

                star = xo.LimbDarkLightCurve(u_star)

                # Loop over "instruments" (TESS, then each ground-based lightcurve)
                parameters = dict()
                lc_models = dict()
                roughdepths = dict()

                for n, (name, (x, y, yerr, texp)) in enumerate(self.data.items()):

                    # Define per-instrument parameters in a submodel, to not need
                    # to prefix the names. Yields e.g., "TESS_mean",
                    # "elsauce_0_mean", "elsauce_2_a2"
                    with pm.Model(name=name, model=model):

                        mean = pm.Normal(
                            f'{name}_mean', mu=p[f'{name}_mean'][1],
                            sd=p[f'{name}_mean'][2], testval=p[f'{name}_mean'][1]
                        )
                        a1 =  pm.Uniform(
                            f'{name}_a1', lower=p[f'{name}_a1'][1],
                            upper=p[f'{name}_a1'][2], testval=0
                        )
                        a2 = pm.Uniform(
                            f'{name}_a2', lower=p[f'{name}_a2'][1],
                            upper=p[f'{name}_a2'][2], testval=0
                        )

                        # midpoint for this definition of the quadratic trend
                        _tmid = np.nanmedian(x)

                        lc_models[name] = pm.Deterministic(
                            f'{name}_mu_transit',
                            mean +
                            a1*(x-_tmid) +
                            a2*(x-_tmid)**2 +
                            star.get_light_curve(
                                orbit=orbit, r=r_pl, t=x, texp=texp
                            ).T.flatten()
                        )

                    likelihood = pm.Normal(
                        f'{name}_obs', mu=lc_models[name],
                        sigma=pm.math.sqrt(yerr**2 + tt.exp(2 * log_jitter)),
                        observed=y
                    )

                #
                # Derived parameters
                #

                # planet radius in jupiter radii
                r_planet = pm.Deterministic(
                    "r_planet", (ror*r_star)*( 1*units.Rsun/(1*units.Rjup) ).cgs.value
                )

                #
                # eq 30 of winn+2010, ignoring planet density.
                #
                a_Rs = pm.Deterministic(
                    "a_Rs",
                    (rho_star * period**2)**(1/3)
                    *
                    (( (1*units.gram/(1*units.cm)**3) * (1*units.day**2)
                      * const.G / (3*np.pi)
                    )**(1/3)).cgs.value
                )

                #
                # cosi. assumes e=0 (e.g., Winn+2010 eq 7)
                #
                cosi = pm.Deterministic("cosi", b / a_Rs)

                # probably safer than tt.arccos(cosi)
                sini = pm.Deterministic("sini", pm.math.sqrt( 1 - cosi**2 ))

                #
                # transit durations (T_14, T_13) for circular orbits. Winn+2010 Eq 14, 15.
                # units: hours.
                #
                T_14 = pm.Deterministic(
                    'T_14',
                    (period/np.pi)*
                    tt.arcsin(
                        (1/a_Rs) * pm.math.sqrt( (1+ror)**2 - b**2 )
                        * (1/sini)
                    )*24
                )

                T_13 =  pm.Deterministic(
                    'T_13',
                    (period/np.pi)*
                    tt.arcsin(
                        (1/a_Rs) * pm.math.sqrt( (1-ror)**2 - b**2 )
                        * (1/sini)
                    )*24
                )

                # Optimizing
                np.random.seed(42)
                LOGINFO(model.check_test_point())
                if start is None:
                    start = model.test_point
                map_soln = start

                # By default, do full optimization
                LOGINFO('Beginning optimization...')
                map_soln = pmx.optimize(start=map_soln)

                return model, map_soln

        model, map_estimate = build_model()

        LOGINFO(map_estimate)

        LOGINFO('Got MAP estimate. Beginning sampling...')

        with model:
            trace = pmx.sample(
                tune=self.N_samples, draws=self.N_samples,
                start=map_estimate, cores=self.N_cores,
                chains=self.N_chains,
                return_inferencedata=True,
                initial_accept=0.8,
                target_accept=0.95
            )

        with open(pklpath, 'wb') as buff:
            pickle.dump({'model': model, 'trace': trace,
                         'map_estimate': map_estimate}, buff)

        self.model = model
        self.trace = trace
        self.map_estimate = map_estimate


    def run_allindivtransit_inference(self, priordict, pklpath,
                                      make_threadsafe=True):
        """
        Fit all transits (TESS+ground), allowing each transit a quadratic trend. No
        "pre-detrending".
        """

        # NOTE: see timmy.run_allindivtransit_inference
        raise NotImplementedError


    def run_rvorbit_inference(self, priordict, pklpath, make_threadsafe=True):
        """
        Fit RVs for Keplerian orbit.
        """

        raise NotImplementedError
        # NOTE: but see draft in timmy.modelfitter


    def run_rvspotorbit_inference(self, pklpath, make_threadsafe=True):
        """
        Fit RVs for Keplerian orbit.
        """

        p = self.priordict

        # if the model has already been run, pull the result from the
        # pickle. otherwise, run it.
        if os.path.exists(pklpath):
            LOGINFO(f'Found {pklpath}, loading from cache.')
            d = pickle.load(open(pklpath, 'rb'))
            self.model = None
            self.trace = d['trace']
            self.map_estimate = d['map_estimate']
            return 1

        with pm.Model() as model:

            t_offset = 2459000

            # Shared parameters

            # Stellar parameters.
            m_star = pm.Bound(pm.Normal, lower=0.5)(
                "m_star", mu=p['m_star'][1], sd=p['m_star'][2]
            )
            r_star = pm.Bound(pm.Normal, lower=0.5)(
                "r_star", mu=p['r_star'][1], sd=p['r_star'][2]
            )

            # Some orbital parameters
            t0 = pm.Normal(
                "t0", mu=p['t0'][1]-t_offset, sd=p['t0'][2], testval=p['t0'][1]-t_offset
            )
            period = pm.Normal(
                'period', mu=p['period'][1], sd=p['period'][2],
                testval=p['period'][1]
            )
            if not p['b'][0] == 'Normal':
                raise NotImplementedError(
                    'i was writing a quick hack to make this, and did not '
                    'implement the distribution agnosticity that i should have')
            b = pm.Normal(
                "b", mu=p['b'][1], sd=p['b'][2], testval=p['b'][1]
            )

            ecc = pm.Bound(pm.Normal, lower=0.001, upper=0.2)(
                'ecc', mu=p['ecc'][1], sd=p['ecc'][2]
            )
            omega = pm.Uniform(
                'omega', lower=p['omega'][1], upper=p['omega'][2]
            )

            K_orb = pm.Uniform(
                'K_orb', lower=p['K_orb'][1], upper=p['K_orb'][2]
            )

            orbit = xo.orbits.KeplerianOrbit(period=period, b=b, t0=t0,
                                             ecc=ecc, omega=omega,
                                             m_star=m_star, r_star=r_star)

            name = 'minerva' # NOTE: again, it was a hack
            mean = pm.Normal(
                f"{name}_mean", mu=p[f'{name}_mean'][1],
                sd=p[f'{name}_mean'][2], testval=p[f'{name}_mean'][1]
            )

            # the GP model for the stellar variability
            log_amp = pm.Bound(pm.Normal, lower=10, upper=13.8)(
                'log_amp', mu=p['log_amp'][1], sd=p['log_amp'][2]
            )
            P_rot = pm.Normal(
                'P_rot', mu=p['P_rot'][1], sd=p['P_rot'][2]
            )
            log_Q0 = pm.Bound(pm.Normal, upper=5, lower=0.5)(
                'log_Q0', mu=p['log_Q0'][1], sd=p['log_Q0'][2]
            )
            log_deltaQ = pm.Normal(
                'log_deltaQ', mu=p['log_deltaQ'][1], sd=p['log_deltaQ'][2]
            )
            mix = pm.Uniform(
                'mix', lower=p['mix'][1], upper=p['mix'][2]
            )

            kernel = xo.gp.terms.RotationTerm(
                log_amp=log_amp, period=P_rot, log_Q0=log_Q0,
                log_deltaQ=log_deltaQ, mix=mix
            )

            _time = self.data[name][0] - t_offset
            _rv = self.data[name][1]
            _rv_err = self.data[name][2]

            def mean_model(t):
                rv_kep = pm.Deterministic(
                    'rv_kep', orbit.get_radial_velocity(t, K=K_orb)
                )
                return rv_kep + mean

            gp = xo.gp.GP(kernel, _time, _rv_err**2, mean=mean_model)
            gp.marginal('rv_obs', observed=_rv)
            pm.Deterministic('gp_pred', gp.predict())

            # Optimizing
            map_estimate = pm.find_MAP(model=model)

            if make_threadsafe:
                pass
            else:
                # NOTE: would usually plot MAP estimate here, but really
                # there's not a huge need.
                LOGINFO(map_estimate)
                pass

            # sample from the posterior defined by this model.
            trace = pm.sample(
                tune=self.N_samples, draws=self.N_samples,
                start=map_estimate, cores=self.N_cores,
                chains=self.N_chains,
                target_accept=0.95
            )

        savevars = ['t0', 'period', 'b', 'minerva_mean', 'P_rot', 'log_deltaQ',
                    'm_star', 'r_star', 'ecc', 'omega', 'K_orb', 'log_amp',
                    'log_Q0', 'mix']

        trace_dict = {k:trace[k] for k in savevars}

        with open(pklpath, 'wb') as buff:
            pickle.dump({'trace': trace_dict,
                         'map_estimate': map_estimate}, buff)

        self.model = None
        self.trace = trace
        self.map_estimate = map_estimate

"""
ModelParser
ModelFitter
    run_transit_inference
    run_gptransit_inference
    run_rvspotorbit_inference

    Not yet implemented here (but see /timmy/):
        run_alltransit_inference
        run_rvorbit_inference
        run_allindivtransit_inference
"""
import logging
logging.getLogger("filelock").setLevel(logging.ERROR)

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

from aesthetic.plot import set_style, savefig

class ModelParser:

    def __init__(self, modelid):
        self.initialize_model(modelid)

    def initialize_model(self, modelid):
        self.modelid = modelid
        self.modelcomponents = modelid.split('_')
        self.verify_modelcomponents()

    def verify_modelcomponents(self):

        validcomponents = ['simpletransit', 'rvspotorbit', 'gptransit',
                           'allindivtransit', 'alltransit']

        assert len(self.modelcomponents) >= 1

        for modelcomponent in self.modelcomponents:
            if modelcomponent not in validcomponents:
                errmsg = (
                    'Got modelcomponent {}. validcomponents include {}.'
                    .format(modelcomponent, validcomponents)
                )
                raise ValueError(errmsg)


class ModelFitter(ModelParser):
    """
    Given a modelid of the form "*transit", "rv_*" and a dataframe containing
    (time and flux), or (time and rv), run the inference.
    """

    def __init__(self, modelid, data_df, priordict, N_samples=2000, N_cores=16,
                 N_chains=4, plotdir=None, pklpath=None,
                 overwrite=1):

        self.N_samples = N_samples
        self.N_cores = N_cores
        self.N_chains = N_chains
        self.PLOTDIR = plotdir
        self.OVERWRITE = overwrite

        implemented_models = [
            'simpletransit', 'allindivtransit', 'oddindivtransit',
            'evenindivtransit', 'rvorbit', 'rvspotorbit', 'alltransit',
            'gptransit'
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

        if modelid == 'simpletransit':
            print(f'Beginning run_transit_inference for {modelid}...')
            self.run_transit_inference(
                pklpath, make_threadsafe=make_threadsafe
            )
            print(f'Finished run_transit_inference for {modelid}...')

        elif modelid == 'gptransit':
            print(f'Beginning PyMC3 run for {modelid}...')
            self.run_gptransit_inference(
                pklpath, make_threadsafe=make_threadsafe
            )
            print(f'Finished PyMC3 for {modelid}...')

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


    def run_transit_inference(self, pklpath, make_threadsafe=True):
        """
        Fit transit data for an Agol+19 transit. (Ignores any stellar
        variability; believes error bars).  Free parameters are {"period",
        "t0", "r", "b", "u0", "u1"}.
        """

        p = self.priordict

        # if the model has already been run, pull the result from the
        # pickle. otherwise, run it.
        if os.path.exists(pklpath):
            d = pickle.load(open(pklpath, 'rb'))
            self.model = d['model']
            self.trace = d['trace']
            self.map_estimate = d['map_estimate']
            return 1

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

            # fix Rp/Rs across bandpasses
            if p['log_r'][0] == 'Uniform':
                log_r = pm.Uniform('log_r', lower=p['log_r'][1],
                                   upper=p['log_r'][2], testval=p['log_r'][3])
            else:
                raise NotImplementedError
            r = pm.Deterministic('r', tt.exp(log_r))

            # Some orbital parameters
            t0 = pm.Normal(
                "t0", mu=p['t0'][1], sd=p['t0'][2], testval=p['t0'][1]
            )
            period = pm.Normal(
                'period', mu=p['period'][1], sd=p['period'][2],
                testval=p['period'][1]
            )
            b = xo.distributions.ImpactParameter(
                "b", ror=r, testval=p['b'][1]
            )

            orbit = xo.orbits.KeplerianOrbit(
                period=period, t0=t0, b=b, rho_star=rho_star, r_star=r_star
            )

            # limb-darkening
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
            u = [u0, u1]

            star = xo.LimbDarkLightCurve(u)

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
                        orbit=orbit, r=r, t=x, texp=texp
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

                # TODO: add error bar fudge in other models
                likelihood = pm.Normal(
                    f'{name}_obs', mu=lc_models[name], sigma=yerr, observed=y
                )


            #
            # Derived parameters
            #

            # planet radius in jupiter radii
            r_planet = pm.Deterministic(
                "r_planet", (r*r_star)*( 1*units.Rsun/(1*units.Rjup) ).cgs.value
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
                    (1/a_Rs) * pm.math.sqrt( (1+r)**2 - b**2 )
                    * (1/sini)
                )*24
            )

            T_13 =  pm.Deterministic(
                'T_13',
                (period/np.pi)*
                tt.arcsin(
                    (1/a_Rs) * pm.math.sqrt( (1-r)**2 - b**2 )
                    * (1/sini)
                )*24
            )

            # Optimizing
            map_estimate = pm.find_MAP(model=model)

            # start = model.test_point
            # if 'transit' in self.modelcomponents:
            #     map_estimate = xo.optimize(start=start,
            #                                vars=[r, b, period, t0])
            # map_estimate = xo.optimize(start=map_estimate)

            if make_threadsafe:
                pass
            else:
                # NOTE: would usually plot MAP estimate here, but really
                # there's not a huge need.
                print(map_estimate)
                pass

            print('Got MAP estimate. Beginning sampling...')
            # sample from the posterior defined by this model.
            trace = pmx.sample(
                tune=self.N_samples, draws=self.N_samples,
                start=map_estimate, cores=self.N_cores,
                chains=self.N_chains,
                return_inferencedata=True,
                target_accept=0.95
            )

        with open(pklpath, 'wb') as buff:
            pickle.dump({'model': model, 'trace': trace,
                         'map_estimate': map_estimate}, buff)

        self.model = model
        self.trace = trace
        self.map_estimate = map_estimate


    def run_gptransit_inference(self, pklpath, make_threadsafe=True):
        """
        Fit transit data for an Agol+19 transit, + a GP for the stellar
        variability, simultaneously.  (Ignores any stellar variability;
        believes error bars).  Assumes single instrument, for now...
        """

        p = self.priordict

        # if the model has already been run, pull the result from the
        # pickle. otherwise, run it.
        if os.path.exists(pklpath):
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
                    "mean", mu=p[f'{name}_mean'][1], sd=p[f'{name}_mean'][2],
                    testval=p[f'{name}_mean'][1]
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
                u = [u0, u1]

                # fix Rp/Rs across bandpasses
                if p['log_r'][0] == 'Uniform':
                    log_r = pm.Uniform('log_r', lower=p['log_r'][1],
                                       upper=p['log_r'][2], testval=p['log_r'][3])
                else:
                    raise NotImplementedError
                r = pm.Deterministic('r', tt.exp(log_r))

                # orbital parameters for planet (single)
                t0 = pm.Normal(
                    "t0", mu=p['t0'][1], sd=p['t0'][2], testval=p['t0'][1]
                )
                period = pm.Normal(
                    'period', mu=p['period'][1], sd=p['period'][2],
                    testval=p['period'][1]
                )
                b = xo.distributions.ImpactParameter(
                    "b", ror=r, testval=p['b'][1]
                )

                # NOTE: begin copy-pasting in (selectively) from Trevor David's
                # epic216357880 analysis. (Which ofc is based on DFM's
                # tutorials).

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

                #
                # RV stuff: for a future model implementation
                #
                # # RV jitter & a quadratic RV trend
                # log_sigma_rv = pm.Normal(
                #     "log_sigma_rv", mu=np.log(np.median(yerr_rv)), sd=5
                # )
                # trend = pm.Normal(
                #     "trend", mu=0, sd=10.0 ** -np.arange(3)[::-1], shape=3
                # )

                # Transit jitter & GP parameters
                # log_sigma_lc = pm.Normal(
                #     "log_sigma_lc", mu=np.log(np.median(yerr[mask])), sd=10
                # )
                # log_rho_gp = pm.Normal("log_rho_gp", mu=0.0, sd=10)
                # log_sigma_gp = pm.Normal(
                #     "log_sigma_gp", mu=np.log(np.std(y[mask])), sd=10
                # )

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

                star = xo.LimbDarkLightCurve(u)

                # NOTE: could loop over instruments here... (e.g., TESS, keplerllc,
                # keplersc, ground-based instruments...). Instead, opt for simpler
                # single-instrument approach.

                # Compute the model light curve
                light_curves = pm.Deterministic(
                    "light_curves",
                    star.get_light_curve(
                        orbit=orbit, r=r, t=x[mask], texp=texp
                    )
                )

                # Line that adds the transit models of different planets in the system,
                # if relevant
                light_curve = pm.math.sum(light_curves, axis=-1) + mean
                resid = y[mask] - light_curve

                ## Below is the GP model from the "together" tutorial. We'll use the GP
                ## model from the stellar variability tutorial instead.  GP model for
                ## the light curve
                # kernel = terms.SHOTerm(
                #     sigma=tt.exp(log_sigma_gp),
                #     rho=tt.exp(log_rho_gp),
                #     Q=1 / np.sqrt(2),
                # )
                # gp = GaussianProcess(kernel, t=x[mask], yerr=tt.exp(log_sigma_lc))
                # gp.marginal("transit_obs", observed=resid)
                # pm.Deterministic("gp_pred", gp.predict(resid))

                # Use the GP model from the stellar variability tutorial
                # https://gallery.exoplanet.codes/en/latest/tutorials/stellar-variability/
                # Literally the same prior parameters too.

                # A jitter term describing excess white noise
                log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(yerr)),
                                       sd=2.0)

                #A term to describe the non-periodic variability
                sigma = pm.InverseGamma(
                    "sigma", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
                )
                rho = pm.InverseGamma(
                    "rho", **pmx.estimate_inverse_gamma_parameters(0.5, 2.0)
                )

                # The parameters of the RotationTerm kernel
                sigma_rot = pm.InverseGamma(
                    "sigma_rot", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
                )
                log_prot = pm.Normal("log_prot", mu=np.log(p['prot'][1]),
                                     sd=p['prot'][2])
                prot = pm.Deterministic("prot", tt.exp(log_prot))
                log_Q0 = pm.Normal("log_Q0", mu=0.0, sd=2.0)
                log_dQ = pm.Normal("log_dQ", mu=0.0, sd=2.0)
                f = pm.Uniform("f", lower=0.1, upper=1.0)

                # Set up the Gaussian Process model
                kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=1/3.0)
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
                    diag=yerr[mask] ** 2 + tt.exp(2 * log_jitter),
                    mean=mean,
                    quiet=True,
                )

                # Compute the Gaussian Process likelihood and add it into the
                # the PyMC3 model as a "potential"
                gp.marginal("transit_obs", observed=resid)

                # Compute the mean model prediction for plotting purposes
                pm.Deterministic("gp_pred", gp.predict(resid))

                # Commenting out the RV stuff
                # And then include the RVs as in the RV tutorial
                # x_rv_ref = 0.5 * (x_rv.min() + x_rv.max())

                # def get_rv_model(t, name=""):
                #     # First the RVs induced by the planets
                #     vrad = orbit.get_radial_velocity(t)
                #     pm.Deterministic("vrad" + name, vrad)

                #     # Define the background model
                #     A = np.vander(t - x_rv_ref, 3)
                #     bkg = pm.Deterministic("bkg" + name, tt.dot(A, trend))

                #     # Sum over planets and add the background to get the full model
                #     return pm.Deterministic(
                #         "rv_model" + name, tt.sum(vrad, axis=-1) + bkg
                #     )

                # # Define the model
                # rv_model = get_rv_model(x_rv)
                # get_rv_model(t_rv, name="_pred")

                # # The likelihood for the RVs
                # err = tt.sqrt(yerr_rv ** 2 + tt.exp(2 * log_sigma_rv))
                # pm.Normal("obs", mu=rv_model, sd=err, observed=y_rv)

                #
                # Begin: derived parameters
                #

                # planet radius in jupiter radii
                r_planet = pm.Deterministic(
                    "r_planet", (r*r_star)*( 1*units.Rsun/(1*units.Rjup) ).cgs.value
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
                        (1/a_Rs) * pm.math.sqrt( (1+r)**2 - b**2 )
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
                        (1/a_Rs) * pm.math.sqrt( (1-r)**2 - b**2 )
                        * (1/sini)
                    )*24
                    *(
                        pm.math.sqrt(1-ecc**2)/(1+ecc*pm.math.sin(omega))
                    )
                )

                #
                # End: derived parameters
                #

                # Fit for the maximum a posteriori parameters, I've found that I can get
                # a better solution by trying different combinations of parameters in turn
                if start is None:
                    start = model.test_point
                # map_soln = pmx.optimize(start=start, vars=[sigma, rho, sigma_rot])
                # map_soln = pmx.optimize(start=map_soln, vars=[log_r, b])
                # map_soln = pmx.optimize(start=map_soln, vars=[period, t0])
                # map_soln = pmx.optimize(
                #     start=map_soln, vars=[log_sigma_lc, log_sigma_gp]
                # )
                #map_soln = pmx.optimize(start=map_soln, vars=[log_rho_gp])
                #map_soln = pmx.optimize(start=map_soln)

                map_soln = pmx.optimize(start=start)

            return model, map_soln

        model, map_estimate = build_model()

        if make_threadsafe:
            pass
        else:
            # NOTE: would usually plot MAP estimate here, but really
            # there's not a huge need.
            print(map_estimate)
            pass

        def plot_light_curve(data, soln, mask=None):
            assert len(data.keys()) == 1
            name = list(data.keys())[0]
            x,y,yerr,texp = data[name]
            if mask is None:
                mask = np.ones(len(x), dtype=bool)

            plt.close('all')
            set_style()
            fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

            ax = axes[0]

            ax.scatter(x[mask], y[mask], c="k", s=0.5, rasterized=True,
                       label="data", linewidths=0, zorder=42)
            gp_mod = soln["gp_pred"] + soln["mean"]
            ax.plot(x[mask], gp_mod, color="C2", label="MAP gp model",
                    zorder=41)
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
            return fig

        def plot_phased_light_curve(data, soln, mask=None, from_trace=False):
            assert len(data.keys()) == 1
            name = list(data.keys())[0]
            x,y,yerr,texp = data[name]

            if mask is None:
                mask = np.ones(len(x), dtype=bool)

            fig, axes = plt.subplots(2, 1, figsize=(5, 4.5), sharex=True)

            if from_trace==True:
                _t0 = np.median(soln["t0"])
                _per = np.median(soln["period"])
                gp_mod = (
                    np.median(soln["gp_pred"], axis=0) +
                    np.median(soln["mean"], axis=0)
                )
                lc_mod = (
                    np.median(np.sum(soln["light_curves"], axis=-1), axis=0)
                )
                lc_mod_band = (
                    np.percentile(np.sum(soln["light_curves"], axis=-1),
                                  [16.,84.], axis=0)
                )
                _yerr = (
                    np.sqrt(yerr[mask] ** 2 +
                            np.exp(2 * np.median(soln["log_jitter"], axis=0)))
                )

            elif from_trace==False:
                _t0 = soln["t0"]
                _per = soln["period"]
                gp_mod = soln["gp_pred"] + soln["mean"]
                lc_mod = soln["light_curves"][:, 0]
                _yerr = (
                    np.sqrt(yerr[mask] ** 2 + np.exp(2 * soln["log_jitter"]))
                )

            x_fold = (x - _t0 + 0.5 * _per) % _per - 0.5 * _per

            ax = axes[0]

            #For plotting
            lc_modx = x_fold[mask]
            lc_mody = lc_mod[np.argsort(lc_modx)]
            if from_trace==True:
                lc_mod_lo = lc_mod_band[0][np.argsort(lc_modx)]
                lc_mod_hi = lc_mod_band[1][np.argsort(lc_modx)]
            lc_modx = np.sort(lc_modx)

            ax.errorbar(24*x_fold[mask], y[mask]-gp_mod, yerr=_yerr,  fmt=".",
                        color="k", label="data", alpha=0.3)

            ax.plot(24*lc_modx, lc_mody, color="C2", label="transit model",
                    zorder=99)

            if from_trace==True:
                art = ax.fill_between(
                    lc_modx, lc_mod_lo, lc_mod_hi, color="C2", alpha=0.5,
                    zorder=1000
                )
                # NOTE: this will raise an error...
                ert.set_edgecolor("none")

            ax.legend(fontsize=10)
            ax.set_ylabel("relative flux")
            ax.set_xlim(-0.5*24,0.5*24)

            ax = axes[1]
            ax.errorbar(24*x_fold[mask], y[mask] - gp_mod - lc_mod, yerr=_yerr,
                        fmt=".", color="k", label="data - GP - transit", alpha=0.3)
            ax.legend(fontsize=10, loc=3)
            ax.set_xlabel("time from mid-transit [hours]")
            ax.set_ylabel("de-trended flux")
            ax.set_xlim(-0.5*24,0.5*24)

            fig.tight_layout()

            return fig

        fig = plot_light_curve(self.data, map_estimate)
        outpath = os.path.join(self.PLOTDIR, 'flux_vs_time_map_estimate.png')
        savefig(fig, outpath, dpi=350)
        plt.close('all')

        fig = plot_phased_light_curve(self.data, map_estimate)
        outpath = os.path.join(self.PLOTDIR, 'flux_vs_phase_map_estimate.png')
        savefig(fig, outpath, dpi=350)
        plt.close('all')


        #FIXME TODO look at phase-fold of the map_estimate

        #FIXME TODO SOME OF DAN'S TUTORIALS DO SIGMA CLIPPING AT THIS POINT 
        import IPython; IPython.embed()
        assert 0

        print('Got MAP estimate. Beginning sampling...')
        # sample from the posterior defined by this model.

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


    def run_alltransit_inference(self, pklpath, make_threadsafe=True):
        """
        Multi-instrument transit fitter.

        modelids are:
            "alltransit": spline-detrended TESS + raw ground transits

        possible modelid extensions, with some of the groundwork written out
        here copied in from the TOI 837 analysis (timmy.modelfitter):
            "alltransit_quad". huber-spline TESS + quadratic ground transits
            "alltransit_quaddepthvar". huber-spline TESS + quadratic ground
            transits + the ground transit depths are made bandpass-specific.
        """

        raise NotImplementedError(
            'implementation below contains TOI837 specific things'
        )

        p = self.priordict

        # if the model has already been run, pull the result from the
        # pickle. otherwise, run it.
        if os.path.exists(pklpath):
            d = pickle.load(open(pklpath, 'rb'))
            self.model = d['model']
            self.trace = d['trace']
            self.map_estimate = d['map_estimate']
            return 1

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

            # fix Rp/Rs across bandpasses, b/c you're assuming it's a planet
            if 'quaddepthvar' not in self.modelid:

                if p['log_r'][0] == 'Uniform':
                    log_r = pm.Uniform('log_r', lower=p['log_r'][1],
                                       upper=p['log_r'][2], testval=p['log_r'][3])
                    r = pm.Deterministic('r', tt.exp(log_r))
                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError(
                    'boilerplate below was copied in from TOI 837 analysis, '
                    'and needs to be generalized'
                )

                # NOTE: below is boilerplate
                log_r_Tband = pm.Uniform('log_r_Tband', lower=np.log(1e-2),
                                         upper=np.log(1),
                                         testval=p['log_r_Tband'])
                r_Tband = pm.Deterministic('r_Tband', tt.exp(log_r_Tband))

                log_r_Rband = pm.Uniform('log_r_Rband', lower=np.log(1e-2),
                                         upper=np.log(1),
                                         testval=p['log_r_Rband'])
                r_Rband = pm.Deterministic('r_Rband', tt.exp(log_r_Rband))

                log_r_Bband = pm.Uniform('log_r_Bband', lower=np.log(1e-2),
                                         upper=np.log(1),
                                         testval=p['log_r_Bband'])
                r_Bband = pm.Deterministic('r_Bband', tt.exp(log_r_Bband))

                r = r_Tband
                # NOTE: above is boilerplate

            # Some orbital parameters
            t0 = pm.Normal(
                "t0", mu=p['t0'][1], sd=p['t0'][2], testval=p['t0'][1]
            )
            period = pm.Normal(
                'period', mu=p['period'][1], sd=p['period'][2],
                testval=p['period'][1]
            )
            b = xo.distributions.ImpactParameter(
                "b", ror=r, testval=p['b'][1]
            )

            orbit = xo.orbits.KeplerianOrbit(
                period=period, t0=t0, b=b, rho_star=rho_star, r_star=r_star
            )

            # limb-darkening
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
            u = [u0, u1]

            star = xo.LimbDarkLightCurve(u)

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

                    if 'quad' in self.modelid:
                        raise NotImplementedError('boilerplate for this case')
                        if name != 'tess':
                            # units: rel flux per day.
                            a1 = pm.Normal(
                                "a1", mu=p[f'{name}_a1'], sd=1,
                                testval=p[f'{name}_a1']
                            )
                            # units: rel flux per day^2.
                            a2 = pm.Normal(
                                "a2", mu=p[f'{name}_a2'], sd=1,
                                testval=p[f'{name}_a2']
                            )

                if self.modelid == 'alltransit':
                    transit_lc = star.get_light_curve(
                        orbit=orbit, r=r, t=x, texp=texp
                    ).T.flatten()

                    lc_models[name] = pm.Deterministic(
                        f'{name}_mu_transit',
                        mean +
                        transit_lc
                    )

                elif self.modelid == 'alltransit_quad':
                    raise NotImplementedError('boilerplate for this case')
                    if name != 'tess':
                        # midpoint for this definition of the quadratic trend
                        _tmid = np.nanmedian(x)
                        lc_models[name] = pm.Deterministic(
                            f'{name}_mu_transit',
                            mean +
                            a1*(x-_tmid) +
                            a2*(x-_tmid)**2 +
                            star.get_light_curve(
                                orbit=orbit, r=r, t=x, texp=texp
                            ).T.flatten()
                        )
                    elif name == 'tess':
                        lc_models[name] = pm.Deterministic(
                            f'{name}_mu_transit',
                            mean +
                            star.get_light_curve(
                                orbit=orbit, r=r, t=x, texp=texp
                            ).T.flatten()
                        )

                elif self.modelid == 'alltransit_quaddepthvar':
                    if name != 'tess':
                        # midpoint for this definition of the quadratic trend
                        _tmid = np.nanmedian(x)

                        raise NotImplementedError('boilerplate')

                        # do custom depth-to-
                        if (name == 'elsauce_20200401' or
                            name == 'elsauce_20200426'
                        ):
                            r = r_Rband
                        elif name == 'elsauce_20200521':
                            r = r_Tband
                        elif name == 'elsauce_20200614':
                            r = r_Bband

                        transit_lc = star.get_light_curve(
                            orbit=orbit, r=r, t=x, texp=texp
                        ).T.flatten()

                        lc_models[name] = pm.Deterministic(
                            f'{name}_mu_transit',
                            mean +
                            a1*(x-_tmid) +
                            a2*(x-_tmid)**2 +
                            transit_lc
                        )

                        roughdepths[name] = pm.Deterministic(
                            f'{name}_roughdepth',
                            pm.math.abs_(transit_lc).max()
                        )

                    elif name == 'tess':

                        r = r_Tband

                        transit_lc = star.get_light_curve(
                            orbit=orbit, r=r, t=x, texp=texp
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

                # TODO: add error bar fudge
                likelihood = pm.Normal(
                    f'{name}_obs', mu=lc_models[name], sigma=yerr, observed=y
                )


            #
            # Derived parameters
            #
            if self.modelid == 'alltransit_quaddepthvar':
                raise NotImplementedError
                r = r_Tband

            # planet radius in jupiter radii
            r_planet = pm.Deterministic(
                "r_planet", (r*r_star)*( 1*units.Rsun/(1*units.Rjup) ).cgs.value
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
                    (1/a_Rs) * pm.math.sqrt( (1+r)**2 - b**2 )
                    * (1/sini)
                )*24
            )

            T_13 =  pm.Deterministic(
                'T_13',
                (period/np.pi)*
                tt.arcsin(
                    (1/a_Rs) * pm.math.sqrt( (1-r)**2 - b**2 )
                    * (1/sini)
                )*24
            )

            # Optimizing
            map_estimate = pm.find_MAP(model=model)

            # start = model.test_point
            # if 'transit' in self.modelcomponents:
            #     map_estimate = xo.optimize(start=start,
            #                                vars=[r, b, period, t0])
            # map_estimate = xo.optimize(start=map_estimate)

            if make_threadsafe:
                pass
            else:
                # NOTE: would usually plot MAP estimate here, but really
                # there's not a huge need.
                print(map_estimate)
                pass

            # sample from the posterior defined by this model.
            trace = pmx.sample(
                tune=self.N_samples, draws=self.N_samples,
                start=map_estimate, cores=self.N_cores,
                chains=self.N_chains,
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

            # ecs = xo.UnitDisk("ecs", shape=(2, 1), testval=0.01 * np.ones((2, 1)))
            # ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2, axis=0))
            # omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
            # xo.eccentricity.vaneylen19("ecc_prior", multi=False, shape=1, observed=ecc)

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

            # start = model.test_point
            # if 'transit' in self.modelcomponents:
            #     map_estimate = xo.optimize(start=start,
            #                                vars=[r, b, period, t0])
            # map_estimate = xo.optimize(start=map_estimate)

            if make_threadsafe:
                pass
            else:
                # NOTE: would usually plot MAP estimate here, but really
                # there's not a huge need.
                print(map_estimate)
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


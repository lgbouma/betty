import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import pickle, os
from astropy import units as units, constants as const
from numpy import array as nparr
from functools import partial
from collections import OrderedDict

import exoplanet as xo
from exoplanet.gp import terms, GP
import theano.tensor as tt

from betty.constants import factor

class ModelParser:

    def __init__(self, modelid):
        self.initialize_model(modelid)

    def initialize_model(self, modelid):
        self.modelid = modelid
        self.modelcomponents = modelid.split('_')
        self.verify_modelcomponents()

    def verify_modelcomponents(self):

        validcomponents = ['simpletransit', 'rv', 'allindivtransit']

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
    Given a modelid of the form "*transit", or "rv" and a dataframe containing
    (time and flux), or (time and rv), run the inference.
    """

    def __init__(self, modelid, data_df, priordict, N_samples=2000, N_cores=16,
                 target_accept=0.8, N_chains=4, plotdir=None, pklpath=None,
                 overwrite=1, rvdf=None):

        self.N_samples = N_samples
        self.N_cores = N_cores
        self.N_chains = N_chains
        self.PLOTDIR = plotdir
        self.OVERWRITE = overwrite

        implemented_models = ['simpletransit', 'allindivtransit',
                              'oddindivtransit', 'evenindivtransit']

        if modelid in implemented_models:
            assert isinstance(data_df, OrderedDict)
            self.data = data_df
            self.priordict = priordict
            #FIXME FIXME FIXME probably can remove from other functions, if you
            # assign it here...

        if 'rv' in modelid:
            raise NotImplementedError

        self.initialize_model(modelid)

        if modelid not in implemented_models:
            raise NotImplementedError

        # hard-code against thread safety. (PyMC3 + matplotlib).
        make_threadsafe = False

        if modelid == 'simpletransit':
            self.run_transit_inference(
                pklpath, make_threadsafe=make_threadsafe
            )

        elif modelid == 'rv':
            self.run_rv_inference(
                pklpath, make_threadsafe=make_threadsafe
            )

        elif modelid in ['allindivtransit', 'oddindivtransit',
                         'evenindivtransit']:
            self.run_allindivtransit_inference(
                pklpath, make_threadsafe=make_threadsafe,
                target_accept=target_accept
            )


    def run_transit_inference(self, pklpath, make_threadsafe=True):
        """
        Fit transit data for an Agol+19 transit. (Ignores any stellar
        variability; believes error bars).
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
                period=period, t0=t0, b=b, rho_star=rho_star
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

            # sample from the posterior defined by this model.
            trace = pm.sample(
                tune=self.N_samples, draws=self.N_samples,
                start=map_estimate, cores=self.N_cores,
                chains=self.N_chains,
                step=xo.get_dense_nuts_step(target_accept=0.8),
            )

        with open(pklpath, 'wb') as buff:
            pickle.dump({'model': model, 'trace': trace,
                         'map_estimate': map_estimate}, buff)

        self.model = model
        self.trace = trace
        self.map_estimate = map_estimate


    def run_allindivtransit_inference(self, priordict, pklpath,
                                      make_threadsafe=True, target_accept=0.8):
        """
        Fit all transits (TESS+ground), allowing each transit a quadratic trend. No
        "pre-detrending".
        """

        # NOTE: see timmy.run_allindivtransit_inference
        raise NotImplementedError


    def run_rv_inference(self, priordict, pklpath, make_threadsafe=True):
        """
        Fit RVs for Keplerian orbit.
        """

        raise NotImplementedError
        # NOTE: but see draft in timmy.modelfitter

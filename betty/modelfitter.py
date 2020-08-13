import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import pickle, os
from astropy import units as units, constants as const
from numpy import array as nparr
from functools import partial
from collections import OrderedDict

import exoplanet as xo
from exoplanet.gp import terms, GP
import theano.tensor as tt

from betty.plotting import plot_MAP_data as plot_MAP_phot
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
    Given a modelid of the form "transit", or "rv" and a dataframe containing
    (time and flux), or (time and rv), run the inference.
    """

    def __init__(self, modelid, data_df, prior_d, N_samples=2000, N_cores=16,
                 target_accept=0.8, N_chains=4, plotdir=None, pklpath=None,
                 overwrite=1, rvdf=None):

        self.N_samples = N_samples
        self.N_cores = N_cores
        self.N_chains = N_chains
        self.PLOTDIR = plotdir
        self.OVERWRITE = overwrite

        if 'transit' == modelid:
            self.data = data_df
            self.x_obs = nparr(data_df['x_obs'])
            self.y_obs = nparr(data_df['y_obs'])
            self.y_err = nparr(data_df['y_err'])
            self.t_exp = np.nanmedian(np.diff(self.x_obs))

        #FIXME remove
        if modelid in ['alltransit', 'alltransit_quad',
                       'alltransit_quaddepthvar', 'onetransit',
                       'allindivtransit', 'tessindivtransit',
                       'oddindivtransit', 'evenindivtransit']:
            assert isinstance(data_df, OrderedDict)
            self.data = data_df

        if 'rv' in modelid:
            raise NotImplementedError

        self.initialize_model(modelid)

        #FIXME remove
        if modelid not in ['alltransit', 'alltransit_quad',
                           'alltransit_quaddepthvar', 'onetransit',
                           'allindivtransit', 'tessindivtransit',
                           'oddindivtransit', 'evenindivtransit']:
            self.verify_inputdata()

        #NOTE threadsafety needn't be hardcoded
        make_threadsafe = False

        #FIXME remove
        if modelid == 'transit':
            self.run_transit_inference(
                prior_d, pklpath, make_threadsafe=make_threadsafe
            )

        elif modelid == 'onetransit':
            self.run_onetransit_inference(
                prior_d, pklpath, make_threadsafe=make_threadsafe
            )

        elif modelid == 'rv':
            self.run_rv_inference(
                prior_d, pklpath, make_threadsafe=make_threadsafe
            )

        elif modelid in ['alltransit', 'alltransit_quad',
                         'alltransit_quaddepthvar']:
            self.run_alltransit_inference(
                prior_d, pklpath, make_threadsafe=make_threadsafe
            )

        elif modelid in ['allindivtransit', 'tessindivtransit',
                         'oddindivtransit', 'evenindivtransit']:
            self.run_allindivtransit_inference(
                prior_d, pklpath, make_threadsafe=make_threadsafe,
                target_accept=target_accept
            )


    def verify_inputdata(self):
        np.testing.assert_array_equal(
            self.x_obs,
            self.x_obs[np.argsort(self.x_obs)]
        )
        assert len(self.x_obs) == len(self.y_obs)
        assert isinstance(self.x_obs, np.ndarray)
        assert isinstance(self.y_obs, np.ndarray)


    def run_transit_inference(self, prior_d, pklpath, make_threadsafe=True):
        """
        Fit all data together for a Mandel-Agol transit. (Ignores any stellar
        variability; believes error bars).
        """

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
            logg_star = pm.Normal("logg_star", mu=LOGG, sd=LOGG_STDEV)
            r_star = pm.Bound(pm.Normal, lower=0.0)(
                "r_star", mu=RSTAR, sd=RSTAR_STDEV
            )
            rho_star = pm.Deterministic(
                "rho_star", factor*10**logg_star / r_star
            )

            # fix Rp/Rs across bandpasses, b/c you're assuming it's a planet
            if 'quaddepthvar' not in self.modelid:
                log_r = pm.Uniform('log_r', lower=np.log(1e-2),
                                   upper=np.log(1), testval=prior_d['log_r'])
                r = pm.Deterministic('r', tt.exp(log_r))
            else:

                log_r_Tband = pm.Uniform('log_r_Tband', lower=np.log(1e-2),
                                         upper=np.log(1),
                                         testval=prior_d['log_r_Tband'])
                r_Tband = pm.Deterministic('r_Tband', tt.exp(log_r_Tband))

                log_r_Rband = pm.Uniform('log_r_Rband', lower=np.log(1e-2),
                                         upper=np.log(1),
                                         testval=prior_d['log_r_Rband'])
                r_Rband = pm.Deterministic('r_Rband', tt.exp(log_r_Rband))

                log_r_Bband = pm.Uniform('log_r_Bband', lower=np.log(1e-2),
                                         upper=np.log(1),
                                         testval=prior_d['log_r_Bband'])
                r_Bband = pm.Deterministic('r_Bband', tt.exp(log_r_Bband))

                r = r_Tband


            # Some orbital parameters
            t0 = pm.Normal(
                "t0", mu=prior_d['t0'], sd=5e-3, testval=prior_d['t0']
            )
            period = pm.Normal(
                'period', mu=prior_d['period'], sd=5e-3,
                testval=prior_d['period']
            )
            b = xo.distributions.ImpactParameter(
                "b", ror=r, testval=prior_d['b']
            )
            orbit = xo.orbits.KeplerianOrbit(
                period=period, t0=t0, b=b, rho_star=rho_star
            )

            # NOTE: limb-darkening should be bandpass specific, but we don't
            # have the SNR to justify that, so go with TESS-dominated
            u0 = pm.Uniform(
                'u[0]', lower=prior_d['u[0]']-0.15,
                upper=prior_d['u[0]']+0.15,
                testval=prior_d['u[0]']
            )
            u1 = pm.Uniform(
                'u[1]', lower=prior_d['u[1]']-0.15,
                upper=prior_d['u[1]']+0.15,
                testval=prior_d['u[1]']
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
                        "mean", mu=prior_d[f'{name}_mean'], sd=1e-2,
                        testval=prior_d[f'{name}_mean']
                    )

                    if 'quad' in self.modelid:

                        if name != 'tess':

                            # units: rel flux per day.
                            a1 = pm.Normal(
                                "a1", mu=prior_d[f'{name}_a1'], sd=1,
                                testval=prior_d[f'{name}_a1']
                            )
                            # units: rel flux per day^2.
                            a2 = pm.Normal(
                                "a2", mu=prior_d[f'{name}_a2'], sd=1,
                                testval=prior_d[f'{name}_a2']
                            )

                if self.modelid == 'alltransit':
                    lc_models[name] = pm.Deterministic(
                        f'{name}_mu_transit',
                        mean +
                        star.get_light_curve(
                            orbit=orbit, r=r, t=x, texp=texp
                        ).T.flatten()
                    )

                elif self.modelid == 'alltransit_quad':

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


    def run_allindivtransit_inference(self, prior_d, pklpath,
                                      make_threadsafe=True, target_accept=0.8):
        """
        Fit all transits (TESS+ground), allowing each transit a quadratic trend. No
        "pre-detrending".
        """

        # NOTE: see timmy.run_allindivtransit_inference
        raise NotImplementedError


    def run_rv_inference(self, prior_d, pklpath, make_threadsafe=True):
        """
        Fit RVs for Keplerian orbit.
        """

        raise NotImplementedError
        # NOTE: but see draft in timmy.modelfitter

"""
ModelParser
ModelFitter
"""
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

        validcomponents = ['simpletransit', 'rvspotorbit', 'allindivtransit',
                           'alltransit']

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
                 target_accept=0.8, N_chains=4, plotdir=None, pklpath=None,
                 overwrite=1):

        self.N_samples = N_samples
        self.N_cores = N_cores
        self.N_chains = N_chains
        self.PLOTDIR = plotdir
        self.OVERWRITE = overwrite

        implemented_models = ['simpletransit', 'allindivtransit',
                              'oddindivtransit', 'evenindivtransit',
                              'rvorbit', 'rvspotorbit', 'alltransit']

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
            self.run_transit_inference(
                pklpath, make_threadsafe=make_threadsafe
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
                step=xo.get_dense_nuts_step(target_accept=0.8),
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


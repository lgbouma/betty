"""
MAP fitting recipes. Contents include:

    flatten_starspots

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

def flatten_starspots(time, flux, flux_err, p_rotation):

    x,y,yerr = time,flux,flux_err

    def build_model(mask=None, start=None):

        if mask is None:
            mask = np.ones(len(x), dtype=bool)

        with pm.Model() as model:

            mean = pm.Normal("mean", mu=1., sigma=0.2)

            # A jitter term describing excess white noise
            #log_jitter = pm.Normal(
            #    "log_jitter", mu=np.log(np.nanmean(yerr[mask])), sigma=2.0
            #)

            # The parameters of the RotationTerm kernel
            sigma_rot = pm.InverseGamma(
                "sigma_rot", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
            )
            log_period = pm.Normal("log_period", mu=np.log(p_rotation), sigma=1.0)
            period = pm.Deterministic("period", tt.exp(log_period))
            log_Q0 = pm.HalfNormal("log_Q0", sigma=2.0)
            log_dQ = pm.Normal("log_dQ", mu=0.0, sigma=0.5)
            f = pm.Uniform("f", lower=0.001, upper=1.0)

            # Set up the Gaussian Process model
            kernel = terms.RotationTerm(
                sigma=sigma_rot, period=period, Q0=tt.exp(log_Q0),
                dQ=tt.exp(log_dQ), f=f,
            )
            gp = GaussianProcess(
                kernel, t=x[mask],
                diag=yerr[mask] ** 2,# + tt.exp(2 * log_jitter),
                mean=mean, quiet=True,
            )

            # Compute the Gaussian Process likelihood and add it into the
            # the PyMC3 model as a "potential"
            gp.marginal("gp", observed=y[mask])

            # Compute the mean model prediction for plotting purposes
            # [without masking!]
            pm.Deterministic("pred", gp.predict(y[mask], t=x))

            # Optimize to find the maximum a posteriori parameters
            np.random.seed(42)
            if start is None:
                start = model.test_point

            map_soln = start

            map_soln = pmx.optimize(
                start=map_soln,
                vars=[sigma_rot, f, log_period]
            )

            map_soln = pmx.optimize(start=map_soln)

        return model, map_soln

    model0, map_soln0 = build_model()
    print(map_soln0)

    resid = (y - map_soln0['pred'])
    mad = np.nanmedian(np.abs(resid))
    mask = np.abs(resid) < 3 * mad

    model1, map_soln1 = build_model(mask=mask, start=None)
    print(map_soln1)

    trend_flux = map_soln1['pred']
    flat_flux = flux/trend_flux

    return flat_flux, trend_flux

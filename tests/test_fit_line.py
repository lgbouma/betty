import pytest
import numpy as np, matplotlib.pyplot as plt, pymc3 as pm
import pickle, os, corner

from os.path import join
from pymc3.backends.tracetab import trace_to_dataframe
from betty.paths import TESTRESULTSDIR, BETTYDIR

n_cpu = os.cpu_count()

@pytest.mark.skip(reason="This is a boiler-plate test with no added value.")
def test_fit_line():

    #################
    # generate data #
    #################
    size = 200
    true_intercept = 1
    true_slope = 2
    true_sigma = 0.5

    x = np.linspace(0, 1, size)
    # y = a + b*x
    true_regression_line = true_intercept + true_slope * x
    # add noise
    y = true_regression_line + np.random.normal(scale=true_sigma, size=size)

    data = dict(x=x, y=y)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, xlabel='x', ylabel='y',
                         title='Generated data and underlying model')

    ax.plot(x, y, 'x', label='sampled data')
    ax.plot(x, true_regression_line, label='true regression line', lw=2.)
    plt.legend(loc=0)
    path0 = join(TESTRESULTSDIR, 'test_fit_line_data.png')
    plt.savefig(path0)
    plt.close('all')

    assert os.path.exists(path0)

    ####################################
    # fit a line and sample parameters #
    ####################################
    pklpath = join(BETTYDIR, 'test_fit_model_line.pkl')

    if not os.path.exists(pklpath):
        with pm.Model() as model:
            # Define priors
            sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
            intercept = pm.Normal('Intercept', 0, sigma=20)
            x_coeff = pm.Normal('x', 0, sigma=20)

            # Define likelihood
            # Here Y ~ N(Xβ, σ^2), for β the coefficients of the model. Note
            # though the error bars are not _observed_ in this case; they are
            # part of the model!
            likelihood = pm.Normal('y', mu=intercept + x_coeff * x,
                                   sigma=sigma, observed=y)

            # Inference! Draw 1000 posterior samples using NUTS sampling.
            trace = pm.sample(1000, cores=n_cpu)

        with open(pklpath, 'wb') as buff:
            pickle.dump({'model': model, 'trace': trace}, buff)
    else:
        d = pickle.load(open(pklpath, 'rb'))
        model, trace = d['model'], d['trace']

    assert os.path.exists(pklpath)

    ##################
    # analyze output #
    ##################
    # trace plot from PyMC3
    plt.figure(figsize=(7, 7))
    pm.plot_trace(trace[100:])
    plt.tight_layout()
    path1 = join(TESTRESULTSDIR, 'test_fit_line_traceplot.png')
    plt.savefig(path1)
    plt.close('all')

    assert os.path.exists(path1)

    # overplot posterior samples in data space
    plt.figure(figsize=(7, 7))
    plt.plot(x, y, 'x', label='data')
    pm.plot_posterior_predictive_glm(
        trace, samples=100, label='posterior predictive lines'
    )
    plt.plot(x, true_regression_line, label='true line', lw=3., c='y')
    plt.title('Posterior predictive regression lines')
    plt.legend(loc=0)
    plt.xlabel('x')
    plt.ylabel('y')
    path2 = join(TESTRESULTSDIR, 'test_fit_line_posterior_samples.png')
    plt.savefig(path2)
    plt.close('all')

    assert os.path.exists(path2)

    # corner
    trace_df = trace_to_dataframe(trace)
    fig = corner.corner(trace_df, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        truths=[true_intercept, true_slope, true_sigma])
    path3 = join(TESTRESULTSDIR, 'test_fit_line_corner.png')
    fig.savefig(path3)

    assert os.path.exists(path3)


if __name__ == "__main__":
    test_fit_line()

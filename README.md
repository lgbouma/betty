# betty

<img src="https://github.com/lgbouma/betty/workflows/Tests/badge.svg">

This repo contains reusable recipes for the
[exoplanet](https://github.com/exoplanet-dev/exoplanet) fitting code.

Many routines are similar to those in
[astrobase.lcfit](https://astrobase.readthedocs.io/en/latest/astrobase.lcfit.html#),
except we're using PyMC3 to sample instead of emcee. This improves runtime
speed significantly.

### Install

Clone + `python setup.py install`.

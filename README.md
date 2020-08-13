# betty

<img src="https://github.com/lgbouma/betty/workflows/Tests/badge.svg">


This repo contains cookbook routines for using the
[exoplanet](https://github.com/exoplanet-dev/exoplanet) fitting code.

The main routines are similar to those implemented in
[astrobase.lcfit](https://astrobase.readthedocs.io/en/latest/astrobase.lcfit.html#),
except we're using PyMC3 to sample instead of emcee. This affects runtime speed
significantly.

### Install

Clone + `python setup.py install` from the repo. (or develop!)

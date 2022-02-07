"""
Contents:
    given_priordict_make_priorfile
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

#############
## IMPORTS ##
#############

import os

def given_priordict_make_priorfile(priordict, priorpath):
    """
    The betty prior file sometimes is easier made first in an analysis script
    (e.g., "fit_models_to_gold", in the CDIPS project).  But then, it's a good
    idea to cache it for future reference of what you fitted.
    """
    # by default, will overwrite
    firstline = 'import numpy as np\n\n'
    nextlines = "),\n".join(repr(priordict).split("),"))
    outlines = firstline+'priordict = '+nextlines
    with open(priorpath, mode='w') as f:
        f.writelines(outlines)
    print(f'Wrote {priorpath}')

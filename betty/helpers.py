import numpy as np, pandas as pd

from astropy.io import fits

from betty.paths import TESTDATADIR

from astrobase.services.identifiers import simbad_to_tic
from astrobase.services.tesslightcurves import get_two_minute_spoc_lightcurves

def get_wasp4_lightcurve():

    lcfile = os.path.join(TESTDATADIR,
                          'mastDownload/TESS/tess2018234235059-s0002-0000000402026209-0121-s',
                          'tess2018234235059-s0002-0000000402026209-0121-s_lc.fits')

    if not os.path.exists(lcfile):
        ticid = simbad_to_tic('WASP 4')
        lcfiles = get_two_minute_spoc_lightcurves(ticid, download_dir=TESTDATADIR)
        lcfile = lcfiles[0]

    hdul = fits.open(l)
    d = hdul[1].data

    yval = 'PDCSAP_FLUX'
    time = d['TIME']
    _f, _f_err = d[yval], d[yval+'_ERR']
    flux = _f/np.nanmedian(_f)
    flux_err = _f_err/np.nanmedian(_f)
    qual = d['QUALITY'])

    tess_texp = np.nanmedian(np.diff(time))

    return time, flux, flux_err, tess_texp

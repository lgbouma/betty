import numpy as np
from astropy import units as u

# normal: mu, sd
# uniform: lower, upper, testval
# ImpactParameter: testval

# teff 4780K
# logg 3.50

u0 = 0.46
u1 = 0.32
delta_u = 0.001

priordict = {
'period': ('Normal', 3.0, 0.1), # BLS on HATNet (2.99524)
't0': ('Normal', 56197.75437, 0.01), # BLS on HATNet
'log_ror': ('Uniform', np.log(1e-3), np.log(1), np.log(  ((0.9*u.Rjup)/(3.220*u.Rsun)).cgs.value )),
'b': ('ImpactParameter', 0.52),
# 'u[0]': ('Uniform', u0-delta_u, u0+delta_u, u0),
# 'u[1]': ('Uniform', u1-delta_u, u1+delta_u, u1),
'u_star': ('QuadLimbDark',),
'r_star': ('Normal', 3.220, 0.062), # Table1 Grunblatt+22
'logg_star': ('Normal', 3.50, 0.06), # Table1 Grunblatt+22
'hatnet_mean': ('Normal', 1, 0.1),
'log_jitter': ('Normal', r"\log\langle \sigma_f \rangle", 2.0),
}

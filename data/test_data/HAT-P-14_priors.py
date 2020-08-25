import numpy as np

# normal: mu, sd
# uniform: lower, upper, testval
# ImpactParameter: testval

u0 = 0.2996 # Claret+17, Teff 6727, logg below
u1 = 0.2316
delta_u = 0.2

priordict = {
'period': ('Normal', 4.62787, 0.01), # SPOC widened
't0': ('Normal', 1984.6530, 0.05), # SPOC widened
'log_r': ('Uniform', np.log(1e-2), np.log(1), np.log(0.0803)),
'b': ('ImpactParameter', 0.8),
'u[0]': ('Uniform', u0-delta_u, u0+delta_u, u0),
'u[1]': ('Uniform', u1-delta_u, u1+delta_u, u1),
# 'r_star': ('Normal', 1.54522, 0.069), # TIC8
# 'logg_star': ('Normal', 4.21633, 0.09015), # TIC8
'r_star': ('Normal', 1.468, 0.054), # Torres+10
'logg_star': ('Normal', 4.25, 0.03), # Torres+10
'tess_mean': ('Normal', 1, 0.1)
}

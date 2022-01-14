import numpy as np

# normal: mu, sd
# uniform: lower, upper, testval
# ImpactParameter: testval

priordict = {
'period': ('Normal', 1.338231466, 0.005),
't0': ('Normal', 1355.1845, 0.02),
'log_ror': ('Uniform', np.log(1e-2), np.log(1), np.log(0.15201)),
'b': ('ImpactParameter', 0.5),
'u[0]': ('Uniform', 0.382-0.2, 0.382+0.2, 0.382),
'u[1]': ('Uniform', 0.210-0.2, 0.210+0.2, 0.210),
'r_star': ('Normal', 0.893, 0.034),
'logg_star': ('Normal', 4.47, 0.11),
'tess_mean': ('Normal', 1, 0.1)
}

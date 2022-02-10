import numpy as np

priordict = {'period': ('Normal', 2.158716092397785, 0.02158716092397785),
 't0': ('Normal', 2458684.7037610332, 0.020833333333333332),
 'log_ror': ('Uniform', -3.180656504322065, 0.0, -2.029363957825042),
 'b': ('ImpactParameter', 0.5),
 'u_star': ('QuadLimbDark',),
 'r_star': ('Normal', 1.04535, 0.0500503),
 'logg_star': ('Normal', 4.42485, 0.0792532),
 'log_jitter': ('Normal', '\\log\\langle \\sigma_f \\rangle', 2.0)}

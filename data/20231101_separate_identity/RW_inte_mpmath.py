###########################################################
# Integration with mpmath
# for the shifted Pareto, no nugget
# pRW_mpmath, dRW_mpmath, qRW_mpmath
###########################################################

from mpmath import mp
import numpy as np
mp.dps = 30

# mpmath dRW
def dRW_integrand_mpmath(r, x, phi, gamma):
    numerator = mp.power(r, phi-1.5)
    denominator = mp.power(x + mp.power(r, phi), 2)
    exp = mp.exp(-(gamma/(2*r)))
    return numerator / denominator * exp
def dRW_mpmath(x, phi, gamma, **kwargs):
    return mp.sqrt(gamma/(2 * mp.pi)) * mp.quad(lambda r : dRW_integrand_mpmath(r, x, phi, gamma), [0, mp.inf], method='tanh-sinh', **kwargs)
dRW_mpmath_vec = np.vectorize(dRW_mpmath) # return a np.array of mpf
def dRW_mpmath_vec_float(x, phi, gamma): # return a np.array of floats
    return(dRW_mpmath_vec(x, phi, gamma).astype(float))

# mpmath pRW
def pRW_integrand_mpmath(r, x, phi, gamma):
    numerator = mp.power(r, phi-1.5)
    denominator = x + mp.power(r, phi)
    exp = mp.exp(-(gamma/(2*r)))
    return numerator / denominator * exp
def pRW_mpmath(x, phi, gamma):
    return 1 - mp.sqrt(gamma/(2*mp.pi)) * mp.quad(lambda r : pRW_integrand_mpmath(r, x, phi, gamma), [0, mp.inf], method='tanh-sinh')
pRW_mpmath_vec = np.vectorize(pRW_mpmath) # return a np.array of mpf
def pRW_mpmath_vec_float(x, phi, gamma): # return a np.array of floats
    return(pRW_mpmath_vec(x, phi, gamma).astype(float))

# mpmath transform dRW -- no significant gain in terms of accuracy as compared to dRW_mpmath
# mpmath with high dps can handle integration from [0, mp.inf] well
def dRW_integrand_transformed_mpmath(t, x, phi, gamma):
    numerator = mp.power((1-t)/t, phi-1.5)
    denominator = mp.power(x + mp.power((1-t)/t, phi), 2)
    exp = mp.exp(-gamma/(2*(1-t)/t))
    jacobian = 1 / mp.power(t, 2)
    return (numerator / denominator) * exp * jacobian
def dRW_transformed_mpmath(x, phi, gamma):
    return mp.sqrt(gamma/(2 * mp.pi)) * mp.quad(lambda t : dRW_integrand_transformed_mpmath(t, x, phi, gamma), [0, 1], method='tanh-sinh')

# mpmath transform pRW -- no significant gain in terms of accuracy, as compared to pRW_mpmath
# mpmath with high dps can handle integration from [0, mp.inf] well
def pRW_integrand_transformed_mpmath(t, x, phi, gamma):
    numerator = mp.power((1-t)/t, phi-1.5)
    denominator = x + mp.power((1-t)/t, phi)
    exp = mp.exp(- gamma / (2 * (1-t)/t))
    jacobian = 1 / mp.power(t, 2)
    return numerator / denominator * exp * jacobian
def pRW_transformed_mpmath(x, phi, gamma):
    return 1 - mp.sqrt(gamma/(2*mp.pi)) * mp.quad(lambda t: pRW_integrand_transformed_mpmath(t, x, phi, gamma), [0, 1], method='tanh-sinh')

# mpmath quantile function
def qRW_mpmath(p, phi, gamma):
    return mp.findroot(lambda x : pRW_mpmath(x, phi, gamma) - p,
                       [0,1e12],
                       solver='anderson')
qRW_mpmath_vec = np.vectorize(qRW_mpmath) # return a np.array of mpf
def qRW_mpmath_vec_float(p, phi, gamma): # return a np.array of floats
    return(qRW_mpmath_vec(p, phi, gamma).astype(float))
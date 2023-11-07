# %%
import numpy as np
import scipy
from scipy.integrate import quad
from mpmath import mp
import model_sim
from numba import jit
# %%
###########################################################
# Integration with GSL
# for the shifted Pareto, no nugget
# pRW_cpp, dRW_cpp, qRW_cpp
###########################################################
dRW_cpp = model_sim.dRW
pRW_cpp = model_sim.pRW
def qRW_cpp(p, phi, gamma):
    return(model_sim.qRW_Newton(p, phi, gamma, 100)) # qRW_Newton is vectorized


# %%
###########################################################
# Integration with mpmath
# for the shifted Pareto, no nugget
# pRW_mpmath, dRW_mpmath, qRW_mpmath
###########################################################
mp.dps = 15

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

# %%
###########################################################
# Integration with scipy QUAD                             #
# for the shifted Pareto, no nugget                       #
# pRW_scipy, dRW_scipy, qRW_scipy                         #
###########################################################

# scipy dRW
@jit(nopython=True)
def dRW_integrand_scipy(r, x, phi, gamma):
    return (r**(phi-3/2)) / ((x+r**phi)**2) * np.exp(-(gamma/(2*r)))
def dRW_scipy(x, phi, gamma):
    return np.sqrt(gamma/(2*np.pi)) * quad(dRW_integrand_scipy, 0, np.inf, args=(x, phi, gamma))[0]
dRW_scipy_vec = np.vectorize(dRW_scipy, otypes=[float])

# scipy pRW
@jit(nopython=True)
def pRW_integrand_scipy(r, x, phi, gamma):
    numerator = np.power(r, phi-1.5)
    denominator = x + np.power(r, phi)
    exp = np.exp(-(gamma/(2*r)))
    return numerator / denominator * exp
def pRW_scipy(x, phi, gamma):
    return 1 - np.sqrt(gamma/(2*np.pi)) * quad(pRW_integrand_scipy, 0, np.inf, args=(x, phi, gamma))[0]
pRW_scipy_vec = np.vectorize(pRW_scipy, otypes=[float]) # return a np.array of mpf

# scipy dRW transformed between [0,1]
@jit(nopython=True)
def dRW_integrand_transformed_scipy(t, x, phi, gamma):
    ratio_numerator = np.power((1-t)/t, phi-1.5)
    ratio_denominator = (x + np.power((1-t)/t, phi))**2
    exponential_term = np.exp(-gamma/(2*((1-t)/t)))
    jacobian = 1/(t**2)
    return (ratio_numerator/ratio_denominator) * exponential_term * jacobian
def dRW_transformed_scipy(x, phi, gamma):
    return np.sqrt(gamma/(2*np.pi)) * quad(dRW_integrand_transformed_scipy, 0, 1, args=(x, phi, gamma))[0]
dRW_transformed_scipy_vec = np.vectorize(dRW_transformed_scipy, otypes=[float])

# scipy pRW transformed between [0,1]
@jit(nopython=True)
def pRW_integrand_transformed_scipy(t, x, phi, gamma):
    numerator = np.power((1-t)/t, phi-1.5)
    denominator = x + np.power((1-t)/t, phi)
    exp = np.exp(- gamma / (2 * (1-t)/t))
    jacobian = 1 / np.power(t, 2)
    return numerator / denominator * exp * jacobian
def pRW_transformed_scipy(x, phi, gamma):
    return 1 - np.sqrt(gamma/(2*np.pi)) * quad(pRW_integrand_transformed_scipy, 0, 1, args=(x, phi, gamma))[0]
pRW_transformed_scipy_vec = np.vectorize(pRW_transformed_scipy, otypes=[float])

def qRW_scipy(p, phi, gamma):
    return scipy.optimize.root_scalar(lambda x: pRW_transformed_scipy(x, phi, gamma) - p,
                                     bracket=[0.1, 1e12],
                                     fprime = lambda x: dRW_transformed_scipy(x, phi, gamma),
                                     x0 = 10).root
qRW_scipy_vec = np.vectorize(qRW_scipy, otypes=[float])

# %%
###########################################################
# Incomplete gamma functions with GSL                     #
# for the nonshifted Pareto, no nugget                    #
# pRW_stdPareto, dRW_stdPareto, qRW_stdPareto             #
###########################################################

# using the GSL incomplete gamma function is much (10x) faster than mpmath
upper_gamma_C = np.vectorize(model_sim.lib.upper_gamma_C, otypes = [float])
lower_gamma_C = np.vectorize(model_sim.lib.lower_gamma_C, otypes = [float])
def dRW_stdPareto(x, phi, gamma):
    upper_gamma = upper_gamma_C(0.5 - phi, gamma / (2*np.power(x, 1/phi)))
    # upper_gamma = float(mpmath.gammainc(0.5 - phi, a = gamma / (2*np.power(x, 1/phi))))
    return (1/np.power(x,2)) * np.sqrt(1/np.pi) * np.power(gamma/2, phi) * upper_gamma
def pRW_stdPareto(x, phi, gamma):
    lower_gamma = lower_gamma_C(0.5, gamma / (2*np.power(x, 1/phi)))
    upper_gamma = upper_gamma_C(0.5 - phi, gamma / (2*np.power(x, 1/phi)))
    # lower_gamma = float(mpmath.gammainc(0.5, b = gamma / (2*np.power(x, 1/phi))))
    # upper_gamma = float(mpmath.gammainc(0.5 - phi, a = gamma / (2*np.power(x, 1/phi))))
    survival = np.sqrt(1/np.pi) * lower_gamma + (1/x) * np.sqrt(1/np.pi) * np.power(gamma/2, phi) * upper_gamma
    return 1 - survival
def qRW_stdPareto(p, phi, gamma):
    return scipy.optimize.root_scalar(lambda x: pRW_stdPareto(x, phi, gamma) - p,
                                      bracket=[0.1,1e12],
                                      fprime = lambda x: dRW_stdPareto(x, phi, gamma),
                                      x0 = 10,
                                      method='ridder').root
dRW_stdPareto_vec = np.vectorize(dRW_stdPareto, otypes=[float])
pRW_stdPareto_vec = np.vectorize(pRW_stdPareto, otypes=[float])
qRW_stdPareto_vec = np.vectorize(qRW_stdPareto, otypes=[float])
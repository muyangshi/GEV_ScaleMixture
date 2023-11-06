# combined utilitlies helpful to MCMC sampler
# grab the functions from model_sim.py and data_simulation_1_radius.py
# Require:
#   - p_inte.cpp & p_inte.so
#   - model_sim.py, ns_cov.py
#   - RW_inte_mpmath.py
# %%
# general imports and ubiquitous utilities
import numpy as np
from scipy.linalg import lapack
from ns_cov import *
import RW_inte_mpmath
import model_sim
import scipy
import mpmath

# specify integration and transformation
#############################
dRW_method = 'cpp'          #
pRW_method = 'cpp'          #
qRW_method = 'cpp'          #
# norm_pareto = 'shifted'   #
norm_pareto = 'standard'    #
#############################

weights_fun = model_sim.weights_fun
wendland_weights_fun = model_sim.wendland_weights_fun
rlevy = model_sim.rlevy
pgev = model_sim.pgev
qgev = model_sim.qgev
dgev = model_sim.dgev

# %%
# specify g(Z) transformation

def norm_to_stdPareto(Z):
    pNorm = scipy.stats.norm.cdf(x = Z)
    return(scipy.stats.pareto.ppf(q = pNorm, b = 1))
norm_to_stdPareto_vec = np.vectorize(norm_to_stdPareto)
def stdPareto_to_Norm(W):
    pPareto = scipy.stats.pareto.cdf(x = W, b = 1)
    return(scipy.stats.norm.ppf(q = pPareto))
stdPareto_to_Norm_vec = np.vectorize(stdPareto_to_Norm)

if norm_pareto == 'shifted':
    norm_to_Pareto = model_sim.norm_to_Pareto1
    pareto_to_Norm = model_sim.pareto1_to_Norm
else: # norm_pareto == 'standard'
    norm_to_Pareto = norm_to_stdPareto_vec
    pareto_to_Norm = stdPareto_to_Norm_vec

# %%
# specify which dRW, pRW, and qRW to use

dRW_cpp = model_sim.dRW
pRW_cpp = model_sim.pRW
def qRW_cpp(p, phi, gamma):
    return(model_sim.qRW_Newton(p, phi, gamma, 100)) # qRW_Newton is vectorized

dRW_mpmath = RW_inte_mpmath.dRW_mpmath_vec_float
pRW_mpmath = RW_inte_mpmath.pRW_mpmath_vec_float
qRW_mpmath = RW_inte_mpmath.qRW_mpmath_vec_float

# using the GSL incomplete gamma function is much (10x) faster than mpmath
upper_gamma_C = np.vectorize(model_sim.lib.upper_gamma_C, otypes = [float])
lower_gamma_C = np.vectorize(model_sim.lib.lower_gamma_C, otypes = [float])
# mpmath.mp.dps = 15
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
                                      method='ridder').root
dRW_stdPareto_vec = np.vectorize(dRW_stdPareto, otypes=[float])
pRW_stdPareto_vec = np.vectorize(pRW_stdPareto, otypes=[float])
qRW_stdPareto_vec = np.vectorize(qRW_stdPareto, otypes=[float])

if norm_pareto == 'standard':
    dRW = dRW_stdPareto_vec
    pRW = pRW_stdPareto_vec
    qRW = qRW_stdPareto_vec
else: # norm_pareto == 'shifted'
    dRW = dRW_cpp if dRW_method == 'cpp' else dRW_mpmath
    pRW = pRW_cpp if pRW_method == 'cpp' else pRW_mpmath
    qRW = qRW_cpp if qRW_method == 'cpp' else qRW_mpmath

# %%
# marginal likelihood using integration and transformation specified above 
def marg_transform_data_mixture_likelihood_1t_shifted(Y, X, Loc, Scale, Shape, phi_vec, gamma_vec, R_vec, cholesky_U):
    if(isinstance(Y, (int, np.int64, float))): Y=np.array([Y], dtype='float64')

    ## Initialize space to store the log-likelihoods for each observation:
    W_vec = X/R_vec**phi_vec

    Z_vec = pareto_to_Norm(W_vec)
    # part1 = -0.5*eig2inv_quadform_vector(V, 1/d, Z_vec)-0.5*np.sum(np.log(d)) # multivariate density
    cholesky_inv = lapack.dpotrs(cholesky_U,Z_vec)
    part1 = -0.5*np.sum(Z_vec*cholesky_inv[0])-np.sum(np.log(np.diag(cholesky_U))) # multivariate density

    ## Jacobian determinant
    part21 = 0.5*np.sum(Z_vec**2) # 1/standard Normal densities of each Z_j
    part22 = np.sum(-phi_vec*np.log(R_vec)-2*np.log(W_vec+1)) # R_j^phi_j/X_j^2
    part23 = np.sum(dgev(Y, Loc=Loc, Scale=Scale, Shape=Shape, log=True)-np.log(dRW(X, phi_vec, gamma_vec)))

    return part1 + part21 + part22 + part23

def marg_transform_data_mixture_likelihood_1t_standard(Y, X, Loc, Scale, Shape, phi_vec, gamma_vec, R_vec, cholesky_U):
    if(isinstance(Y, (int, np.int64, float))): Y=np.array([Y], dtype='float64')

    ## Initialize space to store the log-likelihoods for each observation:
    W_vec = X/R_vec**phi_vec

    # for standard Pareto, check for W out of range (W < 1 ?)
    if any(W_vec < 1):
        return np.NINF

    Z_vec = pareto_to_Norm(W_vec)
    # part1 = -0.5*eig2inv_quadform_vector(V, 1/d, Z_vec)-0.5*np.sum(np.log(d)) # multivariate density
    cholesky_inv = lapack.dpotrs(cholesky_U,Z_vec)
    part1 = -0.5*np.sum(Z_vec*cholesky_inv[0])-np.sum(np.log(np.diag(cholesky_U))) # multivariate density

    ## Jacobian determinant
    part21 = 0.5*np.sum(Z_vec**2) # 1/standard Normal densities of each Z_j
    part22 = np.sum(-phi_vec*np.log(R_vec)-2*np.log(W_vec+1)) # R_j^phi_j/X_j^2
    part23 = np.sum(dgev(Y, Loc=Loc, Scale=Scale, Shape=Shape, log=True)-np.log(dRW(X, phi_vec, gamma_vec)))

    return part1 + part21 + part22 + part23

if norm_pareto == 'shifted':
    marg_transform_data_mixture_likelihood_1t = marg_transform_data_mixture_likelihood_1t_shifted
else: # norm_pareto == 'standard'
    marg_transform_data_mixture_likelihood_1t = marg_transform_data_mixture_likelihood_1t_standard

# likelihood by parts
# for plotting and debug purposes
def marg_transform_data_mixture_likelihood_1t_detail(Y, X, Loc, Scale, Shape, phi_vec, gamma_vec, R_vec, cholesky_U):
    if(isinstance(Y, (int, np.int64, float))): Y=np.array([Y], dtype='float64')


    ## Initialize space to store the log-likelihoods for each observation:
    W_vec = X/R_vec**phi_vec

    Z_vec = pareto_to_Norm(W_vec)
    # part1 = -0.5*eig2inv_quadform_vector(V, 1/d, Z_vec)-0.5*np.sum(np.log(d)) # multivariate density
    cholesky_inv = lapack.dpotrs(cholesky_U,Z_vec)
    part1 = -0.5*np.sum(Z_vec*cholesky_inv[0])-np.sum(np.log(np.diag(cholesky_U))) # multivariate density

    ## Jacobian determinant
    part21 = 0.5*np.sum(Z_vec**2) # 1/standard Normal densities of each Z_j
    part22 = np.sum(-phi_vec*np.log(R_vec)-2*np.log(W_vec+1)) # R_j^phi_j/X_j^2
    part23 = np.sum(dgev(Y, Loc=Loc, Scale=Scale, Shape=Shape, log=True))
    part24 = np.sum(-np.log(dRW(X, phi_vec, gamma_vec)))

    return np.array([part1,part21 ,part22, part23, part24])
# combined utilitlies helpful to MCMC sampler
# grab the functions from model_sim.py and data_simulation_1_radius.py
# Require:
#   - model_sim.py, p_inte.cpp & p_inte.so
#   - RW_inte.py
#   - ns_cov.py
# %%
# general imports and ubiquitous utilities
import numpy as np
from scipy.linalg import lapack
from ns_cov import *
import RW_inte
import model_sim
import scipy

# specify integration and transformation
#############################
# inte_method = 'cpp'     # this is also bad beyond 5e3 don't use
# inte_method = 'scipy'   # Scipy QUAD is bad don't use 
# inte_method = 'mpmath'  # super slow don't use
inte_method = 'cpp_transformed'
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

if norm_pareto == 'standard':
    dRW = RW_inte.dRW_stdPareto_vec
    pRW = RW_inte.pRW_stdPareto_vec
    qRW = RW_inte.qRW_stdPareto_vec
else: # norm_pareto == 'shifted'
    if inte_method == 'cpp_transformed':
        dRW = RW_inte.dRW_transformed_cpp
        pRW = RW_inte.pRW_transformed_cpp
        qRW = RW_inte.qRW_transformed_cpp
    elif inte_method == 'cpp':
        dRW = RW_inte.dRW_cpp
        pRW = RW_inte.pRW_cpp
        qRW = RW_inte.qRW_cpp
    elif inte_method == 'mpmath':
        dRW = RW_inte.dRW_mpmath_vec_float
        pRW = RW_inte.pRW_mpmath_vec_float
        qRW = RW_inte.qRW_mpmath_vec_float
    elif inte_method == 'scipy':
        dRW = RW_inte.dRW_scipy_vec
        pRW = RW_inte.pRW_scipy_vec
        qRW = RW_inte.qRW_scipy_vec

# %%
# marginal likelihood using integration and transformation specified above 
def marg_transform_data_mixture_likelihood_1t_shifted(Y, X, Loc, Scale, Shape, phi_vec, gamma_vec, R_vec, cholesky_U):
    if(isinstance(Y, (int, np.int64, float))): Y=np.array([Y], dtype='float64')

    ## Initialize space to store the log-likelihoods for each observation:
    W_vec = X/R_vec**phi_vec

    Z_vec = pareto_to_Norm(W_vec)
    # part1 = -0.5*eig2inv_quadform_vector(V, 1/d, Z_vec)-0.5*np.sum(np.log(d)) # multivariate density
    # cholesky_inv = lapack.dpotrs(cholesky_U,Z_vec)
    # part1 = -0.5*np.sum(Z_vec*cholesky_inv[0])-np.sum(np.log(np.diag(cholesky_U))) # multivariate density
    Z_standard_normal = scipy.linalg.solve_triangular(cholesky_U, Z_vec, trans=1)
    part1 = -0.5*np.sum(Z_standard_normal**2) - np.sum(np.log(np.diag(cholesky_U)))

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
    # cholesky_inv = lapack.dpotrs(cholesky_U,Z_vec)
    # part1 = -0.5*np.sum(Z_vec*cholesky_inv[0])-np.sum(np.log(np.diag(cholesky_U))) # multivariate density
    Z_standard_normal = scipy.linalg.solve_triangular(cholesky_U, Z_vec, trans=1)
    part1 = -0.5*np.sum(Z_standard_normal**2) - np.sum(np.log(np.diag(cholesky_U)))

    ## Jacobian determinant
    part21 = 0.5*np.sum(Z_vec**2) # 1/standard Normal densities of each Z_j
    # part22 = np.sum(-phi_vec*np.log(R_vec)-2*np.log(W_vec+1)) # R_j^phi_j/X_j^2
    part22 = np.sum(-phi_vec*np.log(R_vec)-2*np.log(W_vec)) # standar Pareto no need W+1
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

    # for standard Pareto, check for W out of range (W < 1 ?)
    if norm_pareto == 'standard' and any(W_vec < 1):
        return np.NINF

    Z_vec = pareto_to_Norm(W_vec)
    # part1 = -0.5*eig2inv_quadform_vector(V, 1/d, Z_vec)-0.5*np.sum(np.log(d)) # multivariate density
    # cholesky_inv = lapack.dpotrs(cholesky_U,Z_vec)
    # part1 = -0.5*np.sum(Z_vec*cholesky_inv[0])-np.sum(np.log(np.diag(cholesky_U))) # multivariate density
    Z_standard_normal = scipy.linalg.solve_triangular(cholesky_U, Z_vec, trans=1)
    part1 = -0.5*np.sum(Z_standard_normal**2) - np.sum(np.log(np.diag(cholesky_U)))

    ## Jacobian determinant
    part21 = 0.5*np.sum(Z_vec**2) # 1/standard Normal densities of each Z_j
    if norm_pareto == 'standard':
        part22 = np.sum(-phi_vec*np.log(R_vec)-2*np.log(W_vec))
    else:
        part22 = np.sum(-phi_vec*np.log(R_vec)-2*np.log(W_vec+1)) # R_j^phi_j/X_j^2
    part23 = np.sum(dgev(Y, Loc=Loc, Scale=Scale, Shape=Shape, log=True))
    part24 = np.sum(-np.log(dRW(X, phi_vec, gamma_vec)))

    return np.array([part1,part21 ,part22, part23, part24])
# %%

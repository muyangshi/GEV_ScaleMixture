# combined utilitlies helpful to MCMC sampler
# grab the functions from model_sim.py and data_simulation_1_radius.py

# %%
# general imports
import numpy as np
from scipy.linalg import lapack
from ns_cov import *
import RW_inte_mpmath
import model_sim

# utility functions used to simulate dataset
weights_fun = model_sim.weights_fun
wendland_weights_fun = model_sim.wendland_weights_fun
norm_to_Pareto1 = model_sim.norm_to_Pareto1
pareto1_to_Norm = model_sim.pareto1_to_Norm
rlevy = model_sim.rlevy
pgev = model_sim.pgev
qgev = model_sim.qgev
dgev = model_sim.dgev

# %%
# specify which dRW, pRW, and qRW to use
# then specify the likelihoods to use that integral

dRW = model_sim.dRW
pRW = model_sim.pRW
qRW = model_sim.qRW_Newton

dRW = RW_inte_mpmath.dRW_mpmath_vec_float
pRW = RW_inte_mpmath.pRW_mpmath_vec_float
qRW = RW_inte_mpmath.qRW_mpmath_vec_float

def marg_transform_data_mixture_likelihood_1t(Y, X, Loc, Scale, Shape, phi_vec, gamma_vec, R_vec, cholesky_U):
  if(isinstance(Y, (int, np.int64, float))): Y=np.array([Y], dtype='float64')
  
  
  ## Initialize space to store the log-likelihoods for each observation:
  W_vec = X/R_vec**phi_vec
  
  Z_vec = pareto1_to_Norm(W_vec)
  # part1 = -0.5*eig2inv_quadform_vector(V, 1/d, Z_vec)-0.5*np.sum(np.log(d)) # multivariate density
  cholesky_inv = lapack.dpotrs(cholesky_U,Z_vec)
  part1 = -0.5*np.sum(Z_vec*cholesky_inv[0])-np.sum(np.log(np.diag(cholesky_U))) # multivariate density
  
  ## Jacobian determinant
  part21 = 0.5*np.sum(Z_vec**2) # 1/standard Normal densities of each Z_j
  part22 = np.sum(-phi_vec*np.log(R_vec)-2*np.log(W_vec+1)) # R_j^phi_j/X_j^2
  part23 = np.sum(dgev(Y, Loc=Loc, Scale=Scale, Shape=Shape, log=True)-np.log(dRW(X, phi_vec, gamma_vec)))
  
  return part1 + part21 + part22 + part23
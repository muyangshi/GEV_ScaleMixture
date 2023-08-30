#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 13:22:31 2021

Model Simulation & Grid Interpolation

@authors: Likun Zhang & Mark Risser
"""
import numpy as np
import sys
from scipy.stats import norm
from scipy.stats import uniform
import scipy.special as sc
import mpmath
import scipy.integrate as si
# import scipy.interpolate as interp
# import scipy.optimize as optim
from scipy.stats import genextreme




## integration.cpp
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##                        Import C++ function library
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##
##
## i.e., RW_marginal, pRW_me_interp, find_xrange_pRW_me
##
import os, ctypes
# from scipy import LowLevelCallable

# g++ -std=c++11 -shared -fPIC -o p_inte.so p_inte.cpp
lib = ctypes.CDLL(os.path.abspath('./nonstat_Pareto1/p_inte.so'))
i_and_o_type = np.ctypeslib.ndpointer(ndim=1, dtype=np.float64)
# grid_type  = np.ctypeslib.ndpointer(ndim=1, dtype=np.float64)
# bool_type  = np.ctypeslib.ndpointer(ndim=1, dtype='bool')

lib.p_integrand.restype = ctypes.c_double
lib.p_integrand.argtypes = (ctypes.c_int, ctypes.c_void_p)

lib.p_integrand1.restype = ctypes.c_double
lib.p_integrand1.argtypes = (ctypes.c_int, ctypes.c_void_p)

lib.pmixture_C_inf_integration.restype = ctypes.c_double
lib.pmixture_C_inf_integration.argtypes = (ctypes.c_double,ctypes.c_double,ctypes.c_double)

lib.pmixture_C.restype = ctypes.c_double
lib.pmixture_C.argtypes = (ctypes.c_double,ctypes.c_double,ctypes.c_double)


# No gain compared to np.vectorize()
# lib.pmixture_C_vec.restype = ctypes.c_int
# lib.pmixture_C_vec.argtypes = (i_and_o_type, ctypes.c_double, ctypes.c_double, 
#                                ctypes.c_int, i_and_o_type)

lib.d_integrand.restype = ctypes.c_double
lib.d_integrand.argtypes = (ctypes.c_int, ctypes.c_void_p)

lib.dmixture_C.restype = ctypes.c_double
lib.dmixture_C.argtypes = (ctypes.c_double,ctypes.c_double,ctypes.c_double)

lib.find_xrange_pRW_C.restype = ctypes.c_int
lib.find_xrange_pRW_C.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                      ctypes.c_double, ctypes.c_double, i_and_o_type)

lib.qRW_bisection_C.restype = ctypes.c_double
lib.qRW_bisection_C.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int)

lib.qRW_newton_C.restype = ctypes.c_double
lib.qRW_newton_C.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int)

# lib.p_integrand.restype = ctypes.c_double
# lib.p_integrand.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))

# func_p = LowLevelCallable(lib.p_integrand)

# lib.d_integrand.restype = ctypes.c_double
# lib.d_integrand.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))

# func_d = LowLevelCallable(lib.d_integrand)





## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##                       Generate Levy random samples
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##
##
## i.e., Stable variables with alpha=1/2
##
def rlevy(n, m = 0, s = 1):
  if np.any(s < 0):
    sys.exit("s must be positive")
  return s/norm.ppf(1-uniform.rvs(0,1,n)/2)**2 + m

def dlevy(r, m=0, s=1, log=False):
    if np.any(s < 0):
        sys.exit("s must be positive")
    if np.any(r < m):
        sys.exit("y must be greater than m")
        
    tmp = np.log(s/(2 * np.pi))/2 - 3 * np.log(r - m)/2 - s/(2 * (r - m))
    if not log: 
        tmp = np.exp(tmp)
    return tmp
    

## The density for R^phi in which R is levy distributed
def dR_power_phi(x, phi, m=0, s=1, log=False):
    x_phi = x**(1/phi)
    if np.any(x_phi <= m):
        sys.exit("some x**phi <= m")
    if np.any(s <= 0): 
        sys.exit("s must be positive")
    tmp = np.sum(np.log(s/(2 * np.pi))/2 - 3 * np.log(x_phi - m)/2 - s/(2 * (x_phi - 
        m)) + (1/phi-1)*np.log(x)-np.log(phi))
    if np.invert(log): 
        tmp = np.exp(tmp)
    return tmp
        
## -------------------------------------------------------------------------- ##






## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##        Calculate unregularized upper incomplete gamma function
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##
##
## The negative a values are allowed
##
def gammaincc_unregulized(a,x):
    if(isinstance(x, (int, np.int64, float))): x=np.array([x])
    if x.any()<0: sys.exit("x must be positive")
    if a>0: 
        return sc.gamma(a)*sc.gammaincc(a,x)
    elif a<0: 
        return gammaincc_unregulized(a+1,x)/a-(x**a)*np.exp(-x)/a
    else:
        return mpmath.gammainc(0,x)

## Compare with mpmath.gammainc
## gammaincc_unregulized is more efficient
# import time
# 
# start_time = time.time()
# gammaincc_unregulized(-3.62,5)
# time.time() - start_time

# start_time = time.time()
# mpmath.gammainc(-3.62,5)
# time.time() - start_time
## -------------------------------------------------------------------------- ##





## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##         Calculate the exact marginal survival function for R^phi*W
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##
##
##
## Approach 1: define integrand in Python
## Did not fix the numerical integral issue when xval is less 1e-15
def p_integrand(x, xval, phi, gamma):
   
    return (x**(phi-1.5))*np.exp(-gamma /(2*x))*np.sqrt(gamma/(2*np.pi))/(xval+x**phi)

def pmixture_py(xval, phi, gamma):
    I_1 = si.quad(p_integrand, 0,np.inf, args=(xval, phi, gamma))   
    return 1-I_1[0]

## Approach 2: use LowLevelCallable as integrand
# def pmixture_C_quad(xval, phi, gamma):
#     tmp = np.sqrt(gamma/(2*np.pi))
#     I_1 = si.quad(func_p, 0,np.inf, args=(xval, phi, gamma))   
#     return 1-I_1[0]*tmp

## Approach 3: integrate in C using gsl
pRW = np.vectorize(lib.pmixture_C)


def RW_marginal_asymp(x,phi,gamma):
    if phi<0.5:
        moment = ((2*gamma)**phi)*sc.gamma(1-2*phi)/sc.gamma(1-phi)
        return moment/x
    elif phi>0.5:
        return np.sqrt(2*gamma/np.pi)*(x**(-1/(2*phi)))/(1-1/(2*phi))
    else:
        return np.sqrt(2*gamma/np.pi)*np.log(x)/x

def RW_quantile_asymp(p,phi,gamma):
    if phi<0.5:
        moment = ((2*gamma)**phi)*sc.gamma(1-2*phi)/sc.gamma(1-phi)
        return moment/(1-p)
    elif phi>0.5:
        return (np.sqrt(2*gamma/np.pi)/(1-1/(2*phi))/(1-p))**(2*phi)
    else:
        tmp = (1-p)/np.sqrt(2*gamma/np.pi)
        return tmp/sc.lambertw(tmp)

# # Compare the exact and asymptotic CDF
# gamma = 1.2; x =10; phi=0.3

# import matplotlib.pyplot as plt
# axes = plt.gca()
# axes.set_ylim([0,0.125])
# X_vals = np.linspace(100,1500,num=200)
# P_vals = RW_marginal(X_vals,phi,gamma)
# P_asym = RW_marginal_asymp(X_vals,phi,gamma)
# plt.plot(X_vals, P_vals, 'b')
# plt.plot(X_vals, P_asym, 'r',linestyle='--');
## -------------------------------------------------------------------------- ##




## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##         Calculate the exact marginal density function for R^phi*W
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##
##
##
## Approach 1: define integrand in Python
def d_integrand(x, xval, phi, gamma):
   
    return (x**(phi-1.5))*np.exp(-gamma /(2*x))*np.sqrt(gamma/(2*np.pi))/(xval+x**phi)**2

def dmixture_py(xval, phi, gamma):
    I_1 = si.quad(d_integrand, 0,np.inf, args=(xval, phi, gamma))   
    return I_1[0]

## Approach 2: use LowLevelCallable as integrand
# def dmixture_C_quad(xval, phi, gamma):
#     tmp = np.sqrt(gamma/(2*np.pi))
#     I_1 = si.quad(func_d, 0,np.inf, args=(xval, phi, gamma))   
#     return I_1[0]*tmp

## Approach 3: integrate in C using gsl
dRW = np.vectorize(lib.dmixture_C)


def RW_density_asymp(x,phi,gamma):
    if phi<0.5:
        moment = ((2*gamma)**phi)*sc.gamma(1-2*phi)/sc.gamma(1-phi)
        return moment/(x**2)
    elif phi>0.5:
        return np.sqrt(2*gamma/np.pi)*(x**(-1/(2*phi)-1))/(2*phi-1)
    else:
        return np.sqrt(2*gamma/np.pi)*(np.log(x)-1)/(x**2)

# # Compare the exact and asymptotic CDF
# gamma = 1.2; x =10; phi=0.3

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# X_vals = np.linspace(100,1500,num=200)
# # X_vals = np.linspace(150,350,num=200) # For phi=0.3
# P_vals = RW_density(X_vals,phi,gamma,log=False)
# P_asym = RW_density_asymp(X_vals,phi,gamma)
# ax.plot(X_vals, P_vals, 'b', label="R^phi*W density")
# ax.plot(X_vals, P_asym, 'r',linestyle='--', label="R^phi*W tail approx")
# legend = ax.legend(loc = "upper right",shadow=True)
# plt.title(label="phi=0.3") 
# plt.show()
## -------------------------------------------------------------------------- ##




## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##       Calculate the quantile inverse function for R^phi*W + epsilon
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##
##
def find_xrange_pRW(min_p, max_p, x_init, phi, gamma):
    x_range = np.zeros(2)
    min_x = x_init[0]
    max_x = x_init[1]
    # if min_x <= 0 or min_p <= 0.15:
    #     sys.exit('This will only work for x > 0, which corresponds to p > 0.15.')
    if min_x >= max_x:
        sys.exit('x_init[0] must be smaller than x_init[1].')

    ## First the min
    p_min_x = pRW(min_x, phi, gamma)
    while p_min_x > min_p:
        min_x = min_x/2 # RW must be positive
        p_min_x = pRW(min_x, phi, gamma)
    x_range[0] = min_x

    ## Now the max
    p_max_x = pRW(max_x, phi, gamma)
    while p_max_x < max_p:
        max_x = max_x*2 # Upper will set to 20 initially
        p_max_x = pRW(max_x, phi, gamma)
    x_range[1] = max_x
    return x_range

def qRW_bisection(p, phi, gamma, n_x=100):
    x_range = np.empty(2)
    x_range = find_xrange_pRW(p, p, np.array([1., 5.]), phi, gamma)
    m = (x_range[0]+x_range[1])/2
    iter=0
    new_F = pRW(m, phi, gamma)-p
    while iter<100 and np.abs(new_F) > 1e-04:
        if new_F>0: 
            x_range[1] = m 
        else: 
            x_range[0]=m
        m = (x_range[0]+x_range[1])/2
        new_F = pRW(m, phi, gamma)-p
        iter += 1
        
    return m

def qRW_newton(p, phi, gamma, n_x=400):
    if p<1e-15: return 5.32907052e-15
    x_range = np.empty(2)
    x_range = find_xrange_pRW(p, p, np.array([1., 5.]), phi, gamma)
    current_x = x_range[0]; iter=0; error=1
    while iter<400 and error > 1e-08:
        tmp = (pRW(current_x, phi, gamma)-p)/dRW(current_x, phi, gamma)
        new_x = current_x - tmp
        error = np.abs(new_x-current_x)
        iter += 1
        current_x = max(x_range[0],new_x)
        if(current_x==x_range[0]): current_x = qRW_bisection(p, phi, gamma)
        
    return current_x

# C++ implementation
qRW_Newton = np.vectorize(lib.qRW_newton_C, excluded=['n_x'])
    









## scalemix_utils.R

## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##                     Compute the Matern correlation function
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##
##
## Input from a matrix of pairwise distances and a vector of parameters
##

def corr_fn(r, theta):
    if type(r).__module__!='numpy' or isinstance(r, np.float64):
      r = np.array(r)
    if np.any(r<0):
      sys.exit('Distance argument must be nonnegative.')
    r[r == 0] = 1e-10

    # range = theta[0]
    range = np.sqrt(theta[0])  # Mark's generic Matern
    nu = theta[1]
    part1 = 2 ** (1 - nu) / sc.gamma(nu)
    part2 = (r / range) ** nu
    part3 = sc.kv(nu, r / range)
    # part2 = (np.sqrt(2 * nu) * r / range) ** nu
    # part3 = sc.kv(nu, np.sqrt(2 * nu) * r / range)
    return part1 * part2 * part3
## -------------------------------------------------------------------------- ##




## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##                                    For MVN
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##
## Assumes that A = VDV', where D is a diagonal vector of eigenvectors of A, and
## V is a matrix of normalized eigenvectors of A.

## Computes A^{-1}x
##
def eig2inv_times_vector(V, d_inv, x):
  return V@np.diag(d_inv)@V.T@x

## Computes y=A^{-1}x via solving linear system Ay=x
from scipy.linalg import lapack
def inv_times_vector(A, x):
  inv = lapack.dposv(A,x)
  return inv


## Assumes that A = VDV', where D is a diagonal vector of eigenvectors of A, and
## V is a matrix of normalized eigenvectors of A.
##
## log(|A|)
##
def eig2logdet(d):
  return np.sum(np.log(d))


## Multivariate normal log density of R, where each column of 
## R iid N(mean,VDV'), where VDV' is the covariance matrix
## It essentially computes the log density of each column of R, then takes
## the sum.  Faster than looping over the columns, but not as transparent.
##                                     
## Ignore the coefficient: -p/2*log(2*pi)
##
def dmvn_eig(R, V, d_inv, mean=0):
  if len(R.shape)==1: 
    n_rep = 1 
  else: 
    n_rep = R.shape[1]
  res = -0.5*n_rep*eig2logdet(1/d_inv) - 0.5 * np.sum((R-mean) * eig2inv_times_vector(V, d_inv, R-mean))
  return res

def dmvn(R, Cor, mean=0, cholesky_inv = None):## cholesky_inv is the output of inv_times_vector()
  if len(R.shape)==1: 
    n_rep = 1 
  else: 
    n_rep = R.shape[1]
    
  if cholesky_inv is None: 
      inv = lapack.dposv(Cor, R-mean)
  else:
      sol = lapack.dpotrs(cholesky_inv[0],R-mean) #Solve Ax = b using factorization
      inv = (cholesky_inv[0],sol[0])
  logdet = 2*np.sum(np.log(np.diag(inv[0])))
  res = -0.5*n_rep*logdet - 0.5 * np.sum((R-mean) * inv[1])
  return res

## Assumes that A = VDV', where D is a diagonal vector of eigenvectors of A, and
## V is a matrix of normalized eigenvectors of A.
##
## Computes x'A^{-1}x
##
def eig2inv_quadform_vector(V, d_inv, x):
  cp = V@np.diag(d_inv)@V.T@x
  return sum(x*cp)

def inv_quadform_vector(Cor, x, cholesky_inv = None):
  if cholesky_inv is None:
      inv = lapack.dposv(Cor, x)
      cp = inv[1]
  else:
      sol = lapack.dpotrs(cholesky_inv[0],x) #Solve Ax = b using factorization
      cp = sol[0]
  return sum(x*cp)

## -------------------------------------------------------------------------- ##





## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##                For generalized extreme value (GEV) distribution
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
## 
## Negative shape parametrization in scipy.genextreme
## 

def dgev(yvals, Loc, Scale, Shape, log=False):
    if log:
        return genextreme.logpdf(yvals, c=-Shape, loc=Loc, scale=Scale)  # Opposite shape
    else:
        return genextreme.pdf(yvals, c=-Shape, loc=Loc, scale=Scale)  # Opposite shape

def pgev(yvals, Loc, Scale, Shape, log=False):
    if log:
        return genextreme.logcdf(yvals, c=-Shape, loc=Loc, scale=Scale)  # Opposite shape
    else:
        return genextreme.cdf(yvals, c=-Shape, loc=Loc, scale=Scale)  # Opposite shape

def qgev(p, Loc, Scale, Shape):
    if type(p).__module__!='numpy':
      p = np.array(p)  
    return genextreme.ppf(p, c=-Shape, loc=Loc, scale=Scale)  # Opposite shape


## -------------------------------------------------------------------------- ##






## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##                    Transform Normal to Standard Pareto
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##
##
## i.e., Stable variables with alpha=1/2
##
## Also, we use shifted Pareto here
##
def norm_to_Pareto(z):
    if(isinstance(z, (int, np.int64, float))): z=np.array([z])
    tmp = norm.cdf(z)
    if np.any(tmp==1): tmp[tmp==1]=1-1e-9
    return 1/(1-tmp)-1


def pareto_to_Norm(W):
    if(isinstance(W, (int, np.int64, float))): W=np.array([W])
    tmp = 1-1/(W+1)
    return norm.ppf(tmp)
        
## -------------------------------------------------------------------------- ##








## scalemix_likelihoods.R 

## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##                           Marginal transformations
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
## 
## Transforms observations from a Gaussian scale mixture to a GPD, or vice versa
## Only takes one location
## 
def RW_2_gev(X, phi, gamma, Loc, Scale, Shape):
    probs = pRW(X, phi, gamma)
    gevs = qgev(probs, Loc=Loc, Scale=Scale, Shape=Shape)
    return gevs

def gev_2_RW(Y, phi, gamma, Loc, Scale, Shape):
    unifs = pgev(Y, Loc, Scale, Shape)
    scalemixes = qRW_Newton(unifs, phi, gamma, 400)
    return scalemixes 

## -------------------------------------------------------------------------- ##






## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##                           Gaussian moothing kernel
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
## When d > fit radius, the weight will be zero
## h is the bandwidth parameter
##

def weights_fun(d,radius,h=1, cutoff=True):
  if(isinstance(d, (int, np.int64, float))): d=np.array([d])
  tmp = np.exp(-d**2/(2*h))
  if cutoff: tmp[d>radius] = 0
  
  return tmp/np.sum(tmp)

                                                                             
##
## -------------------------------------------------------------------------- ##








## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##                       Wendland compactly-supported basis
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
## fields_Wendland(d, theta = 1, dimension, k, derivative=0, phi=NA)
## theta: the range where the basis value is non-zero, i.e. [0, theta]
## dimension: dimension of locations 
## k: smoothness of the function at zero.

def wendland_weights_fun(d, theta, k=0, dimension=2, derivative=0):
  if(isinstance(d, (int, np.int64, float))): d=np.array([d])      
  d = d/theta
  l = np.floor(dimension/2) + k + 1
  if (k==0): 
      res = np.where(d < 1, (1-d)**l, 0)

  if (k==1):
      res = np.where(d < 1, (1-d)**(l+k) * ((l+1)*d + 1), 0)
  
  if (k==2):
      res = np.where(d < 1, (1-d)**(l+k) * ((l**2+4*l+3)*d**2 + (3*l+6) * d + 3), 0)
      
  if (k==3):
      res = np.where(d < 1, (1-d)**(l+k) * ((l**3+9*l**2+23*l+15)*d**3 + 
                                            (6*l**2+36*l+45) * d**2 + (15*l+45) * d + 15), 0)
  
  if (k>3):
      sys.exit("k must be less than 4")
  return res/np.sum(res)

                                                                             
##
## -------------------------------------------------------------------------- ##



## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##                             Censored likelihood
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
## The log likelihood of the data, where the data comes from a scale mixture
## of Gaussians, transformed to GEV (matrix/vector input)
##

## Without the nugget: one-time replicate
## --- cholesky_U is the output of linalg.cholesky(lower=False)
def marg_transform_data_mixture_likelihood_1t(Y, X, Loc, Scale, Shape, phi_vec, gamma_vec, R_vec, cholesky_U):
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

## Without the nugget: all time
def marg_transform_data_mixture_likelihood(Y, X, Loc, Scale, Shape, phi_vec, gamma_vec, R, cholesky_U):
  ## Initialize space to store the log-likelihoods for each observation:
  n_t = Y.shape[1]
  ll = np.empty(n_t); ll[:] = np.nan
  
  for idx in np.arange(n_t):
      ll[idx] = marg_transform_data_mixture_likelihood_1t(Y[:,idx], X[:,idx], Loc[:,idx], Scale[:,idx], 
                                                Shape[:,idx], phi_vec, gamma_vec, R[:,idx], cholesky_U)
  return np.sum(ll)

##
## -------------------------------------------------------------------------- ##
















## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##                 Full likelihood for latent Gaussian field
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
## For the generic Metropolis sampler
## Samples from the parameters of the mixing distribution, for the scale 
## mixture of Gaussians.
##
## Both functions in this section admit column vectors only.
##

def Z_likelihood_conditional_eigen(Z, V, d):
    # R_powered = R**phi
    part1 = -0.5*eig2inv_quadform_vector(V, 1/d, Z)-0.5*np.sum(np.log(d))
    return part1

def Z_likelihood_conditional(Z, Cor, cholesky_inv):
    # R_powered = R**phi
    part1 = -0.5*inv_quadform_vector(Cor, Z, cholesky_inv)-0.5*2*np.sum(np.log(np.diag(cholesky_inv[0])))
    return part1


##
## -------------------------------------------------------------------------- ##







## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##                   Full likelihood for Matern parameters
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
## Update covariance parameters. 
##
##

def theta_c_update_mixture_me_likelihood_eigen(data, params, S, V=np.nan, d=np.nan):
  Z = data
  range = params[0]
  nu = params[1]
  if len(Z.shape)==1:
      Z = Z.reshape((Z.shape[0],1))
  n_t = Z.shape[1]
  
  if np.any(np.isnan(V)):
    Cor = corr_fn(S, np.array([range,nu]))
    eig_Cor = np.linalg.eigh(Cor) #For symmetric matrices
    V = eig_Cor[1]
    d = eig_Cor[0]

  ll = np.empty(n_t)
  ll[:]=np.nan
  for idx in np.arange(n_t):
    ll[idx] = Z_likelihood_conditional_eigen(Z[:,idx], V, d)
  return np.sum(ll)

def range_update_mixture_me_likelihood_eigen(data, params, nu, S, V=np.nan, d=np.nan):
  Z = data
  range = params
  
  if len(Z.shape)==1:
      Z = Z.reshape((Z.shape[0],1))
  n_t = Z.shape[1]
  
  if np.any(np.isnan(V)):
    Cor = corr_fn(S, np.array([range,nu]))
    eig_Cor = np.linalg.eigh(Cor) #For symmetric matrices
    V = eig_Cor[1]
    d = eig_Cor[0]

  ll = np.empty(n_t)
  ll[:]=np.nan
  for idx in np.arange(n_t):
    ll[idx] = Z_likelihood_conditional_eigen(Z[:,idx], V, d)
  return np.sum(ll)


def theta_c_update_mixture_me_likelihood(data, params, S, Cor=None, cholesky_inv=None):
  Z = data
  range = params[0]
  nu = params[1]
  if len(Z.shape)==1:
      Z = Z.reshape((Z.shape[0],1))
  n_t = Z.shape[1]
  
  if Cor is None:
    Cor = corr_fn(S, np.array([range,nu]))
    cholesky_inv = lapack.dposv(Cor,Z[:,0])

  ll = np.empty(n_t)
  ll[:]=np.nan
  for idx in np.arange(n_t):
    ll[idx] = Z_likelihood_conditional(Z[:,idx], Cor, cholesky_inv)
  return np.sum(ll)

def range_update_mixture_me_likelihood(data, params, nu, S, Cor=None, cholesky_inv=None):
  Z = data
  range = params
  
  if len(Z.shape)==1:
      Z = Z.reshape((Z.shape[0],1))
  n_t = Z.shape[1]
  
  if Cor is None:
    Cor = corr_fn(S, np.array([range,nu]))
    cholesky_inv = lapack.dposv(Cor,Z[:,0])

  ll = np.empty(n_t)
  ll[:]=np.nan
  for idx in np.arange(n_t):
    ll[idx] = Z_likelihood_conditional(Z[:,idx], Cor, cholesky_inv)
  return np.sum(ll)

##
## -------------------------------------------------------------------------- ##










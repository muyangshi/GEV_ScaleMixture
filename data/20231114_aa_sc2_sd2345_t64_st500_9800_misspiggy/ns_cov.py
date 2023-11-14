import numpy as np
import sys
import scipy.special as sc
from scipy.spatial import distance

## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##               Implement the Matern correlation function (stationary)
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##
##
## Input from a matrix of pairwise distances and a vector of parameters
##

def cov_spatial(r, cov_model = "exponential", cov_pars = np.array([1,1]), kappa = 0.5):
    if type(r).__module__!='numpy' or isinstance(r, np.float64):
      r = np.array(r)
    if np.any(r<0):
      sys.exit('Distance argument must be nonnegative.')
    r[r == 0] = 1e-10
    
    if cov_model != "matern" and cov_model != "gaussian" and cov_model != "exponential" :
        sys.exit("Please specify a valid covariance model (matern, gaussian, or exponential).")
    
    if cov_model == "exponential":
        C = np.exp(-r)
    
    if cov_model == "gaussian" :
        C = np.exp(-(r^2))
  
    if cov_model == "matern" :
        range = 1
        nu = kappa
        part1 = 2 ** (1 - nu) / sc.gamma(nu)
        part2 = (r / range) ** nu
        part3 = sc.kv(nu, r / range)
        C = part1 * part2 * part3
    return C

## -------------------------------------------------------------------------- ##




## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##               Calculate a locally isotropic spatial covariance
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
## Arguments:
##    range_vec = N-vector of range parameters (one for each location) 
##    sigsq_vec = N-vector of marginal variance parameters (one for each location)
##    coords = N x 2 matrix of coordinates
##    cov.model = "matern" --> underlying covariance model: "gaussian", "exponential", or "matern"
##    kappa = 0.5 --> Matern smoothness, scalar
##

def ns_cov(range_vec, sigsq_vec, coords, kappa = 0.5, cov_model = "matern"):
    if type(range_vec).__module__!='numpy' or isinstance(range_vec, np.float64):
      range_vec = np.array(range_vec)
      sigsq_vec = np.array(sigsq_vec)
    
    N = range_vec.shape[0] # Number of spatial locations
    if coords.shape[0]!=N: 
      sys.exit('Number of spatial locations should be equal to the number of range parameters.')
  
    # Scale matrix
    arg11 = range_vec
    arg22 = range_vec
    arg12 = np.repeat(0,N)
    ones = np.repeat(1,N)
    det1  = arg11*arg22 - arg12**2
  
    ## --- Outer product: matrix(arg11, nrow = N) %x% matrix(1, ncol = N) --- 
    mat11_1 = np.reshape(arg11, (N, 1)) * ones
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg11, ncol = N) ---
    mat11_2 = np.reshape(ones, (N, 1)) * arg11
    ## --- Outer product: matrix(arg22, nrow = N) %x% matrix(1, ncol = N) ---
    mat22_1 = np.reshape(arg22, (N, 1)) * ones  
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg22, ncol = N) ---
    mat22_2 = np.reshape(ones, (N, 1)) * arg22
    ## --- Outer product: matrix(arg12, nrow = N) %x% matrix(1, ncol = N) ---
    mat12_1 = np.reshape(arg12, (N, 1)) * ones 
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg12, ncol = N) ---
    mat12_2 = np.reshape(ones, (N, 1)) * arg12
  
    mat11 = 0.5*(mat11_1 + mat11_2)
    mat22 = 0.5*(mat22_1 + mat22_2)
    mat12 = 0.5*(mat12_1 + mat12_2)
  
    det12 = mat11*mat22 - mat12**2
  
    Scale_mat = np.diag(det1**(1/4)).dot(np.sqrt(1/det12)).dot(np.diag(det1**(1/4)))
  
    # Distance matrix
    inv11 = mat22/det12
    inv22 = mat11/det12
    inv12 = -mat12/det12
  
    dists1 = distance.squareform(distance.pdist(np.reshape(coords[:,0], (N, 1))))
    dists2 = distance.squareform(distance.pdist(np.reshape(coords[:,1], (N, 1))))
  
    temp1_1 = np.reshape(coords[:,0], (N, 1)) * ones
    temp1_2 = np.reshape(ones, (N, 1)) * coords[:,0]
    temp2_1 = np.reshape(coords[:,1], (N, 1)) * ones
    temp2_2 = np.reshape(ones, (N, 1)) * coords[:,1]
  
    sgn_mat1 = ( temp1_1 - temp1_2 >= 0 )
    sgn_mat1[~sgn_mat1] = -1
    sgn_mat2 = ( temp2_1 - temp2_2 >= 0 )
    sgn_mat2[~sgn_mat2] = -1
  
    dists1_sq = dists1**2
    dists2_sq = dists2**2
    dists12 = sgn_mat1*dists1*sgn_mat2*dists2
  
    Dist_mat_sqd = inv11*dists1_sq + 2*inv12*dists12 + inv22*dists2_sq
    Dist_mat = np.zeros(Dist_mat_sqd.shape)
    Dist_mat[Dist_mat_sqd>0] = np.sqrt(Dist_mat_sqd[Dist_mat_sqd>0])
  
    # Combine
    Unscl_corr = cov_spatial(Dist_mat, cov_model = cov_model, cov_pars = np.array([1,1]), kappa = kappa)
    NS_corr = Scale_mat*Unscl_corr
  
    Spatial_cov = np.diag(sigsq_vec).dot(NS_corr).dot(np.diag(sigsq_vec)) 
    return(Spatial_cov)
    
# Using the grid of values to interpolate because sc.special.kv is computationally expensive
# tck is the output function of sc.interpolate.pchip (Contains information about roughness kappa)
# ** Has to be Matern model **
def ns_cov_interp(range_vec, sigsq_vec, coords, tck):
    if type(range_vec).__module__!='numpy' or isinstance(range_vec, np.float64):
      range_vec = np.array(range_vec)
      sigsq_vec = np.array(sigsq_vec)
    
    N = range_vec.shape[0] # Number of spatial locations
    if coords.shape[0]!=N:
      sys.exit('Number of spatial locations should be equal to the number of range parameters.')
  
    # Scale matrix
    arg11 = range_vec
    arg22 = range_vec
    arg12 = np.repeat(0,N)
    ones = np.repeat(1,N)
    det1  = arg11*arg22 - arg12**2
  
    ## --- Outer product: matrix(arg11, nrow = N) %x% matrix(1, ncol = N) ---
    mat11_1 = np.reshape(arg11, (N, 1)) * ones
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg11, ncol = N) ---
    mat11_2 = np.reshape(ones, (N, 1)) * arg11
    ## --- Outer product: matrix(arg22, nrow = N) %x% matrix(1, ncol = N) ---
    mat22_1 = np.reshape(arg22, (N, 1)) * ones
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg22, ncol = N) ---
    mat22_2 = np.reshape(ones, (N, 1)) * arg22
    ## --- Outer product: matrix(arg12, nrow = N) %x% matrix(1, ncol = N) ---
    mat12_1 = np.reshape(arg12, (N, 1)) * ones
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg12, ncol = N) ---
    mat12_2 = np.reshape(ones, (N, 1)) * arg12
  
    mat11 = 0.5*(mat11_1 + mat11_2)
    mat22 = 0.5*(mat22_1 + mat22_2)
    mat12 = 0.5*(mat12_1 + mat12_2)
  
    det12 = mat11*mat22 - mat12**2
  
    Scale_mat = np.diag(det1**(1/4)).dot(np.sqrt(1/det12)).dot(np.diag(det1**(1/4)))
  
    # Distance matrix
    inv11 = mat22/det12
    inv22 = mat11/det12
    inv12 = -mat12/det12
  
    dists1 = distance.squareform(distance.pdist(np.reshape(coords[:,0], (N, 1))))
    dists2 = distance.squareform(distance.pdist(np.reshape(coords[:,1], (N, 1))))
  
    temp1_1 = np.reshape(coords[:,0], (N, 1)) * ones
    temp1_2 = np.reshape(ones, (N, 1)) * coords[:,0]
    temp2_1 = np.reshape(coords[:,1], (N, 1)) * ones
    temp2_2 = np.reshape(ones, (N, 1)) * coords[:,1]
  
    sgn_mat1 = ( temp1_1 - temp1_2 >= 0 )
    sgn_mat1[~sgn_mat1] = -1
    sgn_mat2 = ( temp2_1 - temp2_2 >= 0 )
    sgn_mat2[~sgn_mat2] = -1
  
    dists1_sq = dists1**2
    dists2_sq = dists2**2
    dists12 = sgn_mat1*dists1*sgn_mat2*dists2
  
    Dist_mat_sqd = inv11*dists1_sq + 2*inv12*dists12 + inv22*dists2_sq
    Dist_mat = np.zeros(Dist_mat_sqd.shape)
    Dist_mat[Dist_mat_sqd>0] = np.sqrt(Dist_mat_sqd[Dist_mat_sqd>0])
  
    # Combine
    Unscl_corr = np.ones(Dist_mat_sqd.shape)
    Unscl_corr[Dist_mat_sqd>0] = tck(Dist_mat[Dist_mat_sqd>0])
    NS_corr = Scale_mat*Unscl_corr
  
    Spatial_cov = np.diag(sigsq_vec).dot(NS_corr).dot(np.diag(sigsq_vec))
    return(Spatial_cov)
## -------------------------------------------------------------------------- ##







## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##                                    Example
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##

# ## expand.grid()
# x,y = np.meshgrid(np.linspace(0,1,num=25), np.linspace(0,1,num=25))
# coords = np.c_[x.flatten(), y.flatten()]
# range_vec = np.exp(-2 + 1*coords[:,0] + 1*coords[:,1])

# ## Look at range field:
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# c = ax.pcolormesh(np.linspace(0,1,num=25), np.linspace(0,1,num=25), 
#                   range_vec.reshape((25,25),order='F'), cmap='jet', vmin=0, vmax=1)
# ax.set_title('Quilt plot')
# # set the limits of the plot to the limits of the data
# ax.axis([0,1,0,1])
# fig.colorbar(c, ax=ax)
# plt.show()

# ## Calculate covariance:
# Cov = ns_cov(range_vec, np.repeat(1, coords.shape[0]), coords, kappa = 1.5, cov_model = "matern")

# ## Look at covariance
# import seaborn as sns; sns.set()
# ax = sns.heatmap(Cov, cmap='jet')
# ax.invert_yaxis()

# ## Correlation plots for fixed locations
# fig, ax = plt.subplots()
# c = ax.pcolormesh(np.linspace(0,1,num=25), np.linspace(0,1,num=25), 
#                   Cov[30,].reshape((25,25),order='F'), cmap='jet', vmin=0, vmax=1)
# ax.set_title('Quilt plot')
# # set the limits of the plot to the limits of the data
# ax.axis([0,1,0,1])
# fig.colorbar(c, ax=ax)
# plt.show() # Shorter correlation length-scale

# fig, ax = plt.subplots()
# c = ax.pcolormesh(np.linspace(0,1,num=25), np.linspace(0,1,num=25), 
#                   Cov[495,].reshape((25,25),order='F'), cmap='jet', vmin=0, vmax=1)
# ax.set_title('Quilt plot')
# # set the limits of the plot to the limits of the data
# ax.axis([0,1,0,1])
# fig.colorbar(c, ax=ax)
# plt.show() # Much longer correlation length-scale

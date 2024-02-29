"""
Summary informaiton regarding the posterior draws
- covariance
- summary statistics
- etc.
"""
# %%
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import geopandas as gpd
state_map = gpd.read_file('./cb_2018_us_state_20m/cb_2018_us_state_20m.shp')
import matplotlib as mpl
from matplotlib import colormaps

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))

def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
from mpi4py import MPI
from time import strftime, localtime
from utilities import *
import gstools as gs
import rpy2.robjects as robjects
from rpy2.robjects import r 
from rpy2.robjects.numpy2ri import numpy2rpy
from rpy2.robjects.packages import importr

data_seed = 2345
np.random.seed(data_seed)


# %% Load Dataset and Setup -----------------------------------------------------------------------------------------------
# Load Dataset and Setup   -----------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------
# data

mgcv = importr('mgcv')
r('''load('JJA_precip_maxima.RData')''')
GEV_estimates      = np.array(r('GEV_estimates')).T
mu0_estimates      = GEV_estimates[:,0]
mu1_estimates      = GEV_estimates[:,1]
logsigma_estimates = GEV_estimates[:,2]
ksi_estimates      = GEV_estimates[:,3]
JJA_maxima         = np.array(r('JJA_maxima')).T
stations           = np.array(r('stations')).T
elevations         = np.array(r('elev')).T/200

# truncate for easier run on misspiggy
Nt                 = 32
Ns                 = 300
times_subset       = np.arange(Nt)
sites_subset       = np.random.default_rng(data_seed).choice(JJA_maxima.shape[0],size=Ns,replace=False,shuffle=False)
GEV_estimates      = GEV_estimates[sites_subset,:]
mu0_estimates      = GEV_estimates[:,0]
mu1_estimates      = GEV_estimates[:,1]
logsigma_estimates = GEV_estimates[:,2]
ksi_estimates      = GEV_estimates[:,3]
JJA_maxima         = JJA_maxima[sites_subset,:][:,times_subset]
stations           = stations[sites_subset]
elevations         = elevations[sites_subset]

Y = JJA_maxima

# Setup (Covariates and Constants)    ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------
# Ns, Nt

Nt = JJA_maxima.shape[1] # number of time replicates
Ns = JJA_maxima.shape[0] # number of sites/stations
start_year = 1950
end_year   = 2017
all_years  = np.linspace(start_year, end_year, Nt)
# Note, to use the mu1 estimates from Likun, the `Time`` must be standardized the same way
# Time = np.linspace(-Nt/2, Nt/2-1, Nt)
Time       = (all_years - np.mean(all_years))/np.std(all_years, ddof=1) # delta degress of freedom, to match the n-1 in R
Time       = Time[0:Nt] # if there is any truncation

# ----------------------------------------------------------------------------------------------------------------
# Sites

sites_xy = stations
sites_x = sites_xy[:,0]
sites_y = sites_xy[:,1]

# define the lower and upper limits for x and y
minX, maxX = np.floor(np.min(sites_x)), np.ceil(np.max(sites_x))
minY, maxY = np.floor(np.min(sites_y)), np.ceil(np.max(sites_y))

# ----------------------------------------------------------------------------------------------------------------
# Knots

# res_x = 3
# res_y = 3
# k = res_x * res_y # number of knots
# # create one-dimensional arrays for x and y
# x_pos = np.linspace(minX, maxX, res_x+2)[1:-1]
# y_pos = np.linspace(minY, maxY, res_y+2)[1:-1]
# # create the mesh based on these arrays
# X_pos, Y_pos = np.meshgrid(x_pos,y_pos)
# knots_xy = np.vstack([X_pos.ravel(), Y_pos.ravel()]).T
# knots_x = knots_xy[:,0]
# knots_y = knots_xy[:,1]    

# isometric knot grid
N_outer_grid = 9
x_pos                    = np.linspace(minX + 1, maxX + 1, num = int(2*np.sqrt(N_outer_grid)))
y_pos                    = np.linspace(minY + 1, maxY + 1, num = int(2*np.sqrt(N_outer_grid)))
x_outer_pos              = x_pos[0::2]
x_inner_pos              = x_pos[1::2]
y_outer_pos              = y_pos[0::2]
y_inner_pos              = y_pos[1::2]
X_outer_pos, Y_outer_pos = np.meshgrid(x_outer_pos, y_outer_pos)
X_inner_pos, Y_inner_pos = np.meshgrid(x_inner_pos, y_inner_pos)
knots_outer_xy           = np.vstack([X_outer_pos.ravel(), Y_outer_pos.ravel()]).T
knots_inner_xy           = np.vstack([X_inner_pos.ravel(), Y_inner_pos.ravel()]).T
knots_xy                 = np.vstack((knots_outer_xy, knots_inner_xy))
knots_id_in_domain       = [row for row in range(len(knots_xy)) if (minX < knots_xy[row,0] < maxX and minY < knots_xy[row,1] < maxY)]
knots_xy                 = knots_xy[knots_id_in_domain]
knots_x                  = knots_xy[:,0]
knots_y                  = knots_xy[:,1]
k                        = len(knots_id_in_domain)


# ----------------------------------------------------------------------------------------------------------------
# Copula Splines

# Basis Parameters - for the Gaussian and Wendland Basis
bandwidth = 4 # range for the gaussian kernel
radius = 4 # radius of infuence for basis, 3.5 might make some points closer to the edge of circle, might lead to numerical issues
radius_from_knots = np.repeat(radius, k) # influence radius from a knot

# Generate the weight matrices
# Weight matrix generated using Gaussian Smoothing Kernel
gaussian_weight_matrix = np.full(shape = (Ns, k), fill_value = np.nan)
for site_id in np.arange(Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                    XB = knots_xy)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
    gaussian_weight_matrix[site_id, :] = weight_from_knots

# Weight matrix generated using wendland basis
wendland_weight_matrix = np.full(shape = (Ns,k), fill_value = np.nan)
for site_id in np.arange(Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                    XB = knots_xy)
    # influence coming from each of the knots
    weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
    wendland_weight_matrix[site_id, :] = weight_from_knots

# # constant weight matrix
# constant_weight_matrix = np.full(shape = (Ns, k), fill_value = np.nan)
# for site_id in np.arange(Ns):
#     # Compute distance between each pair of the two collections of inputs
#     d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
#                                     XB = knots_xy)
#     # influence coming from each of the knots
#     weight_from_knots = np.repeat(1, k)/k
#     constant_weight_matrix[site_id, :] = weight_from_knots

# ----------------------------------------------------------------------------------------------------------------
# Setup For the Marginal Model - GEV(mu, sigma, ksi)

# ----- using splines for mu0 and mu1 ---------------------------------------------------------------------------
# "knots" and prediction sites for splines 
gs_x        = np.linspace(minX, maxX, 50)
gs_y        = np.linspace(minY, maxY, 50)
gs_xy       = np.vstack([coords.ravel() for coords in np.meshgrid(gs_x, gs_y, indexing='ij')]).T # indexing='ij' fill vertically, need .T in imshow

gs_x_ro     = numpy2rpy(gs_x)        # Convert to R object
gs_y_ro     = numpy2rpy(gs_y)        # Convert to R object
gs_xy_ro    = numpy2rpy(gs_xy)       # Convert to R object
sites_xy_ro = numpy2rpy(sites_xy)    # Convert to R object

r.assign("gs_x_ro", gs_x_ro)         # Note: this is a matrix in R, not df
r.assign("gs_y_ro", gs_y_ro)         # Note: this is a matrix in R, not df
r.assign("gs_xy_ro", gs_xy_ro)       # Note: this is a matrix in R, not df
r.assign('sites_xy_ro', sites_xy_ro) # Note: this is a matrix in R, not df

r('''
    gs_xy_df <- as.data.frame(gs_xy_ro)
    colnames(gs_xy_df) <- c('x','y')
    sites_xy_df <- as.data.frame(sites_xy_ro)
    colnames(sites_xy_df) <- c('x','y')
    ''')

# Location mu_0(s) ----------------------------------------------------------------------------------------------

Beta_mu0_splines_m = 12 - 1 # number of splines basis, -1 b/c drop constant column
Beta_mu0_m         = Beta_mu0_splines_m + 2 # adding intercept and elevation
C_mu0_splines      = np.array(r('''
                                basis      <- smoothCon(s(x, y, k = {Beta_mu0_splines_m}, fx = TRUE), data = gs_xy_df)[[1]]
                                basis_site <- PredictMat(basis, data = sites_xy_df)
                                # basis_site
                                basis_site[,c(-(ncol(basis_site)-2))] # dropped the 3rd to last column of constant
                                '''.format(Beta_mu0_splines_m = Beta_mu0_splines_m+1))) # shaped(Ns, Beta_mu0_splines_m)
C_mu0_1t           = np.column_stack((np.ones(Ns),  # intercept
                                    elevations,     # elevation
                                    C_mu0_splines)) # splines (excluding intercept)
C_mu0              = np.tile(C_mu0_1t.T[:,:,None], reps = (1, 1, Nt))

# Location mu_1(s) ----------------------------------------------------------------------------------------------

Beta_mu1_splines_m = 18 - 1 # drop the 3rd to last column of constant
Beta_mu1_m         = Beta_mu1_splines_m + 2 # adding intercept and elevation
C_mu1_splines      = np.array(r('''
                                basis      <- smoothCon(s(x, y, k = {Beta_mu1_splines_m}, fx = TRUE), data = gs_xy_df)[[1]]
                                basis_site <- PredictMat(basis, data = sites_xy_df)
                                # basis_site
                                basis_site[,c(-(ncol(basis_site)-2))] # drop the 3rd to last column of constant
                                '''.format(Beta_mu1_splines_m = Beta_mu1_splines_m+1))) # shaped(Ns, Beta_mu1_splines_m)
C_mu1_1t           = np.column_stack((np.ones(Ns),  # intercept
                                    elevations,     # elevation
                                    C_mu1_splines)) # splines (excluding intercept)
C_mu1              = np.tile(C_mu1_1t.T[:,:,None], reps = (1, 1, Nt))

# Scale logsigma(s) ----------------------------------------------------------------------------------------------

Beta_logsigma_m   = 2 # just intercept and elevation
C_logsigma        = np.full(shape = (Beta_logsigma_m, Ns, Nt), fill_value = np.nan)
C_logsigma[0,:,:] = 1.0 
C_logsigma[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T

# Shape ksi(s) ----------------------------------------------------------------------------------------------

Beta_ksi_m   = 2 # just intercept and elevation
C_ksi        = np.full(shape = (Beta_ksi_m, Ns, Nt), fill_value = np.nan) # ksi design matrix
C_ksi[0,:,:] = 1.0
C_ksi[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T

# ----------------------------------------------------------------------------------------------------------------
# Setup For the Copula/Data Model - X_star = R^phi * g(Z)

# Covariance K for Gaussian Field g(Z) --------------------------------------------------------------------------
nu = 0.5 # exponential kernel for matern with nu = 1/2
sigsq = 1.0 # sill for Z
sigsq_vec = np.repeat(sigsq, Ns) # hold at 1

# Scale Mixture R^phi --------------------------------------------------------------------------------------------
## phi and gamma
gamma = 0.5 # this is the gamma that goes in rlevy, gamma_at_knots
delta = 0.0 # this is the delta in levy, stays 0
alpha = 0.5
gamma_at_knots = np.repeat(gamma, k)
gamma_vec = np.sum(np.multiply(wendland_weight_matrix, gamma_at_knots)**(alpha), 
                    axis = 1)**(1/alpha) # bar{gamma}, axis = 1 to sum over K knots

# %% generate data & setup
# generate data & setup

# ----------------------------------------------------------------------------------------------------------------
# Numbers - Ns, Nt, n_iters

np.random.seed(data_seed)
Nt = 32 # number of time replicates
Ns = 125 # number of sites/stations
Time = np.linspace(-Nt/2, Nt/2-1, Nt)/np.std(np.linspace(-Nt/2, Nt/2-1, Nt), ddof=1)

# ----------------------------------------------------------------------------------------------------------------
# Sites - random uniformly (x,y) generate site locations

sites_xy = np.random.random((Ns, 2)) * 10
sites_x = sites_xy[:,0]
sites_y = sites_xy[:,1]

# # define the lower and upper limits for x and y
minX, maxX = np.floor(np.min(sites_x)), np.ceil(np.max(sites_x))
minY, maxY = np.floor(np.min(sites_y)), np.ceil(np.max(sites_y))

# ----------------------------------------------------------------------------------------------------------------
# Elevation Function - 
# Note: the simple elevation function 1/5(|x-5| + |y-5|) is way too similar to the first basis
#       this might cause identifiability issue
# def elevation_func(x,y):
    # return(np.abs(x-5)/5 + np.abs(y-5)/5)
elev_surf_generator = gs.SRF(gs.Gaussian(dim=2, var = 1, len_scale = 2), seed=data_seed)
elevations = elev_surf_generator((sites_x, sites_y))

# ----------------------------------------------------------------------------------------------------------------
# Knots - uniform grid of 9 knots, should do this programatically...

# k = 9 # number of knots
# x_pos = np.linspace(0,10,5,True)[1:-1]
# y_pos = np.linspace(0,10,5,True)[1:-1]
# X_pos, Y_pos = np.meshgrid(x_pos,y_pos)
# knots_xy = np.vstack([X_pos.ravel(), Y_pos.ravel()]).T
# knots_x = knots_xy[:,0]
# knots_y = knots_xy[:,1]

# isometric knot grid
N_outer_grid = 9
x_pos                    = np.linspace(minX + 1, maxX + 1, num = int(2*np.sqrt(N_outer_grid)))
y_pos                    = np.linspace(minY + 1, maxY + 1, num = int(2*np.sqrt(N_outer_grid)))
x_outer_pos              = x_pos[0::2]
x_inner_pos              = x_pos[1::2]
y_outer_pos              = y_pos[0::2]
y_inner_pos              = y_pos[1::2]
X_outer_pos, Y_outer_pos = np.meshgrid(x_outer_pos, y_outer_pos)
X_inner_pos, Y_inner_pos = np.meshgrid(x_inner_pos, y_inner_pos)
knots_outer_xy           = np.vstack([X_outer_pos.ravel(), Y_outer_pos.ravel()]).T
knots_inner_xy           = np.vstack([X_inner_pos.ravel(), Y_inner_pos.ravel()]).T
knots_xy                 = np.vstack((knots_outer_xy, knots_inner_xy))
knots_id_in_domain       = [row for row in range(len(knots_xy)) if (minX < knots_xy[row,0] < maxX and minY < knots_xy[row,1] < maxY)]
knots_xy                 = knots_xy[knots_id_in_domain]
knots_x                  = knots_xy[:,0]
knots_y                  = knots_xy[:,1]
k                        = len(knots_id_in_domain)

# ----------------------------------------------------------------------------------------------------------------
# Copula Splines

bandwidth = 4 # range for the gaussian kernel
radius = 4 # radius of infuence for basis, 3.5 might make some points closer to the edge of circle, might lead to numerical issues
radius_from_knots = np.repeat(radius, k) # ?influence radius from a knot?
assert k == len(knots_xy)

# Weight matrix generated using Gaussian Smoothing Kernel
gaussian_weight_matrix = np.full(shape = (Ns, k), fill_value = np.nan)
for site_id in np.arange(Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                    XB = knots_xy)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
    gaussian_weight_matrix[site_id, :] = weight_from_knots

# Weight matrix generated using wendland basis
wendland_weight_matrix = np.full(shape = (Ns,k), fill_value = np.nan)
for site_id in np.arange(Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                    XB = knots_xy)
    # influence coming from each of the knots
    weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
    wendland_weight_matrix[site_id, :] = weight_from_knots

# # constant weight matrix
# constant_weight_matrix = np.full(shape = (Ns, k), fill_value = np.nan)
# for site_id in np.arange(Ns):
#     # Compute distance between each pair of the two collections of inputs
#     d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
#                                     XB = knots_xy)
#     # influence coming from each of the knots
#     weight_from_knots = np.repeat(1, k)/k
#     constant_weight_matrix[site_id, :] = weight_from_knots

# ----------------------------------------------------------------------------------------------------------------
# Setup For the Marginal Model - GEV(mu, sigma, ksi)

# ----- using splines for mu0 and mu1 ---------------------------------------------------------------------------
# "knots" and prediction sites for splines 
gs_x        = np.linspace(minX, maxX, 50)
gs_y        = np.linspace(minY, maxY, 50)
gs_xy       = np.vstack([coords.ravel() for coords in np.meshgrid(gs_x, gs_y, indexing='ij')]).T # indexing='ij' fill vertically, need .T in imshow

gs_x_ro     = numpy2rpy(gs_x)        # Convert to R object
gs_y_ro     = numpy2rpy(gs_y)        # Convert to R object
gs_xy_ro    = numpy2rpy(gs_xy)       # Convert to R object
sites_xy_ro = numpy2rpy(sites_xy)    # Convert to R object

r.assign("gs_x_ro", gs_x_ro)         # Note: this is a matrix in R, not df
r.assign("gs_y_ro", gs_y_ro)         # Note: this is a matrix in R, not df
r.assign("gs_xy_ro", gs_xy_ro)       # Note: this is a matrix in R, not df
r.assign('sites_xy_ro', sites_xy_ro) # Note: this is a matrix in R, not df

mgcv = importr('mgcv')
r('''
    gs_xy_df <- as.data.frame(gs_xy_ro)
    colnames(gs_xy_df) <- c('x','y')
    sites_xy_df <- as.data.frame(sites_xy_ro)
    colnames(sites_xy_df) <- c('x','y')
    ''')

# r("save(gs_x_ro, file='gs_x_ro.gzip', compress=TRUE)")
# r("save(gs_y_ro, file='gs_y_ro.gzip', compress=TRUE)")
# r("save(gs_xy_df, file='gs_xy_df.gzip', compress=TRUE)")
# r("save(sites_xy_df, file='sites_xy_df.gzip',compress=TRUE)")

# Location mu_0(s) ----------------------------------------------------------------------------------------------
Beta_mu0_splines_m = 12 - 1 # number of splines basis, -1 b/c drop constant column
Beta_mu0_m         = Beta_mu0_splines_m + 2 # adding intercept and elevation
C_mu0_splines      = np.array(r('''
                                basis      <- smoothCon(s(x, y, k = {Beta_mu0_splines_m}, fx = TRUE), data = gs_xy_df)[[1]]
                                basis_site <- PredictMat(basis, data = sites_xy_df)
                                # basis_site
                                basis_site[,c(-(ncol(basis_site)-2))] # dropped the 3rd to last column of constant
                                '''.format(Beta_mu0_splines_m = Beta_mu0_splines_m+1))) # shaped(Ns, Beta_mu0_splines_m)
C_mu0_1t           = np.column_stack((np.ones(Ns),  # intercept
                                    elevations,     # elevation
                                    C_mu0_splines)) # splines (excluding intercept)
C_mu0              = np.tile(C_mu0_1t.T[:,:,None], reps = (1, 1, Nt))

# Location mu_1(s) ----------------------------------------------------------------------------------------------

Beta_mu1_splines_m = 18 - 1 # drop the 3rd to last column of constant
Beta_mu1_m         = Beta_mu1_splines_m + 2 # adding intercept and elevation
C_mu1_splines      = np.array(r('''
                                basis      <- smoothCon(s(x, y, k = {Beta_mu1_splines_m}, fx = TRUE), data = gs_xy_df)[[1]]
                                basis_site <- PredictMat(basis, data = sites_xy_df)
                                # basis_site
                                basis_site[,c(-(ncol(basis_site)-2))] # drop the 3rd to last column of constant
                                '''.format(Beta_mu1_splines_m = Beta_mu1_splines_m+1))) # shaped(Ns, Beta_mu1_splines_m)
C_mu1_1t           = np.column_stack((np.ones(Ns),  # intercept
                                    elevations,     # elevation
                                    C_mu1_splines)) # splines (excluding intercept)
C_mu1              = np.tile(C_mu1_1t.T[:,:,None], reps = (1, 1, Nt))

# Scale logsigma(s) ----------------------------------------------------------------------------------------------

Beta_logsigma_m   = 2 # just intercept and elevation
C_logsigma        = np.full(shape = (Beta_logsigma_m, Ns, Nt), fill_value = np.nan)
C_logsigma[0,:,:] = 1.0 
C_logsigma[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T

# Shape ksi(s) ----------------------------------------------------------------------------------------------

Beta_ksi_m   = 2 # just intercept and elevation
C_ksi        = np.full(shape = (Beta_ksi_m, Ns, Nt), fill_value = np.nan) # ksi design matrix
C_ksi[0,:,:] = 1.0
C_ksi[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T

# ----------------------------------------------------------------------------------------------------------------
# Setup For the Copula/Data Model - X_star = R^phi * g(Z)

# Covariance K for Gaussian Field g(Z) --------------------------------------------------------------------------
nu = 0.5 # exponential kernel for matern with nu = 1/2
sigsq = 1.0 # sill for Z
sigsq_vec = np.repeat(sigsq, Ns) # hold at 1

# Scale Mixture R^phi --------------------------------------------------------------------------------------------
## phi and gamma
gamma = 0.5 # this is the gamma that goes in rlevy, gamma_at_knots
delta = 0.0 # this is the delta in levy, stays 0
alpha = 0.5
gamma_at_knots = np.repeat(gamma, k)
gamma_vec = np.sum(np.multiply(wendland_weight_matrix, gamma_at_knots)**(alpha), 
                    axis = 1)**(1/alpha) # bar{gamma}, axis = 1 to sum over K knots


# ----------------------------------------------------------------------------------------------------------------
# Marginal Parameters - GEV(mu, sigma, ksi)
Beta_mu0            = np.concatenate(([0], [0.1], np.array([0.05]*Beta_mu0_splines_m)))
Beta_mu1            = np.concatenate(([0], [0.01], np.array([0.01] * Beta_mu1_splines_m)))
Beta_logsigma       = np.array([0.0, 0.01])
Beta_ksi            = np.array([0.2, 0.05])
sigma_Beta_mu0      = 1
sigma_Beta_mu1      = 1
sigma_Beta_logsigma = 1
sigma_Beta_ksi      = 1

# ----------------------------------------------------------------------------------------------------------------
# Data Model Parameters - X_star = R^phi * g(Z)

range_at_knots = np.sqrt(0.3*knots_x + 0.4*knots_y)/2 # range for spatial Matern Z

### scenario 1
# phi_at_knots = 0.65-np.sqrt((knots_x-3)**2/4 + (knots_y-3)**2/3)/10
### scenario 2
phi_at_knots = 0.65-np.sqrt((knots_x-5.1)**2/5 + (knots_y-5.3)**2/4)/11.6
### scenario 3
# phi_at_knots = 0.37 + 5*(scipy.stats.multivariate_normal.pdf(knots_xy, mean = np.array([2.5,3]), cov = 2*np.matrix([[1,0.2],[0.2,1]])) + 
                        #  scipy.stats.multivariate_normal.pdf(knots_xy, mean = np.array([7,7.5]), cov = 2*np.matrix([[1,-0.2],[-0.2,1]])))

# ----------------------------------------------------------------------------------------------------------------
# Generate Data

# W = g(Z), Z ~ MVN(0, K)
range_vec = gaussian_weight_matrix @ range_at_knots
K         = ns_cov(range_vec = range_vec, sigsq_vec = sigsq_vec,
                    coords = sites_xy, kappa = nu, cov_model = "matern")
Z         = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(Ns,)),cov=K,size=Nt).T
W         = norm_to_Pareto(Z) 

# R^phi Scaling Factor
phi_vec    = gaussian_weight_matrix @ phi_at_knots
R_at_knots = np.full(shape = (k, Nt), fill_value = np.nan)
for t in np.arange(Nt):
    R_at_knots[:,t] = rlevy(n = k, m = delta, s = gamma) # generate R at time t, spatially varying k knots
    # should need to vectorize rlevy so in future s = gamma_at_knots (k,) vector
    # R_at_knots[:,t] = scipy.stats.levy.rvs(delta, gamma, k)
    # R_at_knots[:,t] = np.repeat(rlevy(n = 1, m = delta, s = gamma), k) # generate R at time t, spatially constant k knots
R_at_sites = wendland_weight_matrix @ R_at_knots
R_phi      = np.full(shape = (Ns, Nt), fill_value = np.nan)
for t in np.arange(Nt):
    R_phi[:,t] = np.power(R_at_sites[:,t], phi_vec)

# F_Y(y) = F_Xstar(Xstar = R^phi * g(Z))
mu_matrix    = (C_mu0.T @ Beta_mu0).T + (C_mu1.T @ Beta_mu1).T * Time
sigma_matrix = np.exp((C_logsigma.T @ Beta_logsigma).T)
ksi_matrix   = (C_ksi.T @ Beta_ksi).T
X_star       = R_phi * W
Y            = np.full(shape = (Ns, Nt), fill_value = np.nan)
for t in np.arange(Nt):
    Y[:,t] = qgev(pRW(X_star[:,t], phi_vec, gamma_vec), mu_matrix[:,t], sigma_matrix[:,t], ksi_matrix[:,t])

mu0_estimates = None
mu1_estimates = None
logsigma_estimates = None
ksi_estimates = None

# %% load traceplot
# load traceplots

# folder                    = './data/20240221_t32_s125_shifted_isogrid_elev200_postcov_20k/'
# folder                    = './data/20240225_t32_s125_standard_isogrid_elev200_r0234_150k/'

# folder = './data/20240226_2345_sc1_t32_s300_standard_100k/'
# folder = './data/20240224_2345_sc2_t32_s300_standard_100k/'
# folder = './data/20240226_2345_sc3_t32_s300_standard_100k/'

folder = './data/20240229_t32_s300_standard_impute/'

phi_knots_trace           = np.load(folder + 'phi_knots_trace.npy')
R_trace_log               = np.load(folder + 'R_trace_log.npy')
range_knots_trace         = np.load(folder + 'range_knots_trace.npy')
Beta_mu0_trace            = np.load(folder + 'Beta_mu0_trace.npy')
Beta_mu1_trace            = np.load(folder + 'Beta_mu1_trace.npy')
Beta_logsigma_trace       = np.load(folder + 'Beta_logsigma_trace.npy')
Beta_ksi_trace            = np.load(folder + 'Beta_ksi_trace.npy')
sigma_Beta_mu0_trace      = np.load(folder + 'sigma_Beta_mu0_trace.npy')
sigma_Beta_mu1_trace      = np.load(folder + 'sigma_Beta_mu1_trace.npy')
sigma_Beta_logsigma_trace = np.load(folder + 'sigma_Beta_logsigma_trace.npy')
sigma_Beta_ksi_trace      = np.load(folder + 'sigma_Beta_ksi_trace.npy')

k               = R_trace_log.shape[1]
Nt              = R_trace_log.shape[2]
Beta_mu0_m      = Beta_mu0_trace.shape[1]
Beta_mu1_m      = Beta_mu1_trace.shape[1]
Beta_logsigma_m = Beta_logsigma_trace.shape[1]
Beta_ksi_m      = Beta_ksi_trace.shape[1]

# %%
# burnins
burnin = 4000

phi_knots_trace           = phi_knots_trace[burnin:]
R_trace_log               = R_trace_log[burnin:]
range_knots_trace         = range_knots_trace[burnin:]
Beta_mu0_trace            = Beta_mu0_trace[burnin:]
Beta_mu1_trace            = Beta_mu1_trace[burnin:]
Beta_logsigma_trace       = Beta_logsigma_trace[burnin:]
Beta_ksi_trace            = Beta_ksi_trace[burnin:]
sigma_Beta_mu0_trace      = sigma_Beta_mu0_trace[burnin:]
sigma_Beta_mu1_trace      = sigma_Beta_mu1_trace[burnin:]
sigma_Beta_logsigma_trace = sigma_Beta_logsigma_trace[burnin:]
sigma_Beta_ksi_trace      = sigma_Beta_ksi_trace[burnin:]


# %%
# remove unfinished cells

R_trace_log               = R_trace_log[~np.isnan(R_trace_log)].reshape((-1,k,Nt))
phi_knots_trace           = phi_knots_trace[~np.isnan(phi_knots_trace)].reshape((-1,k))
range_knots_trace         = range_knots_trace[~np.isnan(range_knots_trace)].reshape((-1,k))
Beta_mu0_trace            = Beta_mu0_trace[~np.isnan(Beta_mu0_trace)].reshape((-1,Beta_mu0_m))
Beta_mu1_trace            = Beta_mu1_trace[~np.isnan(Beta_mu1_trace)].reshape((-1,Beta_mu1_m))
Beta_logsigma_trace       = Beta_logsigma_trace[~np.isnan(Beta_logsigma_trace)].reshape((-1,Beta_logsigma_m))
Beta_ksi_trace            = Beta_ksi_trace[~np.isnan(Beta_ksi_trace)].reshape((-1,Beta_ksi_m))
sigma_Beta_mu0_trace      = sigma_Beta_mu0_trace[~np.isnan(sigma_Beta_mu0_trace)].reshape((-1,1))
sigma_Beta_mu1_trace      = sigma_Beta_mu1_trace[~np.isnan(sigma_Beta_mu1_trace)].reshape((-1,1))
sigma_Beta_logsigma_trace = sigma_Beta_logsigma_trace[~np.isnan(sigma_Beta_logsigma_trace)].reshape((-1,1))
sigma_Beta_ksi_trace      = sigma_Beta_ksi_trace[~np.isnan(sigma_Beta_ksi_trace)].reshape((-1,1))


#######################################
##### Posterior Covariance Matrix #####
#######################################
# %%
# posterior covariance matrix
phi_cov           = np.cov(phi_knots_trace.T)
R_log_cov         = np.full(shape=(k,k,R_trace_log.shape[2]), fill_value = np.nan)
for t in range(R_trace_log.shape[2]):
    R_log_cov[:,:,t] = np.cov(R_trace_log[:,:,t].T)
range_cov         = np.cov(range_knots_trace.T)
Beta_mu0_cov      = np.cov(Beta_mu0_trace.T)
Beta_mu1_cov      = np.cov(Beta_mu1_trace.T)
Beta_logsigma_cov = np.cov(Beta_logsigma_trace.T)
Beta_ksi_cov      = np.cov(Beta_ksi_trace.T)
sigma_Beta_mu0_cov = np.cov(sigma_Beta_mu0_trace.T)
sigma_Beta_mu1_cov = np.cov(sigma_Beta_mu1_trace.T)
sigma_Beta_logsigma_cov = np.cov(sigma_Beta_logsigma_trace.T)
sigma_Beta_ksi_cov = np.cov(sigma_Beta_ksi_trace.T)

#######################################
##### Posterior Median            #####
#######################################
# Potentially use these as initial values
# %%
# posterior median
phi_median                 = np.median(phi_knots_trace, axis = 0)
R_log_median               = np.full(shape=(k,R_trace_log.shape[2]), fill_value = np.nan)
for t in range(R_trace_log.shape[2]):
    R_log_median[:,t] = np.median(R_trace_log[:,:,t], axis = 0)
range_median               = np.median(range_knots_trace, axis = 0)
Beta_mu0_median            = np.median(Beta_mu0_trace, axis = 0)
Beta_mu1_median            = np.median(Beta_mu1_trace, axis = 0)
Beta_logsigma_median       = np.median(Beta_logsigma_trace, axis = 0)
Beta_ksi_median            = np.median(Beta_ksi_trace, axis = 0)
sigma_Beta_mu0_median      = np.median(sigma_Beta_mu0_trace, axis = 0)
sigma_Beta_mu1_median      = np.median(sigma_Beta_mu1_trace, axis = 0)
sigma_Beta_logsigma_median = np.median(sigma_Beta_logsigma_trace, axis = 0)
sigma_Beta_ksi_median      = np.median(sigma_Beta_ksi_trace, axis = 0)

#######################################
##### Posterior mean            #####
#######################################
# Potentially use these as initial values
# %%
# posterior mean
phi_mean                 = np.mean(phi_knots_trace, axis = 0)
R_log_mean               = np.full(shape=(k,R_trace_log.shape[2]), fill_value = np.nan)
for t in range(R_trace_log.shape[2]):
    R_log_mean[:,t] = np.mean(R_trace_log[:,:,t], axis = 0)
range_mean               = np.mean(range_knots_trace, axis = 0)
Beta_mu0_mean            = np.mean(Beta_mu0_trace, axis = 0)
Beta_mu1_mean            = np.mean(Beta_mu1_trace, axis = 0)
Beta_logsigma_mean       = np.mean(Beta_logsigma_trace, axis = 0)
Beta_ksi_mean            = np.mean(Beta_ksi_trace, axis = 0)
sigma_Beta_mu0_mean      = np.mean(sigma_Beta_mu0_trace, axis = 0)
sigma_Beta_mu1_mean      = np.mean(sigma_Beta_mu1_trace, axis = 0)
sigma_Beta_logsigma_mean = np.mean(sigma_Beta_logsigma_trace, axis = 0)
sigma_Beta_ksi_mean      = np.mean(sigma_Beta_ksi_trace, axis = 0)


#######################################
##### Posterior Last Iteration    #####
#######################################
# %%
# last iteration values
phi_knots_last           = phi_knots_trace[-1]
R_last_log               = R_trace_log[-1]
range_knots_last         = range_knots_trace[-1]
Beta_mu0_last            = Beta_mu0_trace[-1]
Beta_mu1_last            = Beta_mu1_trace[-1]
Beta_logsigma_last       = Beta_logsigma_trace[-1]
Beta_ksi_last            = Beta_ksi_trace[-1]
sigma_Beta_mu0_last      = sigma_Beta_mu0_trace[-1]
sigma_Beta_mu1_last      = sigma_Beta_mu1_trace[-1]
sigma_Beta_logsigma_last = sigma_Beta_logsigma_trace[-1]
sigma_Beta_ksi_last      = sigma_Beta_ksi_trace[-1]

#######################################
##### Posterior Plotting          #####
#######################################
# %%
# Print traceplot thinned by 10
iter = phi_knots_trace.shape[0]
xs       = np.arange(iter)
xs_thin  = xs[0::10] # index 1, 11, 21, ...
xs_thin2 = np.arange(len(xs_thin)) # index 1, 2, 3, ...

R_trace_log_thin               = R_trace_log[0:iter:10,:,:]
phi_knots_trace_thin           = phi_knots_trace[0:iter:10,:]
range_knots_trace_thin         = range_knots_trace[0:iter:10,:]
Beta_mu0_trace_thin            = Beta_mu0_trace[0:iter:10,:]
Beta_mu1_trace_thin            = Beta_mu1_trace[0:iter:10,:]
Beta_logsigma_trace_thin       = Beta_logsigma_trace[0:iter:10,:]
Beta_ksi_trace_thin            = Beta_ksi_trace[0:iter:10,:]
sigma_Beta_mu0_trace_thin      = sigma_Beta_mu0_trace[0:iter:10,:]
sigma_Beta_mu1_trace_thin      = sigma_Beta_mu1_trace[0:iter:10,:]
sigma_Beta_logsigma_trace_thin = sigma_Beta_logsigma_trace[0:iter:10,:]
sigma_Beta_ksi_trace_thin      = sigma_Beta_ksi_trace[0:iter:10,:]


for j in range(Beta_mu1_m):
    plt.plot(xs_thin2, Beta_mu1_trace_thin[:,j], label = 'Beta_'+str(j))
    plt.annotate('Beta_' + str(j), xy=(xs_thin2[-1], Beta_mu1_trace_thin[:,j][-1]))
plt.title('traceplot for Beta_mu1')
plt.xlabel('iter thinned by 10')
plt.ylabel('Beta_mu1')
plt.legend()  

# side by side mu0
vmin = min(np.floor(min(mu0_estimates)), np.floor(min((C_mu0.T @ Beta_mu0_mean).T[:,0])))
vmax = max(np.ceil(max(mu0_estimates)), np.ceil(max((C_mu0.T @ Beta_mu0_mean).T[:,0])))
# mpnorm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
divnorm = mpl.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)

fig, ax     = plt.subplots(1,2)
mu0_scatter = ax[0].scatter(sites_x, sites_y, s = 10, c = mu0_estimates,
                            cmap = colormaps['bwr'], norm = divnorm)
ax[0].set_aspect('equal', 'box')
ax[0].title.set_text('mu0 data estimates')
mu0_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, c = (C_mu0.T @ Beta_mu0_mean).T[:,0],
                                cmap = colormaps['bwr'], norm = divnorm)
ax[1].set_aspect('equal', 'box')
ax[1].title.set_text('mu0 post mean estimates')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(mu0_est_scatter, cax = cbar_ax)
plt.show()

# side by side mu1
vmin = min(np.floor(min(mu1_estimates)), np.floor(min((C_mu1.T @ Beta_mu1_mean).T[:,0])))
vmax = max(np.ceil(max(mu1_estimates)), np.ceil(max((C_mu1.T @ Beta_mu1_mean).T[:,0])))
# mpnorm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
divnorm = mpl.colors.TwoSlopeNorm(vcenter = 0, vmin = vmin, vmax = vmax)

fig, ax     = plt.subplots(1,2)
mu1_scatter = ax[0].scatter(sites_x, sites_y, s = 10, c = mu1_estimates,
                            cmap = colormaps['bwr'], norm = divnorm)
ax[0].set_aspect('equal', 'box')
ax[0].title.set_text('mu1 data estimates')
mu1_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, c = (C_mu1.T @ Beta_mu1_mean).T[:,0],
                                cmap = colormaps['bwr'], norm = divnorm)
ax[1].set_aspect('equal', 'box')
ax[1].title.set_text('mu1 post mean estimates')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(mu1_est_scatter, cax = cbar_ax)
plt.show()

# side by side for mu = mu0 + mu1
this_year = 31
vmin = min(np.floor(min(mu0_estimates + mu1_estimates * Time[this_year])), 
           np.floor(min(((C_mu0.T @ Beta_mu0_mean).T + (C_mu1.T @ Beta_mu1_mean).T * Time)[:,this_year])))
vmax = max(np.ceil(max(mu0_estimates + mu1_estimates * Time[this_year])), 
           np.ceil(max(((C_mu0.T @ Beta_mu0_mean).T + (C_mu1.T @ Beta_mu1_mean).T * Time)[:,this_year])))
divnorm = mpl.colors.TwoSlopeNorm(vcenter = (vmin+vmax)/2, vmin = vmin, vmax = vmax)

fig, ax     = plt.subplots(1,2)
mu0_scatter = ax[0].scatter(sites_x, sites_y, s = 10, c = mu0_estimates + mu1_estimates * Time[this_year],
                            cmap = colormaps['bwr'], norm = divnorm)
ax[0].set_aspect('equal', 'box')
ax[0].title.set_text('mu data year: ' + str(1950+this_year))
mu0_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, c = ((C_mu0.T @ Beta_mu0_mean).T + (C_mu1.T @ Beta_mu1_mean).T * Time)[:,this_year],
                                cmap = colormaps['bwr'], norm = divnorm)
ax[1].set_aspect('equal', 'box')
ax[1].title.set_text('mu post mean year: ' + str(1950+this_year))
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(mu0_est_scatter, cax = cbar_ax)
plt.show()

# side by side logsigma
vmin = min(my_floor(min(logsigma_estimates), 1), my_floor(min((C_logsigma.T @ Beta_logsigma_mean).T[:,0]), 1))
vmax = max(my_ceil(max(logsigma_estimates), 1), my_ceil(max((C_logsigma.T @ Beta_logsigma_mean).T[:,0]), 1))
divnorm = mpl.colors.TwoSlopeNorm(vcenter = (vmin+vmax)/2, vmin = vmin, vmax = vmax)

fig, ax     = plt.subplots(1,2)
logsigma_scatter = ax[0].scatter(sites_x, sites_y, s = 10, c = logsigma_estimates,
                            cmap = colormaps['bwr'], norm = divnorm)
ax[0].set_aspect('equal', 'box')
ax[0].title.set_text('logsigma data estimates')
logsigma_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, c = (C_logsigma.T @ Beta_logsigma_mean).T[:,0],
                                cmap = colormaps['bwr'], norm = divnorm)
ax[1].set_aspect('equal', 'box')
ax[1].title.set_text('logsigma post mean estimates')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(logsigma_est_scatter, cax = cbar_ax)
plt.show()

# side by side ksi
vmin = min(my_floor(min(ksi_estimates), 1), my_floor(min((C_ksi.T @ Beta_ksi_mean).T[:,0]), 1))
vmax = max(my_ceil(max(ksi_estimates), 1), my_ceil(max((C_ksi.T @ Beta_ksi_mean).T[:,0]), 1))
divnorm = mpl.colors.TwoSlopeNorm(vcenter = (vmin+vmax)/2, vmin = vmin, vmax = vmax)

fig, ax     = plt.subplots(1,2)
ksi_scatter = ax[0].scatter(sites_x, sites_y, s = 10, c = ksi_estimates,
                            cmap = colormaps['bwr'], norm = divnorm)
ax[0].set_aspect('equal', 'box')
ax[0].title.set_text('ksi data estimates')
ksi_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, c = (C_ksi.T @ Beta_ksi_mean).T[:,0],
                                cmap = colormaps['bwr'], norm = divnorm)
ax[1].set_aspect('equal', 'box')
ax[1].title.set_text('ksi post mean estimates')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(ksi_est_scatter, cax = cbar_ax)
plt.show()

plotgrid_res_x = 50
plotgrid_res_y = 75
plotgrid_res_xy = plotgrid_res_x * plotgrid_res_y
plotgrid_x = np.linspace(minX,maxX,plotgrid_res_x)
plotgrid_y = np.linspace(minY,maxY,plotgrid_res_y)
plotgrid_X, plotgrid_Y = np.meshgrid(plotgrid_x, plotgrid_y)
plotgrid_xy = np.vstack([plotgrid_X.ravel(), plotgrid_Y.ravel()]).T

gaussian_weight_matrix_for_plot = np.full(shape = (plotgrid_res_xy, k), fill_value = np.nan)
for site_id in np.arange(plotgrid_res_xy):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)), 
                                    XB = knots_xy)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
    gaussian_weight_matrix_for_plot[site_id, :] = weight_from_knots

wendland_weight_matrix_for_plot = np.full(shape = (plotgrid_res_xy,k), fill_value = np.nan)
for site_id in np.arange(plotgrid_res_xy):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)), 
                                    XB = knots_xy)
    # influence coming from each of the knots
    weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
    wendland_weight_matrix_for_plot[site_id, :] = weight_from_knots

# 3. phi surface

# heatplot of phi surface
phi_vec_for_plot = gaussian_weight_matrix_for_plot @ phi_mean
graph, ax = plt.subplots()
heatmap = ax.imshow(phi_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x), 
                    cmap ='bwr', interpolation='nearest', extent = [minX, maxX, maxY, minY])
ax.invert_yaxis()
graph.colorbar(heatmap)
plt.show()

phi_vec_for_plot = gaussian_weight_matrix_for_plot @ phi_mean
fig, ax = plt.subplots()
state_map.boundary.plot(ax=ax, color = 'black')
heatmap = ax.imshow(phi_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x), 
                    cmap ='bwr', interpolation='nearest', extent = [minX, maxX, maxY, minY])
ax.invert_yaxis()
fig.colorbar(heatmap)
plt.xlim([-105,-90])
plt.ylim([30,50])
plt.show()

# 4. Plot range surface

# heatplot of range surface
range_vec_for_plot = gaussian_weight_matrix_for_plot @ range_mean
graph, ax = plt.subplots()
heatmap = ax.imshow(range_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x), 
                    cmap ='bwr', interpolation='nearest', extent = [minX, maxX, maxY, minY])
ax.invert_yaxis()
graph.colorbar(heatmap)
plt.show()

range_vec_for_plot = gaussian_weight_matrix_for_plot @ range_mean
fig, ax = plt.subplots()
state_map.boundary.plot(ax=ax, color = 'black')
heatmap = ax.imshow(range_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x), 
                    cmap ='bwr', interpolation='nearest', extent = [minX, maxX, maxY, minY])
ax.invert_yaxis()
fig.colorbar(heatmap)
plt.xlim([-105,-90])
plt.ylim([30,50])
plt.show()



# %%
# GEV_post_cov = np.cov(np.array([GEV_knots_trace[:,0,0].ravel(), # mu location
#                                 GEV_knots_trace[:,1,0].ravel()])) # tau scale

# phi_post_cov = np.cov(np.array([phi_knots_trace[:,i].ravel() for i in range(k)]))
# range_post_cov = np.cov(np.array([range_knots_trace[:,i].ravel() for i in range(k)]))

# mu = GEV_knots_trace[:,0,0].ravel()
# tau = GEV_knots_trace[:,1,0].ravel()
# phi = [phi_knots_trace[:,i].ravel() for i in range(k)]

# np.corrcoef(np.insert(phi, 0, [mu,tau], 0))
# # np.cov(np.vstack((mu, tau,phi)))
# # np.corrcoef(np.vstack((mu, tau,phi)))

# %%
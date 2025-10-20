# %%
data_seed = 2345

# imports
import os
os.environ["OMP_NUM_THREADS"]        = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"]   = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"]        = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"]    = "1"  # export NUMEXPR_NUM_THREADS=1
import numpy as np
import matplotlib
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
import pickle
mgcv = importr('mgcv')
import geopandas as gpd
state_map = gpd.read_file('./cb_2018_us_state_20m/cb_2018_us_state_20m.shp').to_crs(epsg=4326)

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
random_generator = np.random.RandomState((rank+1)*7) # use of this avoids impacting the global np state

try: # data_seed is defined when python MCMC.py
    data_seed
except: # when running on local machine interactively
    data_seed = 2345
finally:
    if rank == 0: print('data_seed: ', data_seed)
np.random.seed(data_seed)

if rank == 0: print('Pareto: ', norm_pareto)

try:
    with open('iter.pkl','rb') as file:
        start_iter = pickle.load(file) + 1
        if rank == 0: print('start_iter loaded from pickle, set to be:', start_iter)
except Exception as e:
    if rank == 0: 
        print('Exception loading iter.pkl:', e)
        print('Setting start_iter to 1')
    start_iter = 1

if norm_pareto == 'shifted':  n_iters = 20000
if norm_pareto == 'standard': n_iters = 200000

# Load Dataset    -----------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------
# data

r('''load('JJA_precip_maxima_nonimputed.RData')''')
GEV_estimates      = np.array(r('GEV_estimates')).T
mu0_estimates      = GEV_estimates[:,0]
mu1_estimates      = GEV_estimates[:,1]
logsigma_estimates = GEV_estimates[:,2]
ksi_estimates      = GEV_estimates[:,3]
JJA_maxima         = np.array(r('JJA_maxima_nonimputed'))
stations           = np.array(r('stations')).T
elevations         = np.array(r('elev')).T/200

# # truncate if only running a random subset
# Nt                 = 24
# Ns                 = 125
# times_subset       = np.arange(Nt)
# sites_subset       = np.random.default_rng(data_seed).choice(JJA_maxima.shape[0],size=Ns,replace=False,shuffle=False)
# GEV_estimates      = GEV_estimates[sites_subset,:]
# mu0_estimates      = GEV_estimates[:,0]
# mu1_estimates      = GEV_estimates[:,1]
# logsigma_estimates = GEV_estimates[:,2]
# ksi_estimates      = GEV_estimates[:,3]
# JJA_maxima         = JJA_maxima[sites_subset,:][:,times_subset]
# stations           = stations[sites_subset]
# elevations         = elevations[sites_subset]

Y           = JJA_maxima.copy()
miss_matrix = np.isnan(Y)

# Setup (Covariates and Constants)    ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------
# Ns, Nt

Nt = JJA_maxima.shape[1] # number of time replicates
Ns = JJA_maxima.shape[0] # number of sites/stations
start_year = 1949
end_year   = 2023
all_years  = np.linspace(start_year, end_year, Nt)
# Note, to use the mu1 estimates from Likun, the `Time` must be standardized the same way
Time       = (all_years - np.mean(all_years))/np.std(all_years, ddof=1) # delta degress of freedom, to match the n-1 in R
Time       = Time[0:Nt] # if there is any truncation specified above
assert len(all_years) == Nt

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

# isometric knot grid - for R and phi
N_outer_grid = 16
h_dist_between_knots     = (maxX - minX) / (int(2*np.sqrt(N_outer_grid))-1)
v_dist_between_knots     = (maxY - minY) / (int(2*np.sqrt(N_outer_grid))-1)
x_pos                    = np.linspace(minX + h_dist_between_knots/2, maxX + h_dist_between_knots/2, 
                                       num = int(2*np.sqrt(N_outer_grid)))
y_pos                    = np.linspace(minY + v_dist_between_knots/2, maxY + v_dist_between_knots/2, 
                                       num = int(2*np.sqrt(N_outer_grid)))
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

# isometric knot grid - for rho (de-coupled from phi)
N_outer_grid_rho = 16
h_dist_between_knots_rho     = (maxX - minX) / (int(2*np.sqrt(N_outer_grid_rho))-1)
v_dist_between_knots_rho     = (maxY - minY) / (int(2*np.sqrt(N_outer_grid_rho))-1)
x_pos_rho                    = np.linspace(minX + h_dist_between_knots_rho/2, maxX + h_dist_between_knots_rho/2, 
                                       num = int(2*np.sqrt(N_outer_grid_rho)))
y_pos_rho                    = np.linspace(minY + v_dist_between_knots_rho/2, maxY + v_dist_between_knots_rho/2, 
                                       num = int(2*np.sqrt(N_outer_grid_rho)))
x_outer_pos_rho              = x_pos_rho[0::2]
x_inner_pos_rho              = x_pos_rho[1::2]
y_outer_pos_rho              = y_pos_rho[0::2]
y_inner_pos_rho              = y_pos_rho[1::2]
X_outer_pos_rho, Y_outer_pos_rho = np.meshgrid(x_outer_pos_rho, y_outer_pos_rho)
X_inner_pos_rho, Y_inner_pos_rho = np.meshgrid(x_inner_pos_rho, y_inner_pos_rho)
knots_outer_xy_rho           = np.vstack([X_outer_pos_rho.ravel(), Y_outer_pos_rho.ravel()]).T
knots_inner_xy_rho           = np.vstack([X_inner_pos_rho.ravel(), Y_inner_pos_rho.ravel()]).T
knots_xy_rho                 = np.vstack((knots_outer_xy_rho, knots_inner_xy_rho))
knots_id_in_domain_rho       = [row for row in range(len(knots_xy_rho)) if (minX < knots_xy_rho[row,0] < maxX and minY < knots_xy_rho[row,1] < maxY)]
knots_xy_rho                 = knots_xy_rho[knots_id_in_domain_rho]
knots_x_rho                  = knots_xy_rho[:,0]
knots_y_rho                  = knots_xy_rho[:,1]
k_rho                        = len(knots_id_in_domain_rho)

# ----------------------------------------------------------------------------------------------------------------
# Copula Splines

# Basis Parameters - for the Gaussian and Wendland Basis

radius            = 4                    # radius of Wendland Basis for R
radius_from_knots = np.repeat(radius, k) # influence radius from a knot

bandwidth         = 4               # range for the gaussian basis for phi
# eff_range       = radius                    # range for the gaussian basis s.t. effective range is `radius`: exp(-3) = 0.05
# bandwidth       = eff_range**2/6       

bandwidth_rho     = 4                    # range for the gaussian basis for rho
# eff_range_rho   = radius                    # range for the gaussian basis for rho s.t. effective range is `radius`
# bandwidth_rho   = eff_range_rho**2/6

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

# Gaussian weight matrix specific to the rho/range surface
gaussian_weight_matrix_rho = np.full(shape = (Ns, k_rho), fill_value = np.nan)
for site_id in np.arange(Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                                XB = knots_xy_rho)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_rho, cutoff = False)
    gaussian_weight_matrix_rho[site_id, :] = weight_from_knots

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

Beta_mu1_splines_m = 12 - 1 # drop the 3rd to last column of constant
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

Beta_mu0 = np.linalg.lstsq(a=C_mu0[:,:,0].T, b=mu0_estimates,rcond=None)[0]
Beta_mu1 = np.linalg.lstsq(a=C_mu1[:,:,0].T, b=mu1_estimates,rcond=None)[0]
Beta_logsigma = np.linalg.lstsq(a=C_logsigma[:,:,0].T, b=logsigma_estimates,rcond=None)[0]
Beta_ksi = np.linalg.lstsq(a=C_ksi[:,:,0].T, b=ksi_estimates,rcond=None)[0]

## Note: these sigma_Beta_xx must be values, can't be arrays
sigma_Beta_mu0      = 9.62944645
sigma_Beta_mu1      = 0.22947093
sigma_Beta_logsigma = 1.79421561
sigma_Beta_ksi      = 0.13111096

mu_matrix    = (C_mu0.T @ Beta_mu0).T + (C_mu1.T @ Beta_mu1).T * Time
sigma_matrix = np.exp((C_logsigma.T @ Beta_logsigma).T)
ksi_matrix   = (C_ksi.T @ Beta_ksi).T

# ----------------------------------------------------------------------------------------------------------------
# Data Model Parameters - X_star = R^phi * g(Z)

# Covariance K for Gaussian Field g(Z) --------------------------------------------------------------------------------------------

# Estimate range: using sites within the radius of each knot
range_at_knots = np.array([])
distance_matrix = np.full(shape=(Ns, k_rho), fill_value=np.nan)
# distance from knots
for site_id in np.arange(Ns):
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), XB = knots_xy_rho)
    distance_matrix[site_id,:] = d_from_knots
# each knot's "own" sites
sites_within_knots = {}
for knot_id in np.arange(k_rho):
    knot_name = 'knot_' + str(knot_id)
    sites_within_knots[knot_name] = np.where(distance_matrix[:,knot_id] <= bandwidth_rho)[0]

# empirical variogram estimates
for key in sites_within_knots.keys():
    selected_sites           = sites_within_knots[key]
    demeaned_Y               = Y - mu_matrix
    bin_center, gamma_variog = gs.vario_estimate((sites_x[selected_sites], sites_y[selected_sites]), 
                                                np.nanmean(demeaned_Y[selected_sites], axis=1))
    fit_model = gs.Exponential(dim=2)
    fit_model.fit_variogram(bin_center, gamma_variog, nugget=False)
    # ax = fit_model.plot(x_max = 4)
    # ax.scatter(bin_center, gamma_variog)
    range_at_knots = np.append(range_at_knots, fit_model.len_scale)
if rank == 0:
    print('estimated range:',range_at_knots)

# check for unreasonably large values, intialize at some smaller ones
range_upper_bound = 4
if len(np.where(range_at_knots > range_upper_bound)[0]) > 0:
    if rank == 0: print('estimated range >', range_upper_bound, ' at:', np.where(range_at_knots > range_upper_bound)[0])
    if rank == 0: print('range at those knots set to be at', range_upper_bound)
    range_at_knots[np.where(range_at_knots > range_upper_bound)[0]] = range_upper_bound

# check for unreasonably small values, initialize at some larger ones
range_lower_bound = 0.01
if len(np.where(range_at_knots < range_lower_bound)[0]) > 0:
    if rank == 0: print('estimated range <', range_lower_bound, ' at:', np.where(range_at_knots < range_lower_bound)[0])
    if rank == 0: print('range at those knots set to be at', range_lower_bound)
    range_at_knots[np.where(range_at_knots < range_lower_bound)[0]] = range_lower_bound    

# range_vec = gaussian_weight_matrix @ range_at_knots

# Scale Mixture R^phi --------------------------------------------------------------------------------------------

phi_at_knots = np.array([0.4] * k)
phi_vec = gaussian_weight_matrix @ phi_at_knots

if norm_pareto == 'standard':
    R_at_knots = np.full(shape = (k, Nt), fill_value = np.nan)
    for t in np.arange(Nt):
        # R_at_knots[:,t] = (np.min(qRW(pgev(Y[:,t], mu_matrix[:,t], sigma_matrix[:,t], ksi_matrix[:,t]), 
        #                         phi_vec, gamma_vec))/1.5)**2
        
        # only use non-missing values
        miss_index_1t = np.where(miss_matrix[:,t] == True)[0]
        obs_index_1t  = np.where(miss_matrix[:,t] == False)[0]
        R_at_knots[:,t] = (np.min(qRW(pgev(Y[obs_index_1t,t], 
                                            mu_matrix[obs_index_1t,t], sigma_matrix[obs_index_1t,t], ksi_matrix[obs_index_1t,t]), 
                                    phi_vec[obs_index_1t], gamma_vec[obs_index_1t]))/1.5)**(1/phi_at_knots)
else: # norm_pareto == 'shifted':
    # Calculate Rt in Parallel, only use non-missing values
    comm.Barrier()
    miss_index_1t = np.where(miss_matrix[:,rank] == True)[0]
    obs_index_1t  = np.where(miss_matrix[:,rank] == False)[0]
    X_1t       = qRW(pgev(Y[obs_index_1t,rank], mu_matrix[obs_index_1t,rank], sigma_matrix[obs_index_1t,rank], ksi_matrix[obs_index_1t,rank]),
                        phi_vec[obs_index_1t], gamma_vec[obs_index_1t])
    R_1t       = np.array([np.median(X_1t)**2] * k)
    R_gathered = comm.gather(R_1t, root = 0)
    R_at_knots = np.array(R_gathered).T if rank == 0 else None
    R_at_knots = comm.bcast(R_at_knots, root = 0)

# %%

# 2. Elevation
fig, ax = plt.subplots()
elev_scatter = ax.scatter(sites_x, sites_y, s=10, c = elevations,
                            cmap = 'bwr')
ax.set_aspect('equal', 'box')
plt.colorbar(elev_scatter)
plt.savefig('Plot_station_elevation.pdf')
plt.show()
# plt.close()       


# 5. GEV Surfaces
mu0_matrix      = (C_mu0.T @ Beta_mu0).T  
mu1_matrix      = (C_mu1.T @ Beta_mu1).T
mu_matrix       = mu0_matrix + mu1_matrix * Time
logsigma_matrix = (C_logsigma.T @ Beta_logsigma).T
sigma_matrix    = np.exp(logsigma_matrix)
ksi_matrix      = (C_ksi.T @ Beta_ksi).T

def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

# Scale # -------------------------------------------------------------------------------------
## logsigma(s) plot stations
vmin = min(my_floor(min(logsigma_estimates), 1), my_floor(min(logsigma_matrix[:,0]), 1))
vmax = max(my_ceil(max(logsigma_estimates), 1), my_ceil(max(logsigma_matrix[:,0]), 1))
divnorm = matplotlib.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)
fig, ax = plt.subplots(1,2)
logsigma_scatter = ax[0].scatter(sites_x, sites_y, s = 10, cmap = 'bwr', c = logsigma_estimates, norm = divnorm)
ax[0].set_aspect('equal', 'box')
ax[0].title.set_text('GEV logsigma estimates')
logsigma_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, cmap = 'bwr', c = logsigma_matrix[:,0], norm = divnorm)
ax[1].set_aspect('equal','box')
ax[1].title.set_text('spline logsigma fit')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(logsigma_est_scatter, cax = cbar_ax)
plt.show()
# plt.savefig('Plot_initial_logsigma_estimates.pdf')

# Shape # -------------------------------------------------------------------------------------
# ksi(s) plot stations
vmin = min(my_floor(min(ksi_estimates), 1), my_floor(min(ksi_matrix[:,0]), 1))
vmax = max(my_ceil(max(ksi_estimates), 1), my_ceil(max(ksi_matrix[:,0]), 1))
divnorm = matplotlib.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)
fig, ax = plt.subplots(1,2)
ksi_scatter = ax[0].scatter(sites_x, sites_y, s = 10, cmap = 'bwr', c = ksi_estimates, norm = divnorm)
ax[0].set_aspect('equal', 'box')
ax[0].title.set_text('GEV ksi estimates')
ksi_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, cmap = 'bwr', c = ksi_matrix[:,0], norm = divnorm)
ax[1].set_aspect('equal','box')
ax[1].title.set_text('spline ksi fit')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(ksi_est_scatter, cax = cbar_ax)
plt.show()
# plt.savefig('Plot_initial_ksi_estimates.pdf')


# %% 5-panel summary: Elevation vs smoothed GEV surfaces
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig = plt.figure(figsize=(16.5, 11))
gs  = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1],
                       wspace=0.15, hspace=0.15)

ax_ls_est  = fig.add_subplot(gs[0, 0])
ax_ls_fit  = fig.add_subplot(gs[0, 1])
ax_ksi_est = fig.add_subplot(gs[1, 0])
ax_ksi_fit = fig.add_subplot(gs[1, 1])
ax_elev    = fig.add_subplot(gs[0, 2])     # spans two rows

def add_cb(ax, mappable):
    """Append a right-side colorbar of fixed relative size for uniform axes sizes."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad="2%")
    return fig.colorbar(mappable, cax=cax)

# --- log sigma: estimates (own color scale) ---
vmin_ls_est = my_floor(np.nanmin(logsigma_estimates), 1)
vmax_ls_est = my_ceil (np.nanmax(logsigma_estimates), 1)
norm_ls_est = matplotlib.colors.TwoSlopeNorm(vcenter=(vmin_ls_est+vmax_ls_est)/2,
                                             vmin=vmin_ls_est, vmax=vmax_ls_est)
sc_ls_est = ax_ls_est.scatter(sites_x, sites_y, s=10, c=logsigma_estimates,
                              cmap='viridis', norm=norm_ls_est, rasterized=True)
state_map.plot(ax=ax_ls_est, facecolor='none', edgecolor='0.3', linewidth=0.4, zorder=0)
ax_ls_est.set_aspect('equal', 'box')
ax_ls_est.set_title('log(\u03C3) estimates', fontsize = 20)
# fig.colorbar(sc_ls_est, ax=ax_ls_est, fraction=0.046, pad=0.02)

# --- log sigma splint smoothed ---
vmin_ls = min(my_floor(np.nanmin(logsigma_matrix[:,0]), 1), my_floor(np.nanmin(logsigma_matrix[:,0]), 1))
vmax_ls = max(my_ceil (np.nanmax(logsigma_matrix[:,0]), 1), my_ceil (np.nanmax(logsigma_matrix[:,0]), 1))
norm_ls = matplotlib.colors.TwoSlopeNorm(vcenter=(vmin_ls+vmax_ls)/2, vmin=vmin_ls, vmax=vmax_ls)
sc_ls_fit = ax_ls_fit.scatter(sites_x, sites_y, s=10, c=logsigma_matrix[:,0],
                              cmap='viridis', norm=norm_ls, rasterized=True)
state_map.plot(ax=ax_ls_fit, facecolor='none', edgecolor='0.3', linewidth=0.4, zorder=0)
ax_ls_fit.set_aspect('equal', 'box')
ax_ls_fit.set_title('log(\u03C3) spline-smoothed', fontsize = 20)
# fig.colorbar(sc_ls_fit, ax=ax_ls_fit, fraction=0.046, pad=0.02)

# --- xi initial estimates ---
vmin_k_est = min(my_floor(np.nanmin(ksi_estimates), 1), my_floor(np.nanmin(ksi_estimates), 1))
vmax_k_est = max(my_ceil (np.nanmax(ksi_estimates), 1), my_ceil (np.nanmax(ksi_estimates), 1))
norm_k_est = matplotlib.colors.TwoSlopeNorm(vcenter=(vmin_k_est+vmax_k_est)/2,vmin=vmin_k_est, vmax=vmax_k_est)

sc_ks_est = ax_ksi_est.scatter(sites_x, sites_y, s=10, c=ksi_estimates,
                               cmap='viridis', norm=norm_k_est, rasterized=True)
state_map.plot(ax=ax_ksi_est, facecolor='none', edgecolor='0.3', linewidth=0.4, zorder=0)
ax_ksi_est.set_aspect('equal', 'box')
ax_ksi_est.set_title('\u03BE estimates', fontsize = 20)
# fig.colorbar(sc_ks_est, ax=ax_ksi_est, fraction=0.046, pad=0.02)

# --- xi spline-smoothed ---
# vmin_k = min(my_floor(np.nanmin(ksi_matrix[:,0]), 1), my_floor(np.nanmin(ksi_matrix[:,0]), 1))
# vmax_k = max(my_ceil (np.nanmax(ksi_matrix[:,0]), 1), my_ceil (np.nanmax(ksi_matrix[:,0]), 1))
vmin_k = np.nanmin(ksi_matrix[:,0])
vmax_k = np.nanmax(ksi_matrix[:,0])
norm_k = matplotlib.colors.TwoSlopeNorm(vcenter=(vmin_k+vmax_k)/2,vmin=vmin_k, vmax=vmax_k)
sc_ks_fit = ax_ksi_fit.scatter(sites_x, sites_y, s=10, c=ksi_matrix[:,0],
                               cmap='viridis', norm=norm_k, rasterized=True)
state_map.plot(ax=ax_ksi_fit, facecolor='none', edgecolor='0.3', linewidth=0.4, zorder=0)
ax_ksi_fit.set_aspect('equal', 'box')
ax_ksi_fit.set_title('\u03BE spline-smoothed', fontsize = 20)
# fig.colorbar(sc_ks_fit, ax=ax_ksi_fit, fraction=0.046, pad=0.02)

# --- Elevation (scaled, you defined elevations = elev/200 above) ---
vmin_e = my_floor(np.nanmin(elevations), 1)
vmax_e = my_ceil (np.nanmax(elevations), 1)
elev_sc = ax_elev.scatter(sites_x, sites_y, s=10, c=-elevations, cmap='viridis',
                          rasterized=True)

# state boundaries overlay (already loaded as state_map)
state_map.plot(ax=ax_elev, facecolor='none', edgecolor='0.3', linewidth=0.4, zorder=0)
ax_elev.set_aspect('equal', 'box')
ax_elev.set_title('(Negated) Elevation (scaled)', fontsize = 20)
# fig.colorbar(elev_sc, ax=ax_elev, fraction=0.046, pad=0.02)

for ax in [ax_elev, ax_ls_est, ax_ls_fit, ax_ksi_est, ax_ksi_fit]:
    ax.set_xlim([-102, -92])
    ax.set_ylim([32, 44])
    # ax.set_aspect('equal', adjustable='box')  # remove this
    ax.set_box_aspect((44-32)/(-92 - (-102)))  # 12/10 = 1.2
    ax.tick_params(labelsize=14)

cbar = fig.colorbar(sc_ls_est, ax=ax_ls_est, fraction=0.06, pad=0.02)
cbar.ax.tick_params(labelsize=14)
cbar = fig.colorbar(sc_ls_fit, ax=ax_ls_fit, fraction=0.06, pad=0.02)
cbar.ax.tick_params(labelsize=14)
cbar = fig.colorbar(sc_ks_est, ax=ax_ksi_est, fraction=0.06, pad=0.02)
cbar.ax.tick_params(labelsize=14)
cbar = fig.colorbar(sc_ks_fit, ax=ax_ksi_fit, fraction=0.06, pad=0.02)
cbar.ax.tick_params(labelsize=14)
cbar = fig.colorbar(elev_sc, ax=ax_elev, fraction=0.06, pad=0.02)
cbar.ax.tick_params(labelsize=14)

plt.savefig('Fig_elev_logsigma_xi_5panel.pdf', bbox_inches='tight')
plt.show()

# %%

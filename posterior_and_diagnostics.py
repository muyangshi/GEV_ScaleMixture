"""
-------- Results Surface --------

Make surface of posterior mean for 
    - GEV
    - phi
    - rho

Model estimate chi surface
    - April 19 Ben: use R drawn from stable prior

-------- Diagnostics --------

Make QQplot of gumbel (April 14)
    - mean(per MCMC iter GEV)
    Note that CDF(Y) and transformation to gumbel should be performed on misspiggy
    require lots of memory to load the pY per MCMC iteration

loglikelihood at testing sites (April 15)
    - April 19 Ben: ll calculation isn't linear, use per-iteration parameter value


"""
# %% imports
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import geopandas as gpd
state_map = gpd.read_file('./cb_2018_us_state_20m/cb_2018_us_state_20m.shp')
import matplotlib as mpl
from matplotlib import colormaps
import os
# os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
# os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
# os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
# os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
import numpy as np
import matplotlib.pyplot as plt
import scipy
from utilities import *
from rpy2.robjects import r 
from rpy2.robjects.numpy2ri import numpy2rpy
from rpy2.robjects.packages import importr
import multiprocessing
from math import sin, cos, sqrt, atan2, radians, asin
import math

# the training dataset
mgcv = importr('mgcv')
r('''load('JJA_precip_maxima_nonimputed.RData')''')
GEV_estimates      = np.array(r('GEV_estimates')).T
mu0_estimates      = GEV_estimates[:,0]
mu1_estimates      = GEV_estimates[:,1]
logsigma_estimates = GEV_estimates[:,2]
ksi_estimates      = GEV_estimates[:,3]
JJA_maxima         = np.array(r('JJA_maxima_nonimputed'))
stations           = np.array(r('stations')).T
elevations         = np.array(r('elev')).T/200
Y = JJA_maxima.copy()
miss_matrix = np.isnan(Y)
Nt = 75
Ns = 590
start_year = 1949
end_year   = 2023
all_years  = np.linspace(start_year, end_year, Nt)
Time       = (all_years - np.mean(all_years))/np.std(all_years, ddof=1)

# helper functions
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

def coord_to_dist(coord1: tuple, coord2: tuple):
    R = 6373.0 # Approximate radius of earth in km

    long1 = radians(coord1[0])
    lat1  = radians(coord1[1])
    long2 = radians(coord2[0])
    lat2  = radians(coord2[1])

    dlong = long2 - long1
    dlat  = lat2  - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlong / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def random_point_at_dist(coord1: tuple, h): # return the longitude and latitudes
    R = 6373.0
    
    lat_rad = radians(coord1[1])
    lon_rad = radians(coord1[0])
    
    angular_distance = h / R

    theta = np.random.uniform(0, 2*np.pi)

    lat_b_rad = asin(sin(lat_rad) * cos(angular_distance) + 
                     cos(lat_rad) * sin(angular_distance) * cos(theta))
    lon_b_rad = lon_rad + atan2(sin(theta) * sin(angular_distance) * cos(lat_rad),
                                cos(angular_distance) - sin(lat_rad) * sin(lat_b_rad))

    lat_b = math.degrees(lat_b_rad)
    lon_b = math.degrees(lon_b_rad)

    return np.array([lon_b, lat_b])

# %%
# Specify which chain

folder           = './data_alpine/CONVERGED/20240306_realdata_t75_s590_k13_r4/'
name             = 'k13_r4'
fixGEV           = False
radius           = 4
bandwidth_phi    = 4
bandwidth_rho    = 4
N_outer_grid_phi = 9
N_outer_grid_rho = 9
mark             = True
burnin           = 5000

# folder           = './data_alpine/CONVERGED/20240320_realdata_t75_s590_k13_r4_fixGEV/'
# name             = 'k13_r4_fixGEV'
# fixGEV           = True
# radius           = 4
# bandwidth_phi    = 4
# bandwidth_rho    = 4
# N_outer_grid_phi = 9
# N_outer_grid_rho = 9
# mark             = True
# burnin           = 5000

# folder           = './data_alpine/CONVERGED/20240328_realdata_t75_s590_k25_r2/'
# name             = 'k25_r2'
# fixGEV           = False
# radius           = 2
# bandwidth_phi    = 2
# bandwidth_rho    = 2
# N_outer_grid_phi = 16
# N_outer_grid_rho = 16
# mark             = False
# burnin           = 6000

# folder           = './data_alpine/CONVERGED/20240402_realdata_t75_s590_k25_r2_fixGEV/'
# name             = 'k25_r2_fixGEV'
# fixGEV           = True
# radius           = 2 # radius of infuence for basis, 3.5 might make some points closer to the edge of circle, might lead to numerical issues
# bandwidth_phi    = 2 # range for the gaussian kernel
# bandwidth_rho    = 2
# N_outer_grid_phi = 16
# N_outer_grid_rho = 16
# mark             = False
# burnin           = 6000

# folder           = './data_alpine/CONVERGED/20240402_realdata_t75_s590_k25_r4_fixGEV/'
# name             = 'k25_r4_fixGEV'
# fixGEV           = True
# radius           = 4 # radius of infuence for basis, 3.5 might make some points closer to the edge of circle, might lead to numerical issues
# bandwidth_phi    = 4 # range for the gaussian kernel
# bandwidth_rho    = 4
# N_outer_grid_phi = 16
# N_outer_grid_rho = 16
# mark             = False
# burnin           = 5000

# folder           = './data_alpine/CONVERGED/20240406_realdata_t75_s590_k25_r4/'
# name             = 'k25_r4'
# fixGEV           = False
# radius           = 4
# bandwidth_phi    = 4
# bandwidth_rho    = 4
# N_outer_grid_phi = 16
# N_outer_grid_rho = 16
# mark             = False
# burnin           = 5000

# folder           = './data_alpine/CONVERGED/20240410_realdata_t75_s590_k25_efr2_fixksi/'
# name             = 'k25_efr2_fixksi'
# fixGEV           = False
# radius           = 2
# bandwidth_phi    = radius**2/6
# bandwidth_rho    = radius**2/6
# N_outer_grid_phi = 16
# N_outer_grid_rho = 16
# mark             = False
# burnin           = 5000

# folder           = './data_alpine/20240428_copy/20240410_realdata_t75_s590_k25_efr2/'
# name             = 'k25_efr2'
# fixGEV           = False
# radius           = 2
# bandwidth_phi    = radius**2/6 # effective range for gaussian kernel: exp(-3) = 0.05
# bandwidth_rho    = radius**2/6
# N_outer_grid_phi = 16
# N_outer_grid_rho = 16
# mark             = False
# burnin           = 0

# folder           = './data_alpine/20240428_copy/20240419_realdata_t75_s590_k41_efr2/'
# name             = 'k41_efr2'
# fixGEV           = False
# radius           = 2
# bandwidth_phi    = radius**2/6 # effective range for gaussian kernel: exp(-3) = 0.05
# bandwidth_rho    = radius**2/6
# N_outer_grid_phi = 25
# N_outer_grid_rho = 25
# mark             = False
# burnin           = 0

# folder           = './data_alpine/20240517_copy/20240504_realdata_t75_s590_phik41efr2_rhok13r4/'
# name             = 'phik41efr2_rhok13r4'
# fixGEV           = False
# radius           = 2
# bandwidth_phi    = radius**2/6
# bandwidth_rho    = 4
# N_outer_grid_phi = 25
# N_outer_grid_rho = 9
# mark             = False
# burnin           = 0

# %% load traceplots
# load traceplots

phi_knots_trace   = np.load(folder + 'phi_knots_trace.npy')
R_trace_log       = np.load(folder + 'R_trace_log.npy') # shape [niters, k, Nt]
range_knots_trace = np.load(folder + 'range_knots_trace.npy')
Nt                = R_trace_log.shape[2]
k_phi             = phi_knots_trace.shape[1]
k_rho             = range_knots_trace.shape[1]
k_R               = R_trace_log.shape[1]

if not fixGEV:
    Beta_mu0_trace            = np.load(folder + 'Beta_mu0_trace.npy')
    Beta_mu1_trace            = np.load(folder + 'Beta_mu1_trace.npy')
    Beta_logsigma_trace       = np.load(folder + 'Beta_logsigma_trace.npy')
    sigma_Beta_mu0_trace      = np.load(folder + 'sigma_Beta_mu0_trace.npy')
    sigma_Beta_mu1_trace      = np.load(folder + 'sigma_Beta_mu1_trace.npy')
    sigma_Beta_logsigma_trace = np.load(folder + 'sigma_Beta_logsigma_trace.npy')

    Beta_mu0_m                = Beta_mu0_trace.shape[1]
    Beta_mu1_m                = Beta_mu1_trace.shape[1]
    Beta_logsigma_m           = Beta_logsigma_trace.shape[1]

    try:
        Beta_ksi_trace       = np.load(folder + 'Beta_ksi_trace.npy')
        sigma_Beta_ksi_trace = np.load(folder + 'sigma_Beta_ksi_trace.npy')
        Beta_ksi_m           = Beta_ksi_trace.shape[1]
    except:
        pass

# remove burnins and unfinished cells

phi_knots_trace   = phi_knots_trace[burnin:]
R_trace_log       = R_trace_log[burnin:]
range_knots_trace = range_knots_trace[burnin:]
R_trace_log       = R_trace_log[~np.isnan(R_trace_log)].reshape((-1,k_R,Nt))
phi_knots_trace   = phi_knots_trace[~np.isnan(phi_knots_trace)].reshape((-1,k_phi))
range_knots_trace = range_knots_trace[~np.isnan(range_knots_trace)].reshape((-1,k_rho))
if not fixGEV:
    Beta_mu0_trace            = Beta_mu0_trace[burnin:]
    Beta_mu1_trace            = Beta_mu1_trace[burnin:]
    Beta_logsigma_trace       = Beta_logsigma_trace[burnin:]
    sigma_Beta_mu0_trace      = sigma_Beta_mu0_trace[burnin:]
    sigma_Beta_mu1_trace      = sigma_Beta_mu1_trace[burnin:]
    sigma_Beta_logsigma_trace = sigma_Beta_logsigma_trace[burnin:]
    Beta_mu0_trace            = Beta_mu0_trace[~np.isnan(Beta_mu0_trace)].reshape((-1,Beta_mu0_m))
    Beta_mu1_trace            = Beta_mu1_trace[~np.isnan(Beta_mu1_trace)].reshape((-1,Beta_mu1_m))
    Beta_logsigma_trace       = Beta_logsigma_trace[~np.isnan(Beta_logsigma_trace)].reshape((-1,Beta_logsigma_m))
    sigma_Beta_mu0_trace      = sigma_Beta_mu0_trace[~np.isnan(sigma_Beta_mu0_trace)].reshape((-1,1))
    sigma_Beta_mu1_trace      = sigma_Beta_mu1_trace[~np.isnan(sigma_Beta_mu1_trace)].reshape((-1,1))
    sigma_Beta_logsigma_trace = sigma_Beta_logsigma_trace[~np.isnan(sigma_Beta_logsigma_trace)].reshape((-1,1))
    try:
        Beta_ksi_trace            = Beta_ksi_trace[burnin:]
        sigma_Beta_ksi_trace      = sigma_Beta_ksi_trace[burnin:]
        Beta_ksi_trace            = Beta_ksi_trace[~np.isnan(Beta_ksi_trace)].reshape((-1,Beta_ksi_m))
        sigma_Beta_ksi_trace      = sigma_Beta_ksi_trace[~np.isnan(sigma_Beta_ksi_trace)].reshape((-1,1))
    except:
        pass


# posterior mean
phi_mean                 = np.mean(phi_knots_trace, axis = 0)
R_log_mean               = np.full(shape=(k_R,R_trace_log.shape[2]), fill_value = np.nan)
for t in range(R_trace_log.shape[2]):
    # R_log_mean[:,t] = np.mean(R_trace_log[:,:,t], axis = 0)
    R_log_mean[:,t] = np.log(np.mean(np.exp(R_trace_log[:,:,t]), axis = 0))
range_mean               = np.mean(range_knots_trace, axis = 0)
if not fixGEV:
    Beta_mu0_mean            = np.mean(Beta_mu0_trace, axis = 0)
    Beta_mu1_mean            = np.mean(Beta_mu1_trace, axis = 0)
    Beta_logsigma_mean       = np.mean(Beta_logsigma_trace, axis = 0)
    sigma_Beta_mu0_mean      = np.mean(sigma_Beta_mu0_trace, axis = 0)
    sigma_Beta_mu1_mean      = np.mean(sigma_Beta_mu1_trace, axis = 0)
    sigma_Beta_logsigma_mean = np.mean(sigma_Beta_logsigma_trace, axis = 0)
    try:
        Beta_ksi_mean            = np.mean(Beta_ksi_trace, axis = 0)
        sigma_Beta_ksi_mean      = np.mean(sigma_Beta_ksi_trace, axis = 0)
    except:
        pass

# posterior median
phi_median                 = np.median(phi_knots_trace, axis = 0)
R_log_median               = np.full(shape=(k_R,R_trace_log.shape[2]), fill_value = np.nan)
for t in range(R_trace_log.shape[2]):
    R_log_median[:,t] = np.median(R_trace_log[:,:,t], axis = 0)
range_median               = np.median(range_knots_trace, axis = 0)
if not fixGEV:
    Beta_mu0_median            = np.median(Beta_mu0_trace, axis = 0)
    Beta_mu1_median            = np.median(Beta_mu1_trace, axis = 0)
    Beta_logsigma_median       = np.median(Beta_logsigma_trace, axis = 0)
    sigma_Beta_mu0_median      = np.median(sigma_Beta_mu0_trace, axis = 0)
    sigma_Beta_mu1_median      = np.median(sigma_Beta_mu1_trace, axis = 0)
    sigma_Beta_logsigma_median = np.median(sigma_Beta_logsigma_trace, axis = 0)
    try:
        Beta_ksi_median            = np.median(Beta_ksi_trace, axis = 0)
        sigma_Beta_ksi_median      = np.median(sigma_Beta_ksi_trace, axis = 0)
    except:
        pass

# thinned by 10
iter = phi_knots_trace.shape[0]
xs       = np.arange(iter)
xs_thin  = xs[0::10] # index 1, 11, 21, ...
xs_thin2 = np.arange(len(xs_thin)) # index 1, 2, 3, ...

R_trace_log_thin               = R_trace_log[0:iter:10,:,:]
phi_knots_trace_thin           = phi_knots_trace[0:iter:10,:]
range_knots_trace_thin         = range_knots_trace[0:iter:10,:]
if not fixGEV:
    Beta_mu0_trace_thin            = Beta_mu0_trace[0:iter:10,:]
    Beta_mu1_trace_thin            = Beta_mu1_trace[0:iter:10,:]
    Beta_logsigma_trace_thin       = Beta_logsigma_trace[0:iter:10,:]
    sigma_Beta_mu0_trace_thin      = sigma_Beta_mu0_trace[0:iter:10,:]
    sigma_Beta_mu1_trace_thin      = sigma_Beta_mu1_trace[0:iter:10,:]
    sigma_Beta_logsigma_trace_thin = sigma_Beta_logsigma_trace[0:iter:10,:]
    try:
        Beta_ksi_trace_thin            = Beta_ksi_trace[0:iter:10,:]
        sigma_Beta_ksi_trace_thin      = sigma_Beta_ksi_trace[0:iter:10,:]
    except:
        pass

# %% Splines setup 
# Splines setup 

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

if not mark:
    # isometric grid for phi (de-coupled from rho)
    h_dist_between_knots_phi     = (maxX - minX) / (int(2*np.sqrt(N_outer_grid_phi))-1)
    v_dist_between_knots_phi     = (maxY - minY) / (int(2*np.sqrt(N_outer_grid_phi))-1)
    x_pos_phi                    = np.linspace(minX + h_dist_between_knots_phi/2, 
                                                maxX + h_dist_between_knots_phi/2, 
                                                num = int(2*np.sqrt(N_outer_grid_phi)))
    y_pos_phi                    = np.linspace(minY + v_dist_between_knots_phi/2, 
                                                maxY + v_dist_between_knots_phi/2, 
                                                num = int(2*np.sqrt(N_outer_grid_phi)))
    x_outer_pos_phi              = x_pos_phi[0::2]
    x_inner_pos_phi              = x_pos_phi[1::2]
    y_outer_pos_phi              = y_pos_phi[0::2]
    y_inner_pos_phi              = y_pos_phi[1::2]
    X_outer_pos_phi, Y_outer_pos_phi = np.meshgrid(x_outer_pos_phi, y_outer_pos_phi)
    X_inner_pos_phi, Y_inner_pos_phi = np.meshgrid(x_inner_pos_phi, y_inner_pos_phi)
    knots_outer_xy_phi           = np.vstack([X_outer_pos_phi.ravel(), Y_outer_pos_phi.ravel()]).T
    knots_inner_xy_phi           = np.vstack([X_inner_pos_phi.ravel(), Y_inner_pos_phi.ravel()]).T
    knots_xy_phi                 = np.vstack((knots_outer_xy_phi, knots_inner_xy_phi))
    knots_id_in_domain_phi       = [row for row in range(len(knots_xy_phi)) if (minX < knots_xy_phi[row,0] < maxX and minY < knots_xy_phi[row,1] < maxY)]
    knots_xy_phi                 = knots_xy_phi[knots_id_in_domain_phi]
    knots_x_phi                  = knots_xy_phi[:,0]
    knots_y_phi                  = knots_xy_phi[:,1]
    k_phi                        = len(knots_id_in_domain_phi)

    # isometric grid for rho (de-coupled from phi)
    h_dist_between_knots_rho     = (maxX - minX) / (int(2*np.sqrt(N_outer_grid_rho))-1)
    v_dist_between_knots_rho     = (maxY - minY) / (int(2*np.sqrt(N_outer_grid_rho))-1)
    x_pos_rho                    = np.linspace(minX + h_dist_between_knots_rho/2, 
                                                maxX + h_dist_between_knots_rho/2, 
                                                num = int(2*np.sqrt(N_outer_grid_rho)))
    y_pos_rho                    = np.linspace(minY + v_dist_between_knots_rho/2, 
                                                maxY + v_dist_between_knots_rho/2, 
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

if mark == True:
    # isometric knot grid - Mark's
    x_pos_phi                    = np.linspace(minX + 1, maxX + 1, num = int(2*np.sqrt(N_outer_grid_phi)))
    y_pos_phi                    = np.linspace(minY + 1, maxY + 1, num = int(2*np.sqrt(N_outer_grid_phi)))
    x_outer_pos_phi              = x_pos_phi[0::2]
    x_inner_pos_phi              = x_pos_phi[1::2]
    y_outer_pos_phi              = y_pos_phi[0::2]
    y_inner_pos_phi              = y_pos_phi[1::2]
    X_outer_pos_phi, Y_outer_pos_phi = np.meshgrid(x_outer_pos_phi, y_outer_pos_phi)
    X_inner_pos_phi, Y_inner_pos_phi = np.meshgrid(x_inner_pos_phi, y_inner_pos_phi)
    knots_outer_xy_phi           = np.vstack([X_outer_pos_phi.ravel(), Y_outer_pos_phi.ravel()]).T
    knots_inner_xy_phi           = np.vstack([X_inner_pos_phi.ravel(), Y_inner_pos_phi.ravel()]).T
    knots_xy_phi                 = np.vstack((knots_outer_xy_phi, knots_inner_xy_phi))
    knots_id_in_domain_phi       = [row for row in range(len(knots_xy_phi)) if (minX < knots_xy_phi[row,0] < maxX and minY < knots_xy_phi[row,1] < maxY)]
    knots_xy_phi                 = knots_xy_phi[knots_id_in_domain_phi]
    knots_x_phi                  = knots_xy_phi[:,0]
    knots_y_phi                  = knots_xy_phi[:,1]
    k_phi                        = len(knots_id_in_domain_phi)

    x_pos_rho                    = np.linspace(minX + 1, maxX + 1, num = int(2*np.sqrt(N_outer_grid_rho)))
    y_pos_rho                    = np.linspace(minY + 1, maxY + 1, num = int(2*np.sqrt(N_outer_grid_rho)))
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
# Copula Splines - Generate the weight matrices

radius_from_knots = np.repeat(radius, k_phi) # influence radius from a knot

# Weight matrix using wendland basis
wendland_weight_matrix = np.full(shape = (Ns,k_phi), fill_value = np.nan)
for site_id in np.arange(Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                                XB = knots_xy_phi)
    # influence coming from each of the knots
    weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
    wendland_weight_matrix[site_id, :] = weight_from_knots

# Weight matrix using Gaussian Smoothing Kernel
gaussian_weight_matrix_phi = np.full(shape = (Ns, k_phi), fill_value = np.nan)
for site_id in np.arange(Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                                XB = knots_xy_phi)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_phi, cutoff = False)
    gaussian_weight_matrix_phi[site_id, :] = weight_from_knots

gaussian_weight_matrix_rho = np.full(shape = (Ns, k_rho), fill_value = np.nan)
for site_id in np.arange(Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                                XB = knots_xy_rho)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_rho, cutoff = False)
    gaussian_weight_matrix_rho[site_id, :] = weight_from_knots


plotgrid_res_x = 150
plotgrid_res_y = 275
plotgrid_res_xy = plotgrid_res_x * plotgrid_res_y
plotgrid_x = np.linspace(minX,maxX,plotgrid_res_x)
plotgrid_y = np.linspace(minY,maxY,plotgrid_res_y)
plotgrid_X, plotgrid_Y = np.meshgrid(plotgrid_x, plotgrid_y)
plotgrid_xy = np.vstack([plotgrid_X.ravel(), plotgrid_Y.ravel()]).T

wendland_weight_matrix_for_plot = np.full(shape = (plotgrid_res_xy,k_phi), fill_value = np.nan)
for site_id in np.arange(plotgrid_res_xy):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)), 
                                                XB = knots_xy_phi)
    # influence coming from each of the knots
    weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
    wendland_weight_matrix_for_plot[site_id, :] = weight_from_knots

gaussian_weight_matrix_for_plot_phi = np.full(shape = (plotgrid_res_xy, k_phi), 
                                              fill_value = np.nan)
for site_id in np.arange(plotgrid_res_xy):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)), 
                                                XB = knots_xy_phi)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_phi, cutoff = False)
    gaussian_weight_matrix_for_plot_phi[site_id, :] = weight_from_knots

gaussian_weight_matrix_for_plot_rho = np.full(shape = (plotgrid_res_xy, k_rho), 
                                                fill_value = np.nan)
for site_id in np.arange(plotgrid_res_xy):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)), 
                                                XB = knots_xy_rho)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_rho, cutoff = False)
    gaussian_weight_matrix_for_plot_rho[site_id, :] = weight_from_knots        

# gaussian_weight_matrix_for_plot = gaussian_weight_matrix_for_plot_phi


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


# # %% Plot traceplot

# for i in range(k):
#     plt.subplots()
#     plt.plot(xs_thin2, phi_knots_trace_thin[:,i], label = 'knot'+str(i))
#     plt.title('traceplot for phi knot' + str(i))
#     plt.xlabel('iter thinned by 10')
#     plt.ylabel('phi')
#     plt.show()
# plt.close()

# for j in range(Beta_mu1_m):
#     plt.plot(xs_thin2, Beta_mu1_trace_thin[:,j], label = 'Beta_'+str(j))
#     plt.annotate('Beta_' + str(j), xy=(xs_thin2[-1], Beta_mu1_trace_thin[:,j][-1]))
# plt.title('traceplot for Beta_mu1')
# plt.xlabel('iter thinned by 10')
# plt.ylabel('Beta_mu1')
# plt.legend()  

# %% marginal parameter surface
# marginal parameter surface

if not fixGEV:
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
    plt.savefig('Surface:mu0.pdf')
    plt.show()
    plt.close()

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
    plt.savefig('Surface:mu1.pdf')
    plt.show()
    plt.close()

    # side by side for mu = mu0 + mu1
    this_year = 50
    vmin = min(np.floor(min(mu0_estimates + mu1_estimates * Time[this_year])), 
            np.floor(min(((C_mu0.T @ Beta_mu0_mean).T + (C_mu1.T @ Beta_mu1_mean).T * Time)[:,this_year])))
    vmax = max(np.ceil(max(mu0_estimates + mu1_estimates * Time[this_year])), 
            np.ceil(max(((C_mu0.T @ Beta_mu0_mean).T + (C_mu1.T @ Beta_mu1_mean).T * Time)[:,this_year])))
    divnorm = mpl.colors.TwoSlopeNorm(vcenter = (vmin+vmax)/2, vmin = vmin, vmax = vmax)

    fig, ax     = plt.subplots(1,2)
    mu0_scatter = ax[0].scatter(sites_x, sites_y, s = 10, c = mu0_estimates + mu1_estimates * Time[this_year],
                                cmap = colormaps['bwr'], norm = divnorm)
    ax[0].set_aspect('equal', 'box')
    ax[0].title.set_text('mu data year: ' + str(start_year+this_year))
    mu0_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, c = ((C_mu0.T @ Beta_mu0_mean).T + (C_mu1.T @ Beta_mu1_mean).T * Time)[:,this_year],
                                    cmap = colormaps['bwr'], norm = divnorm)
    ax[1].set_aspect('equal', 'box')
    ax[1].title.set_text('mu post mean year: ' + str(start_year+this_year))
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(mu0_est_scatter, cax = cbar_ax)
    plt.savefig('Surface:mu'+str(1949+this_year)+'.pdf')
    plt.show()
    plt.close()

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
    plt.savefig('Surface:logsigma.pdf')
    plt.show()
    plt.close()

    try:
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
        plt.savefig('Surface:ksi.pdf')
        plt.show()
        plt.close()
    except:
        pass

# %% Copula Posterior Surface Plotting
# copula parameter surface

# phi surface
phi_vec_for_plot = gaussian_weight_matrix_for_plot_phi @ phi_mean
fig, ax = plt.subplots()
fig.set_size_inches(8,6)
ax.set_aspect('equal', 'box')
state_map.boundary.plot(ax=ax, color = 'black')
heatmap = ax.imshow(phi_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x), 
                    vmin = 0, vmax = 0.5,
                    cmap ='seismic', interpolation='nearest', extent = [minX, maxX, maxY, minY])
ax.set_xticks(np.linspace(minX, maxX,num=3))
ax.set_yticks(np.linspace(minY, maxY,num=5))
ax.invert_yaxis()
cbar = fig.colorbar(heatmap, ax=ax)
cbar.ax.tick_params(labelsize=20)  # Set the fontsize here
plt.xlim([-104,-90])
plt.ylim([30,47])
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('longitude', fontsize = 20)
plt.ylabel('latitude', fontsize = 20)
plt.title(r'Posterior mean $\phi$ surface', fontsize = 20)
plt.savefig('Surface:phi.pdf')
plt.show()
plt.close()

# range surface
range_vec_for_plot = gaussian_weight_matrix_for_plot_rho @ range_mean
fig, ax = plt.subplots()
fig.set_size_inches(8,6)
ax.set_aspect('equal', 'box')
state_map.boundary.plot(ax=ax, color = 'black')
heatmap = ax.imshow(range_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x),
                    vmin = 0, vmax = 2, 
                    cmap ='seismic', interpolation='nearest', extent = [minX, maxX, maxY, minY])
ax.set_xticks(np.linspace(minX, maxX,num=3))
ax.set_yticks(np.linspace(minY, maxY,num=5))
ax.invert_yaxis()
cbar = fig.colorbar(heatmap, ax=ax)
cbar.ax.tick_params(labelsize=20)  # Set the fontsize here
plt.xlim([-104,-90])
plt.ylim([30,47])
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('longitude', fontsize = 20)
plt.ylabel('latitude', fontsize = 20)
plt.title(r'Posterior mean $\rho$ surface', fontsize = 20)
plt.savefig('Surface:range.pdf')
plt.show()
plt.close()

# %% Empirical Model estimated chi plot -----------------------------------------------------------------------------------------------
# Empirical Model estimated chi plot

"""
like with the moving window empirical chi plot before, we assume local stationarity
engineering two points inside that window, draw 100,000 bivariate gaussian, estiamte chi empirically
then make a "moving window chi plot" -- one that is estimated by the model
"""

# place knots for chi plot
res_x_chi = 7
res_y_chi = 17
k_chi     = res_x_chi * res_y_chi # number of knots
x_pos_chi = np.linspace(minX, maxX, res_x_chi+4)[2:-2]
y_pos_chi = np.linspace(minY, maxY, res_y_chi+4)[2:-2]
X_pos_chi, Y_pos_chi = np.meshgrid(x_pos_chi,y_pos_chi) # create the mesh based on these arrays
knots_xy_chi = np.vstack([X_pos_chi.ravel(), Y_pos_chi.ravel()]).T
knots_x_chi = knots_xy_chi[:,0]
knots_y_chi = knots_xy_chi[:,1]

# make a plot of the sites and knots
fig, ax = plt.subplots()
fig.set_size_inches(6,8)
ax.set_aspect('equal', 'box')
state_map.boundary.plot(ax=ax, color = 'black', linewidth = 0.5)
ax.scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.5)
ax.scatter(knots_x_chi, knots_y_chi, s = 15, color = 'blue', marker = '+')
rect_width = (knots_xy_chi[0][0] - minX)*2
rect_height = (knots_xy_chi[0][1] - minY)*2
chi_i = 118 # select a rectangle (0, 1, ..., 118) to draw
rect_i = plt.Rectangle((knots_xy_chi[chi_i][0] - rect_width/2, knots_xy_chi[chi_i][1] - rect_height/2), 
                       width = rect_width, height = rect_height,
                       fill = False, ec = 'black', linewidth = 2) # Rectangle location spsecified by lower left corner
ax.add_patch(rect_i)
plt.xlim([-105,-90])
plt.ylim([30,50])
plt.show()
plt.close()

# # function to calculate chi for a knot_chi
# def calc_chi(args):
#     point_A, u, h = args
#     point_B = random_point_at_dist(point_A, h)
#     sites_AB = np.row_stack([point_A, point_B])
#     gaussian_weight_matrix_AB = np.full(shape = (2, k), fill_value = np.nan)
#     for site_id in np.arange(2):
#         # Compute distance between each pair of the two collections of inputs
#         d_from_knots = scipy.spatial.distance.cdist(XA = sites_AB[site_id,:].reshape((-1,2)), 
#                                                     XB = knots_xy)
#         # influence coming from each of the knots
#         weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
#         gaussian_weight_matrix_AB[site_id, :] = weight_from_knots
#     wendland_weight_matrix_AB = np.full(shape = (2,k), fill_value = np.nan)
#     for site_id in np.arange(2):
#         # Compute distance between each pair of the two collections of inputs
#         d_from_knots = scipy.spatial.distance.cdist(XA = sites_AB[site_id,:].reshape((-1,2)), 
#                                                     XB = knots_xy)
#         # influence coming from each of the knots
#         weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
#         wendland_weight_matrix_AB[site_id, :] = weight_from_knots

#     # Need: R(s), phi(s), rho(s) --> K
#     phi_vec_AB     = gaussian_weight_matrix_AB @ phi_mean
#     range_vec_AB   = gaussian_weight_matrix_AB @ range_mean
#     gamma_at_knots = np.repeat(0.5, k)
#     alpha          = 0.5
#     gamma_vec_AB   = np.sum(np.multiply(wendland_weight_matrix_AB, gamma_at_knots)**(alpha),
#                             axis = 1)**(1/alpha)
#     R_matrix_AB    = wendland_weight_matrix_AB @ np.exp(R_log_mean) # shape (k, Nt)
#     sigsq_vec      = np.repeat(1.0, 2)
#     nu             = 0.5
#     K_AB           = ns_cov(range_vec = range_vec_AB,
#                             sigsq_vec = sigsq_vec,
#                             coords    = sites_AB,
#                             kappa     = nu, cov_model = "matern")
#     # cholesky_U_AB  = scipy.linalg.cholesky(K_AB, lower = False)

#     # Draw a lot of bivariate Z --> X
#     n_draw  = 10000
#     Z_bivar = np.full(shape = (n_draw, 2, Nt), fill_value = np.nan)
#     for i in range(Nt):
#         Z_bivar[:,:,i] = scipy.stats.multivariate_normal.rvs(mean = None, cov = K_AB, size = n_draw)
#     W_bivar  = norm_to_Pareto(Z_bivar)
#     X_bivar  = np.full(shape = (n_draw, 2, Nt), fill_value = np.nan)
#     for i in range(n_draw):
#         X_bivar[i,:,:] = (R_matrix_AB.T ** phi_vec_AB).T * W_bivar[i,:,:]

#     # calculate chi
#     #     Calculate F(X) is costly, just calculate threshold once and use threshold
#     u_AB = qRW(u, phi_vec_AB, gamma_vec_AB)

#     # using theoretical denominator
#     # chi = np.mean(np.logical_and(X_bivar[:,0,:] > u_AB[0], X_bivar[:,1,:] > u_AB[1])) / (1-u)
#     # using empirical denominator
#     chi = np.mean(np.logical_and(X_bivar[:,0,:] > u_AB[0], X_bivar[:,1,:] > u_AB[1])) / np.mean(X_bivar[:,1,:] > u_AB[1])
#     return chi

"""
April 19, Ben: use R drawn from Stables directly, don't use posterior mean
"""
# function to calculate chi for a knot_chi
def calc_chi(args):
    n_draw = 10000 # number of R and bivariate gaussian to draw
    point_A, u, h = args

    # engineer new point
    point_B = random_point_at_dist(point_A, h)
    sites_AB = np.row_stack([point_A, point_B])

    # new weight matrices
    # gaussian_weight_matrix_AB = np.full(shape = (2, k), fill_value = np.nan)
    # for site_id in np.arange(2):
    #     # Compute distance between each pair of the two collections of inputs
    #     d_from_knots = scipy.spatial.distance.cdist(XA = sites_AB[site_id,:].reshape((-1,2)), 
    #                                                 XB = knots_xy)
    #     # influence coming from each of the knots
    #     weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
    #     gaussian_weight_matrix_AB[site_id, :] = weight_from_knots
    gaussian_weight_matrix_AB_phi = np.full(shape = (2, k_phi), fill_value = np.nan)
    for site_id in np.arange(2):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_AB[site_id,:].reshape((-1,2)), 
                                                    XB = knots_xy_phi)
        # influence coming from each of the knots
        weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_phi, cutoff = False)
        gaussian_weight_matrix_AB_phi[site_id, :] = weight_from_knots
    gaussian_weight_matrix_AB_rho = np.full(shape = (2, k_rho), fill_value = np.nan)
    for site_id in np.arange(2):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_AB[site_id,:].reshape((-1,2)), 
                                                    XB = knots_xy_rho)
        # influence coming from each of the knots
        weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_rho, cutoff = False)
        gaussian_weight_matrix_AB_rho[site_id, :] = weight_from_knots
    wendland_weight_matrix_AB = np.full(shape = (2,k_phi), fill_value = np.nan)
    for site_id in np.arange(2):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_AB[site_id,:].reshape((-1,2)), 
                                                    XB = knots_xy_phi)
        # influence coming from each of the knots
        weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
        wendland_weight_matrix_AB[site_id, :] = weight_from_knots

    # Need: R(s), phi(s), rho(s) --> K
    phi_vec_AB     = gaussian_weight_matrix_AB_phi @ phi_mean
    range_vec_AB   = gaussian_weight_matrix_AB_rho @ range_mean
    gamma_at_knots = np.repeat(0.5, k_phi)
    alpha          = 0.5
    gamma_vec_AB   = np.sum(np.multiply(wendland_weight_matrix_AB, gamma_at_knots)**(alpha),
                            axis = 1)**(1/alpha)
    if any(np.isnan(gamma_vec_AB)):
        print(wendland_weight_matrix_AB)
    sigsq_vec      = np.repeat(1.0, 2)
    nu             = 0.5
    K_AB           = ns_cov(range_vec = range_vec_AB,
                            sigsq_vec = sigsq_vec,
                            coords    = sites_AB,
                            kappa     = nu, cov_model = "matern")
    # cholesky_U_AB  = scipy.linalg.cholesky(K_AB, lower = False)

    # Draw R and bivariate Z
    S_vec   = np.array([scipy.stats.levy.rvs(loc = 0, scale = 0.5, size = k_phi) for _ in range(n_draw)])
    Z_bivar = scipy.stats.multivariate_normal.rvs(mean = None, cov = K_AB, size = n_draw)

    # calculate X
    R_vec_AB = (wendland_weight_matrix_AB @ S_vec.T).T # shape (n_draw, 2)
    W_bivar  = norm_to_Pareto(Z_bivar)
    X_bivar  = (R_vec_AB ** phi_vec_AB) * W_bivar

    # calculate chi - Calculate F(X) is costly, just calculate threshold once and use threshold
    u_AB = qRW(u, phi_vec_AB, gamma_vec_AB)

    # using theoretical denominator
    # chi = np.mean(np.logical_and(X_bivar[:,0,:] > u_AB[0], X_bivar[:,1,:] > u_AB[1])) / (1-u)
    # using empirical denominator
    chi = np.mean(np.logical_and(X_bivar[:,0] > u_AB[0], X_bivar[:,1] > u_AB[1])) / np.mean(X_bivar[:,1] > u_AB[1])
    return chi

# Create a LinearSegmentedColormap from white to red
colors = ["#ffffff", "#ff0000"]
min_chi = 0.0
max_chi = 1.0
n_bins = 30  # Number of discrete bins
n_ticks = 10
cmap_name = "white_to_red"
colormap = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
ticks = np.linspace(min_chi, max_chi, n_ticks+1).round(3)


# Make a heatplot of chi single u h

# u = 0.95
# h = 75 # km
# np.random.seed(417)
# args_list = []
# for i in range(knots_xy_chi.shape[0]):
#     args = (knots_xy_chi[i], u, h)
#     args_list.append(args)
# with multiprocessing.Pool(processes=30) as pool:
#     results = pool.map(calc_chi, args_list)

# chi_mat2 = np.full(shape = (len(y_pos_chi), len(x_pos_chi)), fill_value = np.nan)
# for i in range(knots_xy_chi.shape[0]):
#     chi_mat2[-1 - i//len(x_pos_chi), i%len(x_pos_chi)] = results[i]

# fig, ax = plt.subplots()
# fig.set_size_inches(6,8)
# ax.set_aspect('equal', 'box')
# state_map.boundary.plot(ax=ax, color = 'black', linewidth = 0.5)
# heatmap = ax.imshow(chi_mat2, cmap =colormap, interpolation='nearest', 
#                     vmin = 0, vmax = 1,
#                     extent = [min(x_pos_chi - rect_width/8), max(x_pos_chi + rect_width/8), 
#                               min(y_pos_chi - rect_height/8), max(y_pos_chi + rect_height/8)])
# # ax.scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.8)
# ax.scatter(knots_x_chi, knots_y_chi, s = 15, color = 'white', marker = '+')
# fig.colorbar(heatmap)
# plt.xlim([-105,-90])
# plt.ylim([30,50])
# plt.title(rf'model empirical $\chi_{{{u}}}$, h $\approx$ {h}km')
# plt.savefig('model_empirical_chi_u={}_h={}.pdf'.format(u,h))
# plt.show()
# plt.close()

# Parallel compute chi across several values of u h 
np.random.seed(417)
for h in [75, 150, 225]:

    fig, axes = plt.subplots(1,3)
    fig.set_size_inches(10,6)
    for ax_id, u in enumerate([0.9, 0.95, 0.99]):
        args_list = []
        for i in range(knots_xy_chi.shape[0]):
            args = (knots_xy_chi[i], u, h)
            args_list.append(args)
        with multiprocessing.Pool(processes=30) as pool:
            results = pool.map(calc_chi, args_list)
        
        chi_mat2 = np.full(shape = (len(y_pos_chi), len(x_pos_chi)), fill_value = np.nan)
        for i in range(knots_xy_chi.shape[0]):
            chi_mat2[-1 - i//len(x_pos_chi), i%len(x_pos_chi)] = results[i]

        ax = axes[ax_id]
        ax.set_aspect('equal', 'box')
        state_map.boundary.plot(ax=ax, color = 'black', linewidth = 0.5)
        heatmap = ax.imshow(chi_mat2, cmap = colormap, interpolation = 'nearest',
                            vmin = 0.0, vmax = 1.0,
                            extent = [min(x_pos_chi - rect_width/8), max(x_pos_chi + rect_width/8), 
                                    min(y_pos_chi - rect_height/8), max(y_pos_chi + rect_height/8)])
        # ax.scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.8)
        ax.scatter(knots_x_chi, knots_y_chi, s = 15, color = 'white', marker = '+')
        ax.set_xlim(-101,-93)
        ax.set_ylim(32.5, 45)
        ax.title.set_text(rf'$\chi_{{{u}}}$, h $\approx$ {h}km')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
    fig.colorbar(heatmap, cax = cbar_ax, ticks = ticks)
    plt.savefig('Surface:model_empirical_chi_h={}.pdf'.format(h))
    plt.show()
    plt.close()

# %% Diagnostics and Model selection

#################################################
#####               Diagnostics             #####
#################################################

# %% Testing sample and splines
# Testing sample and splines

Beta_mu0_initSmooth      = np.linalg.lstsq(a=C_mu0[:,:,0].T, b=mu0_estimates,rcond=None)[0]
Beta_mu1_initSmooth      = np.linalg.lstsq(a=C_mu1[:,:,0].T, b=mu1_estimates,rcond=None)[0]
Beta_logsigma_initSmooth = np.linalg.lstsq(a=C_logsigma[:,:,0].T, b=logsigma_estimates,rcond=None)[0]
Beta_ksi_initSmooth      = np.linalg.lstsq(a=C_ksi[:,:,0].T, b=ksi_estimates,rcond=None)[0]

extRemes = importr('extRemes')
ordinal  = importr('ordinal')

r('''load('blockMax_JJA_centralUS_test.RData')''')
r('''load('stations_test.RData')''')
JJA_maxima_99  = np.array(r('blockMax_JJA_centralUS_test')).T
stations_99    = np.array(r('stations_test')).T[:,[0,1]].astype('f')
elevations_99  = np.array(r('stations_test')).T[:,3].astype('f')/200
Y_99           = JJA_maxima_99.copy()

test_sites_xy = stations_99
test_Ns       = 99
Nt            = 75

fig, ax = plt.subplots()
fig.set_size_inches(8,6)
ax.set_aspect('equal', 'box')
state_map.boundary.plot(ax=ax, color = 'black')
ax.scatter(test_sites_xy[:, 0], test_sites_xy[:, 1], color='blue')  # Scatter plot of points
for index, (x, y) in enumerate(test_sites_xy):
    ax.text(x, y, f'{index}', fontsize=12, ha='right')
plt.xlim([-102,-92])
plt.ylim([32,45])
plt.title('Scatter Plot with Labels')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(8,6)
ax.set_aspect('equal', 'box')
state_map.boundary.plot(ax=ax, color = 'black')
ax.scatter(test_sites_xy[(23, 41, 82, 94), 0], test_sites_xy[(23, 41, 82, 94), 1], color='blue')  # Scatter plot of points
labels = (23, 41, 82, 94)
for index, (x, y) in enumerate(test_sites_xy[(23, 41, 82, 94),:]):
    ax.text(x, y, f'{labels[index]}', fontsize=12, ha='right')
plt.xlim([-102,-92])
plt.ylim([32,45])
ax.set_xticks(np.linspace(-102, -92,num=3))
ax.set_yticks(np.linspace(32, 45,num=5))
# plt.title('Scatter Plot with Labels')
plt.xlabel('longitude', fontsize = 20)
plt.ylabel('latitude', fontsize = 20)
plt.savefig('out-of-sample stations.pdf',bbox_inches="tight")
plt.show()

# Create GEV Splines --------------------------------------------------------------------------------------------------

test_sites_xy_ro = numpy2rpy(test_sites_xy)    # Convert to R object
r.assign('test_sites_xy_ro', test_sites_xy_ro) # Note: this is a matrix in R, not df
r('''
    test_sites_xy_df <- as.data.frame(test_sites_xy_ro)
    colnames(test_sites_xy_df) <- c('x','y')
    ''')
C_mu0_splines      = np.array(r('''
                                basis      <- smoothCon(s(x, y, k = {Beta_mu0_splines_m}, fx = TRUE), data = gs_xy_df)[[1]]
                                basis_site <- PredictMat(basis, data = test_sites_xy_df)
                                # basis_site
                                basis_site[,c(-(ncol(basis_site)-2))] # dropped the 3rd to last column of constant
                                '''.format(Beta_mu0_splines_m = Beta_mu0_splines_m+1))) # shaped(test_Ns, Beta_mu0_splines_m)
C_mu0_1t           = np.column_stack((np.ones(test_Ns),  # intercept
                                    elevations_99,     # elevation
                                    C_mu0_splines)) # splines (excluding intercept)
C_mu0              = np.tile(C_mu0_1t.T[:,:,None], reps = (1, 1, Nt))

Beta_mu1_splines_m = 12 - 1 # drop the 3rd to last column of constant
Beta_mu1_m         = Beta_mu1_splines_m + 2 # adding intercept and elevation
C_mu1_splines      = np.array(r('''
                                basis      <- smoothCon(s(x, y, k = {Beta_mu1_splines_m}, fx = TRUE), data = gs_xy_df)[[1]]
                                basis_site <- PredictMat(basis, data = test_sites_xy_df)
                                # basis_site
                                basis_site[,c(-(ncol(basis_site)-2))] # drop the 3rd to last column of constant
                                '''.format(Beta_mu1_splines_m = Beta_mu1_splines_m+1))) # shaped(test_Ns, Beta_mu1_splines_m)
C_mu1_1t           = np.column_stack((np.ones(test_Ns),  # intercept
                                    elevations_99,     # elevation
                                    C_mu1_splines)) # splines (excluding intercept)
C_mu1              = np.tile(C_mu1_1t.T[:,:,None], reps = (1, 1, Nt))

Beta_logsigma_m   = 2 # just intercept and elevation
C_logsigma        = np.full(shape = (Beta_logsigma_m, test_Ns, Nt), fill_value = np.nan)
C_logsigma[0,:,:] = 1.0 
C_logsigma[1,:,:] = np.tile(elevations_99, reps = (Nt, 1)).T

Beta_ksi_m   = 2 # just intercept and elevation
C_ksi        = np.full(shape = (Beta_ksi_m, test_Ns, Nt), fill_value = np.nan) # ksi design matrix
C_ksi[0,:,:] = 1.0
C_ksi[1,:,:] = np.tile(elevations_99, reps = (Nt, 1)).T

# %% Gumbel QQPlot
# Gumbel QQPlot

# Gumbel QQPlot with initial smoothed MLE -----------------------------------------------------------------------------

mu0_initSmooth = (C_mu0.T @ Beta_mu0_initSmooth).T
mu1_initSmooth = (C_mu1.T @ Beta_mu1_initSmooth).T
mu_initSmooth  = mu0_initSmooth + mu1_initSmooth * Time
sigma_initSmooth = np.exp((C_logsigma.T @ Beta_logsigma_initSmooth).T)
ksi_initSmooth = (C_ksi.T @ Beta_ksi_initSmooth).T

pY_initSmooth_test = np.full(shape = (test_Ns, Nt), fill_value = np.nan)
for t in range(Nt):
    pY_initSmooth_test[:,t] = pgev(Y_99[:,t], mu_initSmooth[:,t], sigma_initSmooth[:,t], ksi_initSmooth[:,t])
pY_initSmooth_test_ro = numpy2rpy(pY_initSmooth_test)
r.assign('pY_initSmooth_test_ro', pY_initSmooth_test_ro)
r("save(pY_initSmooth_test_ro, file='pY_initSmooth_test_ro.gzip', compress=TRUE)")

gumbel_pY_initSmooth_test = np.full(shape = (test_Ns, Nt), fill_value = np.nan)
for t in range(Nt):
    gumbel_pY_initSmooth_test[:,t] = scipy.stats.gumbel_r.ppf(pY_initSmooth_test[:,t])
gumbel_pY_initSmooth_test_ro = numpy2rpy(gumbel_pY_initSmooth_test)
r.assign('gumbel_pY_initSmooth_test_ro', gumbel_pY_initSmooth_test_ro)
r("save(gumbel_pY_initSmooth_test_ro, file='gumbel_pY_initSmooth_test_ro.gzip', compress=TRUE)")

# Gumbel QQPlot with mean(each MCMC iter GEV --> Gumbel) --------------------------------------------------------------

if not fixGEV:
    # with per MCMC iterations of marginal GEV params
    n = Beta_mu0_trace_thin.shape[0]

    mu0_matrix_mcmc = (C_mu0.T @ Beta_mu0_trace_thin.T).T # shape (n, test_Ns, Nt)
    mu1_matrix_mcmc = (C_mu1.T @ Beta_mu1_trace_thin.T).T # shape (n, test_Ns, Nt)
    mu_matrix_mcmc  = mu0_matrix_mcmc + mu1_matrix_mcmc * Time
    sigma_matrix_mcmc = np.exp((C_logsigma.T @ Beta_logsigma_trace_thin.T).T)
    ksi_matrix_mcmc = (C_ksi.T @ Beta_ksi_trace_thin.T).T

    pY_mcmc_test = np.full(shape = (n, test_Ns, Nt), fill_value = np.nan)
    for i in range(n):
        for t in range(Nt):
            pY_mcmc_test[i,:,t] = pgev(Y_99[:,t], mu_matrix_mcmc[i,:,t],
                                        sigma_matrix_mcmc[i,:,t],
                                        ksi_matrix_mcmc[i,:,t])
    pY_mcmc_test_ro = numpy2rpy(pY_mcmc_test)
    r.assign('pY_mcmc_test_ro',pY_mcmc_test_ro)
    r("save(pY_mcmc_test_ro, file='pY_mcmc_test_ro.gzip', compress=TRUE)")

    gumbel_pY_mcmc_test = np.full(shape = (n, test_Ns, Nt), fill_value = np.nan)
    for i in range(n):
        for t in range(Nt):
            gumbel_pY_mcmc_test[i,:,t] = scipy.stats.gumbel_r.ppf(pY_mcmc_test[i,:,t])
    gumbel_pY_mcmc_test_ro = numpy2rpy(gumbel_pY_mcmc_test)
    r.assign('gumbel_pY_mcmc_test_ro',gumbel_pY_mcmc_test_ro)
    r("save(gumbel_pY_mcmc_test_ro, file='gumbel_pY_mcmc_test_ro.gzip', compress=TRUE)")

# Drawing the QQ Plots ------------------------------------------------------------------------------------------------

for _ in range(10):
    # with MLE initial smooth
    r('''
        test_Ns <- 99
        s <- floor(runif(1, min = 1, max = test_Ns + 1))
        print(test_sites_xy_ro[s,]) # print coordinates
        gumbel_s = sort(gumbel_pY_initSmooth_test_ro[s,])
        nquants = length(gumbel_s)
        emp_p = seq(1/nquants, 1-1/nquants, length=nquants)
        emp_q = qgumbel(emp_p)
        qq_gumbel_s <- extRemes::qqplot(gumbel_s, emp_q, regress=FALSE, legend=NULL,
                                        xlab="Observed", ylab="Gumbel", main=paste("GEVfit-QQPlot of Site:",s),
                                        lwd=3)
        pdf(file=paste("QQPlot_R_Test_initSmooth_Site_",s,".pdf", sep=""), width = 6, height = 5)
        par(mgp=c(1.5,0.5,0), mar=c(3,3,1,1))
        plot(type="n",qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$y, pch = 20, xlab="Observed", ylab="Gumbel")
        points(qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$y, pch=20)
        lines(qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$lower, lty=2, col="blue", lwd=3)
        lines(qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$upper, lty=2, col="blue", lwd=3)
        abline(a=0, b=1, lty=3, col="gray80", lwd=3)
        legend("topleft", lty=c(2, 3), lwd=3, legend=c("95% confidence bands", "1:1 line"), col=c("blue", "gray80"), bty="n")
        dev.off()
    ''')
    # with per MCMC iteration transformed
    if not fixGEV:
        r('''
            # s <- floor(runif(1, min = 1, max = test_Ns+1))
            print(test_sites_xy_ro[s,]) # print coordinates
            gumbel_s_mcmc = sort(apply(gumbel_pY_mcmc_test_ro[,s,],2, mean))
            nquants = length(gumbel_s_mcmc)
            emp_p = seq(1/nquants, 1-1/nquants, length=nquants)
            emp_q = qgumbel(emp_p)
            qq_gumbel_s_mcmc <- extRemes::qqplot(gumbel_s_mcmc, emp_q, regress=FALSE, legend=NULL,
                                            xlab="Observed", ylab="Gumbel", main=paste("Modelfit-QQPlot of Site:",s),
                                            lwd=3)
            pdf(file=paste("QQPlot_R_Test_MCMC_Site_",s,".pdf", sep=""), width = 6, height = 5)
            par(mgp=c(1.5,0.5,0), mar=c(3,3,1,1))
            plot(type="n",qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$y, pch = 20, xlab="Observed", ylab="Gumbel")
            points(qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$y, pch=20)
            lines(qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$lower, lty=2, col="blue", lwd=3)
            lines(qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$upper, lty=2, col="blue", lwd=3)
            abline(a=0, b=1, lty=3, col="gray80", lwd=3)
            legend("topleft", lty=c(2, 3), lwd=3, legend=c("95% confidence bands", "1:1 line"), col=c("blue", "gray80"), bty="n")
            dev.off()
            ''')

# %% loglikelihood at Testing sample ----------------------------------------------------------------------------------
# loglikelihood at Testing sample


# Calculate step 1-4 with a single core and no need to parallelize
# Calculate step 5 parallel, using multiprocessing (so one node)
# We need:
#     - 1 Extract the non NA Y at the testing sites
#     - 2 Marginal Parameters. Get the Loc, Scale, Shape at the testing sites:
#             This can be done by directly applying the posterior mean estimates, because
#             the thin-plate basis expansion is linear
#     - 3 Copula Parameters, Get the phi_vec, range_vec, gamma_vec, R_vec at the testing sites:
#             Using posterior mean, because basis expansion is linear
#     - 4 Using the range_vec, calculate the cholesky_U at the testing sites
#     - 5 X should be calculated from the Loc, Scale, and Shape at the testing sites
#         Note that in 5 the marginal transformation isn't linear, so take a sample of the Loc, Scale, Shape
#         and then we have a sample of the X
#         and then we make a sample of the ll using those X
# Finally, 
#     - 6 apply the marg_transform_data over each time replicate and sum
# If parallelize, can set the contribution of np.nan Y to be 0

"""
April 19, Ben: ll calculation is also non-linear, so really, should take per iteration parameters
    - 1 Extract the non NA Y at the testing sites
    - 2 Marginal Parameters: 
            initSmooth if fixGEV
            per iteration if not fixGEV
    - 3 Copula Parameters per iteration
    - 4 Calculate X_star using the per iteration transformation -- parallelize this
    - 5 Likelihood using the per iteration parameter values -- parallelize this
"""

# marg_transform_data_mixture_likelihood_1t(Y, X, Loc, Scale, Shape, phi_vec, gamma_vec, R_vec, cholesky_U)

# 1. Y_99 -------------------------------------------------------------------------------------------------------------

Y_99_noNA = Y_99[~np.isnan(Y_99)]

# 2. Marginal Parameters ----------------------------------------------------------------------------------------------

mu0_initSmooth = (C_mu0.T @ Beta_mu0_initSmooth).T
mu1_initSmooth = (C_mu1.T @ Beta_mu1_initSmooth).T
mu_initSmooth  = mu0_initSmooth + mu1_initSmooth * Time
sigma_initSmooth = np.exp((C_logsigma.T @ Beta_logsigma_initSmooth).T)
ksi_initSmooth = (C_ksi.T @ Beta_ksi_initSmooth).T

if not fixGEV: # these are the per iteration marginal parameters
    Beta_mu0_trace_thin100      = Beta_mu0_trace[0:iter:100,:]
    Beta_mu1_trace_thin100      = Beta_mu1_trace[0:iter:100,:]
    Beta_logsigma_trace_thin100 = Beta_logsigma_trace[0:iter:100,:]
    Beta_ksi_trace_thin100      = Beta_ksi_trace[0:iter:100,:]

    mu0_matrix_thin100   = (C_mu0.T @ Beta_mu0_trace_thin100.T).T # shape (n, test_Ns, Nt)
    mu1_matrix_thin100   = (C_mu1.T @ Beta_mu1_trace_thin100.T).T # shape (n, test_Ns, Nt)
    mu_matrix_thin100    = mu0_matrix_thin100 + mu1_matrix_thin100 * Time
    sigma_matrix_thin100 = np.exp((C_logsigma.T @ Beta_logsigma_trace_thin100.T).T)
    ksi_matrix_thin100   = (C_ksi.T @ Beta_ksi_trace_thin100.T).T

# if not fixGEV: # these are the posterior mean estimates, which we shouldn't use
#     mu0_matrix_test = (C_mu0.T @ Beta_mu0_mean.T).T
#     mu1_matrix_test = (C_mu1.T @ Beta_mu1_mean.T).T
#     mu_matrix_test  = mu0_matrix_test + mu1_matrix_test * Time
#     sigma_matrix_test = np.exp((C_logsigma.T @ Beta_logsigma_mean.T).T)
#     ksi_matrix_test = (C_ksi.T @ Beta_ksi_mean.T).T

# 3. Copula Parameters - should also be per iterations ----------------------------------------------------------------

# weight matrices at the testing sites
# gaussian_weight_matrix_test = np.full(shape = (test_Ns, k), fill_value = np.nan)
# for site_id in np.arange(test_Ns):
#     # Compute distance between each pair of the two collections of inputs
#     d_from_knots = scipy.spatial.distance.cdist(XA = test_sites_xy[site_id,:].reshape((-1,2)), 
#                                     XB = knots_xy)
#     # influence coming from each of the knots
#     weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
#     gaussian_weight_matrix_test[site_id, :] = weight_from_knots
gaussian_weight_matrix_test_phi = np.full(shape = (test_Ns, k_phi), fill_value = np.nan)
for site_id in np.arange(test_Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = test_sites_xy[site_id,:].reshape((-1,2)), 
                                                XB = knots_xy_phi)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_phi, cutoff = False)
    gaussian_weight_matrix_test_phi[site_id, :] = weight_from_knots
gaussian_weight_matrix_test_rho = np.full(shape = (test_Ns, k_rho), fill_value = np.nan)
for site_id in np.arange(test_Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = test_sites_xy[site_id,:].reshape((-1,2)), 
                                                XB = knots_xy_rho)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_rho, cutoff = False)
    gaussian_weight_matrix_test_rho[site_id, :] = weight_from_knots
wendland_weight_matrix_test = np.full(shape = (test_Ns,k_phi), fill_value = np.nan)
for site_id in np.arange(test_Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = test_sites_xy[site_id,:].reshape((-1,2)), 
                                                XB = knots_xy_phi)
    # influence coming from each of the knots
    weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
    wendland_weight_matrix_test[site_id, :] = weight_from_knots

# constants
gamma_at_knots = np.repeat(0.5, k_phi)
alpha          = 0.5
gamma_vec_test = np.sum(np.multiply(wendland_weight_matrix_test, gamma_at_knots)**(alpha),
                        axis = 1)**(1/alpha)
sigsq_vec      = np.repeat(1.0, test_Ns)
nu             = 0.5

# making the per iterations parameters (chains thinned by 100) at testing sites
#   - mu, sigma, ksi, and 
#   - phi, range, R
n_iter      = phi_knots_trace.shape[0]
idx_thin100 = np.arange(n_iter)[0::100] # thin by 100
n_thin100   = len(idx_thin100)
idx_thin100 = np.arange(n_thin100)

phi_knots_trace_thin100 = phi_knots_trace[0:n_iter:100,:]
phi_vec_test_thin100    = (gaussian_weight_matrix_test_phi @ phi_knots_trace_thin100.T).T

range_knots_trace_thin100 = range_knots_trace[0:n_iter:100,:]
range_vec_test_thin100    = (gaussian_weight_matrix_test_rho @ range_knots_trace_thin100.T).T

R_trace_log_thin100 = R_trace_log[0:n_iter:100,:,:]
R_vec_test_thin100 = np.full(shape = (n_thin100, test_Ns, Nt), fill_value = np.nan)
for t in range(Nt):
    R_vec_test_thin100[:,:,t]  = (wendland_weight_matrix_test @ np.exp(R_trace_log_thin100[:,:,t]).T).T

# # Posterior mean of these parameters at the testing sites
# phi_vec_test   = gaussian_weight_matrix_test @ phi_mean
# range_vec_test = gaussian_weight_matrix_test @ range_mean
# gamma_at_knots = np.repeat(0.5, k)
# alpha          = 0.5
# gamma_vec_test = np.sum(np.multiply(wendland_weight_matrix_test, gamma_at_knots)**(alpha),
#                         axis = 1)**(1/alpha)
# R_matrix_test     = wendland_weight_matrix_test @ np.exp(R_log_mean) # shape (k, Nt)

# # 4. K or Cholesky_U
# sigsq_vec = np.repeat(1.0, test_Ns)
# nu        = 0.5
# K_test    = ns_cov(range_vec = range_vec_test,
#                    sigsq_vec = sigsq_vec,
#                    coords    = test_sites_xy,
#                    kappa     = nu, cov_model = "matern")
# cholesky_U_test = scipy.linalg.cholesky(K_test, lower = False)

# 4. Calculate X per iteration -- could really use parallelization here... --------------------------------------------
print('link function g:', norm_pareto)

def qRW_pgev(args):
    Y     = args[:,0]
    Loc   = args[:,1]
    Scale = args[:,2]
    Shape = args[:,3]
    Phi   = args[:,4]
    Gamma = args[:,5]
    # Y, Loc, Scale, Shape, Phi, Gamma = args.T # args shaped (noNA, 6)
    return qRW(pgev(Y, Loc, Scale, Shape), Phi, Gamma)

X_99_thin100 = np.full(shape = (n_thin100, test_Ns, Nt), fill_value = np.nan)
for i in range(n_thin100):
    print('qRW_pgev:', i)
    args_list = []
    for t in range(Nt): # parallel compute the times (distribute the Nt times to n_processes)
        noNA = ~np.isnan(Y_99[:,t])
        if not fixGEV:
            args = np.column_stack((Y_99[noNA, t], 
                                    mu_matrix_thin100[i, noNA, t], 
                                    sigma_matrix_thin100[i,noNA,t],
                                    ksi_matrix_thin100[i,noNA,t],
                                    phi_vec_test_thin100[i,noNA],
                                    gamma_vec_test[noNA]))
        if fixGEV:
            args = np.column_stack((Y_99[noNA, t],
                                    mu_initSmooth[noNA, t],
                                    sigma_initSmooth[noNA, t],
                                    ksi_initSmooth[noNA, t],
                                    phi_vec_test_thin100[i, noNA],
                                    gamma_vec_test[noNA]))
        args_list.append(args)
    with multiprocessing.Pool(processes=30) as pool:
        results = pool.map(qRW_pgev, args_list)
    for t in range(Nt):
        noNA = ~np.isnan(Y_99[:,t])
        X_99_thin100[i,noNA,t] = results[t]

# 5. loglikelihood -- calculation can also be parallelized! -----------------------------------------------------------

def ll(args):
    Y, X, Loc, Scale, Shape, phi, gamma, R, K_chol = args
    return(marg_transform_data_mixture_likelihood_1t(Y, X, Loc, Scale, Shape, phi, gamma, R, K_chol))

# Using per iteration for the parameters
ll_test_thin100 = np.full(shape= (n_thin100, test_Ns, 75), fill_value = 0.0)
for i in range(n_thin100):
    print('ll:', i)
    args_list = []
    for t in range(Nt): # parallel compute the times (distribute the Nt times to n_processes)
        noNA       = ~np.isnan(Y_99[:,t])
        K_test     = ns_cov(range_vec = range_vec_test_thin100[i,:],
                            sigsq_vec = sigsq_vec,
                            coords    = test_sites_xy,
                            kappa     = nu, cov_model = "matern")
        K_subset   = K_test[noNA,:][:,noNA]
        cholesky_U = scipy.linalg.cholesky(K_subset, lower = False)
        if not fixGEV:
            args = (Y_99[noNA, t], X_99_thin100[i, noNA, t],
                    mu_matrix_thin100[i, noNA, t], sigma_matrix_thin100[i,noNA,t], ksi_matrix_thin100[i,noNA,t],
                    phi_vec_test_thin100[i,noNA], gamma_vec_test[noNA], R_vec_test_thin100[i,noNA,t], cholesky_U)
        if fixGEV:
            args = (Y_99[noNA,t], X_99_thin100[i,noNA,t],
                    mu_initSmooth[noNA,t], sigma_initSmooth[noNA,t], ksi_initSmooth[noNA,t],
                    phi_vec_test_thin100[i,noNA], gamma_vec_test[noNA], R_vec_test_thin100[i,noNA,t], cholesky_U)
        args_list.append(args)
    with multiprocessing.Pool(processes=30) as pool:
        results = pool.map(ll, args_list)
    for t in range(Nt):
        noNA = ~np.isnan(Y_99[:,t])
        ll_test_thin100[i,noNA,t] = results[t]

# # Using the posterior mean for the parameters
# ll_test_thin100 = np.full(shape= (n_thin100, test_Ns, 75), fill_value = 0.0)
# for i in range(n_thin100):
#     print('ll:', i)
#     args_list = []
#     for t in range(Nt): # parallel compute the times (distribute the Nt times to n_processes)
#         noNA       = ~np.isnan(Y_99[:,t])
#         K_subset   = K_test[noNA,:][:,noNA]
#         cholesky_U = scipy.linalg.cholesky(K_subset, lower = False)
#         if not fixGEV:
#             args = (Y_99[noNA, t], X_99_thin100[i, noNA, t],
#                     mu_matrix_test[noNA,t], sigma_matrix_test[noNA,t], ksi_matrix_test[noNA,t],
#                     phi_vec_test[noNA], gamma_vec_test[noNA], R_matrix_test[noNA,t], cholesky_U)
#         if fixGEV:
#             args = (Y_99[noNA,t], X_99_thin100[i,noNA,t],
#                     mu_initSmooth[noNA,t], sigma_initSmooth[noNA,t], ksi_initSmooth[noNA,t],
#                     phi_vec_test[noNA], gamma_vec_test[noNA], R_matrix_test[noNA,t], cholesky_U)
#         args_list.append(args)
#     with multiprocessing.Pool(processes=30) as pool:
#         results = pool.map(ll, args_list)
#     for t in range(Nt):
#         noNA = ~np.isnan(Y_99[:,t])
#         ll_test_thin100[i,noNA,t] = results[t]

np.save('ll_'+name, ll_test_thin100)

plt.boxplot(np.sum(ll_test_thin100, axis = (1,2)))
plt.xticks([1], [name])
plt.xlabel('Knot Radius Configuration')
plt.ylabel('log-likelihood @ test sites')
plt.savefig('ll_'+name+'_boxplot.pdf')
plt.show()
plt.close()

# %%

######################################
#### fixksi QQPlot and Likelihood ####
######################################

# %% Gumbel QQPlot fix only ksi
# Gumbel QQPlot fix only ksi

# Gumbel QQPlot with initial smoothed MLE -----------------------------------------------------------------------------

mu0_initSmooth = (C_mu0.T @ Beta_mu0_initSmooth).T
mu1_initSmooth = (C_mu1.T @ Beta_mu1_initSmooth).T
mu_initSmooth  = mu0_initSmooth + mu1_initSmooth * Time
sigma_initSmooth = np.exp((C_logsigma.T @ Beta_logsigma_initSmooth).T)
ksi_initSmooth = (C_ksi.T @ Beta_ksi_initSmooth).T

pY_initSmooth_test = np.full(shape = (test_Ns, Nt), fill_value = np.nan)
for t in range(Nt):
    pY_initSmooth_test[:,t] = pgev(Y_99[:,t], mu_initSmooth[:,t], sigma_initSmooth[:,t], ksi_initSmooth[:,t])
pY_initSmooth_test_ro = numpy2rpy(pY_initSmooth_test)
r.assign('pY_initSmooth_test_ro', pY_initSmooth_test_ro)
r("save(pY_initSmooth_test_ro, file='pY_initSmooth_test_ro.gzip', compress=TRUE)")

gumbel_pY_initSmooth_test = np.full(shape = (test_Ns, Nt), fill_value = np.nan)
for t in range(Nt):
    gumbel_pY_initSmooth_test[:,t] = scipy.stats.gumbel_r.ppf(pY_initSmooth_test[:,t])
gumbel_pY_initSmooth_test_ro = numpy2rpy(gumbel_pY_initSmooth_test)
r.assign('gumbel_pY_initSmooth_test_ro', gumbel_pY_initSmooth_test_ro)
r("save(gumbel_pY_initSmooth_test_ro, file='gumbel_pY_initSmooth_test_ro.gzip', compress=TRUE)")

# Gumbel QQPlot with mean(each MCMC iter GEV --> Gumbel) --------------------------------------------------------------

if not fixGEV:
    # with per MCMC iterations of marginal GEV params
    n = Beta_mu0_trace_thin.shape[0]

    mu0_matrix_mcmc = (C_mu0.T @ Beta_mu0_trace_thin.T).T # shape (n, test_Ns, Nt)
    mu1_matrix_mcmc = (C_mu1.T @ Beta_mu1_trace_thin.T).T # shape (n, test_Ns, Nt)
    mu_matrix_mcmc  = mu0_matrix_mcmc + mu1_matrix_mcmc * Time
    sigma_matrix_mcmc = np.exp((C_logsigma.T @ Beta_logsigma_trace_thin.T).T)
    ksi_matrix_mcmc = np.tile(ksi_initSmooth, reps = (n, 1, 1))

    pY_mcmc_test = np.full(shape = (n, test_Ns, Nt), fill_value = np.nan)
    for i in range(n):
        for t in range(Nt):
            pY_mcmc_test[i,:,t] = pgev(Y_99[:,t], mu_matrix_mcmc[i,:,t],
                                        sigma_matrix_mcmc[i,:,t],
                                        ksi_matrix_mcmc[i,:,t])
    pY_mcmc_test_ro = numpy2rpy(pY_mcmc_test)
    r.assign('pY_mcmc_test_ro',pY_mcmc_test_ro)
    r("save(pY_mcmc_test_ro, file='pY_mcmc_test_ro.gzip', compress=TRUE)")

    gumbel_pY_mcmc_test = np.full(shape = (n, test_Ns, Nt), fill_value = np.nan)
    for i in range(n):
        for t in range(Nt):
            gumbel_pY_mcmc_test[i,:,t] = scipy.stats.gumbel_r.ppf(pY_mcmc_test[i,:,t])
    gumbel_pY_mcmc_test_ro = numpy2rpy(gumbel_pY_mcmc_test)
    r.assign('gumbel_pY_mcmc_test_ro',gumbel_pY_mcmc_test_ro)
    r("save(gumbel_pY_mcmc_test_ro, file='gumbel_pY_mcmc_test_ro.gzip', compress=TRUE)")

# Drawing the QQ Plots ------------------------------------------------------------------------------------------------

for _ in range(10):
    # with MLE initial smooth
    r('''
        test_Ns <- 99
        s <- floor(runif(1, min = 1, max = test_Ns + 1))
        print(test_sites_xy_ro[s,]) # print coordinates
        gumbel_s = sort(gumbel_pY_initSmooth_test_ro[s,])
        nquants = length(gumbel_s)
        emp_p = seq(1/nquants, 1-1/nquants, length=nquants)
        emp_q = qgumbel(emp_p)
        qq_gumbel_s <- extRemes::qqplot(gumbel_s, emp_q, regress=FALSE, legend=NULL,
                                        xlab="Observed", ylab="Gumbel", main=paste("GEVfit-QQPlot of Site:",s),
                                        lwd=3)
        pdf(file=paste("QQPlot_R_Test_initSmooth_Site_",s,".pdf", sep=""), width = 6, height = 5)
        par(mgp=c(1.5,0.5,0), mar=c(3,3,1,1))
        plot(type="n",qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$y, pch = 20, xlab="Observed", ylab="Gumbel")
        points(qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$y, pch=20)
        lines(qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$lower, lty=2, col="blue", lwd=3)
        lines(qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$upper, lty=2, col="blue", lwd=3)
        abline(a=0, b=1, lty=3, col="gray80", lwd=3)
        legend("topleft", lty=c(2, 3), lwd=3, legend=c("95% confidence bands", "1:1 line"), col=c("blue", "gray80"), bty="n")
        dev.off()
    ''')
    # with per MCMC iteration transformed
    if not fixGEV:
        r('''
            # s <- floor(runif(1, min = 1, max = test_Ns+1))
            print(test_sites_xy_ro[s,]) # print coordinates
            gumbel_s_mcmc = sort(apply(gumbel_pY_mcmc_test_ro[,s,],2, mean))
            nquants = length(gumbel_s_mcmc)
            emp_p = seq(1/nquants, 1-1/nquants, length=nquants)
            emp_q = qgumbel(emp_p)
            qq_gumbel_s_mcmc <- extRemes::qqplot(gumbel_s_mcmc, emp_q, regress=FALSE, legend=NULL,
                                            xlab="Observed", ylab="Gumbel", main=paste("Modelfit-QQPlot of Site:",s),
                                            lwd=3)
            pdf(file=paste("QQPlot_R_Test_MCMC_Site_",s,".pdf", sep=""), width = 6, height = 5)
            par(mgp=c(1.5,0.5,0), mar=c(3,3,1,1))
            plot(type="n",qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$y, pch = 20, xlab="Observed", ylab="Gumbel")
            points(qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$y, pch=20)
            lines(qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$lower, lty=2, col="blue", lwd=3)
            lines(qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$upper, lty=2, col="blue", lwd=3)
            abline(a=0, b=1, lty=3, col="gray80", lwd=3)
            legend("topleft", lty=c(2, 3), lwd=3, legend=c("95% confidence bands", "1:1 line"), col=c("blue", "gray80"), bty="n")
            dev.off()
            ''')

# %% loglikelihood at Testing sample fix only ksi ----------------------------------------------------------------------------------
# loglikelihood at Testing sample fix only ksi

# 1. Y_99 -------------------------------------------------------------------------------------------------------------

Y_99_noNA = Y_99[~np.isnan(Y_99)]

# 2. Marginal Parameters ----------------------------------------------------------------------------------------------

mu0_initSmooth = (C_mu0.T @ Beta_mu0_initSmooth).T
mu1_initSmooth = (C_mu1.T @ Beta_mu1_initSmooth).T
mu_initSmooth  = mu0_initSmooth + mu1_initSmooth * Time
sigma_initSmooth = np.exp((C_logsigma.T @ Beta_logsigma_initSmooth).T)
ksi_initSmooth = (C_ksi.T @ Beta_ksi_initSmooth).T

if not fixGEV: # these are the per iteration marginal parameters
    Beta_mu0_trace_thin100      = Beta_mu0_trace[0:iter:100,:]
    Beta_mu1_trace_thin100      = Beta_mu1_trace[0:iter:100,:]
    Beta_logsigma_trace_thin100 = Beta_logsigma_trace[0:iter:100,:]
    # Beta_ksi_trace_thin100      = Beta_ksi_trace[0:iter:100,:]

    mu0_matrix_thin100   = (C_mu0.T @ Beta_mu0_trace_thin100.T).T # shape (n, test_Ns, Nt)
    mu1_matrix_thin100   = (C_mu1.T @ Beta_mu1_trace_thin100.T).T # shape (n, test_Ns, Nt)
    mu_matrix_thin100    = mu0_matrix_thin100 + mu1_matrix_thin100 * Time
    sigma_matrix_thin100 = np.exp((C_logsigma.T @ Beta_logsigma_trace_thin100.T).T)
    ksi_matrix_thin100   = np.tile(ksi_initSmooth, reps = (sigma_matrix_thin100.shape[0], 1, 1))

# if not fixGEV: # these are the posterior mean estimates, which we shouldn't use
#     mu0_matrix_test = (C_mu0.T @ Beta_mu0_mean.T).T
#     mu1_matrix_test = (C_mu1.T @ Beta_mu1_mean.T).T
#     mu_matrix_test  = mu0_matrix_test + mu1_matrix_test * Time
#     sigma_matrix_test = np.exp((C_logsigma.T @ Beta_logsigma_mean.T).T)
#     ksi_matrix_test = (C_ksi.T @ Beta_ksi_mean.T).T

# 3. Copula Parameters - should also be per iterations ----------------------------------------------------------------

# weight matrices at the testing sites
# gaussian_weight_matrix_test = np.full(shape = (test_Ns, k), fill_value = np.nan)
# for site_id in np.arange(test_Ns):
#     # Compute distance between each pair of the two collections of inputs
#     d_from_knots = scipy.spatial.distance.cdist(XA = test_sites_xy[site_id,:].reshape((-1,2)), 
#                                     XB = knots_xy)
#     # influence coming from each of the knots
#     weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
#     gaussian_weight_matrix_test[site_id, :] = weight_from_knots
gaussian_weight_matrix_test_phi = np.full(shape = (test_Ns, k_phi), fill_value = np.nan)
for site_id in np.arange(test_Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = test_sites_xy[site_id,:].reshape((-1,2)), 
                                                XB = knots_xy_phi)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_phi, cutoff = False)
    gaussian_weight_matrix_test_phi[site_id, :] = weight_from_knots
gaussian_weight_matrix_test_rho = np.full(shape = (test_Ns, k_rho), fill_value = np.nan)
for site_id in np.arange(test_Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = test_sites_xy[site_id,:].reshape((-1,2)), 
                                                XB = knots_xy_rho)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_rho, cutoff = False)
    gaussian_weight_matrix_test_rho[site_id, :] = weight_from_knots
wendland_weight_matrix_test = np.full(shape = (test_Ns,k_phi), fill_value = np.nan)
for site_id in np.arange(test_Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = test_sites_xy[site_id,:].reshape((-1,2)), 
                                                XB = knots_xy_phi)
    # influence coming from each of the knots
    weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
    wendland_weight_matrix_test[site_id, :] = weight_from_knots

# constants
gamma_at_knots = np.repeat(0.5, k_phi)
alpha          = 0.5
gamma_vec_test = np.sum(np.multiply(wendland_weight_matrix_test, gamma_at_knots)**(alpha),
                        axis = 1)**(1/alpha)
sigsq_vec      = np.repeat(1.0, test_Ns)
nu             = 0.5

# making the per iterations parameters (chains thinned by 100) at testing sites
#   - mu, sigma, ksi, and 
#   - phi, range, R
n_iter      = phi_knots_trace.shape[0]
idx_thin100 = np.arange(n_iter)[0::100] # thin by 100
n_thin100   = len(idx_thin100)
idx_thin100 = np.arange(n_thin100)

phi_knots_trace_thin100 = phi_knots_trace[0:n_iter:100,:]
phi_vec_test_thin100    = (gaussian_weight_matrix_test_phi @ phi_knots_trace_thin100.T).T

range_knots_trace_thin100 = range_knots_trace[0:n_iter:100,:]
range_vec_test_thin100    = (gaussian_weight_matrix_test_rho @ range_knots_trace_thin100.T).T

R_trace_log_thin100 = R_trace_log[0:n_iter:100,:,:]
R_vec_test_thin100 = np.full(shape = (n_thin100, test_Ns, Nt), fill_value = np.nan)
for t in range(Nt):
    R_vec_test_thin100[:,:,t]  = (wendland_weight_matrix_test @ np.exp(R_trace_log_thin100[:,:,t]).T).T

# # Posterior mean of these parameters at the testing sites
# phi_vec_test   = gaussian_weight_matrix_test @ phi_mean
# range_vec_test = gaussian_weight_matrix_test @ range_mean
# gamma_at_knots = np.repeat(0.5, k)
# alpha          = 0.5
# gamma_vec_test = np.sum(np.multiply(wendland_weight_matrix_test, gamma_at_knots)**(alpha),
#                         axis = 1)**(1/alpha)
# R_matrix_test     = wendland_weight_matrix_test @ np.exp(R_log_mean) # shape (k, Nt)

# # 4. K or Cholesky_U
# sigsq_vec = np.repeat(1.0, test_Ns)
# nu        = 0.5
# K_test    = ns_cov(range_vec = range_vec_test,
#                    sigsq_vec = sigsq_vec,
#                    coords    = test_sites_xy,
#                    kappa     = nu, cov_model = "matern")
# cholesky_U_test = scipy.linalg.cholesky(K_test, lower = False)

# 4. Calculate X per iteration -- could really use parallelization here... --------------------------------------------
print('link function g:', norm_pareto)

def qRW_pgev(args):
    Y     = args[:,0]
    Loc   = args[:,1]
    Scale = args[:,2]
    Shape = args[:,3]
    Phi   = args[:,4]
    Gamma = args[:,5]
    # Y, Loc, Scale, Shape, Phi, Gamma = args.T # args shaped (noNA, 6)
    return qRW(pgev(Y, Loc, Scale, Shape), Phi, Gamma)

X_99_thin100 = np.full(shape = (n_thin100, test_Ns, Nt), fill_value = np.nan)
for i in range(n_thin100):
    print('qRW_pgev:', i)
    args_list = []
    for t in range(Nt): # parallel compute the times (distribute the Nt times to n_processes)
        noNA = ~np.isnan(Y_99[:,t])
        if not fixGEV:
            args = np.column_stack((Y_99[noNA, t], 
                                    mu_matrix_thin100[i, noNA, t], 
                                    sigma_matrix_thin100[i,noNA,t],
                                    ksi_matrix_thin100[i,noNA,t],
                                    phi_vec_test_thin100[i,noNA],
                                    gamma_vec_test[noNA]))
        if fixGEV:
            args = np.column_stack((Y_99[noNA, t],
                                    mu_initSmooth[noNA, t],
                                    sigma_initSmooth[noNA, t],
                                    ksi_initSmooth[noNA, t],
                                    phi_vec_test_thin100[i, noNA],
                                    gamma_vec_test[noNA]))
        args_list.append(args)
    with multiprocessing.Pool(processes=30) as pool:
        results = pool.map(qRW_pgev, args_list)
    for t in range(Nt):
        noNA = ~np.isnan(Y_99[:,t])
        X_99_thin100[i,noNA,t] = results[t]

# 5. loglikelihood -- calculation can also be parallelized! -----------------------------------------------------------

def ll(args):
    Y, X, Loc, Scale, Shape, phi, gamma, R, K_chol = args
    return(marg_transform_data_mixture_likelihood_1t(Y, X, Loc, Scale, Shape, phi, gamma, R, K_chol))

# Using per iteration for the parameters
ll_test_thin100 = np.full(shape= (n_thin100, test_Ns, 75), fill_value = 0.0)
for i in range(n_thin100):
    print('ll:', i)
    args_list = []
    for t in range(Nt): # parallel compute the times (distribute the Nt times to n_processes)
        noNA       = ~np.isnan(Y_99[:,t])
        K_test     = ns_cov(range_vec = range_vec_test_thin100[i,:],
                            sigsq_vec = sigsq_vec,
                            coords    = test_sites_xy,
                            kappa     = nu, cov_model = "matern")
        K_subset   = K_test[noNA,:][:,noNA]
        cholesky_U = scipy.linalg.cholesky(K_subset, lower = False)
        if not fixGEV:
            args = (Y_99[noNA, t], X_99_thin100[i, noNA, t],
                    mu_matrix_thin100[i, noNA, t], sigma_matrix_thin100[i,noNA,t], ksi_matrix_thin100[i,noNA,t],
                    phi_vec_test_thin100[i,noNA], gamma_vec_test[noNA], R_vec_test_thin100[i,noNA,t], cholesky_U)
        if fixGEV:
            args = (Y_99[noNA,t], X_99_thin100[i,noNA,t],
                    mu_initSmooth[noNA,t], sigma_initSmooth[noNA,t], ksi_initSmooth[noNA,t],
                    phi_vec_test_thin100[i,noNA], gamma_vec_test[noNA], R_vec_test_thin100[i,noNA,t], cholesky_U)
        args_list.append(args)
    with multiprocessing.Pool(processes=30) as pool:
        results = pool.map(ll, args_list)
    for t in range(Nt):
        noNA = ~np.isnan(Y_99[:,t])
        ll_test_thin100[i,noNA,t] = results[t]

# # Using the posterior mean for the parameters
# ll_test_thin100 = np.full(shape= (n_thin100, test_Ns, 75), fill_value = 0.0)
# for i in range(n_thin100):
#     print('ll:', i)
#     args_list = []
#     for t in range(Nt): # parallel compute the times (distribute the Nt times to n_processes)
#         noNA       = ~np.isnan(Y_99[:,t])
#         K_subset   = K_test[noNA,:][:,noNA]
#         cholesky_U = scipy.linalg.cholesky(K_subset, lower = False)
#         if not fixGEV:
#             args = (Y_99[noNA, t], X_99_thin100[i, noNA, t],
#                     mu_matrix_test[noNA,t], sigma_matrix_test[noNA,t], ksi_matrix_test[noNA,t],
#                     phi_vec_test[noNA], gamma_vec_test[noNA], R_matrix_test[noNA,t], cholesky_U)
#         if fixGEV:
#             args = (Y_99[noNA,t], X_99_thin100[i,noNA,t],
#                     mu_initSmooth[noNA,t], sigma_initSmooth[noNA,t], ksi_initSmooth[noNA,t],
#                     phi_vec_test[noNA], gamma_vec_test[noNA], R_matrix_test[noNA,t], cholesky_U)
#         args_list.append(args)
#     with multiprocessing.Pool(processes=30) as pool:
#         results = pool.map(ll, args_list)
#     for t in range(Nt):
#         noNA = ~np.isnan(Y_99[:,t])
#         ll_test_thin100[i,noNA,t] = results[t]

np.save('ll_'+name, ll_test_thin100)

plt.boxplot(np.sum(ll_test_thin100, axis = (1,2)))
plt.xticks([1], [name])
plt.xlabel('Knot Radius Configuration')
plt.ylabel('log-likelihood @ test sites')
plt.savefig('ll_'+name+'_boxplot.pdf')
plt.show()
plt.close()
# %%
# Draw boxplots of loglikelihoods between different runs --------------------------------------------------------------
try:
    ll_k13_r4          = np.sum(np.load('results/CONVERGED/k13_r4/ll_k13_r4.npy'), axis = (1,2))
except:
    ll_k13_r4          = []
try:
    ll_k13_r4_fixGEV   = np.sum(np.load('results/CONVERGED/k13_r4_fixGEV/ll_k13_r4_fixGEV.npy'), axis = (1,2))
except:
    ll_k13_r4_fixGEV   = []
try:
    ll_k25_r2          = np.sum(np.load('results/CONVERGED/k25_r2/ll_k25_r2.npy'), axis = (1,2))
except:
    ll_k25_r2          = []
try:
    ll_k25_r2_fixGEV   = np.sum(np.load('results/CONVERGED/k25_r2_fixGEV/ll_k25_r2_fixGEV.npy'), axis = (1,2))
except:
    ll_k25_r2_fixGEV   = []
try:
    ll_k25_r4          = np.sum(np.load('results/CONVERGED/k25_r4/ll_k25_r4.npy'), axis = (1,2))
except:
    ll_k25_r4          = []
try: 
    ll_k25_r4_fixGEV   = np.sum(np.load('results/CONVERGED/k25_r4_fixGEV/ll_k25_r4_fixGEV.npy'), axis = (1,2))
except:
    ll_k25_r4_fixGEV   = []
try:
    ll_k25_efr2        = np.sum(np.load('results/CONVERGED/k25_efr2/ll_k25_efr2.npy'), axis = (1,2))
except:
    ll_k25_efr2        = []
try:
    ll_k25_efr2_fixksi = np.sum(np.load('results/CONVERGED/k25_efr2_fixksi/ll_k25_efr2_fixksi.npy'), axis = (1,2))
except:
    ll_k25_efr2_fixksi = []
try:
    ll_k41_efr2        = np.sum(np.load('results/CONVERGED/k41_efr2/ll_k41_efr2.npy'), axis = (1,2))
except:
    ll_k41_efr2        = []
try:
    ll_phik41efr2_rhok13r4 = np.sum(np.load('results/CONVERGED/phik41efr2_rhok13r4/ll_phik41efr2_rhok13r4.npy'), axis = (1,2))
except:
    ll_phik41efr2_rhok13r4 = []

ll_list = [ll_k13_r4, ll_k13_r4_fixGEV,
           ll_k25_r2, ll_k25_r2_fixGEV,
           ll_k25_r4, ll_k25_r4_fixGEV,
           ll_k25_efr2, ll_k25_efr2_fixksi,
           ll_k41_efr2, ll_phik41efr2_rhok13r4]

fig, ax = plt.subplots()
fig.set_size_inches((10,8))
ax.boxplot(ll_list)

ax.set_xticklabels(['k13_r4', 'k13_r4_fixGEV',
                    'k25_r2', 'k25_r2_fixGEV',
                    'k25_r4', 'k25_r4_fixGEV',
                    'k25_efr2', r'k25_efr2_fix$\xi$',
                    'k41_efr2', r'$\phi$k41efr2_$\rho$k13r4'],
                   rotation = 45)
ax.set_ylabel('loglike at (observed) test sites')
plt.title('Boxplots of loglikelihood @ test sites')
plt.savefig('ll_boxplot_all.pdf')
plt.show()
plt.close()


# %%

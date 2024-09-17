# %% imports
import sys
import os
import math
import requests
import multiprocessing
os.environ["OMP_NUM_THREADS"]        = "64" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"]   = "64" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"]        = "64" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "64" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"]    = "64" # export NUMEXPR_NUM_THREADS=1
from math import sin, cos, sqrt, atan2, radians, asin

import numpy as np
import scipy
np.set_printoptions(threshold=sys.maxsize)
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib import colormaps
from rpy2.robjects import r 
from rpy2.robjects.numpy2ri import numpy2rpy
from rpy2.robjects.packages import importr

from utilities import *

state_map = gpd.read_file('./cb_2018_us_state_20m/cb_2018_us_state_20m.shp')


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

# Sites

sites_xy = stations
sites_x = sites_xy[:,0]
sites_y = sites_xy[:,1]

# define the lower and upper limits for x and y
minX, maxX = np.floor(np.min(sites_x)), np.ceil(np.max(sites_x))
minY, maxY = np.floor(np.min(sites_y)), np.ceil(np.max(sites_y))

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

def get_elevation(longitude, latitude):
    url = 'https://api.open-elevation.com/api/v1/lookup'
    params = {
        'locations': f'{np.round(latitude,7)},{np.round(longitude,7)}'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        elevation = response.json()['results'][0]['elevation']
        return float(elevation)
    else:
        return None

# %%
# Specify which chain

# Model 1: k13_r4

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

# Model 2: k13_r4_fixGEV

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

# Model 3: k25_r2

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

# Model 4: k25_r2_fixGEV

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

# Model 5: k25_r4

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

# Model 6: k25_r4_fixGEV

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

# Model 7: k25_efr2

# folder           = './data_alpine/CONVERGED/20240410_realdata_t75_s590_k25_efr2/'
# name             = 'k25_efr2'
# fixGEV           = False
# radius           = 2
# bandwidth_phi    = radius**2/6 # effective range for gaussian kernel: exp(-3) = 0.05
# bandwidth_rho    = radius**2/6
# N_outer_grid_phi = 16
# N_outer_grid_rho = 16
# mark             = False
# burnin           = 5000

# Model 8: k25_efr2_fixksi

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


# Model 9: k41_efr2

# folder           = './data_alpine/20240428_copy/20240419_realdata_t75_s590_k41_efr2/'
# name             = 'k41_efr2'
# fixGEV           = False
# radius           = 2
# bandwidth_phi    = radius**2/6 # effective range for gaussian kernel: exp(-3) = 0.05
# bandwidth_rho    = radius**2/6
# N_outer_grid_phi = 25
# N_outer_grid_rho = 25
# mark             = False
# burnin           = 7000

# Model 10: phik41efr2_rhok13r4

# folder           = './data_alpine/20240624_copy/20240504_realdata_t75_s590_phik41efr2_rhok13r4/'
# name             = 'phik41efr2_rhok13r4'
# fixGEV           = False
# radius           = 2
# bandwidth_phi    = radius**2/6
# bandwidth_rho    = 4
# N_outer_grid_phi = 25
# N_outer_grid_rho = 9
# mark             = False
# burnin           = 6000

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

# posterior CI
phi_lb = np.percentile(phi_knots_trace, 2.5, axis = 0)
phi_ub = np.percentile(phi_knots_trace, 97.5, axis = 0)

rho_lb = np.percentile(range_knots_trace, 2.5, axis = 0)
rho_ub = np.percentile(range_knots_trace, 97.5, axis = 0)

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

# %%
# Empirical chi from Model Realizaiton

'''
Like Likun suggests,
    Engineer a grid of locations
    Draw 10,000 (time) replicates of observations
    Empirically estimate the chi
'''


# Engineer a grid of sites ----------------------------------------------------

# resolution of engineered points

numX_chi     = 50  # Number of points along the X-axis
numY_chi     = 150 # Number of points along the Y-axis
Ns_chi       = numX_chi * numY_chi
x_chi        = np.linspace(minX, maxX, numX_chi)
y_chi        = np.linspace(minY, maxY, numY_chi)
X_chi,Y_chi  = np.meshgrid(x_chi, y_chi)
sites_xy_chi = np.column_stack([X_chi.ravel(), Y_chi.ravel()]) # a grid of engineerin
sites_x_chi  = sites_xy_chi[:,0]
sites_y_chi  = sites_xy_chi[:,1]

# setting up the copula splines

wendland_weight_matrix_chi     = np.full(shape = (Ns_chi, k_phi), fill_value = np.nan)
for site_id in np.arange(Ns_chi):
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy_chi[site_id,:].reshape((-1,2)), 
                                                    XB = knots_xy_phi)
        wendland_weight_matrix_chi[site_id, :] = wendland_weights_fun(d_from_knots, radius_from_knots)
gaussian_weight_matrix_chi     = np.full(shape = (Ns_chi, k_phi), fill_value = np.nan)
for site_id in np.arange(Ns_chi):
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy_chi[site_id,:].reshape((-1,2)), 
                                                    XB = knots_xy_phi)
        gaussian_weight_matrix_chi[site_id, :] = weights_fun(d_from_knots, radius, bandwidth_phi, cutoff=False)
gaussian_weight_matrix_rho_chi = np.full(shape = (Ns_chi, k_rho), fill_value = np.nan)
for site_id in np.arange(Ns_chi):
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy_chi[site_id,:].reshape((-1,2)), 
                                                    XB = knots_xy_rho)
        gaussian_weight_matrix_rho_chi[site_id, :] = weights_fun(d_from_knots, radius, bandwidth_rho, cutoff=False)

alpha          = 0.5
nu             = 0.5
sigsq_vec      = np.repeat(1.0, Ns_chi)
gamma_at_knots = np.repeat(0.5, k_phi)
gamma_vec_chi  = np.sum(np.multiply(wendland_weight_matrix_chi, gamma_at_knots)**(alpha),
                        axis = 1)**(1/alpha)
assert any(np.isnan(gamma_vec_chi)) != True

# model posterior means

phi_vec_chi = gaussian_weight_matrix_chi     @ phi_mean
rho_vec_chi = gaussian_weight_matrix_rho_chi @ range_mean
K_chi       = ns_cov(range_vec = rho_vec_chi,
                     sigsq_vec = sigsq_vec,
                     coords    = sites_xy_chi,
                     kappa     = nu, cov_model = "matern") # 16 secs for Ns_chi = 7,500

# Draw <n_draw> (time) replicates of observations -----------------------------

np.random.seed(910)
n_draw    = 100000 # number of time replicates to draw
S_vec_chi = np.array([scipy.stats.levy.rvs(loc = 0, scale = 0.5, size = k_phi) for _ in range(n_draw)]) # shape(n_draw, k_phi)
Z_vec_chi = scipy.stats.multivariate_normal.rvs(mean = None, 
                                                cov = K_chi, 
                                                size = n_draw) # shape(n_draw, Ns_chi)
# Notes on multivariate gaussian speed:
#   1m 23s for 7,500 x 100
#   no speed difference when drawing size = 1 or 100

R_vec_chi = (wendland_weight_matrix_chi @ S_vec_chi.T) # shape(Ns_chi, n_draw)
W_chi     = norm_to_Pareto(Z_vec_chi.T)                # shape(Ns_chi, n_draw)
X_model_chi = (R_vec_chi.T ** phi_vec_chi).T * W_chi   # shape(Ns_chi, n_draw)

np.save('X_model_chi', X_model_chi)
np.save('phi_vec_chi', phi_vec_chi)
np.save('rho_vec_chi', rho_vec_chi)

# Empirically estimate chi ----------------------------------------------------

res_x_chi = 7
res_y_chi = 17
k_chi     = res_x_chi * res_y_chi # number of knots
x_pos_chi = np.linspace(minX, maxX, res_x_chi+4)[2:-2]
y_pos_chi = np.linspace(minY, maxY, res_y_chi+4)[2:-2]
X_pos_chi, Y_pos_chi = np.meshgrid(x_pos_chi,y_pos_chi) # create the mesh based on these arrays
knots_xy_chi = np.vstack([X_pos_chi.ravel(), Y_pos_chi.ravel()]).T
knots_x_chi = knots_xy_chi[:,0]
knots_y_chi = knots_xy_chi[:,1]

rect_width = (knots_xy_chi[0][0] - minX)*2
rect_height = (knots_xy_chi[0][1] - minY)*2

e_abs = 0.2

# Threshold quantile qu
# Note:
#   Only needs to be calculated once for each u
#   should parallelize the calculation

def qRW_par(args):
    u, phi, rho = args
    return qRW(u, phi, rho)

args_list090 = []
args_list095 = []
args_list099 = []
for i in range(Ns_chi):
    args_list090.append((0.9, phi_vec_chi[i], rho_vec_chi[i]))
    args_list095.append((0.95, phi_vec_chi[i], rho_vec_chi[i]))
    args_list099.append((0.99, phi_vec_chi[i], rho_vec_chi[i]))
with multiprocessing.Pool(processes = 60) as pool:
    results090 = pool.map(qRW_par, args_list090)
with multiprocessing.Pool(processes = 60) as pool:
    results095 = pool.map(qRW_par, args_list095)
with multiprocessing.Pool(processes = 60) as pool:
    results099 = pool.map(qRW_par, args_list099)

qu_090 = np.array(results090)
qu_095 = np.array(results095)
qu_099 = np.array(results099)
qu_all = (qu_090, qu_095, qu_099)

np.save('qu_090', qu_090)
np.save('qu_095', qu_095)
np.save('qu_099', qu_099)

def calc_model_chi_local(args):
    '''
    Given (as function argument): 
        c: center of a local window, 
        h: distance between points,
        u: threshold probability,
    
    Compute the local empirical chi

    Computation across the local windows is parallelized
    '''

    c, h, u, qu_vec_chi = args

    h_low = h * (1 - e_abs)
    h_up  = h * (1 + e_abs)

    # select sites within the rectangle
    # note: sites_in_rect is an array of index on Ns_chi

    rect_left   = c[0] - rect_width/2
    rect_right  = c[0] + rect_width/2
    rect_top    = c[1] + rect_height/2
    rect_bottom = c[1] - rect_height/2
    sites_in_rect_mask = np.logical_and(np.logical_and(rect_left <= sites_x_chi, sites_x_chi <= rect_right), 
                                        np.logical_and(rect_bottom <= sites_y_chi, sites_y_chi <= rect_top))
    sites_in_rect = sites_xy_chi[sites_in_rect_mask]
    
    # calculate the distance between sites inside rectangle (coords -> km)
    # and select pairs of sites that are ~ h km apart
    # note: site_pairs_to_check is an array of (index, index) on sites_in_rect

    n_sites = sites_in_rect.shape[0]
    sites_dist_mat = np.full(shape = (n_sites, n_sites), fill_value = np.nan)
    for si in range(n_sites): # 30 secs for 600 x 600
        for sj in range(n_sites):
            sites_dist_mat[si,sj] = coord_to_dist(sites_in_rect[si], sites_in_rect[sj])    

    sites_h_mask = np.logical_and(np.triu(sites_dist_mat) >= h_low,
                                  np.triu(sites_dist_mat) <= h_up)
    n_pairs = len(np.triu(sites_dist_mat)[sites_h_mask])
    site_pairs_to_check = np.array([(np.where(sites_h_mask)[0][i], 
                                     np.where(sites_h_mask)[1][i]) for i in range(n_pairs)])

    # calculate chi

    X_in_rect     = X_model_chi[sites_in_rect_mask]
    qu_in_rect    = qu_vec_chi[sites_in_rect_mask]

    count_co_extreme = 0
    for site_pair in site_pairs_to_check:
        site1 = X_in_rect[site_pair[0]]
        site2 = X_in_rect[site_pair[1]]
        site1_qu = qu_in_rect[site_pair[0]]
        site2_qu = qu_in_rect[site_pair[1]]
        count_co_extreme += np.sum(np.logical_and(site1 >= site1_qu,
                                                  site2 >= site2_qu))
    prob_joint_ext = count_co_extreme / (n_pairs * n_draw) # numerator
    prob_uni_ext   = np.mean(X_in_rect >= qu_in_rect[:, np.newaxis])
    chi            = prob_joint_ext / prob_uni_ext if prob_uni_ext != 0 else 0.0
    
    return chi

# Plotting --------------------------------------------------------------------

# Create a LinearSegmentedColormap from white to red
colors = ["#ffffff", "#ff0000"]
min_chi = 0.0
max_chi = 0.5
n_bins = 100  # Number of discrete bins
n_ticks = 10
cmap_name = "white_to_red"
colormap = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
ticks = np.linspace(min_chi, max_chi, n_ticks+1).round(3)


for h in [75, 150, 225]:
    fig, axes = plt.subplots(1,3)
    fig.set_size_inches(10,6)
    for ax_id, u in enumerate([0.9, 0.95, 0.99]):

        args_list = []
        for i in range(knots_xy_chi.shape[0]):
            args = (knots_xy_chi[i], h, u, qu_all[ax_id])
            args_list.append(args)
        with multiprocessing.Pool(processes = 60) as pool:
            results = pool.map(calc_model_chi_local, args_list)
        
        chi_mat = np.full(shape = (len(y_pos_chi), len(x_pos_chi)), fill_value = np.nan)
        for i in range(knots_xy_chi.shape[0]):
            chi_mat[-1 - i//len(x_pos_chi), i % len(x_pos_chi)] = results[i]
        
        ax = axes[ax_id]
        ax.set_aspect('equal', 'box')
        state_map.boundary.plot(ax=ax, color = 'black', linewidth = 0.5)
        heatmap = ax.imshow(chi_mat, cmap = colormap, interpolation = 'nearest',
                            vmin = 0.0, vmax = 1.0,
                            extent = [min(x_pos_chi - rect_width/8), max(x_pos_chi + rect_width/8), 
                                    min(y_pos_chi - rect_height/8), max(y_pos_chi + rect_height/8)])
        # ax.scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.8)
        ax.scatter(knots_x_chi, knots_y_chi, s = 15, color = 'white', marker = '+')
        ax.set_xlim(-101,-93)
        ax.set_ylim(32.5, 45)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.title.set_text(rf'$\chi_{{{u}}}$')
        ax.title.set_fontsize(20)
    
    fig.subplots_adjust(right=0.8)
    fig.text(0.5, 0.825, rf'h $\approx$ {h}km', ha='center', fontsize = 20)
    fig.text(0.5, 0.125, 'Longitude', ha='center', fontsize = 20)
    fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical', fontsize = 20)    
    cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
    colorbar = fig.colorbar(heatmap, cax = cbar_ax, ticks = ticks)
    colorbar.ax.tick_params(labelsize=14)
    plt.savefig('Surface:model_realization_empirical_chi_h={}.pdf'.format(h), bbox_inches='tight')
    plt.show()
    plt.close()
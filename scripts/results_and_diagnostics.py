"""
Generate the below results and diagnostics.
Requires:
    - utilities.py
    - predGEV_grid_elev.npy <- the elevation grid for the predicted marginal GEV surface

-------- Numerical Summaries --------

Posterior summaries (posterior mean, 95% CI) for 
    - phi
    - rho

-------- Posterior Surfaces --------

Posterior mean surfaces for 
    - GEV
    - phi
    - rho

Dataset empirical chi surface
    - using per-iteration fitted GEV

Model-Realized empirical chi surface
    - April 19 Ben: use R drawn from stable prior

-------- Diagnostics Plots --------

Marginal fit: 
Make QQplot of gumbel (April 14, 2024)
    - mean(per MCMC iter GEV)
    Note that CDF(Y) and transformation to gumbel should be performed on misspiggy
    require lots of memory to load the pY per MCMC iteration

Predicative performance: 
log-likelihood at testing sites (April 15, 2024)
    - (April 19, 2024) Ben: ll calculation isn't linear, use per-iteration parameter value


"""
# %% imports

# Base Python
import  sys
import  os
import  math
import  requests
import  multiprocessing
from    math                   import sin, cos, sqrt, atan2, radians, asin

# Python Extensions
import  scipy
import  numpy                as np
import  geopandas            as gpd
import  matplotlib           as mpl
import  matplotlib.pyplot    as plt
import matplotlib.ticker     as mtick
from    matplotlib             import colormaps
from    rpy2.robjects          import r 
from    rpy2.robjects.numpy2ri import numpy2rpy
from    rpy2.robjects.packages import importr
from    tqdm                   import tqdm

# Custom Extensions and settings
from    utilities              import *

N_CORES = 30

np.set_printoptions(threshold=sys.maxsize)
state_map = gpd.read_file('./cb_2018_us_state_20m/cb_2018_us_state_20m.shp')

cool_only = plt.get_cmap('coolwarm', 512)(np.linspace(0, 0.5, 256))
warm_only = plt.get_cmap('coolwarm', 512)(np.linspace(0.5, 1.0, 256))
cool_cmap = mpl.colors.ListedColormap(cool_only)
warm_cmap = mpl.colors.ListedColormap(warm_only)

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

# %% Load Chain Traceplots and Setup

# specify the model -----------------------------------------------------------

# # Model 0: HuserWadsworth ---------------------------------
# folder           = './'
# name             = 'HuserWadsworth'
# fixGEV           = False
# radius           = 100
# bandwidth_phi    = 100
# bandwidth_rho    = 100
# N_outer_grid_phi = 1
# N_outer_grid_rho = 1
# burnin           = 1000

# # Model 1: k13r4b4 ----------------------------------------
# folder           = './'
# name             = 'k13r4b4'
# fixGEV           = False
# # fixksi_only      = False
# radius           = 4
# bandwidth_phi    = 4
# bandwidth_rho    = 4
# N_outer_grid_phi = 9
# N_outer_grid_rho = 9
# # mark             = False
# burnin           = 3000

# # Model 2: k13r4b4m ---------------------------------------
# folder           = './'
# name             = 'k13r4b4m'
# fixGEV           = True
# radius           = 4
# bandwidth_phi    = 4
# bandwidth_rho    = 4
# N_outer_grid_phi = 9
# N_outer_grid_rho = 9
# burnin           = 3000

# # Model 3: k25r2b0.67 -------------------------------------
# folder           = './'
# name             = 'k25r2b0.67'
# fixGEV           = False
# radius           = 2
# bandwidth_phi    = 0.67
# bandwidth_rho    = 0.67
# N_outer_grid_phi = 16
# N_outer_grid_rho = 16
# burnin           = 3000

# # Model 4: k25r2b0.67m ------------------------------------
# folder           = './'
# name             = 'k25r2b0.67m'
# fixGEV           = True
# radius           = 2
# bandwidth_phi    = 0.67
# bandwidth_rho    = 0.67
# N_outer_grid_phi = 16
# N_outer_grid_rho = 16
# burnin           = 3000

# # Model 5: k25r2b2 ----------------------------------------
# folder           = './'
# name             = 'k25r2b2'
# fixGEV           = False
# radius           = 2
# bandwidth_phi    = 2
# bandwidth_rho    = 2
# N_outer_grid_phi = 16
# N_outer_grid_rho = 16
# burnin           = 3000

# # Model 6: k25r2b2m ---------------------------------------
# folder           = './'
# name             = 'k25r2b2m'
# fixGEV           = True
# radius           = 2
# bandwidth_phi    = 2
# bandwidth_rho    = 2
# N_outer_grid_phi = 16
# N_outer_grid_rho = 16
# burnin           = 3000

# # Model 7: k25r4b4 ----------------------------------------
# folder           = './'
# name             = 'k25r4b4'
# fixGEV           = False
# radius           = 4
# bandwidth_phi    = 4
# bandwidth_rho    = 4
# N_outer_grid_phi = 16
# N_outer_grid_rho = 16
# burnin           = 3000

# Model 8: k25r4b4m ---------------------------------------
folder           = './'
name             = 'k25r4b4m'
fixGEV           = True
radius           = 4
bandwidth_phi    = 4
bandwidth_rho    = 4
N_outer_grid_phi = 16
N_outer_grid_rho = 16
burnin           = 3000

# # Model 9: k41r1.6b0.43 ----------------------------------
# folder           = './'
# name             = 'k41r1.6b0.43'
# fixGEV           = False
# radius           = 1.6
# bandwidth_phi    = 0.43
# bandwidth_rho    = 0.43
# N_outer_grid_phi = 25
# N_outer_grid_rho = 25
# burnin           = 3000

# # Model 10: k41r1.6b0.43m ---------------------------------
# folder           = './'
# name             = 'k41r1.6b0.43m'
# fixGEV           = True
# radius           = 1.6
# bandwidth_phi    = 0.43
# bandwidth_rho    = 0.43
# N_outer_grid_phi = 25
# N_outer_grid_rho = 25
# burnin           = 3000

# # Model 11: k41r2b0.67 ------------------------------------
# folder           = './'
# name             = 'k41r2b0.67'
# fixGEV           = False
# radius           = 2
# bandwidth_phi    = 0.67
# bandwidth_rho    = 0.67
# N_outer_grid_phi = 25
# N_outer_grid_rho = 25
# burnin           = 3000

# # Model 12: k41r2b0.67m -----------------------------------
# folder           = './'
# name             = 'k41r2b0.67m'
# fixGEV           = True
# radius           = 2
# bandwidth_phi    = 0.67
# bandwidth_rho    = 0.67
# N_outer_grid_phi = 25
# N_outer_grid_rho = 25
# burnin           = 3000

# load traceplots -------------------------------------------------------------

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

# Knots setup -----------------------------------------------------------------

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

# if mark == True:
#     # isometric knot grid - Mark's
#     x_pos_phi                    = np.linspace(minX + 1, maxX + 1, num = int(2*np.sqrt(N_outer_grid_phi)))
#     y_pos_phi                    = np.linspace(minY + 1, maxY + 1, num = int(2*np.sqrt(N_outer_grid_phi)))
#     x_outer_pos_phi              = x_pos_phi[0::2]
#     x_inner_pos_phi              = x_pos_phi[1::2]
#     y_outer_pos_phi              = y_pos_phi[0::2]
#     y_inner_pos_phi              = y_pos_phi[1::2]
#     X_outer_pos_phi, Y_outer_pos_phi = np.meshgrid(x_outer_pos_phi, y_outer_pos_phi)
#     X_inner_pos_phi, Y_inner_pos_phi = np.meshgrid(x_inner_pos_phi, y_inner_pos_phi)
#     knots_outer_xy_phi           = np.vstack([X_outer_pos_phi.ravel(), Y_outer_pos_phi.ravel()]).T
#     knots_inner_xy_phi           = np.vstack([X_inner_pos_phi.ravel(), Y_inner_pos_phi.ravel()]).T
#     knots_xy_phi                 = np.vstack((knots_outer_xy_phi, knots_inner_xy_phi))
#     knots_id_in_domain_phi       = [row for row in range(len(knots_xy_phi)) if (minX < knots_xy_phi[row,0] < maxX and minY < knots_xy_phi[row,1] < maxY)]
#     knots_xy_phi                 = knots_xy_phi[knots_id_in_domain_phi]
#     knots_x_phi                  = knots_xy_phi[:,0]
#     knots_y_phi                  = knots_xy_phi[:,1]
#     k_phi                        = len(knots_id_in_domain_phi)

#     x_pos_rho                    = np.linspace(minX + 1, maxX + 1, num = int(2*np.sqrt(N_outer_grid_rho)))
#     y_pos_rho                    = np.linspace(minY + 1, maxY + 1, num = int(2*np.sqrt(N_outer_grid_rho)))
#     x_outer_pos_rho              = x_pos_rho[0::2]
#     x_inner_pos_rho              = x_pos_rho[1::2]
#     y_outer_pos_rho              = y_pos_rho[0::2]
#     y_inner_pos_rho              = y_pos_rho[1::2]
#     X_outer_pos_rho, Y_outer_pos_rho = np.meshgrid(x_outer_pos_rho, y_outer_pos_rho)
#     X_inner_pos_rho, Y_inner_pos_rho = np.meshgrid(x_inner_pos_rho, y_inner_pos_rho)
#     knots_outer_xy_rho           = np.vstack([X_outer_pos_rho.ravel(), Y_outer_pos_rho.ravel()]).T
#     knots_inner_xy_rho           = np.vstack([X_inner_pos_rho.ravel(), Y_inner_pos_rho.ravel()]).T
#     knots_xy_rho                 = np.vstack((knots_outer_xy_rho, knots_inner_xy_rho))
#     knots_id_in_domain_rho       = [row for row in range(len(knots_xy_rho)) if (minX < knots_xy_rho[row,0] < maxX and minY < knots_xy_rho[row,1] < maxY)]
#     knots_xy_rho                 = knots_xy_rho[knots_id_in_domain_rho]
#     knots_x_rho                  = knots_xy_rho[:,0]
#     knots_y_rho                  = knots_xy_rho[:,1]
#     k_rho                        = len(knots_id_in_domain_rho)

# Splines setup ---------------------------------------------------------------

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

# Marginal Model Splines - GEV(mu, sigma, ksi) --------------------------------

# thin-plate splines for mu0 and mu1 ----------------------

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

# Location mu_0(s) ----------------------------------------

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

# Location mu_1(s) ----------------------------------------

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

# Scale logsigma(s) ---------------------------------------

Beta_logsigma_m   = 2 # just intercept and elevation
C_logsigma        = np.full(shape = (Beta_logsigma_m, Ns, Nt), fill_value = np.nan)
C_logsigma[0,:,:] = 1.0 
C_logsigma[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T

# Shape ksi(s) --------------------------------------------

Beta_ksi_m   = 2 # just intercept and elevation
C_ksi        = np.full(shape = (Beta_ksi_m, Ns, Nt), fill_value = np.nan) # ksi design matrix
C_ksi[0,:,:] = 1.0
C_ksi[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T


# %% Posterior Traceplot Summaries

# posterior mean ------------------------------------------

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

# posterior median ----------------------------------------

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

# posterior CI --------------------------------------------

phi_lb = np.percentile(phi_knots_trace, 2.5, axis = 0)
phi_ub = np.percentile(phi_knots_trace, 97.5, axis = 0)

rho_lb = np.percentile(range_knots_trace, 2.5, axis = 0)
rho_ub = np.percentile(range_knots_trace, 97.5, axis = 0)

# thinned by 10 -------------------------------------------

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

# Plot traceplot ------------------------------------------

for i in range(k_phi):
    plt.subplots()
    plt.plot(xs_thin2, phi_knots_trace_thin[:,i], label = 'knot'+str(i))
    plt.title('traceplot for phi knot' + str(i))
    plt.xlabel('iter thinned by 10')
    plt.ylabel('phi')
    plt.savefig('Traceplot_phi_knot' + str(i) + '.pdf')
    plt.show()
    plt.close()

for i in range(k_rho):
    plt.subplots()
    plt.plot(xs_thin2, range_knots_trace_thin[:,i], label = 'knot'+str(i))
    plt.title('traceplot for range knot' + str(i))
    plt.xlabel('iter thinned by 10')
    plt.ylabel('range')
    plt.savefig('Traceplot_range_knot' + str(i) + '.pdf')
    plt.show()
    plt.close()

if not fixGEV:

    for j in range(Beta_mu0_m):
        plt.plot(xs_thin2, Beta_mu0_trace_thin[:,j], label = 'Beta_'+str(j))
        plt.annotate('Beta_' + str(j), xy=(xs_thin2[-1], Beta_mu0_trace_thin[:,j][-1]))
    plt.title('traceplot for Beta_mu0')
    plt.xlabel('iter thinned by 10')
    plt.ylabel('Beta_mu0')
    plt.legend()
    plt.savefig('Traceplot_Beta_mu0_merged.pdf')
    plt.show()
    plt.close()

    for j in range(Beta_mu1_m):
        plt.plot(xs_thin2, Beta_mu1_trace_thin[:,j], label = 'Beta_'+str(j))
        plt.annotate('Beta_' + str(j), xy=(xs_thin2[-1], Beta_mu1_trace_thin[:,j][-1]))
    plt.title('traceplot for Beta_mu1')
    plt.xlabel('iter thinned by 10')
    plt.ylabel('Beta_mu1')
    plt.legend()
    plt.savefig('Traceplot_Beta_mu1_merged.pdf')
    plt.show()
    plt.close()

    for j in range(Beta_logsigma_m):
        plt.plot(xs_thin2, Beta_logsigma_trace_thin[:,j], label = 'Beta_'+str(j))
        plt.annotate('Beta_' + str(j), xy=(xs_thin2[-1], Beta_logsigma_trace_thin[:,j][-1]))
    plt.title('traceplot for Beta_logsigma')
    plt.xlabel('iter thinned by 10')
    plt.ylabel('Beta_logsigma')
    plt.legend()
    plt.savefig('Traceplot_Beta_logsigma_merged.pdf')
    plt.show()
    plt.close()

    for j in range(Beta_ksi_m):
        plt.plot(xs_thin2, Beta_ksi_trace_thin[:,j], label = 'Beta_'+str(j))
        plt.annotate('Beta_' + str(j), xy=(xs_thin2[-1], Beta_ksi_trace_thin[:,j][-1]))
    plt.title('traceplot for Beta_ksi')
    plt.xlabel('iter thinned by 10')
    plt.ylabel('Beta_ksi')
    plt.legend()
    plt.savefig('Traceplot_Beta_ksi_merged.pdf')
    plt.show()
    plt.close()


# %% Predicted Smooth GEV Surface

# https://likun-stat.shinyapps.io/lab4/#section-retrieve-elevation-values-on-a-grid
# installation error for `elevatr` -- cannot install terra & raster due to lack of gdal

# resolution for plotting the fitted marginal surface -------------------------
#   b/c we need the elevation and its API is expensive

predGEV_res_x = int(maxX - minX) * 3
predGEV_res_y = int(maxY - minY) * 3
predGEV_res_xy = predGEV_res_x * predGEV_res_y
predGEV_grid_x = np.linspace(minX, maxX, predGEV_res_x)
predGEV_grid_y = np.linspace(minY, maxY, predGEV_res_y)
predGEV_grid_X, predGEV_grid_Y = np.meshgrid(predGEV_grid_x, predGEV_grid_y)
predGEV_grid_xy = np.vstack([predGEV_grid_X.ravel(), predGEV_grid_Y.ravel()]).T

# Download the elevation information as covariate -----------------------------

try:
    predGEV_grid_elev = np.load('predGEV_grid_elev.npy')
except Exception as e:
    print(e) 
    predGEV_grid_elev = np.array([get_elevation(long, lat) for long, lat in predGEV_grid_xy]).astype(float)
    NA_elev_id = np.where(np.isnan(np.array(predGEV_grid_elev).astype(float)))[0]
    predGEV_grid_elev[NA_elev_id]
    np.save('predGEV_grid_elev', predGEV_grid_elev)


# Prepare the Splines ---------------------------------------------------------

predGEV_grid_xy_ro = numpy2rpy(predGEV_grid_xy)
r.assign("predGEV_grid_xy", predGEV_grid_xy_ro)
r('''
    predGEV_grid_xy <- as.data.frame(predGEV_grid_xy)
    colnames(predGEV_grid_xy) <- c('x','y')
  ''')

# Location mu_0

C_mu0_splines_pred      = np.array(r('''
                                        basis      <- smoothCon(s(x, y, k = {Beta_mu0_splines_m}, fx = TRUE), data = gs_xy_df)[[1]]
                                        basis_site <- PredictMat(basis, data = predGEV_grid_xy)
                                        # basis_site
                                        basis_site[,c(-(ncol(basis_site)-2))] # dropped the 3rd to last column of constant
                                    '''.format(Beta_mu0_splines_m = Beta_mu0_splines_m+1))) # shaped(Ns, Beta_mu0_splines_m)
C_mu0_1t_pred           = np.column_stack((np.ones(predGEV_res_xy),
                                            predGEV_grid_elev/200,
                                            C_mu0_splines_pred))
C_mu0_pred              = np.tile(C_mu0_1t_pred.T[:,:,None], reps = (1, 1, Nt))


# Location mu_1

C_mu1_splines_pred      = np.array(r('''
                                        basis      <- smoothCon(s(x, y, k = {Beta_mu1_splines_m}, fx = TRUE), data = gs_xy_df)[[1]]
                                        basis_site <- PredictMat(basis, data = predGEV_grid_xy)
                                        # basis_site
                                        basis_site[,c(-(ncol(basis_site)-2))] # drop the 3rd to last column of constant
                                '''.format(Beta_mu1_splines_m = Beta_mu1_splines_m+1))) # shaped(Ns, Beta_mu1_splines_m)
C_mu1_1t_pred           = np.column_stack((np.ones(predGEV_res_xy),  # intercept
                                            predGEV_grid_elev/200,     # elevation
                                            C_mu1_splines_pred)) # splines (excluding intercept)
C_mu1_pred              = np.tile(C_mu1_1t_pred.T[:,:,None], reps = (1, 1, Nt))


# Scale logsigma

C_logsigma_pred        = np.full(shape = (Beta_logsigma_m, predGEV_res_xy, Nt), fill_value = np.nan)
C_logsigma_pred[0,:,:] = 1.0 
C_logsigma_pred[1,:,:] = np.tile(predGEV_grid_elev/200, reps = (Nt, 1)).T

# Shape xi

C_ksi_pred        = np.full(shape = (Beta_ksi_m, predGEV_res_xy, Nt), fill_value = np.nan) # ksi design matrix
C_ksi_pred[0,:,:] = 1.0
C_ksi_pred[1,:,:] = np.tile(predGEV_grid_elev/200, reps = (Nt, 1)).T

# Plotting Predicted GEV Surface ----------------------------------------------

if not fixGEV:

    # mu0 -------------------------------------------------

    predmu0 = (C_mu0_pred.T @ Beta_mu0_mean).T[:,0]
    vmin    = np.floor(min(predmu0))
    vmax    = np.ceil(max(predmu0))
    divnorm = mpl.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)
    # fig, ax = plt.subplots()
    # fig.set_size_inches(8,6)
    fig, ax = plt.subplots(figsize = (10,8), constrained_layout=True)
    ax.set_aspect('equal','box')
    state_map.boundary.plot(ax=ax, color = 'black')
    heatmap = ax.imshow(predmu0.reshape(predGEV_grid_X.shape),
                        extent=[minX, maxX, minY, maxY],
                        origin = 'lower', cmap = warm_cmap, norm = divnorm) # colormaps['OrRd']
    ax.set_xticks(np.linspace(minX, maxX,num=3))
    ax.set_yticks(np.linspace(minY, maxY,num=3))
    cbar    = fig.colorbar(heatmap, ax = ax)
    cbar.ax.tick_params(labelsize=40)
    cbar.locator = mpl.ticker.MaxNLocator(nbins=5)
    cbar.update_ticks()
    plt.xlim([-104,-90])
    plt.ylim([30,47])
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xlabel('longitude', fontsize = 50)
    plt.ylabel('latitude', fontsize = 50)
    plt.title(r'$\mu_0(s)$ surface', fontsize = 50)
    plt.savefig('Surface_mu0_pred.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    # mu1 -------------------------------------------------

    predmu1   = (C_mu1_pred.T @ Beta_mu1_mean).T[:,0]
    vmin      = np.floor(min(predmu1))
    vmax      = np.ceil(max(predmu1))
    tmp_bound = max(np.abs(vmin), np.abs(vmax))
    divnorm   = mpl.colors.TwoSlopeNorm(vcenter = 0, vmin = -tmp_bound, vmax = tmp_bound)
    # fig, ax   = plt.subplots()
    # fig.set_size_inches(8,6)
    fig, ax = plt.subplots(figsize = (10,8), constrained_layout=True)
    ax.set_aspect('equal','box')
    state_map.boundary.plot(ax=ax, color = 'black')
    heatmap = ax.imshow(predmu1.reshape(predGEV_grid_X.shape),
                        extent=[minX, maxX, minY, maxY],
                        origin = 'lower', cmap = colormaps['coolwarm'], norm = divnorm)
    ax.set_xticks(np.linspace(minX, maxX,num=3))
    ax.set_yticks(np.linspace(minY, maxY,num=3))
    cbar    = fig.colorbar(heatmap, ax = ax)
    cbar.ax.tick_params(labelsize=40)
    cbar.locator = mpl.ticker.MaxNLocator(nbins=4)
    cbar.update_ticks()
    plt.xlim([-104,-90])
    plt.ylim([30,47])
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xlabel('longitude', fontsize = 50)
    plt.ylabel('latitude', fontsize = 50)
    plt.title(r'$\mu_1(s)$ surface', fontsize = 50)
    plt.savefig('Surface_mu1_pred.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    # logsigma --------------------------------------------

    predlogsigma = (C_logsigma_pred.T @ Beta_logsigma_mean).T[:,0]
    vmin    = my_floor(min(predlogsigma), 2)
    vmax    = my_ceil(max(predlogsigma), 2)
    divnorm = mpl.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)
    # fig, ax = plt.subplots()
    # fig.set_size_inches(8,6)
    fig, ax = plt.subplots(figsize = (10,8), constrained_layout=True)
    ax.set_aspect('equal','box')
    state_map.boundary.plot(ax=ax, color = 'black')
    heatmap = ax.imshow(predlogsigma.reshape(predGEV_grid_X.shape),
                        extent=[minX, maxX, minY, maxY],
                        origin = 'lower', cmap = warm_cmap, norm = divnorm) # colormaps['OrRd']
    ax.set_xticks(np.linspace(minX, maxX,num=3))
    ax.set_yticks(np.linspace(minY, maxY,num=3))
    cbar    = fig.colorbar(heatmap, ax = ax)
    cbar.ax.tick_params(labelsize=40)
    cbar.locator = mpl.ticker.MaxNLocator(nbins=4)
    cbar.update_ticks()
    plt.xlim([-104,-90])
    plt.ylim([30,47])
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xlabel('longitude', fontsize = 50)
    plt.ylabel('latitude', fontsize = 50)
    plt.title(r'$\log(\sigma(s))$ surface', fontsize = 50)
    plt.savefig('Surface_logsigma_pred.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    # xi --------------------------------------------------

    predksi = (C_ksi_pred.T @ Beta_ksi_mean).T[:,0]
    vmin    = my_floor(min(predksi), 2)
    vmax    = my_ceil(max(predksi), 2)
    divnorm = mpl.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)
    # fig, ax = plt.subplots()
    # fig.set_size_inches(8,6)
    fig, ax = plt.subplots(figsize = (10,8), constrained_layout=True)
    ax.set_aspect('equal','box')
    state_map.boundary.plot(ax=ax, color = 'black')
    heatmap = ax.imshow(predksi.reshape(predGEV_grid_X.shape),
                        extent=[minX, maxX, minY, maxY],
                        origin = 'lower', cmap = warm_cmap, norm = divnorm) # colormaps['OrRd']
    ax.set_xticks(np.linspace(minX, maxX,num=3))
    ax.set_yticks(np.linspace(minY, maxY,num=3))
    cbar    = fig.colorbar(heatmap, ax = ax)
    cbar.ax.tick_params(labelsize=40)
    cbar.locator = mpl.ticker.MaxNLocator(nbins=4)
    cbar.update_ticks()
    plt.xlim([-104,-90])
    plt.ylim([30,47])
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xlabel('longitude', fontsize = 50)
    plt.ylabel('latitude', fontsize = 50)
    plt.title(r'$\xi(s)$ surface', fontsize = 50)
    plt.savefig('Surface_xi_pred.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

if fixGEV:
    Beta_mu0_initSmooth      = np.linalg.lstsq(a=C_mu0[:,:,0].T, b=mu0_estimates,rcond=None)[0]
    Beta_mu1_initSmooth      = np.linalg.lstsq(a=C_mu1[:,:,0].T, b=mu1_estimates,rcond=None)[0]
    Beta_logsigma_initSmooth = np.linalg.lstsq(a=C_logsigma[:,:,0].T, b=logsigma_estimates,rcond=None)[0]
    Beta_ksi_initSmooth      = np.linalg.lstsq(a=C_ksi[:,:,0].T, b=ksi_estimates,rcond=None)[0]

    # mu0 -------------------------------------------------

    predmu0 = (C_mu0_pred.T @ Beta_mu0_initSmooth).T[:,0]
    vmin    = np.floor(min(predmu0))
    vmax    = np.ceil(max(predmu0))
    divnorm = mpl.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)
    # fig, ax = plt.subplots()
    # fig.set_size_inches(8,6)
    fig, ax = plt.subplots(figsize = (10,8), constrained_layout=True)
    ax.set_aspect('equal','box')
    state_map.boundary.plot(ax=ax, color = 'black')
    heatmap = ax.imshow(predmu0.reshape(predGEV_grid_X.shape),
                        extent=[minX, maxX, minY, maxY],
                        origin = 'lower', cmap = warm_cmap, norm = divnorm) # colormaps['OrRd']
    ax.set_xticks(np.linspace(minX, maxX,num=3))
    ax.set_yticks(np.linspace(minY, maxY,num=3))
    cbar    = fig.colorbar(heatmap, ax = ax)
    cbar.ax.tick_params(labelsize=40)
    cbar.locator = mpl.ticker.MaxNLocator(nbins=5)
    cbar.update_ticks()
    plt.xlim([-104,-90])
    plt.ylim([30,47])
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xlabel('longitude', fontsize = 50)
    plt.ylabel('latitude', fontsize = 50)
    plt.title(r'$\mu_0(s)$ surface', fontsize = 50)
    plt.savefig('Surface_mu0_pred.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    # mu1 -------------------------------------------------

    predmu1   = (C_mu1_pred.T @ Beta_mu1_initSmooth).T[:,0]
    vmin      = np.floor(min(predmu1))
    vmax      = np.ceil(max(predmu1))
    tmp_bound = max(np.abs(vmin), np.abs(vmax))
    divnorm   = mpl.colors.TwoSlopeNorm(vcenter = 0, vmin = -tmp_bound, vmax = tmp_bound)
    # fig, ax   = plt.subplots()
    # fig.set_size_inches(8,6)
    fig, ax = plt.subplots(figsize = (10,8), constrained_layout=True)
    ax.set_aspect('equal','box')
    state_map.boundary.plot(ax=ax, color = 'black')
    heatmap = ax.imshow(predmu1.reshape(predGEV_grid_X.shape),
                        extent=[minX, maxX, minY, maxY],
                        origin = 'lower', cmap = colormaps['coolwarm'], norm = divnorm)
    ax.set_xticks(np.linspace(minX, maxX,num=3))
    ax.set_yticks(np.linspace(minY, maxY,num=3))
    cbar    = fig.colorbar(heatmap, ax = ax)
    cbar.ax.tick_params(labelsize=40)
    cbar.locator = mpl.ticker.MaxNLocator(nbins=4)
    cbar.update_ticks()
    plt.xlim([-104,-90])
    plt.ylim([30,47])
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xlabel('longitude', fontsize = 50)
    plt.ylabel('latitude', fontsize = 50)
    plt.title(r'$\mu_1(s)$ surface', fontsize = 50)
    plt.savefig('Surface_mu1_pred.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    # logsigma --------------------------------------------

    predlogsigma = (C_logsigma_pred.T @ Beta_logsigma_initSmooth).T[:,0]
    vmin    = my_floor(min(predlogsigma), 2)
    vmax    = my_ceil(max(predlogsigma), 2)
    divnorm = mpl.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)
    # fig, ax = plt.subplots()
    # fig.set_size_inches(8,6)
    fig, ax = plt.subplots(figsize = (10,8), constrained_layout=True)
    ax.set_aspect('equal','box')
    state_map.boundary.plot(ax=ax, color = 'black')
    heatmap = ax.imshow(predlogsigma.reshape(predGEV_grid_X.shape),
                        extent=[minX, maxX, minY, maxY],
                        origin = 'lower', cmap = warm_cmap, norm = divnorm) # colormaps['OrRd']
    ax.set_xticks(np.linspace(minX, maxX,num=3))
    ax.set_yticks(np.linspace(minY, maxY,num=3))
    cbar    = fig.colorbar(heatmap, ax = ax)
    cbar.ax.tick_params(labelsize=40)
    cbar.locator = mpl.ticker.MaxNLocator(nbins=4)
    cbar.update_ticks()
    plt.xlim([-104,-90])
    plt.ylim([30,47])
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xlabel('longitude', fontsize = 50)
    plt.ylabel('latitude', fontsize = 50)
    plt.title(r'$\log(\sigma(s))$ surface', fontsize = 50)
    plt.savefig('Surface_logsigma_pred.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    # xi --------------------------------------------------

    predksi = (C_ksi_pred.T @ Beta_ksi_initSmooth).T[:,0]
    vmin    = my_floor(min(predksi), 2)
    vmax    = my_ceil(max(predksi), 2)
    divnorm = mpl.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)
    # fig, ax = plt.subplots()
    # fig.set_size_inches(8,6)
    fig, ax = plt.subplots(figsize = (10,8), constrained_layout=True)
    ax.set_aspect('equal','box')
    state_map.boundary.plot(ax=ax, color = 'black')
    heatmap = ax.imshow(predksi.reshape(predGEV_grid_X.shape),
                        extent=[minX, maxX, minY, maxY],
                        origin = 'lower', cmap = warm_cmap, norm = divnorm) # colormaps['OrRd']
    ax.set_xticks(np.linspace(minX, maxX,num=3))
    ax.set_yticks(np.linspace(minY, maxY,num=3))
    cbar    = fig.colorbar(heatmap, ax = ax)
    cbar.ax.tick_params(labelsize=40)
    cbar.locator = mpl.ticker.MaxNLocator(nbins=4)
    cbar.update_ticks()
    plt.xlim([-104,-90])
    plt.ylim([30,47])
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xlabel('longitude', fontsize = 50)
    plt.ylabel('latitude', fontsize = 50)
    plt.title(r'$\xi(s)$ surface', fontsize = 50)
    plt.savefig('Surface_xi_pred.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

# %% Copula Posterior Surface Plotting

# phi surface
phi_vec_for_plot = gaussian_weight_matrix_for_plot_phi @ phi_mean
# fig, ax = plt.subplots()
# fig.set_size_inches(8,6)
fig, ax = plt.subplots(figsize = (10,8), constrained_layout=True)
ax.set_aspect('equal', 'box')
state_map.boundary.plot(ax=ax, color = 'black')
vmin      = my_floor(min(phi_vec_for_plot), 2)
vmax      = my_ceil(max(phi_vec_for_plot), 2)
tmp_bound = max(np.abs(0.5-vmin), np.abs(0.5-vmax))
divnorm = mpl.colors.TwoSlopeNorm(vcenter = 0.5, vmin = 0.5-tmp_bound, vmax = 0.5+tmp_bound)
heatmap = ax.imshow(phi_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x),
                    extent=[minX, maxX, minY, maxY],
                    origin = 'lower',
                    cmap = colormaps['coolwarm'], norm = divnorm)
contour = ax.contour(plotgrid_X, plotgrid_Y, phi_vec_for_plot.reshape(plotgrid_X.shape),
                    levels=[0.5], colors='black', linewidths=1, linestyles='dashed')
ax.set_xticks(np.linspace(minX, maxX,num=3))
ax.set_yticks(np.linspace(minY, maxY,num=3))
cbar = fig.colorbar(heatmap, ax=ax)
cbar.ax.tick_params(labelsize=40)
cbar.locator = mpl.ticker.MaxNLocator(nbins=5)
cbar.update_ticks()
# cbar.set_ticks([my_floor(min(phi_vec_for_plot), 2), 0.50, my_ceil(max(phi_vec_for_plot), 2)])
for i in range(k_phi): # Plot knots and circles
    circle_i = plt.Circle((knots_xy_phi[i, 0], knots_xy_phi[i, 1]), radius_from_knots[i],
                        color='r', fill=False, fc='None', ec='lightgrey')
    ax.add_patch(circle_i)
# Scatter plot for sites and knots
ax.scatter(knots_x_phi, knots_y_phi, marker='+', c='black', label='knot', s=500)
for index, (x, y) in enumerate(knots_xy_phi):
    ax.text(x+0.1, y+0.2, f'{index+1}', fontsize=25, ha='left', c='black')
plt.xlim([-104,-90])
plt.ylim([30,47])
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
plt.xlabel('longitude', fontsize = 50)
plt.ylabel('latitude', fontsize = 50)
plt.title(r'$\phi(s)$ surface', fontsize = 50)
plt.savefig('Surface_phi.pdf')
plt.show()
plt.close()

plt.subplots(figsize = (12,8), constrained_layout=True)
plt.hist(phi_vec_for_plot, color = 'grey', edgecolor = 'white')
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title('Posterior Samples of $\phi(s)$', fontsize = 50)
plt.xlabel(rf'$\phi(s)$', fontsize = 50)
plt.savefig('Surface_phi_hist.pdf')
plt.show()
plt.close()

# range surface
range_vec_for_plot = gaussian_weight_matrix_for_plot_rho @ range_mean
vmin = 0.0
vmax = my_ceil(max(range_vec_for_plot),2)
fig, ax = plt.subplots(figsize = (10,8), constrained_layout=True)
ax.set_aspect('equal', 'box')
state_map.boundary.plot(ax=ax, color = 'black')
heatmap = ax.imshow(range_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x),
                    vmin = vmin, vmax = vmax, cmap = warm_cmap, interpolation='nearest', 
                    origin = 'lower', extent = [minX, maxX, minY, maxY])
ax.set_xticks(np.linspace(minX, maxX,num=3))
ax.set_yticks(np.linspace(minY, maxY,num=3))
cbar = fig.colorbar(heatmap, ax=ax)
cbar.ax.tick_params(labelsize=40)
cbar.locator = mpl.ticker.MaxNLocator(nbins=5)
cbar.update_ticks()
# Plot knots and circles
for i in range(k_rho):
    circle_i = plt.Circle((knots_xy_rho[i, 0], knots_xy_rho[i, 1]), radius_from_knots[i],
                        color='r', fill=False, fc='None', ec='lightgrey')
    ax.add_patch(circle_i)
# Scatter plot for sites and knots
ax.scatter(knots_x_rho, knots_y_rho, marker='+', c='black', label='knot', s=500)
for index, (x, y) in enumerate(knots_xy_rho):
    ax.text(x+0.1, y+0.2, f'{index+1}', fontsize=25, ha='left', c='black')
plt.xlim([-104,-90])
plt.ylim([30,47])
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
plt.xlabel('longitude', fontsize = 50)
plt.ylabel('latitude', fontsize = 50)
plt.title(r'$\rho(s)$ surface', fontsize = 50)
plt.savefig('Surface_range.pdf')
plt.show()
plt.close()

plt.subplots(figsize = (12,8), constrained_layout=True)
plt.hist(range_vec_for_plot, color = 'grey', edgecolor = 'white')
plt.xticks(fontsize=40)
plt.yticks(np.arange(1000, 10000, 2000), fontsize=40)
plt.title('Posterior Samples of $\\rho(s)$', fontsize = 50)
plt.xlabel(rf'$\rho(s)$', fontsize = 50)
plt.savefig('Surface_range_hist.pdf')
plt.show()
plt.close()

# %% Empirical chi of dataset, mean of the chi using per MCMC iter fitted GEV

"""
Moving window empirical chi plot, using mean of per MCMC iter fitted GEV at the observation sites
"""

if not fixGEV: 
    
    # these are the per iteration marginal parameters
    Beta_mu0_trace_thin100      = Beta_mu0_trace[0:iter:100,:]
    Beta_mu1_trace_thin100      = Beta_mu1_trace[0:iter:100,:]
    Beta_logsigma_trace_thin100 = Beta_logsigma_trace[0:iter:100,:]
    Beta_ksi_trace_thin100      = Beta_ksi_trace[0:iter:100,:]

    mu0_fitted_matrix_thin100   = (C_mu0.T @ Beta_mu0_trace_thin100.T).T # shape (n, test_Ns, Nt)
    mu1_fitted_matrix_thin100   = (C_mu1.T @ Beta_mu1_trace_thin100.T).T # shape (n, test_Ns, Nt)
    mu_fitted_matrix_thin100    = mu0_fitted_matrix_thin100 + mu1_fitted_matrix_thin100 * Time
    sigma_fitted_matrix_thin100 = np.exp((C_logsigma.T @ Beta_logsigma_trace_thin100.T).T)
    ksi_fitted_matrix_thin100   = (C_ksi.T @ Beta_ksi_trace_thin100.T).T

    n_thin100  = Beta_mu0_trace_thin100.shape[0]

    pY_mcmc = np.full(shape = (n_thin100, Ns, Nt), fill_value = np.nan)
    for i in range(n_thin100): # this should be parallelize too
        for t in range(Nt):
            pY_mcmc[i,:,t] = pgev(Y[:,t], 
                                  mu_fitted_matrix_thin100[i,:,t],
                                  sigma_fitted_matrix_thin100[i,:,t],
                                  ksi_fitted_matrix_thin100[i,:,t])

    # place knots for chi plot
    res_x_chi = 9
    res_y_chi = 19
    k_chi = res_x_chi * res_y_chi # number of knots
    # create one-dimensional arrays for x and y
    x_pos_chi = np.linspace(minX, maxX, res_x_chi+2)[2:-2]
    y_pos_chi = np.linspace(minY, maxY, res_y_chi+2)[2:-2]
    # create the mesh based on these arrays
    X_pos_chi, Y_pos_chi = np.meshgrid(x_pos_chi,y_pos_chi)
    knots_xy_chi = np.vstack([X_pos_chi.ravel(), Y_pos_chi.ravel()]).T
    knots_x_chi = knots_xy_chi[:,0]
    knots_y_chi = knots_xy_chi[:,1]   

    rect_width = (knots_xy_chi[0][0] - minX)*2
    rect_height = (knots_xy_chi[0][1] - minY)*2

    # Plot chi with same h in same figure

    e_abs = 0.2

    # Create a LinearSegmentedColormap from white to red
    colors = ["#ffffff", "#ff0000"]
    min_chi = 0.0
    max_chi = 0.5
    n_bins = 100  # Number of discrete bins
    n_ticks = 10
    cmap_name = "white_to_red"
    colormap = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    ticks = np.linspace(min_chi, max_chi, n_ticks+1).round(3)

    def calculate_chi_mat(args):
        pY, h, u = args
        # print('h:',h,'u:',u)

        h_low = h * (1 - e_abs)
        h_up  = h * (1 + e_abs)

        chi_mat2 = np.full(shape = (len(y_pos_chi), len(x_pos_chi)), fill_value = np.nan)

        for i in range(knots_xy_chi.shape[0]):

            # select sites within the rectangle
            rect_left   = knots_xy_chi[i][0] - rect_width/2
            rect_right  = knots_xy_chi[i][0] + rect_width/2
            rect_top    = knots_xy_chi[i][1] + rect_height/2
            rect_bottom = knots_xy_chi[i][1] - rect_height/2
            sites_in_rect_mask = np.logical_and(np.logical_and(rect_left <= sites_x, sites_x <= rect_right), 
                                                np.logical_and(rect_bottom <= sites_y, sites_y <= rect_top))
            sites_in_rect = sites_xy[sites_in_rect_mask]

            # calculate the distance between sites inside rectangle (coords --> km)
            n_sites = sites_in_rect.shape[0]
            sites_dist_mat = np.full(shape = (n_sites, n_sites), fill_value = np.nan)
            for si in range(n_sites):
                for sj in range(n_sites):
                    sites_dist_mat[si,sj] = coord_to_dist(sites_in_rect[si], sites_in_rect[sj])

            # select pairs: sites that are ~h km apart
            sites_h_mask = np.logical_and(np.triu(sites_dist_mat) > h_low,
                                        np.triu(sites_dist_mat) < h_up)
            n_pairs = len(np.triu(sites_dist_mat)[sites_h_mask])
            site_pairs_to_check = [(np.where(sites_h_mask)[0][i], np.where(sites_h_mask)[1][i]) for i in range(n_pairs)]

            # large pairs
            # Y_in_rect     = Y[sites_in_rect_mask]
            pY_in_rect    = pY[sites_in_rect_mask]

            # Calculate empirical chi
            count_co_extreme = 0
            for site_pair in site_pairs_to_check:
                # for this pair, over time, how many co-occured extremes?
                count_co_extreme += np.sum(np.logical_and(pY_in_rect[site_pair[0]] >= u,
                                                        pY_in_rect[site_pair[1]] >= u))
            prob_joint_ext = count_co_extreme / (n_pairs * Nt) # numerator
            prob_uni_ext   = np.mean(pY_in_rect >= u)          # denominator
            chi            = prob_joint_ext / prob_uni_ext     # emipircal Chi
            if np.isnan(chi): chi = 0

            # chi_mat[i % len(x_pos_chi), i // len(x_pos_chi)] = chi
            chi_mat2[-1 - i // len(x_pos_chi), i % len(x_pos_chi)] = chi
        
        return chi_mat2


    for h in [75, 150, 225]:

        fig, axes = plt.subplots(1,3)
        fig.set_size_inches(10,6)

        for ax_id, u in enumerate([0.9, 0.95, 0.99]):

            args_list = [(pY_mcmc[i,:,:],h,u) for i in range(n_thin100)]
            with multiprocessing.Pool(processes=N_CORES) as pool:
                results = pool.map(calculate_chi_mat, args_list)
            chi_mats = np.array(results)
            chi_mat_mean = np.mean(chi_mats, axis = 0)


            ax = axes[ax_id]
            ax.set_aspect('equal', 'box')
            state_map.boundary.plot(ax=ax, color = 'black', linewidth = 0.5)
            heatmap = ax.imshow(chi_mat_mean, cmap = colormap, vmin = min_chi, vmax = max_chi,
                                interpolation='nearest', 
                                extent = [min(x_pos_chi - rect_width/8), max(x_pos_chi + rect_width/8), 
                                        min(y_pos_chi - rect_height/8), max(y_pos_chi+rect_height/8)])
            # ax.scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.8)
            ax.scatter(knots_x_chi, knots_y_chi, s = 15, color = 'white', marker = '+')
            ax.set_xlim(-101,-93)
            ax.set_ylim(32.5, 45)
            ax.tick_params(axis='both', which='major', labelsize=14)

            ax.title.set_text(rf'$\chi_{{{u}}}$')
            ax.title.set_fontsize(20)
            # ax.title.set_text(rf'$\chi_{{{u}}}$, h $\approx$ {h}km', fontsize = 20)

        fig.subplots_adjust(right=0.8)
        fig.text(0.5, 0.825, rf'h $\approx$ {h}km', ha='center', fontsize = 20)
        fig.text(0.5, 0.125, 'Longitude', ha='center', fontsize = 20)
        fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical', fontsize = 20)
        cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
        colorbar = fig.colorbar(heatmap, cax = cbar_ax, ticks = ticks)
        colorbar.ax.tick_params(labelsize=14)
        plt.savefig('Surface_mean_empirical_chi_fittedGEV_h={}.pdf'.format(h), bbox_inches='tight')
        plt.show()
        plt.close()

# %% Empirical chi of dataset, initSmooth MLE GEV

if fixGEV:
    
    # these are the initial MLE GEV parameters smoothed
    Beta_mu0_initSmooth      = np.linalg.lstsq(a=C_mu0[:,:,0].T, b=mu0_estimates,rcond=None)[0]
    Beta_mu1_initSmooth      = np.linalg.lstsq(a=C_mu1[:,:,0].T, b=mu1_estimates,rcond=None)[0]
    Beta_logsigma_initSmooth = np.linalg.lstsq(a=C_logsigma[:,:,0].T, b=logsigma_estimates,rcond=None)[0]
    Beta_ksi_initSmooth      = np.linalg.lstsq(a=C_ksi[:,:,0].T, b=ksi_estimates,rcond=None)[0]

    mu0_initSmooth_matrix    = (C_mu0.T @ np.atleast_2d(Beta_mu0_initSmooth).T).T # shape (1, Ns, Nt)
    mu1_initSmooth_matrix    = (C_mu1.T @ np.atleast_2d(Beta_mu1_initSmooth).T).T # shape (1, Ns, Nt)
    mu_initSmooth_matrix     = mu0_initSmooth_matrix + mu1_initSmooth_matrix * Time
    sigma_initSmooth_matrix  = np.exp((C_logsigma.T @ np.atleast_2d(Beta_logsigma_initSmooth).T).T) 
    ksi_initSmooth_matrix    = (C_ksi.T @ np.atleast_2d(Beta_ksi_initSmooth).T).T

    pY_mcmc = np.full(shape = (1, Ns, Nt), fill_value = np.nan)
    for i in range(1):
        for t in range(Nt):
            pY_mcmc[i,:,t] = pgev(Y[:,t], 
                                  mu_initSmooth_matrix[i,:,t],
                                  sigma_initSmooth_matrix[i,:,t],
                                  ksi_initSmooth_matrix[i,:,t])

    # place knots for chi plot
    res_x_chi = 9
    res_y_chi = 19
    k_chi = res_x_chi * res_y_chi # number of knots
    # create one-dimensional arrays for x and y
    x_pos_chi = np.linspace(minX, maxX, res_x_chi+2)[2:-2]
    y_pos_chi = np.linspace(minY, maxY, res_y_chi+2)[2:-2]
    # create the mesh based on these arrays
    X_pos_chi, Y_pos_chi = np.meshgrid(x_pos_chi,y_pos_chi)
    knots_xy_chi = np.vstack([X_pos_chi.ravel(), Y_pos_chi.ravel()]).T
    knots_x_chi = knots_xy_chi[:,0]
    knots_y_chi = knots_xy_chi[:,1]   

    rect_width = (knots_xy_chi[0][0] - minX)*2
    rect_height = (knots_xy_chi[0][1] - minY)*2

    # Plot chi with same h in same figure

    e_abs = 0.2

    # Create a LinearSegmentedColormap from white to red
    colors = ["#ffffff", "#ff0000"]
    min_chi = 0.0
    max_chi = 0.5
    n_bins = 100  # Number of discrete bins
    n_ticks = 10
    cmap_name = "white_to_red"
    colormap = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    ticks = np.linspace(min_chi, max_chi, n_ticks+1).round(3)

    def calculate_chi_mat(args):
        pY, h, u = args
        # print('h:',h,'u:',u)

        h_low = h * (1 - e_abs)
        h_up  = h * (1 + e_abs)

        chi_mat2 = np.full(shape = (len(y_pos_chi), len(x_pos_chi)), fill_value = np.nan)

        for i in range(knots_xy_chi.shape[0]):

            # select sites within the rectangle
            rect_left   = knots_xy_chi[i][0] - rect_width/2
            rect_right  = knots_xy_chi[i][0] + rect_width/2
            rect_top    = knots_xy_chi[i][1] + rect_height/2
            rect_bottom = knots_xy_chi[i][1] - rect_height/2
            sites_in_rect_mask = np.logical_and(np.logical_and(rect_left <= sites_x, sites_x <= rect_right), 
                                                np.logical_and(rect_bottom <= sites_y, sites_y <= rect_top))
            sites_in_rect = sites_xy[sites_in_rect_mask]

            # calculate the distance between sites inside rectangle (coords --> km)
            n_sites = sites_in_rect.shape[0]
            sites_dist_mat = np.full(shape = (n_sites, n_sites), fill_value = np.nan)
            for si in range(n_sites):
                for sj in range(n_sites):
                    sites_dist_mat[si,sj] = coord_to_dist(sites_in_rect[si], sites_in_rect[sj])

            # select pairs: sites that are ~h km apart
            sites_h_mask = np.logical_and(np.triu(sites_dist_mat) > h_low,
                                        np.triu(sites_dist_mat) < h_up)
            n_pairs = len(np.triu(sites_dist_mat)[sites_h_mask])
            site_pairs_to_check = [(np.where(sites_h_mask)[0][i], np.where(sites_h_mask)[1][i]) for i in range(n_pairs)]

            # large pairs
            # Y_in_rect     = Y[sites_in_rect_mask]
            pY_in_rect    = pY[sites_in_rect_mask]

            # Calculate empirical chi
            count_co_extreme = 0
            for site_pair in site_pairs_to_check:
                # for this pair, over time, how many co-occured extremes?
                count_co_extreme += np.sum(np.logical_and(pY_in_rect[site_pair[0]] >= u,
                                                        pY_in_rect[site_pair[1]] >= u))
            prob_joint_ext = count_co_extreme / (n_pairs * Nt) # numerator
            prob_uni_ext   = np.mean(pY_in_rect >= u)          # denominator
            chi            = prob_joint_ext / prob_uni_ext     # emipircal Chi
            if np.isnan(chi): chi = 0

            # chi_mat[i % len(x_pos_chi), i // len(x_pos_chi)] = chi
            chi_mat2[-1 - i // len(x_pos_chi), i % len(x_pos_chi)] = chi
        
        return chi_mat2


    for h in [75, 150, 225]:

        fig, axes = plt.subplots(1,3)
        fig.set_size_inches(10,6)

        for ax_id, u in enumerate([0.9, 0.95, 0.99]):

            args_list = [(pY_mcmc[i,:,:],h,u) for i in range(1)]
            with multiprocessing.Pool(processes=N_CORES) as pool:
                results = pool.map(calculate_chi_mat, args_list)
            chi_mats = np.array(results)
            chi_mat_mean = np.mean(chi_mats, axis = 0)


            ax = axes[ax_id]
            ax.set_aspect('equal', 'box')
            state_map.boundary.plot(ax=ax, color = 'black', linewidth = 0.5)
            heatmap = ax.imshow(chi_mat_mean, cmap = colormap, vmin = min_chi, vmax = max_chi,
                                interpolation='nearest', 
                                extent = [min(x_pos_chi - rect_width/8), max(x_pos_chi + rect_width/8), 
                                        min(y_pos_chi - rect_height/8), max(y_pos_chi+rect_height/8)])
            # ax.scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.8)
            ax.scatter(knots_x_chi, knots_y_chi, s = 15, color = 'white', marker = '+')
            ax.set_xlim(-101,-93)
            ax.set_ylim(32.5, 45)
            ax.tick_params(axis='both', which='major', labelsize=14)

            ax.title.set_text(rf'$\chi_{{{u}}}$')
            ax.title.set_fontsize(20)
            # ax.title.set_text(rf'$\chi_{{{u}}}$, h $\approx$ {h}km', fontsize = 20)

        fig.subplots_adjust(right=0.8)
        fig.text(0.5, 0.825, rf'h $\approx$ {h}km', ha='center', fontsize = 20)
        fig.text(0.5, 0.125, 'Longitude', ha='center', fontsize = 20)
        fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical', fontsize = 20)
        cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
        colorbar = fig.colorbar(heatmap, cax = cbar_ax, ticks = ticks)
        colorbar.ax.tick_params(labelsize=14)
        plt.savefig('Surface_mean_empirical_chi_fittedGEV_h={}.pdf'.format(h), bbox_inches='tight')
        plt.show()
        plt.close()    

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
n_draw    = 100 # number of time replicates to draw
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
    u, phi, gamma = args
    return qRW(u, phi, gamma)

args_list090 = []
args_list095 = []
args_list099 = []
for i in range(Ns_chi):
    args_list090.append((0.9, phi_vec_chi[i], gamma_vec_chi[i]))
    args_list095.append((0.95, phi_vec_chi[i], gamma_vec_chi[i]))
    args_list099.append((0.99, phi_vec_chi[i], gamma_vec_chi[i]))
with multiprocessing.get_context('fork').Pool(processes = 50) as pool:
    results090 = list(tqdm(pool.imap(qRW_par, args_list090), total=len(args_list090), desc='u=0.90'))
with multiprocessing.get_context('fork').Pool(processes = 50) as pool:
    results095 = list(tqdm(pool.imap(qRW_par, args_list095), total=len(args_list095), desc='u=0.95')) #pool.map(qRW_par, args_list095)
with multiprocessing.get_context('fork').Pool(processes = 50) as pool:
    results099 = list(tqdm(pool.imap(qRW_par, args_list099), total=len(args_list099), desc='u=0.99')) #pool.map(qRW_par, args_list099)

qu_090 = np.array(results090)
np.save('qu_090', qu_090)
qu_095 = np.array(results095)
np.save('qu_095', qu_095)
qu_099 = np.array(results099)
np.save('qu_099', qu_099)

qu_all = (qu_090, qu_095, qu_099)

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
max_chi = 1.0
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
    plt.savefig('Surface_model_realization_empirical_chi_h={}.pdf'.format(h), bbox_inches='tight')
    plt.show()
    plt.close()


# %% Diagnostics and Model selection

#####################################################################
#####               Diagnostics with testing dataset            #####
#####################################################################

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

# %% Testing sample and splines
# Testing sample and splines

Beta_mu0_initSmooth      = np.linalg.lstsq(a=C_mu0[:,:,0].T, b=mu0_estimates,rcond=None)[0]
Beta_mu1_initSmooth      = np.linalg.lstsq(a=C_mu1[:,:,0].T, b=mu1_estimates,rcond=None)[0]
Beta_logsigma_initSmooth = np.linalg.lstsq(a=C_logsigma[:,:,0].T, b=logsigma_estimates,rcond=None)[0]
Beta_ksi_initSmooth      = np.linalg.lstsq(a=C_ksi[:,:,0].T, b=ksi_estimates,rcond=None)[0]

# Scatterplot of testing sites with station id

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
plt.close()

# # location of randomly selected stations

# fig, ax = plt.subplots()
# fig.set_size_inches(8,6)
# ax.set_aspect('equal', 'box')
# state_map.boundary.plot(ax=ax, color = 'black')
# ax.scatter(test_sites_xy[(23, 41, 82, 94), 0], test_sites_xy[(23, 41, 82, 94), 1], color='blue')  # Scatter plot of points
# labels = (23, 41, 82, 94)
# for index, (x, y) in enumerate(test_sites_xy[(23, 41, 82, 94),:]):
#     ax.text(x, y, f'{labels[index]}', fontsize=12, ha='right')
# plt.xlim([-102,-92])
# plt.ylim([32,45])
# ax.set_xticks(np.linspace(-102, -92,num=3))
# ax.set_yticks(np.linspace(32, 45,num=5))
# # plt.title('Scatter Plot with Labels')
# plt.xlabel('longitude', fontsize = 20)
# plt.ylabel('latitude', fontsize = 20)
# plt.savefig('out-of-sample stations.pdf',bbox_inches="tight")
# plt.show()

# Create a figure with two horizontally aligned subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [2, 1]})

ax = axes[0]
state_map.boundary.plot(ax=ax, color='lightgrey', zorder = 1)
ax.scatter(sites_x, sites_y, marker='.', c='blue', 
           edgecolor = 'white', label='training', zorder = 2, s = 50)
ax.scatter(test_sites_xy[:,0], test_sites_xy[:,1], marker='^', c='orange', 
           edgecolor='white', s = 50, label='testing', zorder=3)
space_rectangle = plt.Rectangle(xy=(minX, minY), width=maxX-minX, height=maxY-minY,
                                fill=False, color='black', linewidth=4)
ax.add_patch(space_rectangle)
# Set ticks and labels
ax.set_xticks(np.linspace(-130, -70, num=7))
ax.set_yticks(np.linspace(25, 50, num=6))
ax.tick_params(axis='both', which='major', labelsize=25)
ax.set_xlabel('Longitude', fontsize = 40)
ax.set_ylabel('Latitude', fontsize = 40)
ax.set_xlim([-130, -65])
ax.set_ylim([25,50])
ax.set_aspect('auto')

# customize legend
legend_sites = mpl.lines.Line2D([], [], color='blue', marker='.', markersize=30, label='training', linestyle='None')
legend_test_sites = mpl.lines.Line2D([], [], color='orange', marker='^', markersize=20, label='testing', linestyle='None')
ax.legend(handles=[legend_sites, legend_test_sites], 
          fontsize=40, loc = 'upper left')

ax = axes[1]
state_map.boundary.plot(ax=ax, color='lightgrey', zorder = 1)
ax.scatter(sites_x, sites_y, marker='.', c='blue', s = 150,
           edgecolor = 'white', label='training', zorder = 2)
ax.scatter(test_sites_xy[:,0], test_sites_xy[:,1], marker='^', c='orange', 
           edgecolor='white', s = 150, label='testing', zorder=3)
space_rectangle = plt.Rectangle(xy=(minX, minY), width=maxX-minX, height=maxY-minY,
                                fill=False, color='black', linewidth=4)
ax.add_patch(space_rectangle)
# Set ticks and labels
ax.set_xticks(np.linspace(minX, maxX, num=3))
ax.set_yticks(np.linspace(minY, maxY, num=5))
ax.tick_params(axis='both', which='major', labelsize=25)
ax.set_xlabel('Longitude', fontsize = 40)
ax.set_ylabel('Latitude', fontsize = 40)
ax.set_xlim([-105.5, -88.5])
ax.set_ylim([30.75,47.25])
ax.set_aspect('auto')

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the plots
plt.savefig('stations train and test combined.pdf',bbox_inches='tight')
plt.show()
plt.close()


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

# r('''set.seed(822)''')

for s in range(99):

    r.assign('s', int(s+1))

    # with MLE initial smooth
    r('''
        test_Ns <- 99
        # s <- floor(runif(1, min = 1, max = test_Ns + 1))
        print(test_sites_xy_ro[s,]) # print coordinates
        gumbel_s = sort(gumbel_pY_initSmooth_test_ro[s,])
        nquants = length(gumbel_s)
        emp_p = seq(1/nquants, 1-1/nquants, length=nquants)
        emp_q = qgumbel(emp_p)
        qq_gumbel_s <- extRemes::qqplot(gumbel_s, emp_q, regress=FALSE, legend=NULL,
                                        xlab="Observed", ylab="Gumbel", main=paste("GEVfit-QQPlot of Site:",s),
                                        lwd=3)
        pdf(file=paste("QQPlot_R_Test_initSmooth_Site_",s,".pdf", sep=""), width = 6, height = 5)
        par(mgp=c(1.75,0.75,0), mar=c(3,1,1,0), pty = "s")
        plot(type="n",qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$y,
            xlim = c(-2, 5), ylim = c(-2, 5),
            pch = 20, xlab="Observed", ylab="Gumbel", asp = 1)
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
            par(mgp=c(1.75,0.75,0), mar=c(3,1,1,0), pty = "s")
            # plot(type="n",qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$y, asp = 1,
            #     xlim = c(-2, 5), ylim = c(-2, 5),
            #     pch = 20, xlab="Observed", ylab="Gumbel", cex.lab = 1.75, cex.axis = 1.25)
            plot(type="n", x = c(0), y = c(0), asp = 1,
                xlim = c(-2, 5), ylim = c(-2, 5),
                pch = 20, xlab="Observed", ylab="Gumbel", cex.lab = 1.75, cex.axis = 1.25)            
            points(qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$y, pch=20)
            lines(qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$lower, lty=2, col="blue", lwd=3)
            lines(qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$upper, lty=2, col="blue", lwd=3)
            abline(a=0, b=1, lty=3, col="gray80", lwd=3)
            legend("topleft", lty=c(2, 3), lwd=3, legend=c("95% confidence bands", "1:1 line"), col=c("blue", "gray80"), bty="n",
                    cex = 1.5)
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
    with multiprocessing.Pool(processes=N_CORES) as pool:
        results = pool.map(qRW_pgev, args_list)
    for t in range(Nt):
        noNA = ~np.isnan(Y_99[:,t])
        X_99_thin100[i,noNA,t] = results[t]

# 5. loglikelihood -- calculation can also be parallelized! -----------------------------------------------------------

def ll(args):
    Y, X, Loc, Scale, Shape, phi, gamma, R, K_chol = args
    return(marg_transform_data_mixture_likelihood_1t(Y, X, Loc, Scale, Shape, phi, gamma, R, K_chol))

# Using per iteration for the parameters
ll_test_thin100 = np.full(shape= (n_thin100, test_Ns, Nt), fill_value = 0.0)
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
    with multiprocessing.Pool(processes=N_CORES) as pool:
        results = pool.map(ll, args_list)
    for t in range(Nt):
        noNA = ~np.isnan(Y_99[:,t])
        ll_test_thin100[i,noNA,t] = results[t]

# # Using the posterior mean for the parameters
# ll_test_thin100 = np.full(shape= (n_thin100, test_Ns, Nt), fill_value = 0.0)
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
#     with multiprocessing.Pool(processes=N_CORES) as pool:
#         results = pool.map(ll, args_list)
#     for t in range(Nt):
#         noNA = ~np.isnan(Y_99[:,t])
#         ll_test_thin100[i,noNA,t] = results[t]

np.save('ll_'+name, ll_test_thin100)

ax, fig = plt.subplots(figsize = (6,4))
plt.boxplot(np.sum(ll_test_thin100, axis = (1,2)))
# plt.xticks([1], [name])
plt.xlabel('Knot Radius Configuration')
plt.ylabel('log-likelihood @ test sites')
plt.savefig('ll_'+name+'_boxplot.pdf')
plt.show()
plt.close()

# %% Load Models' Predicative loglikelihoods --------------------------------------------------------------

# Make sure corresponding .npy files are put into the result folder

ll_HuserWadsworth  = np.sum(np.load('../results/HuserWadsworth/ll_HuserWadsworth.npy'), axis = (1,2))
ll_k13b4AI         = np.sum(np.load('../results/k13b4AI/ll_k13b4AI.npy'), axis = (1,2))
ll_k13r4b4         = np.sum(np.load('../results/k13r4b4/ll_k13r4b4.npy'), axis = (1,2))
ll_k13r4b4m        = np.sum(np.load('../results/k13r4b4m/ll_k13r4b4m.npy'), axis = (1,2))
ll_k25r2b067       = np.sum(np.load('../results/k25r2b0.67/ll_k25r2b0.67.npy'), axis = (1,2))
ll_k25r2b067m      = np.sum(np.load('../results/k25r2b0.67m/ll_k25r2b0.67m.npy'), axis = (1,2))
ll_k25r2b2         = np.sum(np.load('../results/k25r2b2/ll_k25r2b2.npy'), axis = (1,2))
ll_k25r2b2m        = np.sum(np.load('../results/k25r2b2m/ll_k25r2b2m.npy'), axis = (1,2))
ll_k25r4b4         = np.sum(np.load('../results/k25r4b4/ll_k25r4b4.npy'), axis = (1,2))
ll_k25r4b4m        = np.sum(np.load('../results/k25r4b4m/ll_k25r4b4m.npy'), axis = (1,2))
ll_k41r16b043      = np.sum(np.load('../results_tmp/k41r1.6b0.43/ll_k41r1.6b0.43.npy'), axis = (1,2))
ll_k41r16b043m     = np.sum(np.load('../results/k41r1.6b0.43m/ll_k41r1.6b0.43m.npy'), axis = (1,2))
ll_k41r2b067       = np.sum(np.load('../results_tmp/k41r2b0.67/ll_k41r2b0.67.npy'), axis = (1,2))
ll_k41r2b067m      = np.sum(np.load('../results/k41r2b0.67m/ll_k41r2b0.67m.npy'), axis = (1,2))

ll_list = [ll_HuserWadsworth, #ll_k13b4AI,
           ll_k13r4b4,    ll_k13r4b4m,
           ll_k25r2b067,  ll_k25r2b067m,
           ll_k25r2b2,    ll_k25r2b2m,
           ll_k25r4b4,    ll_k25r4b4m,
           ll_k41r16b043, ll_k41r16b043m,
           ll_k41r2b067,  ll_k41r2b067m]

# %% Vertical Boxplots
fig, ax = plt.subplots()
fig.set_size_inches((40,32))
bp = ax.boxplot(ll_list, patch_artist = True)

# Customize colors
colors = ['deepskyblue', #'deepskyblue',
          'deepskyblue', 'lightsalmon',
          'deepskyblue', 'lightsalmon',
          'deepskyblue', 'lightsalmon',
          'deepskyblue', 'lightsalmon',
          'deepskyblue', 'lightsalmon',
          'deepskyblue', 'lightsalmon']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
for whisker in bp['whiskers']:
    whisker.set(color='black', linewidth=1.5)
for cap in bp['caps']:
    cap.set(color='black', linewidth=1.5)
for median in bp['medians']:
    median.set(color='black', linewidth=3)
for flier in bp['fliers']:
    flier.set(marker='o', color='black', markerfacecolor = 'black')


# Customize axis labels and ticks
ax.set_xticklabels(['HuserWadsworth', #'k13b4AI',
                    'k13r4b4',      'k13r4b4m',
                    'k25r2b0.67',   'k25r2b0.67m',
                    'k25r2b2',      'k25r2b2m',
                    'k25r4b4',      'k25r4b4m',
                    'k41r1.6b0.43', 'k41r1.6b0.43m',
                    'k41r2b0.67',   'k41r2b0.67m'],
                   rotation = 45)
ax.set_ylabel('log likelihood', fontsize = 50)
ax.yaxis.offsetText.set_fontsize(50)
ax.tick_params(axis='both', which='major', labelsize=50)
fig.supxlabel('Models', fontsize = 50)
plt.title('Predicative log likelihoods of out-of-sample observations', 
          fontsize = 50)

# Add gridlines for better readability
ax.grid(True, which='both', linestyle='--',linewidth=0.7)

# Custom legend
legend_elements = [mpl.patches.Patch(facecolor='deepskyblue', edgecolor='black',
                                     label='no marginal restriction'),
                   mpl.patches.Patch(facecolor='lightsalmon', edgecolor='black',
                                     label='restriction on GEV params')]
ax.legend(handles=legend_elements, loc='lower right', fontsize = 50)

# plt.tight_layout()
plt.savefig('ll_boxplot_all.pdf', bbox_inches='tight')
plt.show()
plt.close()


# %% Draw Boxplot
fig, ax = plt.subplots()
fig.set_size_inches((40, 22))

bp = ax.boxplot(list(reversed(ll_list)), patch_artist=True, vert=False)

# Set colors
colors = list(reversed([
    'deepskyblue', 'deepskyblue', 'lightsalmon',
    'deepskyblue', 'lightsalmon',
    'deepskyblue', 'lightsalmon',
    'deepskyblue', 'lightsalmon',
    'deepskyblue', 'lightsalmon',
    'deepskyblue', 'lightsalmon'
]))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
for whisker in bp['whiskers']:
    whisker.set(color='black', linewidth=2)
for cap in bp['caps']:
    cap.set(color='black', linewidth=2)
for median in bp['medians']:
    median.set(color='black', linewidth=4)
for flier in bp['fliers']:
    flier.set(marker='o', color='black', markerfacecolor='black')

# Title and labels
plt.title('Predictive log likelihoods of out-of-sample observations', fontsize=70, pad=40)

# Adjust yticks and spacing
yticklabels = list(reversed([
    'H-W Stationary', 'k13r4b4', 'k13r4b4m',
    'k25r2b0.67', 'k25r2b0.67m',
    'k25r2b2', 'k25r2b2m',
    'k25r4b4', 'k25r4b4m',
    'k41r1.6b0.43', 'k41r1.6b0.43m',
    'k41r2b0.67', 'k41r2b0.67m'
]))
ax.set_yticklabels(yticklabels, fontsize=50)

fig.supylabel('Models', fontsize=70, x=0.005)
ax.set_xlabel('log likelihood', fontsize=70)
ax.tick_params(axis='both', which='major', labelsize=50)

# Turn off scientific notation, format x-axis labels directly
ax.xaxis.get_offset_text().set_visible(False)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e6:.3f}"))

# Optional: add a x-axis scale label manually
ax.annotate(r'$\times 10^6$', xy=(1, 0), xycoords=('axes fraction', 'axes fraction'),
            xytext=(-30, 40), textcoords='offset points',
            fontsize=50, ha='right', va='bottom')

# Add gridlines
ax.grid(True, which='both', linestyle='--', linewidth=1)

# Adjust layout
plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.08)

# Legend
legend_elements = [
    mpl.patches.Patch(facecolor='deepskyblue', edgecolor='black',
                      label='1-step Full Joint Model'),
    mpl.patches.Patch(facecolor='lightsalmon', edgecolor='black',
                      label='2-step Fixed GEV Margins')
]
ax.legend(handles=legend_elements, 
          loc='lower left', fontsize=50, frameon=True)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.savefig('ll_boxplot_all_horizontal.pdf', bbox_inches='tight')
plt.show()
plt.close()












































# %% Archive Code Bank

# Archive Code ----------------------------------------------------------------

# Model 1: k13_r4 -----------------------------------------

# folder           = './data_alpine/CONVERGED/20240306_realdata_t75_s590_k13_r4/'
# name             = 'k13_r4'
# fixGEV           = False
# radius           = 4
# bandwidth_phi    = 4
# bandwidth_rho    = 4
# N_outer_grid_phi = 9
# N_outer_grid_rho = 9
# mark             = True
# burnin           = 5000

# Model 2: k13_r4_fixGEV ----------------------------------

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

# Model 3: k25_r2 -----------------------------------------

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

# Model 4: k25_r2_fixGEV ----------------------------------

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

# Model 5: k25_r4 -----------------------------------------

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

# Model 6: k25_r4_fixGEV ----------------------------------

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

# Model 7: k25_efr2 ---------------------------------------

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

# Model 8: k25_efr2_fixksi --------------------------------

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


# Model 9: k41_efr2 ---------------------------------------

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

# Model 10: phik41efr2_rhok13r4 ---------------------------

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

# Model 11: Stationary Model ------------------------------

# folder           = './data/20241030start_realdata_stationary/'
# name             = 'k1_r100'
# fixGEV           = False
# radius           = 100
# bandwidth_phi    = 100
# bandwidth_rho    = 100
# N_outer_grid_phi = 1
# N_outer_grid_rho = 1
# mark             = True
# burnin           = 3000

# %% (Not used) marginal parameter surface scatterplot
# marginal parameter surface scatterplot

# if not fixGEV:
#     # side by side mu0
#     vmin = min(np.floor(min(mu0_estimates)), np.floor(min((C_mu0.T @ Beta_mu0_mean).T[:,0])))
#     vmax = max(np.ceil(max(mu0_estimates)), np.ceil(max((C_mu0.T @ Beta_mu0_mean).T[:,0])))
#     # mpnorm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
#     divnorm = mpl.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)

#     fig, ax     = plt.subplots(1,2)
#     mu0_scatter = ax[0].scatter(sites_x, sites_y, s = 10, c = mu0_estimates,
#                                 cmap = colormaps['bwr'], norm = divnorm)
#     ax[0].set_aspect('equal', 'box')
#     ax[0].title.set_text('mu0 data estimates')
#     mu0_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, c = (C_mu0.T @ Beta_mu0_mean).T[:,0],
#                                     cmap = colormaps['bwr'], norm = divnorm)
#     ax[1].set_aspect('equal', 'box')
#     ax[1].title.set_text('mu0 post mean estimates')
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     fig.colorbar(mu0_est_scatter, cax = cbar_ax)
#     plt.savefig('Surface:mu0.pdf')
#     plt.show()
#     plt.close()

#     # side by side mu1
#     vmin = min(np.floor(min(mu1_estimates)), np.floor(min((C_mu1.T @ Beta_mu1_mean).T[:,0])))
#     vmax = max(np.ceil(max(mu1_estimates)), np.ceil(max((C_mu1.T @ Beta_mu1_mean).T[:,0])))
#     # mpnorm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
#     divnorm = mpl.colors.TwoSlopeNorm(vcenter = 0, vmin = vmin, vmax = vmax)

#     fig, ax     = plt.subplots(1,2)
#     mu1_scatter = ax[0].scatter(sites_x, sites_y, s = 10, c = mu1_estimates,
#                                 cmap = colormaps['bwr'], norm = divnorm)
#     ax[0].set_aspect('equal', 'box')
#     ax[0].title.set_text('mu1 data estimates')
#     mu1_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, c = (C_mu1.T @ Beta_mu1_mean).T[:,0],
#                                     cmap = colormaps['bwr'], norm = divnorm)
#     ax[1].set_aspect('equal', 'box')
#     ax[1].title.set_text('mu1 post mean estimates')
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     fig.colorbar(mu1_est_scatter, cax = cbar_ax)
#     plt.savefig('Surface:mu1.pdf')
#     plt.show()
#     plt.close()

#     # side by side for mu = mu0 + mu1
#     year = 1999
#     year_adj = year - start_year
#     vmin = min(np.floor(min(mu0_estimates + mu1_estimates * Time[year_adj])), 
#             np.floor(min(((C_mu0.T @ Beta_mu0_mean).T + (C_mu1.T @ Beta_mu1_mean).T * Time)[:,year_adj])))
#     vmax = max(np.ceil(max(mu0_estimates + mu1_estimates * Time[year_adj])), 
#             np.ceil(max(((C_mu0.T @ Beta_mu0_mean).T + (C_mu1.T @ Beta_mu1_mean).T * Time)[:,year_adj])))
#     divnorm = mpl.colors.TwoSlopeNorm(vcenter = (vmin+vmax)/2, vmin = vmin, vmax = vmax)

#     fig, ax     = plt.subplots(1,2)
#     mu0_scatter = ax[0].scatter(sites_x, sites_y, s = 10, c = mu0_estimates + mu1_estimates * Time[year_adj],
#                                 cmap = colormaps['bwr'], norm = divnorm)
#     ax[0].set_aspect('equal', 'box')
#     ax[0].title.set_text('mu data year: ' + str(start_year+year_adj))
#     mu0_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, c = ((C_mu0.T @ Beta_mu0_mean).T + (C_mu1.T @ Beta_mu1_mean).T * Time)[:,year_adj],
#                                     cmap = colormaps['bwr'], norm = divnorm)
#     ax[1].set_aspect('equal', 'box')
#     ax[1].title.set_text('mu post mean year: ' + str(start_year+year_adj))
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     fig.colorbar(mu0_est_scatter, cax = cbar_ax)
#     plt.savefig('Surface:mu'+str(1949+year_adj)+'.pdf')
#     plt.show()
#     plt.close()

#     # side by side logsigma
#     vmin = min(my_floor(min(logsigma_estimates), 1), my_floor(min((C_logsigma.T @ Beta_logsigma_mean).T[:,0]), 1))
#     vmax = max(my_ceil(max(logsigma_estimates), 1), my_ceil(max((C_logsigma.T @ Beta_logsigma_mean).T[:,0]), 1))
#     divnorm = mpl.colors.TwoSlopeNorm(vcenter = (vmin+vmax)/2, vmin = vmin, vmax = vmax)

#     fig, ax     = plt.subplots(1,2)
#     logsigma_scatter = ax[0].scatter(sites_x, sites_y, s = 10, c = logsigma_estimates,
#                                 cmap = colormaps['bwr'], norm = divnorm)
#     ax[0].set_aspect('equal', 'box')
#     ax[0].title.set_text('logsigma data estimates')
#     logsigma_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, c = (C_logsigma.T @ Beta_logsigma_mean).T[:,0],
#                                     cmap = colormaps['bwr'], norm = divnorm)
#     ax[1].set_aspect('equal', 'box')
#     ax[1].title.set_text('logsigma post mean estimates')
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     fig.colorbar(logsigma_est_scatter, cax = cbar_ax)
#     plt.savefig('Surface:logsigma.pdf')
#     plt.show()
#     plt.close()

#     try:
#         # side by side ksi
#         vmin = min(my_floor(min(ksi_estimates), 1), my_floor(min((C_ksi.T @ Beta_ksi_mean).T[:,0]), 1))
#         vmax = max(my_ceil(max(ksi_estimates), 1), my_ceil(max((C_ksi.T @ Beta_ksi_mean).T[:,0]), 1))
#         divnorm = mpl.colors.TwoSlopeNorm(vcenter = (vmin+vmax)/2, vmin = vmin, vmax = vmax)

#         fig, ax     = plt.subplots(1,2)
#         ksi_scatter = ax[0].scatter(sites_x, sites_y, s = 10, c = ksi_estimates,
#                                     cmap = colormaps['bwr'], norm = divnorm)
#         ax[0].set_aspect('equal', 'box')
#         ax[0].title.set_text('ksi data estimates')
#         ksi_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, c = (C_ksi.T @ Beta_ksi_mean).T[:,0],
#                                         cmap = colormaps['bwr'], norm = divnorm)
#         ax[1].set_aspect('equal', 'box')
#         ax[1].title.set_text('ksi post mean estimates')
#         fig.subplots_adjust(right=0.8)
#         cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#         fig.colorbar(ksi_est_scatter, cax = cbar_ax)
#         plt.savefig('Surface:ksi.pdf')
#         plt.show()
#         plt.close()
#     except:
#         pass

# %% (Not used) Externally Smooth GEV Surface

# # mu0
# vmin = np.floor(min((C_mu0.T @ Beta_mu0_mean).T[:,0]))
# vmax = np.ceil(max((C_mu0.T @ Beta_mu0_mean).T[:,0]))
# divnorm = mpl.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)
# # scatter data
# x = sites_x
# y = sites_y
# z = (C_mu0.T @ Beta_mu0_mean).T[:,0]
# # grid for the heatmap
# xi = plotgrid_X
# yi = plotgrid_Y
# # interpolate z values over the grid
# zi = scipy.interpolate.griddata((x,y), z, (xi, yi), method='cubic')
# # create a figure and set a size
# fig, ax = plt.subplots()
# fig.set_size_inches(8,6)
# ax.set_aspect('equal','box')
# # plot the smoothed surface as a heatmap
# state_map.boundary.plot(ax=ax, color = 'black')
# heatmap = ax.imshow(zi, extent=[min(x), max(x), min(y), max(y)], origin='lower', cmap = colormaps['bwr'], norm = divnorm)
# ax.set_xticks(np.linspace(minX, maxX,num=3))
# ax.set_yticks(np.linspace(minY, maxY,num=5))
# cbar = fig.colorbar(heatmap, ax = ax)
# cbar.ax.tick_params(labelsize=20)
# # Overlay the original scatter plot
# ax.scatter(x, y, c=z, edgecolors='black', linewidths=0.1,
#             s = 10, cmap = colormaps['bwr'], norm = divnorm)
# plt.xlim([-104,-90])
# plt.ylim([30,47])
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# plt.xlabel('longitude', fontsize = 20)
# plt.ylabel('latitude', fontsize = 20)
# plt.title(r'Posterior mean $\mu_0$ surface', fontsize = 20)
# plt.savefig('Surface:mu0_smooth.pdf', bbox_inches='tight')
# plt.show()
# plt.close()


# # mu1
# vmin = np.floor(min((C_mu1.T @ Beta_mu1_mean).T[:,0]))
# vmax = np.ceil(max((C_mu1.T @ Beta_mu1_mean).T[:,0]))
# tmp_bound = max(np.abs(vmin), np.abs(vmax))
# divnorm = mpl.colors.TwoSlopeNorm(vcenter = 0, vmin = -tmp_bound, vmax = tmp_bound)
# # scatter data
# x = sites_x
# y = sites_y
# z = (C_mu1.T @ Beta_mu1_mean).T[:,0]
# # grid for the heatmap
# xi = plotgrid_X
# yi = plotgrid_Y
# # interpolate z values over the grid
# zi = scipy.interpolate.griddata((x,y), z, (xi, yi), method='cubic')
# # create a figure and set a size
# fig, ax = plt.subplots()
# fig.set_size_inches(8,6)
# ax.set_aspect('equal','box')
# # plot the smoothed surface as a heatmap
# state_map.boundary.plot(ax=ax, color = 'black')
# heatmap = ax.imshow(zi, extent=[min(x), max(x), min(y), max(y)], origin='lower', cmap = colormaps['bwr'], norm = divnorm)
# ax.set_xticks(np.linspace(minX, maxX,num=3))
# ax.set_yticks(np.linspace(minY, maxY,num=5))
# cbar = fig.colorbar(heatmap, ax = ax)
# cbar.ax.tick_params(labelsize=20)
# # Overlay the original scatter plot
# ax.scatter(x, y, c=z, edgecolors='black', linewidths=0.1,
#             s = 10, cmap = colormaps['bwr'], norm = divnorm)
# plt.xlim([-104,-90])
# plt.ylim([30,47])
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# plt.xlabel('longitude', fontsize = 20)
# plt.ylabel('latitude', fontsize = 20)
# plt.title(r'Posterior mean $\mu_1$ surface', fontsize = 20)
# plt.savefig('Surface:mu1_smooth.pdf', bbox_inches='tight')
# plt.show()
# plt.close()

# # logsigma
# vmin = my_floor(min((C_logsigma.T @ Beta_logsigma_mean).T[:,0]), 1)
# vmax = my_ceil(max((C_logsigma.T @ Beta_logsigma_mean).T[:,0]), 1)
# divnorm = mpl.colors.TwoSlopeNorm(vcenter = (vmin+vmax)/2, vmin = vmin, vmax = vmax)
# # scatter data
# x = sites_x
# y = sites_y
# z = (C_logsigma.T @ Beta_logsigma_mean).T[:,0]
# # grid for the heatmap
# xi = plotgrid_X
# yi = plotgrid_Y
# # interpolate z values over the grid
# zi = scipy.interpolate.griddata((x,y), z, (xi, yi), method='cubic')
# # create a figure and set a size
# fig, ax = plt.subplots()
# fig.set_size_inches(8,6)
# ax.set_aspect('equal','box')
# # plot the smoothed surface as a heatmap
# state_map.boundary.plot(ax=ax, color = 'black')
# heatmap = ax.imshow(zi, extent=[min(x), max(x), min(y), max(y)], origin='lower', cmap = colormaps['bwr'], norm = divnorm)
# ax.set_xticks(np.linspace(minX, maxX,num=3))
# ax.set_yticks(np.linspace(minY, maxY,num=5))
# cbar = fig.colorbar(heatmap, ax = ax)
# cbar.ax.tick_params(labelsize=20)
# # Overlay the original scatter plot
# ax.scatter(x, y, c=z, edgecolors='black', linewidths=0.1,
#             s = 10, cmap = colormaps['bwr'], norm = divnorm)
# plt.xlim([-104,-90])
# plt.ylim([30,47])
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# plt.xlabel('longitude', fontsize = 20)
# plt.ylabel('latitude', fontsize = 20)
# plt.title(r'Posterior mean $\log(\sigma)$ surface', fontsize = 20)
# plt.savefig('Surface:logsigma_smooth.pdf', bbox_inches='tight')
# plt.show()
# plt.close()

# try:
#     # ksi
#     vmin = my_floor(min((C_ksi.T @ Beta_ksi_mean).T[:,0]), 2)
#     vmax = my_ceil(max((C_ksi.T @ Beta_ksi_mean).T[:,0]), 2)
#     divnorm = mpl.colors.TwoSlopeNorm(vcenter = (vmin+vmax)/2, vmin = vmin, vmax = vmax)
#     # scatter data
#     x = sites_x
#     y = sites_y
#     z = (C_ksi.T @ Beta_ksi_mean).T[:,0]
#     # grid for the heatmap
#     xi = plotgrid_X
#     yi = plotgrid_Y
#     # interpolate z values over the grid
#     zi = scipy.interpolate.griddata((x,y), z, (xi, yi), method='cubic')
#     # create a figure and set a size
#     fig, ax = plt.subplots()
#     fig.set_size_inches(8,6)
#     ax.set_aspect('equal','box')
#     # plot the smoothed surface as a heatmap
#     state_map.boundary.plot(ax=ax, color = 'black')
#     heatmap = ax.imshow(zi, extent=[min(x), max(x), min(y), max(y)], origin='lower', cmap = colormaps['bwr'], norm = divnorm)
#     ax.set_xticks(np.linspace(minX, maxX,num=3))
#     ax.set_yticks(np.linspace(minY, maxY,num=5))
#     cbar = fig.colorbar(heatmap, ax = ax)
#     cbar.ax.tick_params(labelsize=20)
#     # Overlay the original scatter plot
#     ax.scatter(x, y, c=z, edgecolors='black', linewidths=0.1,
#                 s = 10, cmap = colormaps['bwr'], norm = divnorm)
#     plt.xlim([-104,-90])
#     plt.ylim([30,47])
#     plt.xticks(fontsize = 20)
#     plt.yticks(fontsize = 20)
#     plt.xlabel('longitude', fontsize = 20)
#     plt.ylabel('latitude', fontsize = 20)
#     plt.title(r'Posterior mean $\xi$ surface', fontsize = 20)
#     plt.savefig('Surface:ksi_smooth.pdf', bbox_inches='tight')
#     plt.show()
#     plt.close()
# except:
#     pass



# %% (Not used) Empirical chi of dataset, using model fitted posterior mean GEV

# """
# Moving window empirical chi plot, using fitted (posterior mean) GEV at the observation sites
# """

# mu0_fitted      = (C_mu0.T @ Beta_mu0_mean).T[:,0]
# mu1_fitted      = (C_mu1.T @ Beta_mu1_mean).T[:,0]
# logsigma_fitted = (C_logsigma.T @ Beta_logsigma_mean).T[:,0]
# ksi_fitted      = (C_ksi.T @ Beta_ksi_mean).T[:,0]

# if not fixGEV:
#     pY   = np.full(shape = (Ns, Nt), fill_value = np.nan)
#     for t in range(Nt):
#         pY[:,t] = pgev(Y[:,t], mu0_fitted + mu1_fitted * Time[t],
#                             np.exp(logsigma_fitted),
#                             ksi_fitted)

#     # place knots for chi plot
#     res_x_chi = 9
#     res_y_chi = 19
#     k_chi = res_x_chi * res_y_chi # number of knots
#     # create one-dimensional arrays for x and y
#     x_pos_chi = np.linspace(minX, maxX, res_x_chi+2)[2:-2]
#     y_pos_chi = np.linspace(minY, maxY, res_y_chi+2)[2:-2]
#     # create the mesh based on these arrays
#     X_pos_chi, Y_pos_chi = np.meshgrid(x_pos_chi,y_pos_chi)
#     knots_xy_chi = np.vstack([X_pos_chi.ravel(), Y_pos_chi.ravel()]).T
#     knots_x_chi = knots_xy_chi[:,0]
#     knots_y_chi = knots_xy_chi[:,1]   

#     rect_width = (knots_xy_chi[0][0] - minX)*2
#     rect_height = (knots_xy_chi[0][1] - minY)*2

#     # Plot chi with same h in same figure

#     # u = 0.99 # 0.9, 0.95, 0.99
#     # h = 225 # 75, 150, 225
#     e_abs = 0.2

#     # Create a LinearSegmentedColormap from white to red
#     colors = ["#ffffff", "#ff0000"]
#     min_chi = 0.0
#     max_chi = 1.0
#     n_bins = 100  # Number of discrete bins
#     n_ticks = 10
#     cmap_name = "white_to_red"
#     colormap = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
#     ticks = np.linspace(min_chi, max_chi, n_ticks+1).round(3)

#     for h in [75, 150, 225]:

#         fig, axes = plt.subplots(1,3)
#         fig.set_size_inches(10,6)

#         for ax_id, u in enumerate([0.9, 0.95, 0.99]):

#             h_low = h * (1 - e_abs)
#             h_up  = h * (1 + e_abs)

#             # e_abs = 20
#             # h_low = h - e_abs
#             # h_up  = h + e_abs

#             chi_mat = np.full(shape = (len(x_pos_chi), len(y_pos_chi)), fill_value = np.nan)
#             chi_mat2 = np.full(shape = (len(y_pos_chi), len(x_pos_chi)), fill_value = np.nan)

#             for i in range(knots_xy_chi.shape[0]):

#                 # select sites within the rectangle
#                 rect_left   = knots_xy_chi[i][0] - rect_width/2
#                 rect_right  = knots_xy_chi[i][0] + rect_width/2
#                 rect_top    = knots_xy_chi[i][1] + rect_height/2
#                 rect_bottom = knots_xy_chi[i][1] - rect_height/2
#                 sites_in_rect_mask = np.logical_and(np.logical_and(rect_left <= sites_x, sites_x <= rect_right), 
#                                                     np.logical_and(rect_bottom <= sites_y, sites_y <= rect_top))
#                 sites_in_rect = sites_xy[sites_in_rect_mask]

#                 # calculate the distance between sites inside rectangle (coords --> km)
#                 n_sites = sites_in_rect.shape[0]
#                 sites_dist_mat = np.full(shape = (n_sites, n_sites), fill_value = np.nan)
#                 for si in range(n_sites):
#                     for sj in range(n_sites):
#                         sites_dist_mat[si,sj] = coord_to_dist(sites_in_rect[si], sites_in_rect[sj])

#                 # select pairs: sites that are ~h km apart
#                 sites_h_mask = np.logical_and(np.triu(sites_dist_mat) > h_low,
#                                             np.triu(sites_dist_mat) < h_up)
#                 n_pairs = len(np.triu(sites_dist_mat)[sites_h_mask])
#                 site_pairs_to_check = [(np.where(sites_h_mask)[0][i], np.where(sites_h_mask)[1][i]) for i in range(n_pairs)]

#                 # large pairs
#                 Y_in_rect     = Y[sites_in_rect_mask]
#                 pY_in_rect    = pY[sites_in_rect_mask]

#                 # Calculate empirical chi
#                 count_co_extreme = 0
#                 for site_pair in site_pairs_to_check:
#                     # for this pair, over time, how many co-occured extremes?
#                     count_co_extreme += np.sum(np.logical_and(pY_in_rect[site_pair[0]] >= u,
#                                                             pY_in_rect[site_pair[1]] >= u))
#                 prob_joint_ext = count_co_extreme / (n_pairs * Nt) # numerator
#                 prob_uni_ext   = np.mean(pY_in_rect >= u)          # denominator
#                 chi            = prob_joint_ext / prob_uni_ext     # emipircal Chi
#                 if np.isnan(chi): chi = 0

#                 chi_mat[i % len(x_pos_chi), i // len(x_pos_chi)] = chi
#                 chi_mat2[-1 - i // len(x_pos_chi), i % len(x_pos_chi)] = chi

#             assert np.all(chi_mat2 <= max_chi)

#             ax = axes[ax_id]
#             ax.set_aspect('equal', 'box')
#             state_map.boundary.plot(ax=ax, color = 'black', linewidth = 0.5)
#             heatmap = ax.imshow(chi_mat2, cmap = colormap, vmin = 0.0, vmax = 1.0,
#                                 interpolation='nearest', 
#                                 extent = [min(x_pos_chi - rect_width/8), max(x_pos_chi + rect_width/8), 
#                                         min(y_pos_chi - rect_height/8), max(y_pos_chi+rect_height/8)])
#             # ax.scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.8)
#             ax.scatter(knots_x_chi, knots_y_chi, s = 15, color = 'white', marker = '+')
#             ax.set_xlim(-101,-93)
#             ax.set_ylim(32.5, 45)
#             ax.tick_params(axis='both', which='major', labelsize=14)

#             ax.title.set_text(rf'$\chi_{{{u}}}$')
#             ax.title.set_fontsize(20)
#             # ax.title.set_text(rf'$\chi_{{{u}}}$, h $\approx$ {h}km', fontsize = 20)

#         fig.subplots_adjust(right=0.8)
#         fig.text(0.5, 0.825, rf'h $\approx$ {h}km', ha='center', fontsize = 20)
#         fig.text(0.5, 0.125, 'Longitude', ha='center', fontsize = 20)
#         fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical', fontsize = 20)
#         cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
#         colorbar = fig.colorbar(heatmap, cax = cbar_ax, ticks = ticks)
#         colorbar.ax.tick_params(labelsize=14)
#         plt.savefig('Surface:empirical_chi_fittedGEV_h={}.pdf'.format(h), bbox_inches='tight')
#         plt.show()
#         plt.close()      





# %% (Not Used) Bivariate Z Empirical Model estimated chi plot -----------------------------------------------------------------------------------------------
# Empirical Model estimated chi plot

# """
# like with the moving window empirical chi plot before, we assume local stationarity
# engineering two points inside that window, draw 100,000 bivariate gaussian, estiamte chi empirically
# then make a "moving window chi plot" -- one that is estimated by the model
# """

# # place knots for chi plot
# res_x_chi = 7
# res_y_chi = 17
# k_chi     = res_x_chi * res_y_chi # number of knots
# x_pos_chi = np.linspace(minX, maxX, res_x_chi+4)[2:-2]
# y_pos_chi = np.linspace(minY, maxY, res_y_chi+4)[2:-2]
# X_pos_chi, Y_pos_chi = np.meshgrid(x_pos_chi,y_pos_chi) # create the mesh based on these arrays
# knots_xy_chi = np.vstack([X_pos_chi.ravel(), Y_pos_chi.ravel()]).T
# knots_x_chi = knots_xy_chi[:,0]
# knots_y_chi = knots_xy_chi[:,1]

# # make a plot of the sites and knots
# fig, ax = plt.subplots()
# fig.set_size_inches(6,8)
# ax.set_aspect('equal', 'box')
# state_map.boundary.plot(ax=ax, color = 'black', linewidth = 0.5)
# ax.scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.5)
# ax.scatter(knots_x_chi, knots_y_chi, s = 15, color = 'blue', marker = '+')
# rect_width = (knots_xy_chi[0][0] - minX)*2
# rect_height = (knots_xy_chi[0][1] - minY)*2
# chi_i = 118 # select a rectangle (0, 1, ..., 118) to draw
# rect_i = plt.Rectangle((knots_xy_chi[chi_i][0] - rect_width/2, knots_xy_chi[chi_i][1] - rect_height/2), 
#                        width = rect_width, height = rect_height,
#                        fill = False, ec = 'black', linewidth = 2) # Rectangle location spsecified by lower left corner
# ax.add_patch(rect_i)
# plt.xlim([-105,-90])
# plt.ylim([30,50])
# plt.show()
# plt.close()

# # # function to calculate chi for a knot_chi, using posterior mean of R
# # def calc_chi(args):
# #     point_A, u, h = args
# #     point_B = random_point_at_dist(point_A, h)
# #     sites_AB = np.row_stack([point_A, point_B])
# #     gaussian_weight_matrix_AB = np.full(shape = (2, k), fill_value = np.nan)
# #     for site_id in np.arange(2):
# #         # Compute distance between each pair of the two collections of inputs
# #         d_from_knots = scipy.spatial.distance.cdist(XA = sites_AB[site_id,:].reshape((-1,2)), 
# #                                                     XB = knots_xy)
# #         # influence coming from each of the knots
# #         weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
# #         gaussian_weight_matrix_AB[site_id, :] = weight_from_knots
# #     wendland_weight_matrix_AB = np.full(shape = (2,k), fill_value = np.nan)
# #     for site_id in np.arange(2):
# #         # Compute distance between each pair of the two collections of inputs
# #         d_from_knots = scipy.spatial.distance.cdist(XA = sites_AB[site_id,:].reshape((-1,2)), 
# #                                                     XB = knots_xy)
# #         # influence coming from each of the knots
# #         weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
# #         wendland_weight_matrix_AB[site_id, :] = weight_from_knots

# #     # Need: R(s), phi(s), rho(s) --> K
# #     phi_vec_AB     = gaussian_weight_matrix_AB @ phi_mean
# #     range_vec_AB   = gaussian_weight_matrix_AB @ range_mean
# #     gamma_at_knots = np.repeat(0.5, k)
# #     alpha          = 0.5
# #     gamma_vec_AB   = np.sum(np.multiply(wendland_weight_matrix_AB, gamma_at_knots)**(alpha),
# #                             axis = 1)**(1/alpha)
# #     R_matrix_AB    = wendland_weight_matrix_AB @ np.exp(R_log_mean) # shape (k, Nt)
# #     sigsq_vec      = np.repeat(1.0, 2)
# #     nu             = 0.5
# #     K_AB           = ns_cov(range_vec = range_vec_AB,
# #                             sigsq_vec = sigsq_vec,
# #                             coords    = sites_AB,
# #                             kappa     = nu, cov_model = "matern")
# #     # cholesky_U_AB  = scipy.linalg.cholesky(K_AB, lower = False)

# #     # Draw a lot of bivariate Z --> X
# #     n_draw  = 10000
# #     Z_bivar = np.full(shape = (n_draw, 2, Nt), fill_value = np.nan)
# #     for i in range(Nt):
# #         Z_bivar[:,:,i] = scipy.stats.multivariate_normal.rvs(mean = None, cov = K_AB, size = n_draw)
# #     W_bivar  = norm_to_Pareto(Z_bivar)
# #     X_bivar  = np.full(shape = (n_draw, 2, Nt), fill_value = np.nan)
# #     for i in range(n_draw):
# #         X_bivar[i,:,:] = (R_matrix_AB.T ** phi_vec_AB).T * W_bivar[i,:,:]

# #     # calculate chi
# #     #     Calculate F(X) is costly, just calculate threshold once and use threshold
# #     u_AB = qRW(u, phi_vec_AB, gamma_vec_AB)

# #     # using theoretical denominator
# #     # chi = np.mean(np.logical_and(X_bivar[:,0,:] > u_AB[0], X_bivar[:,1,:] > u_AB[1])) / (1-u)
# #     # using empirical denominator
# #     chi = np.mean(np.logical_and(X_bivar[:,0,:] > u_AB[0], X_bivar[:,1,:] > u_AB[1])) / np.mean(X_bivar[:,1,:] > u_AB[1])
# #     return chi

# """
# April 19, Ben: use R drawn from Stables directly, don't use posterior mean
# """
# # function to calculate chi for a knot_chi
# def calc_chi(args):
#     n_draw = 10000 # number of R and bivariate gaussian to draw
#     point_A, u, h = args

#     # engineer new point
#     point_B = random_point_at_dist(point_A, h)
#     sites_AB = np.row_stack([point_A, point_B])

#     # new weight matrices

#     gaussian_weight_matrix_AB_phi = np.full(shape = (2, k_phi), fill_value = np.nan)
#     for site_id in np.arange(2):
#         # Compute distance between each pair of the two collections of inputs
#         d_from_knots = scipy.spatial.distance.cdist(XA = sites_AB[site_id,:].reshape((-1,2)), 
#                                                     XB = knots_xy_phi)
#         # influence coming from each of the knots
#         weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_phi, cutoff = False)
#         gaussian_weight_matrix_AB_phi[site_id, :] = weight_from_knots
#     gaussian_weight_matrix_AB_rho = np.full(shape = (2, k_rho), fill_value = np.nan)
#     for site_id in np.arange(2):
#         # Compute distance between each pair of the two collections of inputs
#         d_from_knots = scipy.spatial.distance.cdist(XA = sites_AB[site_id,:].reshape((-1,2)), 
#                                                     XB = knots_xy_rho)
#         # influence coming from each of the knots
#         weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_rho, cutoff = False)
#         gaussian_weight_matrix_AB_rho[site_id, :] = weight_from_knots
#     wendland_weight_matrix_AB = np.full(shape = (2,k_phi), fill_value = np.nan)
#     for site_id in np.arange(2):
#         # Compute distance between each pair of the two collections of inputs
#         d_from_knots = scipy.spatial.distance.cdist(XA = sites_AB[site_id,:].reshape((-1,2)), 
#                                                     XB = knots_xy_phi)
#         # influence coming from each of the knots
#         weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
#         wendland_weight_matrix_AB[site_id, :] = weight_from_knots

#     # Need: R(s), phi(s), rho(s) --> K
#     phi_vec_AB     = gaussian_weight_matrix_AB_phi @ phi_mean
#     range_vec_AB   = gaussian_weight_matrix_AB_rho @ range_mean
#     gamma_at_knots = np.repeat(0.5, k_phi)
#     alpha          = 0.5
#     gamma_vec_AB   = np.sum(np.multiply(wendland_weight_matrix_AB, gamma_at_knots)**(alpha),
#                             axis = 1)**(1/alpha)
#     if any(np.isnan(gamma_vec_AB)):
#         print(wendland_weight_matrix_AB)
#     sigsq_vec      = np.repeat(1.0, 2)
#     nu             = 0.5
#     K_AB           = ns_cov(range_vec = range_vec_AB,
#                             sigsq_vec = sigsq_vec,
#                             coords    = sites_AB,
#                             kappa     = nu, cov_model = "matern")
#     # cholesky_U_AB  = scipy.linalg.cholesky(K_AB, lower = False)

#     # Draw R and bivariate Z
#     S_vec   = np.array([scipy.stats.levy.rvs(loc = 0, scale = 0.5, size = k_phi) for _ in range(n_draw)])
#     Z_bivar = scipy.stats.multivariate_normal.rvs(mean = None, cov = K_AB, size = n_draw)

#     # calculate X
#     R_vec_AB = (wendland_weight_matrix_AB @ S_vec.T).T # shape (n_draw, 2)
#     W_bivar  = norm_to_Pareto(Z_bivar)
#     X_bivar  = (R_vec_AB ** phi_vec_AB) * W_bivar

#     # calculate chi - Calculate F(X) is costly, just calculate threshold once and use threshold
#     u_AB = qRW(u, phi_vec_AB, gamma_vec_AB)

#     # using theoretical denominator
#     # chi = np.mean(np.logical_and(X_bivar[:,0,:] > u_AB[0], X_bivar[:,1,:] > u_AB[1])) / (1-u)
#     # using empirical denominator
#     chi = np.mean(np.logical_and(X_bivar[:,0] > u_AB[0], X_bivar[:,1] > u_AB[1])) / np.mean(X_bivar[:,1] > u_AB[1])
#     if np.isnan(chi): chi = 0.0
#     return chi

# # Create a LinearSegmentedColormap from white to red
# colors = ["#ffffff", "#ff0000"]
# min_chi = 0.0
# max_chi = 1.0
# n_bins = 100  # Number of discrete bins
# n_ticks = 10
# cmap_name = "white_to_red"
# colormap = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
# ticks = np.linspace(min_chi, max_chi, n_ticks+1).round(3)


# # Make a heatplot of chi single u h

# # u = 0.95
# # h = 75 # km
# # np.random.seed(417)
# # args_list = []
# # for i in range(knots_xy_chi.shape[0]):
# #     args = (knots_xy_chi[i], u, h)
# #     args_list.append(args)
# # with multiprocessing.Pool(processes=N_CORES) as pool:
# #     results = pool.map(calc_chi, args_list)

# # chi_mat2 = np.full(shape = (len(y_pos_chi), len(x_pos_chi)), fill_value = np.nan)
# # for i in range(knots_xy_chi.shape[0]):
# #     chi_mat2[-1 - i//len(x_pos_chi), i%len(x_pos_chi)] = results[i]

# # fig, ax = plt.subplots()
# # fig.set_size_inches(6,8)
# # ax.set_aspect('equal', 'box')
# # state_map.boundary.plot(ax=ax, color = 'black', linewidth = 0.5)
# # heatmap = ax.imshow(chi_mat2, cmap =colormap, interpolation='nearest', 
# #                     vmin = 0, vmax = 1,
# #                     extent = [min(x_pos_chi - rect_width/8), max(x_pos_chi + rect_width/8), 
# #                               min(y_pos_chi - rect_height/8), max(y_pos_chi + rect_height/8)])
# # # ax.scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.8)
# # ax.scatter(knots_x_chi, knots_y_chi, s = 15, color = 'white', marker = '+')
# # fig.colorbar(heatmap)
# # plt.xlim([-105,-90])
# # plt.ylim([30,50])
# # plt.title(rf'model empirical $\chi_{{{u}}}$, h $\approx$ {h}km')
# # plt.savefig('model_empirical_chi_u={}_h={}.pdf'.format(u,h))
# # plt.show()
# # plt.close()

# # Parallel compute chi across several values of u h 
# np.random.seed(417)
# for h in [75, 150, 225]:

#     fig, axes = plt.subplots(1,3)
#     fig.set_size_inches(10,6)
#     for ax_id, u in enumerate([0.9, 0.95, 0.99]):
#         args_list = []
#         for i in range(knots_xy_chi.shape[0]):
#             args = (knots_xy_chi[i], u, h)
#             args_list.append(args)
#         with multiprocessing.Pool(processes=N_CORES) as pool:
#             results = pool.map(calc_chi, args_list)
        
#         chi_mat2 = np.full(shape = (len(y_pos_chi), len(x_pos_chi)), fill_value = np.nan)
#         for i in range(knots_xy_chi.shape[0]):
#             chi_mat2[-1 - i//len(x_pos_chi), i%len(x_pos_chi)] = results[i]

#         ax = axes[ax_id]
#         ax.set_aspect('equal', 'box')
#         state_map.boundary.plot(ax=ax, color = 'black', linewidth = 0.5)
#         heatmap = ax.imshow(chi_mat2, cmap = colormap, interpolation = 'nearest',
#                             vmin = 0.0, vmax = 1.0,
#                             extent = [min(x_pos_chi - rect_width/8), max(x_pos_chi + rect_width/8), 
#                                     min(y_pos_chi - rect_height/8), max(y_pos_chi + rect_height/8)])
#         # ax.scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.8)
#         ax.scatter(knots_x_chi, knots_y_chi, s = 15, color = 'white', marker = '+')
#         ax.set_xlim(-101,-93)
#         ax.set_ylim(32.5, 45)
#         ax.tick_params(axis='both', which='major', labelsize=14)
#         ax.title.set_text(rf'$\chi_{{{u}}}$')
#         ax.title.set_fontsize(20)
#         # ax.title.set_text(rf'$\chi_{{{u}}}$, h $\approx$ {h}km')

#     fig.subplots_adjust(right=0.8)
#     fig.text(0.5, 0.825, rf'h $\approx$ {h}km', ha='center', fontsize = 20)
#     fig.text(0.5, 0.125, 'Longitude', ha='center', fontsize = 20)
#     fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical', fontsize = 20)    
#     cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
#     colorbar = fig.colorbar(heatmap, cax = cbar_ax, ticks = ticks)
#     colorbar.ax.tick_params(labelsize=14)
#     plt.savefig('Surface:model_empirical_chi_h={}.pdf'.format(h), bbox_inches='tight')
#     plt.show()
#     plt.close()

# %% fix-xi only qqplot and likelihood

######################################
#### fixksi QQPlot and Likelihood ####
######################################
# if fixksi_only:
#     # %% Gumbel QQPlot fix only ksi
#     # Gumbel QQPlot fix only ksi

#     # Gumbel QQPlot with initial smoothed MLE -----------------------------------------------------------------------------

#     mu0_initSmooth = (C_mu0.T @ Beta_mu0_initSmooth).T
#     mu1_initSmooth = (C_mu1.T @ Beta_mu1_initSmooth).T
#     mu_initSmooth  = mu0_initSmooth + mu1_initSmooth * Time
#     sigma_initSmooth = np.exp((C_logsigma.T @ Beta_logsigma_initSmooth).T)
#     ksi_initSmooth = (C_ksi.T @ Beta_ksi_initSmooth).T

#     pY_initSmooth_test = np.full(shape = (test_Ns, Nt), fill_value = np.nan)
#     for t in range(Nt):
#         pY_initSmooth_test[:,t] = pgev(Y_99[:,t], mu_initSmooth[:,t], sigma_initSmooth[:,t], ksi_initSmooth[:,t])
#     pY_initSmooth_test_ro = numpy2rpy(pY_initSmooth_test)
#     r.assign('pY_initSmooth_test_ro', pY_initSmooth_test_ro)
#     r("save(pY_initSmooth_test_ro, file='pY_initSmooth_test_ro.gzip', compress=TRUE)")

#     gumbel_pY_initSmooth_test = np.full(shape = (test_Ns, Nt), fill_value = np.nan)
#     for t in range(Nt):
#         gumbel_pY_initSmooth_test[:,t] = scipy.stats.gumbel_r.ppf(pY_initSmooth_test[:,t])
#     gumbel_pY_initSmooth_test_ro = numpy2rpy(gumbel_pY_initSmooth_test)
#     r.assign('gumbel_pY_initSmooth_test_ro', gumbel_pY_initSmooth_test_ro)
#     r("save(gumbel_pY_initSmooth_test_ro, file='gumbel_pY_initSmooth_test_ro.gzip', compress=TRUE)")

#     # Gumbel QQPlot with mean(each MCMC iter GEV --> Gumbel) --------------------------------------------------------------

#     if not fixGEV:
#         # with per MCMC iterations of marginal GEV params
#         n = Beta_mu0_trace_thin.shape[0]

#         mu0_matrix_mcmc = (C_mu0.T @ Beta_mu0_trace_thin.T).T # shape (n, test_Ns, Nt)
#         mu1_matrix_mcmc = (C_mu1.T @ Beta_mu1_trace_thin.T).T # shape (n, test_Ns, Nt)
#         mu_matrix_mcmc  = mu0_matrix_mcmc + mu1_matrix_mcmc * Time
#         sigma_matrix_mcmc = np.exp((C_logsigma.T @ Beta_logsigma_trace_thin.T).T)
#         ksi_matrix_mcmc = np.tile(ksi_initSmooth, reps = (n, 1, 1))

#         pY_mcmc_test = np.full(shape = (n, test_Ns, Nt), fill_value = np.nan)
#         for i in range(n):
#             for t in range(Nt):
#                 pY_mcmc_test[i,:,t] = pgev(Y_99[:,t], mu_matrix_mcmc[i,:,t],
#                                             sigma_matrix_mcmc[i,:,t],
#                                             ksi_matrix_mcmc[i,:,t])
#         pY_mcmc_test_ro = numpy2rpy(pY_mcmc_test)
#         r.assign('pY_mcmc_test_ro',pY_mcmc_test_ro)
#         r("save(pY_mcmc_test_ro, file='pY_mcmc_test_ro.gzip', compress=TRUE)")

#         gumbel_pY_mcmc_test = np.full(shape = (n, test_Ns, Nt), fill_value = np.nan)
#         for i in range(n):
#             for t in range(Nt):
#                 gumbel_pY_mcmc_test[i,:,t] = scipy.stats.gumbel_r.ppf(pY_mcmc_test[i,:,t])
#         gumbel_pY_mcmc_test_ro = numpy2rpy(gumbel_pY_mcmc_test)
#         r.assign('gumbel_pY_mcmc_test_ro',gumbel_pY_mcmc_test_ro)
#         r("save(gumbel_pY_mcmc_test_ro, file='gumbel_pY_mcmc_test_ro.gzip', compress=TRUE)")

#     # Drawing the QQ Plots ------------------------------------------------------------------------------------------------

#     for _ in range(10):
#         # with MLE initial smooth
#         r('''
#             test_Ns <- 99
#             s <- floor(runif(1, min = 1, max = test_Ns + 1))
#             print(test_sites_xy_ro[s,]) # print coordinates
#             gumbel_s = sort(gumbel_pY_initSmooth_test_ro[s,])
#             nquants = length(gumbel_s)
#             emp_p = seq(1/nquants, 1-1/nquants, length=nquants)
#             emp_q = qgumbel(emp_p)
#             qq_gumbel_s <- extRemes::qqplot(gumbel_s, emp_q, regress=FALSE, legend=NULL,
#                                             xlab="Observed", ylab="Gumbel", main=paste("GEVfit-QQPlot of Site:",s),
#                                             lwd=3)
#             pdf(file=paste("QQPlot_R_Test_initSmooth_Site_",s,".pdf", sep=""), width = 6, height = 5)
#             par(mgp=c(1.5,0.5,0), mar=c(3,3,1,1))
#             plot(type="n",qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$y, pch = 20, xlab="Observed", ylab="Gumbel")
#             points(qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$y, pch=20)
#             lines(qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$lower, lty=2, col="blue", lwd=3)
#             lines(qq_gumbel_s$qdata$x, qq_gumbel_s$qdata$upper, lty=2, col="blue", lwd=3)
#             abline(a=0, b=1, lty=3, col="gray80", lwd=3)
#             legend("topleft", lty=c(2, 3), lwd=3, legend=c("95% confidence bands", "1:1 line"), col=c("blue", "gray80"), bty="n")
#             dev.off()
#         ''')
#         # with per MCMC iteration transformed
#         if not fixGEV:
#             r('''
#                 # s <- floor(runif(1, min = 1, max = test_Ns+1))
#                 print(test_sites_xy_ro[s,]) # print coordinates
#                 gumbel_s_mcmc = sort(apply(gumbel_pY_mcmc_test_ro[,s,],2, mean))
#                 nquants = length(gumbel_s_mcmc)
#                 emp_p = seq(1/nquants, 1-1/nquants, length=nquants)
#                 emp_q = qgumbel(emp_p)
#                 qq_gumbel_s_mcmc <- extRemes::qqplot(gumbel_s_mcmc, emp_q, regress=FALSE, legend=NULL,
#                                                 xlab="Observed", ylab="Gumbel", main=paste("Modelfit-QQPlot of Site:",s),
#                                                 lwd=3)
#                 pdf(file=paste("QQPlot_R_Test_MCMC_Site_",s,".pdf", sep=""), width = 6, height = 5)
#                 par(mgp=c(1.5,0.5,0), mar=c(3,3,1,1))
#                 plot(type="n",qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$y, pch = 20, xlab="Observed", ylab="Gumbel")
#                 points(qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$y, pch=20)
#                 lines(qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$lower, lty=2, col="blue", lwd=3)
#                 lines(qq_gumbel_s_mcmc$qdata$x, qq_gumbel_s_mcmc$qdata$upper, lty=2, col="blue", lwd=3)
#                 abline(a=0, b=1, lty=3, col="gray80", lwd=3)
#                 legend("topleft", lty=c(2, 3), lwd=3, legend=c("95% confidence bands", "1:1 line"), col=c("blue", "gray80"), bty="n")
#                 dev.off()
#                 ''')

#     # %% loglikelihood at Testing sample fix only ksi ----------------------------------------------------------------------------------
#     # loglikelihood at Testing sample fix only ksi

#     # 1. Y_99 -------------------------------------------------------------------------------------------------------------

#     Y_99_noNA = Y_99[~np.isnan(Y_99)]

#     # 2. Marginal Parameters ----------------------------------------------------------------------------------------------

#     mu0_initSmooth = (C_mu0.T @ Beta_mu0_initSmooth).T
#     mu1_initSmooth = (C_mu1.T @ Beta_mu1_initSmooth).T
#     mu_initSmooth  = mu0_initSmooth + mu1_initSmooth * Time
#     sigma_initSmooth = np.exp((C_logsigma.T @ Beta_logsigma_initSmooth).T)
#     ksi_initSmooth = (C_ksi.T @ Beta_ksi_initSmooth).T

#     if not fixGEV: # these are the per iteration marginal parameters
#         Beta_mu0_trace_thin100      = Beta_mu0_trace[0:iter:100,:]
#         Beta_mu1_trace_thin100      = Beta_mu1_trace[0:iter:100,:]
#         Beta_logsigma_trace_thin100 = Beta_logsigma_trace[0:iter:100,:]
#         # Beta_ksi_trace_thin100      = Beta_ksi_trace[0:iter:100,:]

#         mu0_matrix_thin100   = (C_mu0.T @ Beta_mu0_trace_thin100.T).T # shape (n, test_Ns, Nt)
#         mu1_matrix_thin100   = (C_mu1.T @ Beta_mu1_trace_thin100.T).T # shape (n, test_Ns, Nt)
#         mu_matrix_thin100    = mu0_matrix_thin100 + mu1_matrix_thin100 * Time
#         sigma_matrix_thin100 = np.exp((C_logsigma.T @ Beta_logsigma_trace_thin100.T).T)
#         ksi_matrix_thin100   = np.tile(ksi_initSmooth, reps = (sigma_matrix_thin100.shape[0], 1, 1))

#     # if not fixGEV: # these are the posterior mean estimates, which we shouldn't use
#     #     mu0_matrix_test = (C_mu0.T @ Beta_mu0_mean.T).T
#     #     mu1_matrix_test = (C_mu1.T @ Beta_mu1_mean.T).T
#     #     mu_matrix_test  = mu0_matrix_test + mu1_matrix_test * Time
#     #     sigma_matrix_test = np.exp((C_logsigma.T @ Beta_logsigma_mean.T).T)
#     #     ksi_matrix_test = (C_ksi.T @ Beta_ksi_mean.T).T

#     # 3. Copula Parameters - should also be per iterations ----------------------------------------------------------------

#     # weight matrices at the testing sites
#     # gaussian_weight_matrix_test = np.full(shape = (test_Ns, k), fill_value = np.nan)
#     # for site_id in np.arange(test_Ns):
#     #     # Compute distance between each pair of the two collections of inputs
#     #     d_from_knots = scipy.spatial.distance.cdist(XA = test_sites_xy[site_id,:].reshape((-1,2)), 
#     #                                     XB = knots_xy)
#     #     # influence coming from each of the knots
#     #     weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
#     #     gaussian_weight_matrix_test[site_id, :] = weight_from_knots
#     gaussian_weight_matrix_test_phi = np.full(shape = (test_Ns, k_phi), fill_value = np.nan)
#     for site_id in np.arange(test_Ns):
#         # Compute distance between each pair of the two collections of inputs
#         d_from_knots = scipy.spatial.distance.cdist(XA = test_sites_xy[site_id,:].reshape((-1,2)), 
#                                                     XB = knots_xy_phi)
#         # influence coming from each of the knots
#         weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_phi, cutoff = False)
#         gaussian_weight_matrix_test_phi[site_id, :] = weight_from_knots
#     gaussian_weight_matrix_test_rho = np.full(shape = (test_Ns, k_rho), fill_value = np.nan)
#     for site_id in np.arange(test_Ns):
#         # Compute distance between each pair of the two collections of inputs
#         d_from_knots = scipy.spatial.distance.cdist(XA = test_sites_xy[site_id,:].reshape((-1,2)), 
#                                                     XB = knots_xy_rho)
#         # influence coming from each of the knots
#         weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_rho, cutoff = False)
#         gaussian_weight_matrix_test_rho[site_id, :] = weight_from_knots
#     wendland_weight_matrix_test = np.full(shape = (test_Ns,k_phi), fill_value = np.nan)
#     for site_id in np.arange(test_Ns):
#         # Compute distance between each pair of the two collections of inputs
#         d_from_knots = scipy.spatial.distance.cdist(XA = test_sites_xy[site_id,:].reshape((-1,2)), 
#                                                     XB = knots_xy_phi)
#         # influence coming from each of the knots
#         weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
#         wendland_weight_matrix_test[site_id, :] = weight_from_knots

#     # constants
#     gamma_at_knots = np.repeat(0.5, k_phi)
#     alpha          = 0.5
#     gamma_vec_test = np.sum(np.multiply(wendland_weight_matrix_test, gamma_at_knots)**(alpha),
#                             axis = 1)**(1/alpha)
#     sigsq_vec      = np.repeat(1.0, test_Ns)
#     nu             = 0.5

#     # making the per iterations parameters (chains thinned by 100) at testing sites
#     #   - mu, sigma, ksi, and 
#     #   - phi, range, R
#     n_iter      = phi_knots_trace.shape[0]
#     idx_thin100 = np.arange(n_iter)[0::100] # thin by 100
#     n_thin100   = len(idx_thin100)
#     idx_thin100 = np.arange(n_thin100)

#     phi_knots_trace_thin100 = phi_knots_trace[0:n_iter:100,:]
#     phi_vec_test_thin100    = (gaussian_weight_matrix_test_phi @ phi_knots_trace_thin100.T).T

#     range_knots_trace_thin100 = range_knots_trace[0:n_iter:100,:]
#     range_vec_test_thin100    = (gaussian_weight_matrix_test_rho @ range_knots_trace_thin100.T).T

#     R_trace_log_thin100 = R_trace_log[0:n_iter:100,:,:]
#     R_vec_test_thin100 = np.full(shape = (n_thin100, test_Ns, Nt), fill_value = np.nan)
#     for t in range(Nt):
#         R_vec_test_thin100[:,:,t]  = (wendland_weight_matrix_test @ np.exp(R_trace_log_thin100[:,:,t]).T).T

#     # # Posterior mean of these parameters at the testing sites
#     # phi_vec_test   = gaussian_weight_matrix_test @ phi_mean
#     # range_vec_test = gaussian_weight_matrix_test @ range_mean
#     # gamma_at_knots = np.repeat(0.5, k)
#     # alpha          = 0.5
#     # gamma_vec_test = np.sum(np.multiply(wendland_weight_matrix_test, gamma_at_knots)**(alpha),
#     #                         axis = 1)**(1/alpha)
#     # R_matrix_test     = wendland_weight_matrix_test @ np.exp(R_log_mean) # shape (k, Nt)

#     # # 4. K or Cholesky_U
#     # sigsq_vec = np.repeat(1.0, test_Ns)
#     # nu        = 0.5
#     # K_test    = ns_cov(range_vec = range_vec_test,
#     #                    sigsq_vec = sigsq_vec,
#     #                    coords    = test_sites_xy,
#     #                    kappa     = nu, cov_model = "matern")
#     # cholesky_U_test = scipy.linalg.cholesky(K_test, lower = False)

#     # 4. Calculate X per iteration -- could really use parallelization here... --------------------------------------------
#     print('link function g:', norm_pareto)

#     def qRW_pgev(args):
#         Y     = args[:,0]
#         Loc   = args[:,1]
#         Scale = args[:,2]
#         Shape = args[:,3]
#         Phi   = args[:,4]
#         Gamma = args[:,5]
#         # Y, Loc, Scale, Shape, Phi, Gamma = args.T # args shaped (noNA, 6)
#         return qRW(pgev(Y, Loc, Scale, Shape), Phi, Gamma)

#     X_99_thin100 = np.full(shape = (n_thin100, test_Ns, Nt), fill_value = np.nan)
#     for i in range(n_thin100):
#         print('qRW_pgev:', i)
#         args_list = []
#         for t in range(Nt): # parallel compute the times (distribute the Nt times to n_processes)
#             noNA = ~np.isnan(Y_99[:,t])
#             if not fixGEV:
#                 args = np.column_stack((Y_99[noNA, t], 
#                                         mu_matrix_thin100[i, noNA, t], 
#                                         sigma_matrix_thin100[i,noNA,t],
#                                         ksi_matrix_thin100[i,noNA,t],
#                                         phi_vec_test_thin100[i,noNA],
#                                         gamma_vec_test[noNA]))
#             if fixGEV:
#                 args = np.column_stack((Y_99[noNA, t],
#                                         mu_initSmooth[noNA, t],
#                                         sigma_initSmooth[noNA, t],
#                                         ksi_initSmooth[noNA, t],
#                                         phi_vec_test_thin100[i, noNA],
#                                         gamma_vec_test[noNA]))
#             args_list.append(args)
#         with multiprocessing.Pool(processes=N_CORES) as pool:
#             results = pool.map(qRW_pgev, args_list)
#         for t in range(Nt):
#             noNA = ~np.isnan(Y_99[:,t])
#             X_99_thin100[i,noNA,t] = results[t]

#     # 5. loglikelihood -- calculation can also be parallelized! -----------------------------------------------------------

#     def ll(args):
#         Y, X, Loc, Scale, Shape, phi, gamma, R, K_chol = args
#         return(marg_transform_data_mixture_likelihood_1t(Y, X, Loc, Scale, Shape, phi, gamma, R, K_chol))

#     # Using per iteration for the parameters
#     ll_test_thin100 = np.full(shape= (n_thin100, test_Ns, Nt), fill_value = 0.0)
#     for i in range(n_thin100):
#         print('ll:', i)
#         args_list = []
#         for t in range(Nt): # parallel compute the times (distribute the Nt times to n_processes)
#             noNA       = ~np.isnan(Y_99[:,t])
#             K_test     = ns_cov(range_vec = range_vec_test_thin100[i,:],
#                                 sigsq_vec = sigsq_vec,
#                                 coords    = test_sites_xy,
#                                 kappa     = nu, cov_model = "matern")
#             K_subset   = K_test[noNA,:][:,noNA]
#             cholesky_U = scipy.linalg.cholesky(K_subset, lower = False)
#             if not fixGEV:
#                 args = (Y_99[noNA, t], X_99_thin100[i, noNA, t],
#                         mu_matrix_thin100[i, noNA, t], sigma_matrix_thin100[i,noNA,t], ksi_matrix_thin100[i,noNA,t],
#                         phi_vec_test_thin100[i,noNA], gamma_vec_test[noNA], R_vec_test_thin100[i,noNA,t], cholesky_U)
#             if fixGEV:
#                 args = (Y_99[noNA,t], X_99_thin100[i,noNA,t],
#                         mu_initSmooth[noNA,t], sigma_initSmooth[noNA,t], ksi_initSmooth[noNA,t],
#                         phi_vec_test_thin100[i,noNA], gamma_vec_test[noNA], R_vec_test_thin100[i,noNA,t], cholesky_U)
#             args_list.append(args)
#         with multiprocessing.Pool(processes=N_CORES) as pool:
#             results = pool.map(ll, args_list)
#         for t in range(Nt):
#             noNA = ~np.isnan(Y_99[:,t])
#             ll_test_thin100[i,noNA,t] = results[t]

#     # # Using the posterior mean for the parameters
#     # ll_test_thin100 = np.full(shape= (n_thin100, test_Ns, Nt), fill_value = 0.0)
#     # for i in range(n_thin100):
#     #     print('ll:', i)
#     #     args_list = []
#     #     for t in range(Nt): # parallel compute the times (distribute the Nt times to n_processes)
#     #         noNA       = ~np.isnan(Y_99[:,t])
#     #         K_subset   = K_test[noNA,:][:,noNA]
#     #         cholesky_U = scipy.linalg.cholesky(K_subset, lower = False)
#     #         if not fixGEV:
#     #             args = (Y_99[noNA, t], X_99_thin100[i, noNA, t],
#     #                     mu_matrix_test[noNA,t], sigma_matrix_test[noNA,t], ksi_matrix_test[noNA,t],
#     #                     phi_vec_test[noNA], gamma_vec_test[noNA], R_matrix_test[noNA,t], cholesky_U)
#     #         if fixGEV:
#     #             args = (Y_99[noNA,t], X_99_thin100[i,noNA,t],
#     #                     mu_initSmooth[noNA,t], sigma_initSmooth[noNA,t], ksi_initSmooth[noNA,t],
#     #                     phi_vec_test[noNA], gamma_vec_test[noNA], R_matrix_test[noNA,t], cholesky_U)
#     #         args_list.append(args)
#     #     with multiprocessing.Pool(processes=N_CORES) as pool:
#     #         results = pool.map(ll, args_list)
#     #     for t in range(Nt):
#     #         noNA = ~np.isnan(Y_99[:,t])
#     #         ll_test_thin100[i,noNA,t] = results[t]

#     np.save('ll_'+name, ll_test_thin100)

#     plt.boxplot(np.sum(ll_test_thin100, axis = (1,2)))
#     plt.xticks([1], [name])
#     plt.xlabel('Knot Radius Configuration')
#     plt.ylabel('log-likelihood @ test sites')
#     plt.savefig('ll_'+name+'_boxplot.pdf')
#     plt.show()
#     plt.close()


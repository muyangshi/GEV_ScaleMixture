"""
20251017- Remake chi plots following revision comments

Requires:
    - utilities.py
    - predGEV_grid_elev.npy <- the elevation grid for the predicted marginal GEV surface
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
# from    scipy.stats            import t as tdist
import  numpy                as np
import  geopandas            as gpd
import  matplotlib           as mpl
import  matplotlib.pyplot    as plt
from    matplotlib             import colormaps
from    rpy2.robjects          import r 
from    rpy2.robjects.numpy2ri import numpy2rpy
from    rpy2.robjects.packages import importr
from    tqdm                   import tqdm

# Custom Extensions and settings
from    utilities              import *

N_CORES = 64

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

# Model 7: k25r4b4 ----------------------------------------
folder           = './'
name             = 'k25r4b4'
fixGEV           = False
radius           = 4
bandwidth_phi    = 4
bandwidth_rho    = 4
N_outer_grid_phi = 16
N_outer_grid_rho = 16
burnin           = 3000

# # Model 8: k25r4b4m ---------------------------------------
# folder           = './'
# name             = 'k25r4b4m'
# fixGEV           = True
# radius           = 4
# bandwidth_phi    = 4
# bandwidth_rho    = 4
# N_outer_grid_phi = 16
# N_outer_grid_rho = 16
# burnin           = 3000

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


# %%
# redo chi: helper functions

def build_window_knots(minX, maxX, minY, maxY, res_x, res_y):
    x_pos = np.linspace(minX, maxX, res_x+4)[2:-2]
    y_pos = np.linspace(minY, maxY, res_y+4)[2:-2]
    Xp, Yp = np.meshgrid(x_pos, y_pos)
    knots_xy = np.vstack([Xp.ravel(), Yp.ravel()]).T
    return x_pos, y_pos, knots_xy

def window_mask_for_center(center_xy, rect_w, rect_h, sites_xy):
    cx, cy = center_xy
    left, right  = cx - rect_w/2, cx + rect_w/2
    bottom, top  = cy - rect_h/2, cy + rect_h/2
    return ((left  <= sites_xy[:,0]) & (sites_xy[:,0] <= right) &
            (bottom<= sites_xy[:,1]) & (sites_xy[:,1] <= top))

def select_pairs_by_h(sites_xy_in, h, e_abs, coord_to_dist):
    n = sites_xy_in.shape[0]
    if n < 2: 
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    dmat = np.empty((n,n), dtype=float)
    for i in range(n):
        for j in range(n):
            dmat[i,j] = coord_to_dist(sites_xy_in[i], sites_xy_in[j])
    h_low, h_up = h*(1-e_abs), h*(1+e_abs)
    mask = (np.triu(dmat,1) >= h_low) & (np.triu(dmat,1) <= h_up)
    I, J = np.where(mask)
    return I, J

def precompute_pairs(h_list, e_abs, sites_xy, sites_in_window):
    """Return dict: pairs[(h, win_id)] = (idx_local_i, idx_local_j) for that window."""
    pairs = {}
    for h in h_list:
        for win_id, m in enumerate(sites_in_window):
            sub_xy = sites_xy[m]
            I, J = select_pairs_by_h(sub_xy, h, e_abs, coord_to_dist)
            pairs[(h, win_id)] = (I, J)
    return pairs

# --- TOP-LEVEL WORKER (must be defined at module scope) ---
def _work_one_window_pairs(args):
    """
    Worker to build (I,J) pairs for one window for all h in h_list.
    args = (win_id, mask, h_list, e_abs, sites_xy)
    """
    win_id, mask, h_list, e_abs, sites_xy = args
    sub_xy = sites_xy[mask]
    out = {}
    if sub_xy.shape[0] < 2:
        for h in h_list:
            out[(h, win_id)] = (np.empty(0, dtype=int), np.empty(0, dtype=int))
        return out

    for h in h_list:
        I, J = select_pairs_by_h(sub_xy, h, e_abs, coord_to_dist)  # uses your existing functions
        out[(h, win_id)] = (I, J)
    return out

def precompute_pairs_parallel(h_list, e_abs, sites_xy, sites_in_window, n_jobs=N_CORES):
    """
    Parallel pair precompute by window.
    Returns: dict with keys (h, win_id) -> (I_local, J_local)
    """
    tasks = [(win_id, mask, h_list, e_abs, sites_xy) for win_id, mask in enumerate(sites_in_window)]

    # If you're running as a script, fork is fine; in notebooks, spawn can be safer.
    # Use whichever has worked for you elsewhere:
    ctx = multiprocessing.get_context('fork')

    with ctx.Pool(processes=n_jobs) as pool:
        it = pool.imap(_work_one_window_pairs, tasks, chunksize=1)
        it = tqdm(it, total=len(tasks), desc="precompute_pairs (by window)")
        partials = list(it)

    pairs = {}
    for d in partials:
        pairs.update(d)
    return pairs

def hac_se_time(p_t):
    """HAC-ish SE via Bartlett window with automatic truncation L ~ T^(1/3)."""
    p_t = p_t[np.isfinite(p_t)]
    T = len(p_t)
    if T <= 3:
        return np.nanstd(p_t, ddof=1)/max(np.sqrt(T),1)
    mu = np.nanmean(p_t)
    L  = max(1, int(round(T**(1/3))))
    # autocovariances
    gamma0 = np.nanmean((p_t-mu)*(p_t-mu))
    if gamma0 <= 0:
        return np.nanstd(p_t, ddof=1)/np.sqrt(T)
    acv = []
    for ell in range(1, L+1):
        acv.append(np.nanmean((p_t[:-ell]-mu)*(p_t[ell:]-mu)))
    # Bartlett weights
    var_hac = gamma0 + 2*np.sum([(1-ell/(L+1))*g for ell,g in enumerate(acv, start=1)])
    var_hac = max(var_hac, 0.0)
    return np.sqrt(var_hac / T)

# Build indicators from raw X and per-site quantiles q_i(u)
# X: (Ns, T)
# q_by_u: dict[u] -> q_vec of shape (Ns,)
def indicators_from_quantiles(X, q_by_u):
    I_by_u = {}
    for u, q_vec in q_by_u.items():
        I = (X >= q_vec[:, None]) # broadcast (Ns, T) >= (Ns, 1) -> (Ns, T)
        I_by_u[u] = I
    return I_by_u

# Same computation as before, but takes a dict of boolean indicators
def chi_field_from_indicators(
        I_by_u,              # dict[u] -> (Ns, Nt) bool
        u_list, h_list,
        pairs_by_window,     # dict from precompute_pairs
        sites_in_window,     # list from site masks
        conf=0.95):
    Ns, Nt = next(iter(I_by_u.values())).shape
    Ny, Nx = len(y_pos_chi), len(x_pos_chi)

    chi_hat = np.full((len(h_list), len(u_list), Ny, Nx), np.nan)
    chi_lb  = np.full_like(chi_hat, np.nan)
    chi_ub  = np.full_like(chi_hat, np.nan)

    for hi, h in enumerate(h_list):
        for wi, m in enumerate(sites_in_window):
            if not np.any(m): 
                continue
            idx_global = np.where(m)[0]       # global index of sites in this window
            Ii, Ij = pairs_by_window[(h, wi)] # the pair indexing is local
            n_pairs = len(Ii)
            if n_pairs == 0: continue

            for ui, u in enumerate(u_list):
                I = I_by_u[u][idx_global]  # (n_sites_win, Nt) bool
                joint_t = I[Ii] & I[Ij]    # (n_pairs, Nt)
                p_t = joint_t.sum(axis=0) / float(n_pairs)  # (Nt,)

                p_bar = np.nanmean(p_t)
                se    = hac_se_time(p_t)   # HAC SE over time
                denom = (1.0 - u)
                ch    = p_bar / denom

                # df = max(p_t.size - 1, 1)
                # tcrit = tdist.ppf(0.975, df=df)
                lb = max(0.0, (p_bar - 2*se) / denom)
                ub = min(1.0, (p_bar + 2*se) / denom)

                r = -1 - wi // Nx
                c = wi  % Nx
                chi_hat[hi, ui, r, c] = ch
                chi_lb [hi, ui, r, c] = lb
                chi_ub [hi, ui, r, c] = ub

    return chi_hat, chi_lb, chi_ub

def draw_chi_lb_hat_ub(chi_hat, chi_lb, chi_ub, h, u, h_i,u_i,x_pos, y_pos, rect_w, rect_h, state_map, cmap, vmin, vmax, ticks):
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))
    fig.subplots_adjust(right=0.8)

    def _draw(ax, mat, title_txt):
        ax.set_aspect('equal', 'box')
        state_map.boundary.plot(ax=ax, color='black', linewidth=0.5)
        hm = ax.imshow(
            mat, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest',
            extent=[min(x_pos - rect_w/8), max(x_pos + rect_w/8),
                    min(y_pos - rect_h/8), max(y_pos + rect_h/8)]
        )
        ax.scatter(knots_xy_chi[:,0], knots_xy_chi[:,1], s=25, color='black', marker='+', linewidths=1)
        ax.set_xlim(-101, -93); ax.set_ylim(32.5, 45)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title(title_txt, fontsize=20)
        return hm

    _   = _draw(axes[0], chi_lb[h_i,u_i,:,:],   rf'LB $\chi_{{{u}}}$') if chi_lb is not None else None
    hm0 = _draw(axes[1], chi_hat[h_i,u_i,:,:],  rf'$\hat\chi_{{{u}}}$')
    _   = _draw(axes[2], chi_ub[h_i,u_i,:,:],   rf'UB $\chi_{{{u}}}$') if chi_ub is not None else None

    fig.text(0.5, 0.825, rf'h $\approx$ {h} km,  u = {u}', ha='center', fontsize=20)
    fig.text(0.5, 0.125, 'Longitude', ha='center', fontsize=20)
    fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=20)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
    cbar = fig.colorbar(hm0, cax=cbar_ax, ticks=ticks)
    cbar.ax.tick_params(labelsize=14)
    return fig

def qRW_par(args):
    u, phi, gamma = args
    return qRW(u, phi, gamma)

# # %% Data empirical chi (REDO)

# # A) DATA: build PITs with fitted GEV per iteration, but CI should come from the data.
# # You already built pY_mcmc of shape (n_iter, Ns, Nt); a common, clean choice:
# # use the per-iteration PITs to reduce noise by *averaging indicators in time step*, 
# # but the CI still comes from time variation. Easiest is: pick one consistent pY for data, e.g. the mean CDF over iterations:

# # these are the per iteration marginal parameters
# niter                       = phi_knots_trace.shape[0]
# Beta_mu0_trace_thin100      = Beta_mu0_trace[0:niter:100,:]
# Beta_mu1_trace_thin100      = Beta_mu1_trace[0:niter:100,:]
# Beta_logsigma_trace_thin100 = Beta_logsigma_trace[0:niter:100,:]
# Beta_ksi_trace_thin100      = Beta_ksi_trace[0:niter:100,:]
# mu0_fitted_matrix_thin100   = (C_mu0.T @ Beta_mu0_trace_thin100.T).T # shape (n, test_Ns, Nt)
# mu1_fitted_matrix_thin100   = (C_mu1.T @ Beta_mu1_trace_thin100.T).T # shape (n, test_Ns, Nt)
# mu_fitted_matrix_thin100    = mu0_fitted_matrix_thin100 + mu1_fitted_matrix_thin100 * Time
# sigma_fitted_matrix_thin100 = np.exp((C_logsigma.T @ Beta_logsigma_trace_thin100.T).T)
# ksi_fitted_matrix_thin100   = (C_ksi.T @ Beta_ksi_trace_thin100.T).T
# n_thin100                   = Beta_mu0_trace_thin100.shape[0]

# pY_mcmc = np.full(shape = (n_thin100, Ns, Nt), fill_value = np.nan)
# for i in range(n_thin100):
#     pY_mcmc[i,:,:] = pgev(Y, mu_fitted_matrix_thin100[i,:,:], sigma_fitted_matrix_thin100[i,:,:], ksi_fitted_matrix_thin100[i,:,:])


# u_list = [0.90, 0.95, 0.99]
# h_list = [75, 150, 225]
# e_abs  = 0.2  # relative tolerance for distance band

# # --- grid for χ windows (re-uses your choices) ---
# res_x_chi, res_y_chi = 7, 17
# x_pos_chi, y_pos_chi, knots_xy_chi = build_window_knots(minX, maxX, minY, maxY, res_x_chi, res_y_chi)
# rect_width      = (knots_xy_chi[0][0] - minX)*2
# rect_height     = (knots_xy_chi[0][1] - minY)*2
# sites_in_window = [window_mask_for_center(knots_xy_chi[i], rect_width, rect_height, sites_xy) for i in range(knots_xy_chi.shape[0])] # which sites are in this window, global indexing
# pairs_by_window = precompute_pairs(h_list, e_abs, sites_xy, sites_in_window) # pairs of lag h for each window, local indexing

# # --- calculate \chi ---

# # average the PITs over iterations then compute \chi

# pY_data = np.nanmean(pY_mcmc, axis=0)  # (Ns, Nt)
# q_by_u_pY = {u: np.full(shape=(Ns,), fill_value=u) for u in u_list}
# I_by_u_pY = indicators_from_quantiles(pY_data, q_by_u_pY)

# chi_hat_data, chi_lb_data, chi_ub_data = chi_field_from_indicators(
#     I_by_u=I_by_u_pY, u_list=u_list, h_list=h_list, pairs_by_window=pairs_by_window, sites_in_window=sites_in_window
# )


# # calculate \chi per iteration then average
# q_by_u_pY = {u: np.full(shape=(Ns,), fill_value=u) for u in u_list}

# def chi_per_iter(i):
#     # Build indicators per u from the PITs of this iteration
#     pY_i = pY_mcmc[i, :, :]                 # (Ns, Nt)
#     I_by_u_pY = indicators_from_quantiles(pY_i, q_by_u_pY)
#     # Compute χ fields + time-wise t/HAC CI for this iteration
#     hat_i, lb_i, ub_i = chi_field_from_indicators(
#         I_by_u=I_by_u_pY,
#         u_list=u_list, h_list=h_list,
#         pairs_by_window=pairs_by_window,
#         sites_in_window=sites_in_window,
#         conf=0.95
#     )
#     return hat_i, lb_i, ub_i  # each (H, U, Ny, Nx)

# iter_idx = list(range(n_thin100))
# with multiprocessing.get_context('fork').Pool(processes=N_CORES) as pool:
#     results = list(tqdm(pool.imap(chi_per_iter, iter_idx, chunksize=1), total=len(iter_idx)))

# # Stack along the new "posterior sample" axis: (n_thin100, H, U, Ny, Nx)
# chi_hat_all = np.stack([r[0] for r in results], axis=0)
# chi_lb_all  = np.stack([r[1] for r in results], axis=0)
# chi_ub_all  = np.stack([r[2] for r in results], axis=0)

# # Posterior means (THIS is the “calculate-all-then-mean” you want)
# chi_hat_data = np.nanmean(chi_hat_all, axis=0)  # (H, U, Ny, Nx)
# chi_lb_data  = np.nanmean(chi_lb_all,  axis=0)  # (H, U, Ny, Nx)
# chi_ub_data  = np.nanmean(chi_ub_all,  axis=0)  # (H, U, Ny, Nx)


# # --- drawing of LB and UB for each h and u ---

# for u_i, u in enumerate(u_list):
#     for h_i, h in enumerate(h_list):
#         myfig = draw_chi_lb_hat_ub(chi_hat_data, chi_lb_data, chi_ub_data, 
#                   h, u, h_i=h_i, u_i=u_i,
#                   x_pos=x_pos_chi, y_pos=y_pos_chi, rect_w=rect_width, rect_h=rect_height, 
#                   state_map=state_map, 
#                   cmap=mpl.colors.LinearSegmentedColormap.from_list("white_to_red", ["#ffffff", "#ff0000"], N=100), 
#                   vmin=0.0, vmax=0.5, 
#                   ticks = np.linspace(0.0, 0.5, 10+1).round(3)
#         )
#         myfig.savefig(rf"Surface_data_chi_LBUB_h={h}_u={u}.pdf", bbox_inches="tight")


# # # check pairs within a window ---------------------------
# # # e.g. first endpoint
# # sites_xy[sites_in_window[0]][pairs_by_window[(75,0)][0]]
# # # e.g. second endpoint
# # sites_xy[sites_in_window[0]][pairs_by_window[(75,0)][1]]
# # first119 = dict(list(pairs_by_window.items())[:119])
# # plt.hist([len(pairs_by_window[k][0]) for k in first119])

# # --- manually group by h drawing ---

# for h_i,h in enumerate(h_list):
#     x_pos=x_pos_chi
#     y_pos=y_pos_chi
#     rect_w=rect_width
#     rect_h=rect_height
#     state_map=state_map
#     cmap=mpl.colors.LinearSegmentedColormap.from_list("white_to_red", ["#ffffff", "#ff0000"], N=100)
#     vmin=0.0
#     vmax=0.5
#     ticks = np.linspace(0.0, 0.5, 10+1).round(3)
#     fig, axes = plt.subplots(1, 3, figsize=(10, 6))
#     fig.subplots_adjust(right=0.8)
#     def _draw(ax, mat, title_txt):
#         ax.set_aspect('equal', 'box')
#         state_map.boundary.plot(ax=ax, color='black', linewidth=0.5)
#         hm = ax.imshow(
#             mat, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest',
#             extent=[min(x_pos - rect_w/8), max(x_pos + rect_w/8),
#                     min(y_pos - rect_h/8), max(y_pos + rect_h/8)]
#         )
#         ax.scatter(knots_xy_chi[:,0], knots_xy_chi[:,1], s=25, color='black', marker='+', linewidths=1)
#         ax.set_xlim(-101, -93); ax.set_ylim(32.5, 45)
#         ax.tick_params(axis='both', which='major', labelsize=14)
#         ax.set_title(title_txt, fontsize=20)
#         return hm
#     up90 = _draw(axes[0], chi_hat_data[h_i,0,:,:],   rf'$\chi_{{{u_list[0]}}}$')
#     up95 = _draw(axes[1], chi_hat_data[h_i,1,:,:],  rf'$\hat\chi_{{{u_list[1]}}}$')
#     up99 = _draw(axes[2], chi_hat_data[h_i,2,:,:],   rf'$\chi_{{{u_list[2]}}}$')
#     fig.text(0.5, 0.825, rf'h $\approx$ {h} km', ha='center', fontsize=20)
#     fig.text(0.5, 0.125, 'Longitude', ha='center', fontsize=20)
#     fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=20)
#     cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
#     cbar = fig.colorbar(up95, cax=cbar_ax, ticks=ticks)
#     cbar.ax.tick_params(labelsize=14)
#     plt.savefig(rf'Surface_data_chi_fittedGEV_h={h}.pdf', bbox_inches='tight')

# # %% Model Realization empirical chi (REDO) ===================================

# # %% Engineer a grid of sites -------------------------------------------------

# # --- resolution of engineered points ---

# numX_chi     = 50  # Number of points along the X-axis
# numY_chi     = 150 # Number of points along the Y-axis
# Ns_chi       = numX_chi * numY_chi
# x_chi        = np.linspace(minX, maxX, numX_chi)
# y_chi        = np.linspace(minY, maxY, numY_chi)
# X_chi,Y_chi  = np.meshgrid(x_chi, y_chi)
# sites_xy_chi = np.column_stack([X_chi.ravel(), Y_chi.ravel()]) # a grid of engineerin
# sites_x_chi  = sites_xy_chi[:,0]
# sites_y_chi  = sites_xy_chi[:,1]

# u_list = [0.90, 0.95, 0.99]
# h_list = [75, 150, 225]
# e_abs = 0.2

# # --- grid for χ windows (re-uses your choices) --- 6'30'' single core
# res_x_chi, res_y_chi = 7, 17
# x_pos_chi, y_pos_chi, knots_xy_chi = build_window_knots(minX, maxX, minY, maxY, res_x_chi, res_y_chi)
# rect_width      = (knots_xy_chi[0][0] - minX)*2
# rect_height     = (knots_xy_chi[0][1] - minY)*2
# sites_in_window = [window_mask_for_center(knots_xy_chi[i], rect_width, rect_height, sites_xy_chi) for i in range(knots_xy_chi.shape[0])] # which sites are in this window, global indexing
# pairs_by_window = precompute_pairs(h_list, e_abs, sites_xy_chi, sites_in_window) # pairs of lag h for each window, local indexing


# # %% Simulate data for model realization chi estimation -----------------------

# # --- setting up the copula splines ---

# wendland_weight_matrix_chi     = np.full(shape = (Ns_chi, k_phi), fill_value = np.nan)
# for site_id in np.arange(Ns_chi):
#         d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy_chi[site_id,:].reshape((-1,2)), 
#                                                     XB = knots_xy_phi)
#         wendland_weight_matrix_chi[site_id, :] = wendland_weights_fun(d_from_knots, radius_from_knots)
# gaussian_weight_matrix_chi     = np.full(shape = (Ns_chi, k_phi), fill_value = np.nan)
# for site_id in np.arange(Ns_chi):
#         d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy_chi[site_id,:].reshape((-1,2)), 
#                                                     XB = knots_xy_phi)
#         gaussian_weight_matrix_chi[site_id, :] = weights_fun(d_from_knots, radius, bandwidth_phi, cutoff=False)
# gaussian_weight_matrix_rho_chi = np.full(shape = (Ns_chi, k_rho), fill_value = np.nan)
# for site_id in np.arange(Ns_chi):
#         d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy_chi[site_id,:].reshape((-1,2)), 
#                                                     XB = knots_xy_rho)
#         gaussian_weight_matrix_rho_chi[site_id, :] = weights_fun(d_from_knots, radius, bandwidth_rho, cutoff=False)

# alpha          = 0.5
# nu             = 0.5
# sigsq_vec      = np.repeat(1.0, Ns_chi)
# gamma_at_knots = np.repeat(0.5, k_phi)
# gamma_vec_chi  = np.sum(np.multiply(wendland_weight_matrix_chi, gamma_at_knots)**(alpha),
#                         axis = 1)**(1/alpha)
# assert any(np.isnan(gamma_vec_chi)) != True

# # --- model posterior means ---

# phi_vec_chi = gaussian_weight_matrix_chi     @ phi_mean
# rho_vec_chi = gaussian_weight_matrix_rho_chi @ range_mean
# K_chi       = ns_cov(range_vec = rho_vec_chi,
#                      sigsq_vec = sigsq_vec,
#                      coords    = sites_xy_chi,
#                      kappa     = nu, cov_model = "matern") # 16 secs for Ns_chi = 7,500

# # --- Draw <n_draw> (time) replicates of observations ---

# np.random.seed(910)
# n_draw    = 100 # number of time replicates to draw
# S_vec_chi = np.array([scipy.stats.levy.rvs(loc = 0, scale = 0.5, size = k_phi) for _ in range(n_draw)]) # shape(n_draw, k_phi)
# Z_vec_chi = scipy.stats.multivariate_normal.rvs(mean = None, 
#                                                 cov = K_chi, 
#                                                 size = n_draw) # shape(n_draw, Ns_chi)
# # Notes on multivariate gaussian speed:
# #   1m 23s for 7,500 x 100
# #   no speed difference when drawing size = 1 or 100

# R_vec_chi = (wendland_weight_matrix_chi @ S_vec_chi.T) # shape(Ns_chi, n_draw)
# W_chi     = norm_to_Pareto(Z_vec_chi.T)                # shape(Ns_chi, n_draw)
# X_model_chi = (R_vec_chi.T ** phi_vec_chi).T * W_chi   # shape(Ns_chi, n_draw)

# np.save('X_model_chi', X_model_chi)
# np.save('phi_vec_chi', phi_vec_chi)
# np.save('rho_vec_chi', rho_vec_chi)

# # Threshold quantile qu
# # Note:
# #   Only needs to be calculated once for each u
# #   should parallelize the calculation

# def qRW_par(args):
#     u, phi, gamma = args
#     return qRW(u, phi, gamma)

# args_list090 = []
# args_list095 = []
# args_list099 = []
# for i in range(Ns_chi):
#     args_list090.append((0.9, phi_vec_chi[i], gamma_vec_chi[i]))
#     args_list095.append((0.95, phi_vec_chi[i], gamma_vec_chi[i]))
#     args_list099.append((0.99, phi_vec_chi[i], gamma_vec_chi[i]))
# with multiprocessing.get_context('fork').Pool(processes = 50) as pool:
#     results090 = list(tqdm(pool.imap(qRW_par, args_list090), total=len(args_list090), desc='u=0.90'))
# with multiprocessing.get_context('fork').Pool(processes = 50) as pool:
#     results095 = list(tqdm(pool.imap(qRW_par, args_list095), total=len(args_list095), desc='u=0.95')) #pool.map(qRW_par, args_list095)
# with multiprocessing.get_context('fork').Pool(processes = 50) as pool:
#     results099 = list(tqdm(pool.imap(qRW_par, args_list099), total=len(args_list099), desc='u=0.99')) #pool.map(qRW_par, args_list099)

# qu_090 = np.array(results090)
# np.save('qu_090', qu_090)
# qu_095 = np.array(results095)
# np.save('qu_095', qu_095)
# qu_099 = np.array(results099)
# np.save('qu_099', qu_099)

# # %% \chi estimation ----------------------------------------------------------

# # --- load simulated data ---
# X_model_chi = np.load('X_model_chi.npy')
# qu_090      = np.load('qu_090.npy')
# qu_095      = np.load('qu_095.npy')
# qu_099      = np.load('qu_099.npy')
# qu_all      = (qu_090, qu_095, qu_099)

# # --- calculate χ ---
# q_by_u_model = {u: qu_all[i] for i, u in enumerate(u_list)}
# I_by_u_model = indicators_from_quantiles(X_model_chi, q_by_u_model)
# chi_hat_model, chi_lb_model, chi_ub_model = chi_field_from_indicators(
#     I_by_u = I_by_u_model, 
#     u_list = u_list, h_list = h_list, 
#     pairs_by_window = pairs_by_window, sites_in_window = sites_in_window
# )

# # --- drawing of LB and UB for each h and u ---
# for u_i, u in enumerate(u_list):
#     for h_i, h in enumerate(h_list):
#         myfig = draw_chi_lb_hat_ub(chi_hat_model, chi_lb_model, chi_ub_model, 
#                   h, u, h_i=h_i, u_i=u_i,
#                   x_pos=x_pos_chi, y_pos=y_pos_chi, rect_w=rect_width, rect_h=rect_height, 
#                   state_map=state_map, 
#                   cmap=mpl.colors.LinearSegmentedColormap.from_list("white_to_red", ["#ffffff", "#ff0000"], N=100), 
#                   vmin=0.0, vmax=0.5, 
#                   ticks = np.linspace(0.0, 0.5, 10+1).round(3)
#         )
#         myfig.savefig(rf"Surface_model_chi_LBUB_h={h}_u={u}.pdf", bbox_inches="tight")

# # --- manually group by h drawing ---

# for h_i,h in enumerate(h_list):
#     x_pos=x_pos_chi
#     y_pos=y_pos_chi
#     rect_w=rect_width
#     rect_h=rect_height
#     state_map=state_map
#     cmap=mpl.colors.LinearSegmentedColormap.from_list("white_to_red", ["#ffffff", "#ff0000"], N=100)
#     vmin=0.0
#     vmax=0.5
#     ticks = np.linspace(0.0, 0.5, 10+1).round(3)
#     fig, axes = plt.subplots(1, 3, figsize=(10, 6))
#     fig.subplots_adjust(right=0.8)
#     def _draw(ax, mat, title_txt):
#         ax.set_aspect('equal', 'box')
#         state_map.boundary.plot(ax=ax, color='black', linewidth=0.5)
#         hm = ax.imshow(
#             mat, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest',
#             extent=[min(x_pos - rect_w/8), max(x_pos + rect_w/8),
#                     min(y_pos - rect_h/8), max(y_pos + rect_h/8)]
#         )
#         ax.scatter(knots_xy_chi[:,0], knots_xy_chi[:,1], s=25, color='black', marker='+', linewidths=1)
#         ax.set_xlim(-101, -93); ax.set_ylim(32.5, 45)
#         ax.tick_params(axis='both', which='major', labelsize=14)
#         ax.set_title(title_txt, fontsize=20)
#         return hm
#     up90 = _draw(axes[0], chi_hat_model[h_i,0,:,:],   rf'$\chi_{{{u_list[0]}}}$')
#     up95 = _draw(axes[1], chi_hat_model[h_i,1,:,:],  rf'$\hat\chi_{{{u_list[1]}}}$')
#     up99 = _draw(axes[2], chi_hat_model[h_i,2,:,:],   rf'$\chi_{{{u_list[2]}}}$')
#     fig.text(0.5, 0.825, rf'h $\approx$ {h} km', ha='center', fontsize=20)
#     fig.text(0.5, 0.125, 'Longitude', ha='center', fontsize=20)
#     fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=20)
#     cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
#     cbar = fig.colorbar(up95, cax=cbar_ax, ticks=ticks)
#     cbar.ax.tick_params(labelsize=14)
#     plt.savefig(rf'Surface_model_chi_h={h}.pdf', bbox_inches='tight')

# %% Per MCMC iteration model \chi ============================================

# %% Engineer a grid of sites -------------------------------------------------

numX_chi     = 50  # Number of points along the X-axis
numY_chi     = 150 # Number of points along the Y-axis
Ns_chi       = numX_chi * numY_chi
x_chi        = np.linspace(minX, maxX, numX_chi)
y_chi        = np.linspace(minY, maxY, numY_chi)
X_chi,Y_chi  = np.meshgrid(x_chi, y_chi)
sites_xy_chi = np.column_stack([X_chi.ravel(), Y_chi.ravel()]) # a grid of engineerin
sites_x_chi  = sites_xy_chi[:,0]
sites_y_chi  = sites_xy_chi[:,1]

u_list = [0.90, 0.95, 0.99]
h_list = [75, 150, 225]
e_abs = 0.2

# --- grid for χ windows (re-uses your choices) --- 6'30'' single core
res_x_chi, res_y_chi = 7, 17
x_pos_chi, y_pos_chi, knots_xy_chi = build_window_knots(minX, maxX, minY, maxY, res_x_chi, res_y_chi)
rect_width      = (knots_xy_chi[0][0] - minX)*2
rect_height     = (knots_xy_chi[0][1] - minY)*2
sites_in_window = [window_mask_for_center(knots_xy_chi[i], rect_width, rect_height, sites_xy_chi) for i in range(knots_xy_chi.shape[0])] # which sites are in this window, global indexing
pairs_by_window = precompute_pairs(h_list, e_abs, sites_xy_chi, sites_in_window) # pairs of lag h for each window, local indexing

# --- setting up the copula splines ---

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

# %% Simulate data per MCMC iteration -----------------------------------------

np.random.seed(1019)
n_draw = 100 # number of time replicates per posterior sample/surface

# --- Aggresively thin the posterior samples ---
niter                       = phi_knots_trace.shape[0]
phi_knots_trace_thin100     = phi_knots_trace[0:niter:100,:]
rho_knots_trace_thin100     = range_knots_trace[0:niter:100,:]
n_thin100                   = phi_knots_trace_thin100.shape[0]

# --- posterior samples interpolated surface ---

phi_vec_chi_thin100 = gaussian_weight_matrix_chi     @ phi_knots_trace_thin100.T # shape (Ns_chi, n_thin100)
rho_vec_chi_thin100 = gaussian_weight_matrix_rho_chi @ rho_knots_trace_thin100.T # shape (Ns_chi, n_thin100)

# --- simulate sample ---
# Note: a single K_chi calculation takes 23 secs
#       a Z_vec_chi simulation takes 12 minutes 40 secs
X_model_chi_thin100 = np.full(shape = (Ns_chi, n_draw, n_thin100), fill_value = np.nan)

for i in range(n_thin100):
    print(rf'{i}/{n_thin100}')
    K_chi  = ns_cov(range_vec = rho_vec_chi_thin100[:,i],
                    sigsq_vec = sigsq_vec, coords = sites_xy_chi, kappa = nu, cov_model = "matern")

    # --- Draw <n_draw> (time) replicates, one per posterior sample ---
    S_vec_chi = np.array([scipy.stats.levy.rvs(loc = 0, scale = 0.5, size = k_phi) for _ in range(n_draw)]) # shape(n_draw, k_phi)
    Z_vec_chi = scipy.stats.multivariate_normal.rvs(mean = None, 
                                                    cov = K_chi, 
                                                    size = n_draw) # shape(n_draw, Ns_chi)
    R_vec_chi = (wendland_weight_matrix_chi @ S_vec_chi.T) # shape(Ns_chi, n_draw)
    W_chi     = norm_to_Pareto(Z_vec_chi.T)                # shape(Ns_chi, n_draw)
    X_model_chi_thin100[:,:,i] = (R_vec_chi.T ** phi_vec_chi_thin100[:,i]).T * W_chi   # shape(Ns_chi, n_draw)
    np.save('X_model_chi_thin100', X_model_chi_thin100)

# def _simulate_par(i):
#     K_chi  = ns_cov(range_vec = rho_vec_chi_thin100[:,i],
#                     sigsq_vec = sigsq_vec, coords = sites_xy_chi, kappa = nu, cov_model = "matern")

#     # --- Draw <n_draw> (time) replicates, one per posterior sample ---
#     S_vec_chi = np.array([scipy.stats.levy.rvs(loc = 0, scale = 0.5, size = k_phi) for _ in range(n_draw)]) # shape(n_draw, k_phi)
#     Z_vec_chi = scipy.stats.multivariate_normal.rvs(mean = None, cov = K_chi, size = n_draw) # shape(n_draw, Ns_chi)
#     R_vec_chi = (wendland_weight_matrix_chi @ S_vec_chi.T) # shape(Ns_chi, n_draw)
#     W_chi     = norm_to_Pareto(Z_vec_chi.T)                # shape(Ns_chi, n_draw)
#     return (R_vec_chi.T ** phi_vec_chi_thin100[:,i]).T * W_chi   # shape(Ns_chi, n_draw)

# with multiprocessing.get_context('fork').Pool(processes = N_CORES) as pool:
#     X_model_chi_thin100 = list(tqdm(pool.imap(_simulate_par, np.arange(n_thin100)), total=len(np.arange(n_thin100)), desc='simulate'))

np.save('X_model_chi_thin100', X_model_chi_thin100)
np.save('phi_vec_chi_thin100', phi_vec_chi_thin100)
np.save('rho_vec_chi_thin100', rho_vec_chi_thin100)

# Threshold quantile qu
# Note:
#   Only needs to be calculated once for each u, should parallelize the calculation
#   Roughly 7 secs for 7,500 qRW evals
qu_090_thin100 = np.full(shape = (Ns_chi, n_thin100), fill_value = np.nan)
qu_095_thin100 = np.full(shape = (Ns_chi, n_thin100), fill_value = np.nan)
qu_099_thin100 = np.full(shape = (Ns_chi, n_thin100), fill_value = np.nan)

for i in range(n_thin100):
    print(rf"{i}/{n_thin100}")
    args_list090 = np.vstack([np.full(shape=(Ns_chi,), fill_value=0.9), phi_vec_chi_thin100[:,i], gamma_vec_chi]).T
    args_list090 = [tuple(row) for row in args_list090]
    with multiprocessing.get_context('fork').Pool(processes = N_CORES) as pool:
        results090 = list(tqdm(pool.imap(qRW_par, args_list090), total=len(args_list090), desc='u=0.90'))
    qu_090_thin100[:,i] = np.array(results090)

for i in range(n_thin100):
    print(rf"{i}/{n_thin100}")
    args_list095 = np.vstack([np.full(shape=(Ns_chi,), fill_value=0.95), phi_vec_chi_thin100[:,i], gamma_vec_chi]).T
    args_list095 = [tuple(row) for row in args_list095]
    with multiprocessing.get_context('fork').Pool(processes = N_CORES) as pool:
        results095 = list(tqdm(pool.imap(qRW_par, args_list095), total=len(args_list095), desc='u=0.95'))
    qu_095_thin100[:,i] = np.array(results095)

for i in range(n_thin100):
    print(rf"{i}/{n_thin100}")
    args_list099 = np.vstack([np.full(shape=(Ns_chi,), fill_value=0.99), phi_vec_chi_thin100[:,i], gamma_vec_chi]).T
    args_list099 = [tuple(row) for row in args_list099]
    with multiprocessing.get_context('fork').Pool(processes = N_CORES) as pool:
        results099 = list(tqdm(pool.imap(qRW_par, args_list099), total=len(args_list099), desc='u=0.99'))
    qu_099_thin100[:,i] = np.array(results099)

np.save('qu_090_thin100', qu_090_thin100)
np.save('qu_095_thin100', qu_095_thin100)
np.save('qu_099_thin100', qu_099_thin100)
    
# %% Calculate \chi per MCMC iteration

# --- load simulated data ---
X_model_chi_thin100 = np.load('X_model_chi_thin100.npy')
qu_090_thin100      = np.load('qu_090.npy_thin100')
qu_095_thin100      = np.load('qu_095.npy_thin100')
qu_099_thin100      = np.load('qu_099.npy_thin100')
qu_all_thin100      = (qu_090_thin100, qu_095_thin100, qu_099_thin100)

# --- calculate χ ---

chi_hat_model_thin100 = np.full(shape = (len(h_list), len(u_list), len(y_pos_chi), len(x_pos_chi), n_thin100), fill_value = np.nan)
chi_lb_model_thin100  = np.full_like(chi_hat_model_thin100, np.nan)
chi_ub_model_thin100  = np.full_like(chi_hat_model_thin100, np.nan)

for iter in range(n_thin100):
    print(rf"{iter}/{n_thin100}")
    q_by_u_model = {u: qu_all_thin100[i][:,iter] for i, u in enumerate(u_list)}
    I_by_u_model = indicators_from_quantiles(X_model_chi_thin100[:,iter], q_by_u_model)
    chi_hat_model, chi_lb_model, chi_ub_model = chi_field_from_indicators(
        I_by_u = I_by_u_model, 
        u_list = u_list, h_list = h_list, 
        pairs_by_window = pairs_by_window, sites_in_window = sites_in_window
    )
    chi_hat_model_thin100[:,:,:,iter] = chi_hat_model
    chi_lb_model_thin100 [:,:,:,iter] = chi_lb_model
    chi_ub_model_thin100 [:,:,:,iter] = chi_ub_model
    
np.save('chi_hat_model_thin100', chi_hat_model_thin100)
np.save('chi_lb_model_thin100', chi_lb_model_thin100)
np.save('chi_ub_model_thin100', chi_ub_model_thin100)

# --- drawing of LB and UB for each h and u ---
chi_hat_model = np.mean(chi_hat_model_thin100, axis=-1)
chi_lb_model  = np.mean(chi_lb_model_thin100, axis=-1)
chi_ub_model  = np.mean(chi_ub_model_thin100, axis=-1)
for u_i, u in enumerate(u_list):
    for h_i, h in enumerate(h_list):
        myfig = draw_chi_lb_hat_ub(chi_hat_model, chi_lb_model, chi_ub_model, 
                  h, u, h_i=h_i, u_i=u_i,
                  x_pos=x_pos_chi, y_pos=y_pos_chi, rect_w=rect_width, rect_h=rect_height, 
                  state_map=state_map, 
                  cmap=mpl.colors.LinearSegmentedColormap.from_list("white_to_red", ["#ffffff", "#ff0000"], N=100), 
                  vmin=0.0, vmax=0.5, 
                  ticks = np.linspace(0.0, 0.5, 10+1).round(3)
        )
        myfig.savefig(rf"Surface_model_chi_LBUB_h={h}_u={u}.pdf", bbox_inches="tight")

# --- manually group by h drawing ---

for h_i,h in enumerate(h_list):
    x_pos=x_pos_chi
    y_pos=y_pos_chi
    rect_w=rect_width
    rect_h=rect_height
    state_map=state_map
    cmap=mpl.colors.LinearSegmentedColormap.from_list("white_to_red", ["#ffffff", "#ff0000"], N=100)
    vmin=0.0
    vmax=0.5
    ticks = np.linspace(0.0, 0.5, 10+1).round(3)
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))
    fig.subplots_adjust(right=0.8)
    def _draw(ax, mat, title_txt):
        ax.set_aspect('equal', 'box')
        state_map.boundary.plot(ax=ax, color='black', linewidth=0.5)
        hm = ax.imshow(
            mat, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest',
            extent=[min(x_pos - rect_w/8), max(x_pos + rect_w/8),
                    min(y_pos - rect_h/8), max(y_pos + rect_h/8)]
        )
        ax.scatter(knots_xy_chi[:,0], knots_xy_chi[:,1], s=25, color='black', marker='+', linewidths=1)
        ax.set_xlim(-101, -93); ax.set_ylim(32.5, 45)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title(title_txt, fontsize=20)
        return hm
    up90 = _draw(axes[0], chi_hat_model[h_i,0,:,:],   rf'$\chi_{{{u_list[0]}}}$')
    up95 = _draw(axes[1], chi_hat_model[h_i,1,:,:],  rf'$\hat\chi_{{{u_list[1]}}}$')
    up99 = _draw(axes[2], chi_hat_model[h_i,2,:,:],   rf'$\chi_{{{u_list[2]}}}$')
    fig.text(0.5, 0.825, rf'h $\approx$ {h} km', ha='center', fontsize=20)
    fig.text(0.5, 0.125, 'Longitude', ha='center', fontsize=20)
    fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=20)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
    cbar = fig.colorbar(up95, cax=cbar_ax, ticks=ticks)
    cbar.ax.tick_params(labelsize=14)
    plt.savefig(rf'Surface_model_chi_h={h}.pdf', bbox_inches='tight')
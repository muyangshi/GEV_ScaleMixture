# %%
import sys
data_seed = 20
# %%
# Imports
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
from utilities import *
from time import strftime, localtime

burnins = 6000 # length of burnin iterations
simulation_case = 'scenario2'

#####################################################################################################################
# Generating Dataset ################################################################################################
#####################################################################################################################
# ------- 0. Simulation Setting --------------------------------------

## space setting
np.random.seed(data_seed)
N = 64 # number of time replicates
num_sites = 500 # number of sites/stations
k = 9 # number of knots

## unchanged constants or parameters
gamma = 0.5 # this is the gamma that goes in rlevy
delta = 0.0 # this is the delta in levy, stays 0
mu = 0.0 # GEV location
tau = 1.0 # GEV scale
ksi = 0.2 # GEV shape
nu = 0.5 # exponential kernel for matern with nu = 1/2
sigsq = 1.0 # for Z

n_iters = 1000


# ------- 1. Generate Sites and Knots --------------------------------

sites_xy = np.random.random((num_sites, 2)) * 10
sites_x = sites_xy[:,0]
sites_y = sites_xy[:,1]

## Knots
# creating a grid of knots
x_pos = np.linspace(0,10,5,True)[1:-1]
y_pos = np.linspace(0,10,5,True)[1:-1]
X_pos, Y_pos = np.meshgrid(x_pos,y_pos)
knots_xy = np.vstack([X_pos.ravel(), Y_pos.ravel()]).T
# knots_xy = np.array([[2,2],
#                      [2,8],
#                      [8,2],
#                      [8,8],
#                      [4.5,4.5]])

# putting two knots in the middle
if k == 10:
    knots_xy = np.delete(knots_xy, 4, axis = 0)
    knots_xy = np.insert(knots_xy, 4, np.array([4,5]), axis = 0)
    knots_xy = np.insert(knots_xy, 5, np.array([6,5]), axis = 0)



knots_x = knots_xy[:,0]
knots_y = knots_xy[:,1]

plotgrid_x = np.linspace(0.1,10,25)
plotgrid_y = np.linspace(0.1,10,25)
plotgrid_X, plotgrid_Y = np.meshgrid(plotgrid_x, plotgrid_y)
plotgrid_xy = np.vstack([plotgrid_X.ravel(), plotgrid_Y.ravel()]).T

radius = 4 # 3.5 might make some points closer to the edge of circle
            # might lead to numericla issues
radius_from_knots = np.repeat(radius, k) # ?influence radius from a knot?

# ------- 2. Generate the weight matrices ------------------------------------

# Weight matrix generated using Gaussian Smoothing Kernel
bandwidth = 4 # ?what is bandwidth?
gaussian_weight_matrix = np.full(shape = (num_sites, k), fill_value = np.nan)
for site_id in np.arange(num_sites):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                    XB = knots_xy)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
    gaussian_weight_matrix[site_id, :] = weight_from_knots

# Weight matrix generated using wendland basis
wendland_weight_matrix = np.full(shape = (num_sites,k), fill_value = np.nan)
for site_id in np.arange(num_sites):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                    XB = knots_xy)
    # influence coming from each of the knots
    weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
    wendland_weight_matrix[site_id, :] = weight_from_knots

gaussian_weight_matrix_for_plot = np.full(shape = (625, k), fill_value = np.nan)
for site_id in np.arange(625):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)), 
                                    XB = knots_xy)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
    gaussian_weight_matrix_for_plot[site_id, :] = weight_from_knots

wendland_weight_matrix_for_plot = np.full(shape = (625,k), fill_value = np.nan)
for site_id in np.arange(625):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)), 
                                    XB = knots_xy)
    # influence coming from each of the knots
    weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
    wendland_weight_matrix_for_plot[site_id, :] = weight_from_knots

# ------- 3. Generate covariance matrix, Z, and W --------------------------------

## range_vec
range_at_knots = np.sqrt(0.3*knots_x + 0.4*knots_y)/2 # scenario 2
range_vec = gaussian_weight_matrix @ range_at_knots

## sigsq_vec
sigsq_vec = np.repeat(sigsq, num_sites) # hold at 1

## Covariance matrix K
K = ns_cov(range_vec = range_vec, sigsq_vec = sigsq_vec,
        coords = sites_xy, kappa = nu, cov_model = "matern")
Z = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(num_sites,)),cov=K,size=N).T
W = norm_to_Pareto(Z) 
# W = norm_to_std_Pareto(Z)

# ------- 4. Generate Scaling Factor, R^phi --------------------------------

## phi_vec
match simulation_case:
    case 'scenario1':
        phi_at_knots = 0.65-np.sqrt((knots_x-3)**2/4 + (knots_y-3)**2/3)/10 # scenario 1
    case 'scenario2':
        phi_at_knots = 0.65-np.sqrt((knots_x-5.1)**2/5 + (knots_y-5.3)**2/4)/11.6 # scenario 2
    case 'scenario3':
        phi_at_knots = 10*(0.5*scipy.stats.multivariate_normal.pdf(knots_xy, 
                                                               mean = np.array([2.5,3]), 
                                                               cov = 2*np.matrix([[1,0.2],[0.2,1]])) + 
                        0.5*scipy.stats.multivariate_normal.pdf(knots_xy, 
                                                                mean = np.array([7,7.5]), 
                                                                cov = 2*np.matrix([[1,-0.2],[-0.2,1]]))) + \
                    0.37# scenario 3
# phi_at_knots = np.array([0.3]*k)
phi_vec = gaussian_weight_matrix @ phi_at_knots

## R
## Generate them at the knots
R_at_knots = np.full(shape = (k, N), fill_value = np.nan)
for t in np.arange(N):
    R_at_knots[:,t] = rlevy(n = k, m = delta, s = gamma) # generate R at time t, spatially varying k knots
    # R_at_knots[:,t] = np.repeat(rlevy(n = 1, m = delta, s = gamma), k) # generate R at time t, spatially constant k knots

## Matrix Multiply to the sites
R_at_sites = wendland_weight_matrix @ R_at_knots

## R^phi
R_phi = np.full(shape = (num_sites, N), fill_value = np.nan)
for t in np.arange(N):
    R_phi[:,t] = np.power(R_at_sites[:,t], phi_vec)

# # %%
# # ------- 6. Generate X and Y--------------------------------
X_star = R_phi * W

alpha = 0.5
gamma_at_knots = np.repeat(gamma, k)
gamma_vec = np.sum(np.multiply(wendland_weight_matrix, gamma_at_knots)**(alpha), 
                    axis = 1)**(1/alpha) # axis = 1 to sum over K knots
# gamma_vec is the gamma bar in the overleaf document

# Calculation of Y can(?) be parallelized by time(?)
Y = np.full(shape=(num_sites, N), fill_value = np.nan)
for t in np.arange(N):
    Y[:,t] = qgev(pRW(X_star[:,t], phi_vec, gamma_vec), mu, tau, ksi)

# folder = './data/scenario2/simulation_1/'
# phi_knots_trace = np.load(folder + 'phi_knots_trace.npy')
# R_trace_log = np.load(folder + 'R_trace_log.npy')
# range_knots_trace = np.load(folder + 'range_knots_trace.npy')
# GEV_knots_trace = np.load(folder + 'GEV_knots_trace.npy')














####################################################################################################
# %%
# Coverage setup
####################################################################################################


match simulation_case:
    case 'scenario1':
        assert (phi_at_knots == 0.65-np.sqrt((knots_x-3)**2/4 + (knots_y-3)**2/3)/10).all() # scenario 1
    case 'scenario2':
        assert (phi_at_knots == 0.65-np.sqrt((knots_x-5.1)**2/5 + (knots_y-5.3)**2/4)/11.6).all() # scenario 2
    case 'scenario3':
        assert (phi_at_knots == 10*(0.5*scipy.stats.multivariate_normal.pdf(knots_xy, 
                                                               mean = np.array([2.5,3]), 
                                                               cov = 2*np.matrix([[1,0.2],[0.2,1]])) + 
                        0.5*scipy.stats.multivariate_normal.pdf(knots_xy, 
                                                                mean = np.array([7,7.5]), 
                                                                cov = 2*np.matrix([[1,-0.2],[-0.2,1]]))) + \
                    0.37).all()# scenario 3

sim_id_from = 1
sim_id_to = 50
sim_ids = np.arange(start = sim_id_from, stop = sim_id_to + 1)

# bad sim for scenario 1 with 9 knots CORRECT VERSION
bad_sim_ids = np.array([])

# bad sim for scenario 1 with 9 knots
# bad_sim_ids = np.array([32, # absolutely bad
#                         5, 9, 24, 29, 33, 34, 35, 39, 43, # ah??
#                         3, 26]) # biased

# bad sim for 9 knots scenario 2
# bad_sim_ids = np.array([3,5,6,26,29,32,39, # absolutely bad
#                          8, 9, 11, 12, 13, 20, 22, 23, 24, 25, 33, 41, 42, 43,# biased
#                          1, 4, 19, 34, 35, 46]) # ah??

# bad sim for scenario 3 with 9 knots
# bad_sim_ids = np.array([3, 6, 32, 46, # absolutely bad
#                         4, 5, 12, 23, 25, 26, 33, 34, 36, 39, 41, 42, 43, 44]) # ah??
#                         # 8, 9, 11, 13, 22, 24, 29, 35, 48]) # biased

# bad_sim_ids = np.array([3,4,6,8,10,12,13,24,26]) # bad sim for 10 knots scenario 2

for bad_sim_id in bad_sim_ids:
    sim_ids = np.delete(sim_ids, np.argwhere(sim_ids == bad_sim_id))
nsim = len(sim_ids)


# %%
########################
# individual coverage  #
########################
# load data and calculate statistics
PE_matrix_phi = np.full(shape = (k, nsim), fill_value = np.nan)
lower_bound_matrix_phi = np.full(shape = (k, nsim), fill_value = np.nan)
upper_bound_matrix_phi = np.full(shape = (k, nsim), fill_value = np.nan)
PE_matrix_range = np.full(shape = (k, nsim), fill_value = np.nan)
lower_bound_matrix_range = np.full(shape = (k, nsim), fill_value = np.nan)
upper_bound_matrix_range = np.full(shape = (k, nsim), fill_value = np.nan)
PE_matrix_Rt_log = np.full(shape = (k, N, nsim), fill_value = np.nan)
lower_bound_matrix_Rt_log = np.full(shape = (k, N, nsim), fill_value = np.nan)
upper_bound_matrix_Rt_log = np.full(shape = (k, N, nsim), fill_value = np.nan)
PE_matrix_loc = np.full(shape = (k, nsim), fill_value = np.nan)
lower_bound_matrix_loc = np.full(shape = (k, nsim), fill_value = np.nan)
upper_bound_matrix_loc = np.full(shape = (k, nsim), fill_value = np.nan)
PE_matrix_scale = np.full(shape = (k, nsim), fill_value = np.nan)
lower_bound_matrix_scale = np.full(shape = (k, nsim), fill_value = np.nan)
upper_bound_matrix_scale = np.full(shape = (k, nsim), fill_value = np.nan)
# PE_matrix_shape = np.full(shape = (k, nsim), fill_value = np.nan)
# lower_bound_matrix_shape = np.full(shape = (k, nsim), fill_value = np.nan)
# upper_bound_matrix_shape = np.full(shape = (k, nsim), fill_value = np.nan)

folders = ['./data/'+simulation_case+'/simulation_' + str(sim_id) + '/' for sim_id in sim_ids]
for i in range(nsim):
    folder = folders[i]

    # load
    phi_knots_trace = np.load(folder + 'phi_knots_trace.npy')
    R_trace_log = np.load(folder + 'R_trace_log.npy')
    range_knots_trace = np.load(folder + 'range_knots_trace.npy')
    GEV_knots_trace = np.load(folder + 'GEV_knots_trace.npy')

    # drop burnins
    phi_knots_trace = phi_knots_trace[burnins:]
    R_trace_log = R_trace_log[burnins:]
    range_knots_trace = range_knots_trace[burnins:]
    GEV_knots_trace = GEV_knots_trace[burnins:]

    # phi
    PE_matrix_phi[:,i] = np.mean(phi_knots_trace, axis = 0)
    lower_bound_matrix_phi[:,i] = np.quantile(phi_knots_trace, q = 0.025, axis = 0)
    upper_bound_matrix_phi[:,i] = np.quantile(phi_knots_trace, q = 0.975, axis = 0)
    # range
    PE_matrix_range[:,i] = np.mean(range_knots_trace, axis = 0)
    lower_bound_matrix_range[:,i] = np.quantile(range_knots_trace, q = 0.025, axis = 0)
    upper_bound_matrix_range[:,i] = np.quantile(range_knots_trace, q = 0.975, axis = 0)
    # Rt
    PE_matrix_Rt_log[:,:,i] = np.mean(R_trace_log, axis = 0)
    lower_bound_matrix_Rt_log[:,:,i] = np.quantile(R_trace_log, q = 0.025, axis = 0)
    upper_bound_matrix_Rt_log[:,:,i] = np.quantile(R_trace_log, q = 0.975, axis = 0)
    # loc
    PE_matrix_loc[:,i] = np.mean(GEV_knots_trace[:,0,:], axis = 0)
    lower_bound_matrix_loc[:,i] = np.quantile(GEV_knots_trace[:,0,:], q = 0.025, axis = 0)
    upper_bound_matrix_loc[:,i] = np.quantile(GEV_knots_trace[:,0,:], q = 0.975, axis = 0)
    # scale
    PE_matrix_scale[:,i] = np.mean(GEV_knots_trace[:,1,:], axis = 0)
    lower_bound_matrix_scale[:,i] = np.quantile(GEV_knots_trace[:,1,:], q = 0.025, axis = 0)
    upper_bound_matrix_scale[:,i] = np.quantile(GEV_knots_trace[:,1,:], q = 0.975, axis = 0)
    # # shape
    # PE_matrix_shape[:,i] = np.mean(GEV_knots_trace[:,2,:], axis = 0)
    # lower_bound_matrix_shape[:,i] = np.quantile(GEV_knots_trace[:,2,:], q = 0.025, axis = 0)
    # upper_bound_matrix_shape[:,i] = np.quantile(GEV_knots_trace[:,2,:], q = 0.975, axis = 0)

# %%
# make plots for phi
for knot_id in range(k):
    fig, ax = plt.subplots()
    ax.hlines(y = phi_at_knots[knot_id], xmin = sim_id_from, xmax = sim_id_to,
            color = 'black')
    coloring = ['red' if type1 == True else 'green' 
                for type1 in np.logical_or(lower_bound_matrix_phi[knot_id,:] > phi_at_knots[knot_id], 
                                            upper_bound_matrix_phi[knot_id,:] < phi_at_knots[knot_id])]
    plt.errorbar(x = sim_ids, y = PE_matrix_phi[knot_id,:], 
                yerr = np.vstack((PE_matrix_phi[knot_id,:] - lower_bound_matrix_phi[knot_id,:], 
                                  upper_bound_matrix_phi[knot_id,:] - PE_matrix_phi[knot_id,:])), 
                fmt = 'o',
                ecolor = coloring) # errorbar yerr is for size
    plt.title('knot: ' + str(knot_id) + ' phi = ' + str(round(phi_at_knots[knot_id],3)))
    plt.xlabel('simulation number')
    plt.ylabel('phi')
    plt.show()
    fig.savefig('phi_knot_' + str(knot_id) + '.pdf')
    plt.close()

# %%
# make plots for range
for knot_id in range(k):
    fig, ax = plt.subplots()
    ax.hlines(y = range_at_knots[knot_id], xmin = sim_id_from, xmax = sim_id_to,
            color = 'black')
    coloring = ['red' if type1 == True else 'green' 
            for type1 in np.logical_or(lower_bound_matrix_range[knot_id,:] > range_at_knots[knot_id], 
                                        upper_bound_matrix_range[knot_id,:] < range_at_knots[knot_id])]
    plt.errorbar(x = sim_ids, y = PE_matrix_range[knot_id,:], 
                yerr = np.vstack((PE_matrix_range[knot_id,:] - lower_bound_matrix_range[knot_id,:], 
                                  upper_bound_matrix_range[knot_id,:] - PE_matrix_range[knot_id,:])), 
                fmt = 'o',
                ecolor = coloring)
    plt.title('knot: ' + str(knot_id) + ' range = ' + str(round(range_at_knots[knot_id],3)))
    plt.xlabel('simulation number')
    plt.ylabel('range')
    plt.show()
    fig.savefig('range_knot_' + str(knot_id) + '.pdf')
    plt.close()

# %%
# make plots for loc
# for knot_id in range(k):
knot_id = 0
fig, ax = plt.subplots()
ax.hlines(y = mu, xmin = sim_id_from, xmax = sim_id_to,
        color = 'black')
coloring = ['red' if type1 == True else 'green' 
            for type1 in np.logical_or(lower_bound_matrix_loc[knot_id,:] > mu, 
                                        upper_bound_matrix_loc[knot_id,:] < mu)]
plt.errorbar(x = sim_ids, 
            y = PE_matrix_loc[knot_id,:], 
            yerr = np.vstack((PE_matrix_loc[knot_id,:] - lower_bound_matrix_loc[knot_id,:], 
                                upper_bound_matrix_loc[knot_id,:] - PE_matrix_loc[knot_id,:])), 
            fmt = 'o',
            ecolor = coloring) # errorbar yerr is for size
plt.title('knot: ' + str(knot_id) + ' loc = ' + str(mu))
plt.xlabel('simulation number')
plt.ylabel('loc')
plt.show()
fig.savefig('loc_knot_' + str(knot_id) + '.pdf')
plt.close()

# %%
# make plots for scale
# for knot_id in range(k):
knot_id = 0
fig, ax = plt.subplots()
ax.hlines(y = tau, xmin = sim_id_from, xmax = sim_id_to,
        color = 'black')
coloring = ['red' if type1 == True else 'green' 
            for type1 in np.logical_or(lower_bound_matrix_scale[knot_id,:] > tau, 
                                        upper_bound_matrix_scale[knot_id,:] < tau)]
plt.errorbar(x = sim_ids, 
            y = PE_matrix_scale[knot_id,:], 
            yerr = np.vstack((PE_matrix_scale[knot_id,:] - lower_bound_matrix_scale[knot_id,:], 
                                upper_bound_matrix_scale[knot_id,:] - PE_matrix_scale[knot_id,:])), 
            fmt = 'o',
            ecolor = coloring) # errorbar yerr is for size
plt.title('knot: ' + str(knot_id) + ' scale = ' + str(tau))
plt.xlabel('simulation number')
plt.ylabel('scale')
plt.show()
fig.savefig('scale_knot_' + str(knot_id) + '.pdf')
plt.close()

# # %%
# # make plots for log(R_t)
# #################################################################### #
# Cannot make these plots because R_t is different for each simulation #
# #################################################################### #
# t = 0
# for knot_id in range(k):
#     fig, ax = plt.subplots()
#     ax.hlines(y = np.log(R_at_knots[knot_id,t]), xmin = sim_id_from, xmax = nsim,
#             color = 'black')
#     coloring = ['red' if type1 == True else 'green' 
#             for type1 in np.logical_or(lower_bound_matrix_Rt_log[knot_id,t,:] > np.log(R_at_knots)[knot_id,t], 
#                                         upper_bound_matrix_Rt_log[knot_id,t,:] < np.log(R_at_knots)[knot_id,t])]
#     plt.errorbar(x = 1 + np.arange(nsim), 
#                 y = PE_matrix_Rt_log[knot_id,t,:], 
#                 yerr = np.vstack((PE_matrix_Rt_log[knot_id,t,:] - lower_bound_matrix_Rt_log[knot_id,t,:], 
#                                   upper_bound_matrix_Rt_log[knot_id,t,:] - PE_matrix_Rt_log[knot_id,t,:])), 
#                 fmt = 'o',
#                 ecolor = coloring)
#     plt.title('knot: ' + str(knot_id) + ' t = ' + str(t) +' log(Rt) = ' + str(round(np.log(R_at_knots[knot_id,t]),3)))
#     plt.xlabel('simulation number')
#     plt.ylabel('Rt')
#     plt.show()
#     fig.savefig('R_knot_' + str(knot_id) + '_t_' + str(t) + '.pdf')
#     plt.close()


# %%
# Calculate Coverage
phi_type1 = np.full(shape=(k, nsim), fill_value=np.nan)
range_type1 = np.full(shape=(k,nsim), fill_value=np.nan)
# R_type1 = np.full(shape=(k,N,nsim), fill_value=np.nan)
for knot_id in range(k):
    phi_type1[knot_id,:] = np.logical_or(lower_bound_matrix_phi[knot_id,:] > phi_at_knots[knot_id],
                                         upper_bound_matrix_phi[knot_id,:] < phi_at_knots[knot_id])
    range_type1[knot_id,:] = np.logical_or(lower_bound_matrix_range[knot_id,:] > range_at_knots[knot_id], 
                                           upper_bound_matrix_range[knot_id,:] < range_at_knots[knot_id])
    # for t in range(N):
    #     R_type1[knot_id,t,:] = np.logical_or(lower_bound_matrix_Rt_log[knot_id,t,:] > np.log(R_at_knots)[knot_id,t], 
    #                                     upper_bound_matrix_Rt_log[knot_id,t,:] < np.log(R_at_knots)[knot_id,t])

np.mean(phi_type1) # overall type 1 error
np.mean(phi_type1, axis = 1) # type 1 error of phi at each knot

np.mean(range_type1) # overall type 1 error
np.mean(phi_type1, axis = 1) # type 1 error of range at each knot

# np.mean(R_type1) # overall type 1 error
# np.mean(R_type1, axis = 2) # type 1 error of R at each knot each time



############################
# overall avg coverage phi #
############################
# %%
# overall coverage for phi
alphas = np.flip(np.linspace(0.05, 0.4, 15))
q_low = alphas/2
q_high = 1 - q_low

# summary by credible levels
PE_matrix_phi = np.full(shape = (k, nsim), fill_value = np.nan)
lower_bound_matrix_phi_alpha = np.full(shape = (k, nsim, len(alphas)), fill_value = np.nan)
upper_bound_matrix_phi_alpha = np.full(shape = (k, nsim, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    for i in range(nsim):
        folder = folders[i]
        phi_knots_trace = np.load(folder + 'phi_knots_trace.npy')
        # drop burnins
        phi_knots_trace = phi_knots_trace[burnins:]
        # phi
        PE_matrix_phi[:,i] = np.mean(phi_knots_trace, axis = 0)
        lower_bound_matrix_phi_alpha[:,i, level_i] = np.quantile(phi_knots_trace, q = q_low[level_i], axis = 0)
        upper_bound_matrix_phi_alpha[:,i, level_i] = np.quantile(phi_knots_trace, q = q_high[level_i], axis = 0)

# coverage flag
phi_covers = np.full(shape = (k, nsim, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    for knot_id in range(k):
        phi_covers[knot_id, :, level_i] = \
            np.logical_and(lower_bound_matrix_phi_alpha[knot_id,:,level_i] < phi_at_knots[knot_id], 
                            upper_bound_matrix_phi_alpha[knot_id,:, level_i] > phi_at_knots[knot_id])

# average coverage
avg_phi_covers = np.mean(phi_covers, axis = 1)
se_phi_covers = scipy.stats.sem(phi_covers, axis = 1)

# plotting
for knot_id in range(k):
    fig, ax = plt.subplots()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ax.plot([0,1],[0,1], transform=ax.transAxes, color = 'black')
    plt.errorbar(x = 1 - alphas, 
                y = avg_phi_covers[knot_id,:], 
                yerr = 1.96*se_phi_covers[knot_id,:],
                fmt = 'o')
    plt.title('phi knot ' + str(knot_id))
    plt.ylabel('empirical coverage w/ 1.96*SE')
    plt.xlabel('1-alpha')
    plt.show()
    fig.savefig('phi_knot_' + str(knot_id) + '_avg' + '.pdf')
    plt.close()


##############################
# overall avg coverage range #
##############################
# %%
# overall coverage for range

alphas = np.flip(np.linspace(0.05, 0.4, 15))
q_low = alphas/2
q_high = 1 - q_low

# summary by credible levels
PE_matrix_range = np.full(shape = (k, nsim), fill_value = np.nan)
lower_bound_matrix_range_alpha = np.full(shape = (k, nsim, len(alphas)), fill_value = np.nan)
upper_bound_matrix_range_alpha = np.full(shape = (k, nsim, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    for i in range(nsim):
        folder = folders[i]
        range_knots_trace = np.load(folder + 'range_knots_trace.npy')
        # drop burnins
        range_knots_trace = range_knots_trace[burnins:]
        # range
        PE_matrix_range[:,i] = np.mean(range_knots_trace, axis = 0)
        lower_bound_matrix_range_alpha[:,i, level_i] = np.quantile(range_knots_trace, q = q_low[level_i], axis = 0)
        upper_bound_matrix_range_alpha[:,i, level_i] = np.quantile(range_knots_trace, q = q_high[level_i], axis = 0)

# coverage flag
range_covers = np.full(shape = (k, nsim, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    for knot_id in range(k):
        range_covers[knot_id, :, level_i] = \
            np.logical_and(lower_bound_matrix_range_alpha[knot_id,:,level_i] < range_at_knots[knot_id], 
                            upper_bound_matrix_range_alpha[knot_id,:, level_i] > range_at_knots[knot_id])

# average coverage
avg_range_covers = np.mean(range_covers, axis = 1)
# std_range_covers = np.std(range_covers, axis = 1)
se_range_covers = scipy.stats.sem(range_covers, axis = 1)

# plotting
for knot_id in range(k):
    fig, ax = plt.subplots()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ax.plot([0,1],[0,1], transform=ax.transAxes, color = 'black')
    plt.errorbar(x = 1 - alphas, 
                y = avg_range_covers[knot_id,:], 
                yerr = 1.96*se_range_covers[knot_id,:],
                fmt = 'o')
    plt.title('range knot ' + str(knot_id))
    plt.ylabel('empirical coverage w/ 1.96*SE')
    plt.xlabel('1-alpha')
    plt.show()
    fig.savefig('range_knot_' + str(knot_id) + '_avg' + '.pdf')
    plt.close()

# %%

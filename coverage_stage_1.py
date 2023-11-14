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

#####################################################################################################################
# Generating Dataset ################################################################################################
#####################################################################################################################
# %%
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

## Remember to change below
# knots locations
# radius
# range at knots
# phi_at_knots
# phi_post_cov
# range_post_cov
n_iters = 1000

# %%
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
knots_x = knots_xy[:,0]
knots_y = knots_xy[:,1]

plotgrid_x = np.linspace(0.1,10,25)
plotgrid_y = np.linspace(0.1,10,25)
plotgrid_X, plotgrid_Y = np.meshgrid(plotgrid_x, plotgrid_y)
plotgrid_xy = np.vstack([plotgrid_X.ravel(), plotgrid_Y.ravel()]).T

radius = 4 # 3.5 might make some points closer to the edge of circle
            # might lead to numericla issues
radius_from_knots = np.repeat(radius, k) # ?influence radius from a knot?

# Plot the space
# fig, ax = plt.subplots()
# ax.plot(sites_x, sites_y, 'b.', alpha = 0.4)
# ax.plot(knots_x, knots_y, 'r+')
# space_rectangle = plt.Rectangle(xy = (0,0), width = 10, height = 10,
#                                 fill = False, color = 'black')
# for i in range(k):
#     circle_i = plt.Circle((knots_xy[i,0],knots_xy[i,1]), radius_from_knots[0], 
#                      color='r', fill=True, fc='grey', ec = 'red', alpha = 0.2)
#     ax.add_patch(circle_i)
# ax.add_patch(space_rectangle)
# plt.xlim([-2,12])
# plt.ylim([-2,12])
# plt.show()
# plt.close()

# %%
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


# %%
# ------- 3. Generate covariance matrix, Z, and W --------------------------------

## range_vec
range_at_knots = np.sqrt(0.3*knots_x + 0.4*knots_y)/2 # scenario 2
range_vec = gaussian_weight_matrix @ range_at_knots

# range_vec_for_plot = gaussian_weight_matrix_for_plot @ range_at_knots
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(projection='3d')
# ax2.plot_trisurf(plotgrid_xy[:,0], plotgrid_xy[:,1], range_vec_for_plot, linewidth=0.2, antialiased=True)
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('phi(s)')
# ax2.scatter(knots_x, knots_y, range_at_knots, c='red', marker='o', s=100)
# plt.show()
# plt.close()

## sigsq_vec
sigsq_vec = np.repeat(sigsq, num_sites) # hold at 1

## Covariance matrix K
K = ns_cov(range_vec = range_vec, sigsq_vec = sigsq_vec,
        coords = sites_xy, kappa = nu, cov_model = "matern")
Z = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(num_sites,)),cov=K,size=N).T
W = norm_to_Pareto(Z) 
# W = norm_to_std_Pareto(Z)

# %%
# ------- 4. Generate Scaling Factor, R^phi --------------------------------

## phi_vec
phi_at_knots = 0.65-np.sqrt((knots_x-5.1)**2/5 + (knots_y-5.3)**2/4)/11.6 # scenario 2
# phi_at_knots = np.array([0.3]*k)
phi_vec = gaussian_weight_matrix @ phi_at_knots

# phi_vec_for_plot = gaussian_weight_matrix_for_plot @ phi_at_knots
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(plotgrid_X, plotgrid_Y, np.matrix(phi_vec_for_plot).reshape(25,25))
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('phi(s)')
# ax.scatter(knots_x, knots_y, phi_at_knots, c='red', marker='o', s=100)
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(projection='3d')
# ax2.plot_trisurf(plotgrid_xy[:,0], plotgrid_xy[:,1], phi_vec_for_plot, linewidth=0.2, antialiased=True)
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('phi(s)')
# ax2.scatter(knots_x, knots_y, phi_at_knots, c='red', marker='o', s=100)
# plt.show()
# plt.close()

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
# X_star = R_phi * W

# # Calculation of Y can(?) be parallelized by time(?)
# Y = np.full(shape=(num_sites, N), fill_value = np.nan)
# for t in np.arange(N):
#     Y[:,t] = qgev(pRW(X_star[:,t], phi_vec, gamma), mu, tau, ksi)

# folder = './data/scenario2/simulation_1/'
# phi_knots_trace = np.load(folder + 'phi_knots_trace.npy')
# R_trace_log = np.load(folder + 'R_trace_log.npy')
# range_knots_trace = np.load(folder + 'range_knots_trace.npy')
# GEV_knots_trace = np.load(folder + 'GEV_knots_trace.npy')








########################
# individual coverage  #
########################

# %%
# load data and calculate statistics
sim_id_from = 1
sim_id_to = 30
sim_ids = np.arange(start = sim_id_from, stop = sim_id_to+1)
nsim = len(sim_ids)

PE_matrix_loc = np.full(shape = (k, nsim), fill_value = np.nan)
lower_bound_matrix_loc = np.full(shape = (k, nsim), fill_value = np.nan)
upper_bound_matrix_loc = np.full(shape = (k, nsim), fill_value = np.nan)
PE_matrix_scale = np.full(shape = (k, nsim), fill_value = np.nan)
lower_bound_matrix_scale = np.full(shape = (k, nsim), fill_value = np.nan)
upper_bound_matrix_scale = np.full(shape = (k, nsim), fill_value = np.nan)
PE_matrix_shape = np.full(shape = (k, nsim), fill_value = np.nan)
lower_bound_matrix_shape = np.full(shape = (k, nsim), fill_value = np.nan)
upper_bound_matrix_shape = np.full(shape = (k, nsim), fill_value = np.nan)

folders = ['./data/stage_1/simulation_' + str(sim_id) + '/' for sim_id in sim_ids]
for i in range(nsim):
    folder = folders[i]
    GEV_knots_trace_stage1 = np.load(folder + 'GEV_knots_trace_stage1.npy')
    # loc
    PE_matrix_loc[:,i] = np.mean(GEV_knots_trace_stage1[:,0,:], axis = 0)
    lower_bound_matrix_loc[:,i] = np.quantile(GEV_knots_trace_stage1[:,0,:], q = 0.025, axis = 0)
    upper_bound_matrix_loc[:,i] = np.quantile(GEV_knots_trace_stage1[:,0,:], q = 0.975, axis = 0)
    # scale
    PE_matrix_scale[:,i] = np.mean(GEV_knots_trace_stage1[:,1,:], axis = 0)
    lower_bound_matrix_scale[:,i] = np.quantile(GEV_knots_trace_stage1[:,1,:], q = 0.025, axis = 0)
    upper_bound_matrix_scale[:,i] = np.quantile(GEV_knots_trace_stage1[:,1,:], q = 0.975, axis = 0)
    # shape
    PE_matrix_shape[:,i] = np.mean(GEV_knots_trace_stage1[:,2,:], axis = 0)
    lower_bound_matrix_shape[:,i] = np.quantile(GEV_knots_trace_stage1[:,2,:], q = 0.025, axis = 0)
    upper_bound_matrix_shape[:,i] = np.quantile(GEV_knots_trace_stage1[:,2,:], q = 0.975, axis = 0)

# %%
# make plots for loc
# for knot_id in range(k):
knot_id = 0
fig, ax = plt.subplots()
ax.hlines(y = mu, xmin = 1, xmax = nsim,
        color = 'black')
coloring = ['red' if type1 == True else 'green' 
            for type1 in np.logical_or(lower_bound_matrix_loc[knot_id,:] > mu, 
                                        upper_bound_matrix_loc[knot_id,:] < mu)]
plt.errorbar(x = 1 + np.arange(nsim), 
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

# bias
np.mean(PE_matrix_loc, axis = 1)[knot_id] - mu

# %%
# make plots for scale
# for knot_id in range(k):
knot_id = 0
fig, ax = plt.subplots()
ax.hlines(y = tau, xmin = 1, xmax = nsim,
        color = 'black')
coloring = ['red' if type1 == True else 'green' 
            for type1 in np.logical_or(lower_bound_matrix_scale[knot_id,:] > tau, 
                                        upper_bound_matrix_scale[knot_id,:] < tau)]
plt.errorbar(x = 1 + np.arange(nsim), 
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

# bias
np.mean(PE_matrix_scale, axis = 1)[knot_id] - tau

# %%
# make plots for shpae
# for knot_id in range(k):
knot_id = 0
fig, ax = plt.subplots()
ax.hlines(y = ksi, xmin = 1, xmax = nsim,
        color = 'black')
coloring = ['red' if type1 == True else 'green' 
            for type1 in np.logical_or(lower_bound_matrix_shape[knot_id,:] > ksi, 
                                        upper_bound_matrix_shape[knot_id,:] < ksi)]
plt.errorbar(x = 1 + np.arange(nsim), 
            y = PE_matrix_shape[knot_id,:], 
            yerr = np.vstack((PE_matrix_shape[knot_id,:] - lower_bound_matrix_shape[knot_id,:], 
                                upper_bound_matrix_shape[knot_id,:] - PE_matrix_shape[knot_id,:])), 
            fmt = 'o',
            ecolor = coloring) # errorbar yerr is for size
plt.title('knot: ' + str(knot_id) + ' shape = ' + str(ksi))
plt.xlabel('simulation number')
plt.ylabel('shape')
plt.show()
fig.savefig('shape_knot_' + str(knot_id) + '.pdf')
plt.close()

# bias
np.mean(PE_matrix_shape, axis = 1)[knot_id] - ksi


########################
# overall avg coverage #
########################


# %%
# overall coverage for phi
# load good simulations
sim_id_from = 1
sim_id_to = 30
sim_ids = np.arange(start = sim_id_from, stop = sim_id_to + 1)
bad_sim_ids = np.array([])
for bad_sim_id in bad_sim_ids:
    sim_ids = np.delete(sim_ids, np.argwhere(sim_ids == bad_sim_id))
nsim = len(sim_ids)

folders = ['./data/scenario2_phiprior/simulation_' + str(sim_id) + '/' for sim_id in sim_ids]
alphas = np.flip(np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]))
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

# %%
# overall coverage for range
# load good simulations
sim_id_from = 1
sim_id_to = 30
sim_ids = np.arange(start = sim_id_from, stop = sim_id_to + 1)
bad_sim_ids = np.array([3])
for bad_sim_id in bad_sim_ids:
    sim_ids = np.delete(sim_ids, np.argwhere(sim_ids == bad_sim_id))
nsim = len(sim_ids)

folders = ['./data/scenario2_phiprior/simulation_' + str(sim_id) + '/' for sim_id in sim_ids]
alphas = np.flip(np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]))
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

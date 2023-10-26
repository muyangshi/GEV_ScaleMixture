# This is a MCMC sampler that constantly gets updated
# Scratch work and modifications are done in this file
#%%
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

# %%
# ------- 0. Simulation Setting --------------------------------------

## space setting
np.random.seed(2345) # 1
# np.random.seed(79) # 2
N = 64 # number of time replicates
num_sites = 625 # number of sites/stations
k = 9 # number of knots

## unchanged constants or parameters
gamma = 0.5 # this is the gamma that goes in rlevy
delta = 0.0 # this is the delta in levy, stays 0
mu = 0.0 # GEV location
tau = 1.0 # GEV scale
ksi = 0.2 # GEV shape
nu = 0.5 # exponential kernel for matern with nu = 1/2

## Remember to change below
# knots locations
# radius
# range at knots
# phi_at_knots
# phi_post_cov
# range_post_cov
n_iters = 10000

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
# rho = 2.0 # the rho in matern kernel exp(-rho * x)
# length_scale = 1/rho # scikit/learn parameterization (length_scale)
# range_at_knots = np.full(shape = k, fill_value = length_scale) # array([0.5, 0.5])
# range_at_knots = np.array([0.3,0.3,0.3,
#                            0.3,0.3,0.3,
#                            0.3,0.3,0.3])
# range_vec = gaussian_weight_matrix @ range_at_knots

## range_vec
range_at_knots = np.sqrt(0.3*knots_x + 0.4*knots_y)/2
# range_at_knots = np.array([0.3]*9)
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
sigsq_vec = np.repeat(1, num_sites) # hold at 1

## Covariance matrix K
K = ns_cov(range_vec = range_vec, sigsq_vec = sigsq_vec,
           coords = sites_xy, kappa = nu, cov_model = "matern")
Z = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(num_sites,)),cov=K,size=N).T
W = norm_to_Pareto1(Z) 
# W = norm_to_std_Pareto(Z)

# %%
# ------- 4. Generate Scaling Factor, R^phi --------------------------------

## phi_vec
# phi_at_knots = np.full(shape = k, fill_value = 0.33)
# phi_at_knots = np.array([0.2,0.2,0.2,
#                          0.2,0.8,0.2,
#                          0.2,0.2,0.2])
phi_at_knots = 0.65-np.sqrt((knots_x-5.1)**2/5 + (knots_y-5.3)**2/4)/11.6
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
    R_at_knots[:,t] = rlevy(n = k, m = delta, s = gamma) # generate R at time t

## Matrix Multiply to the sites
R_at_sites = wendland_weight_matrix @ R_at_knots

## R^phi
R_phi = np.full(shape = (num_sites, N), fill_value = np.nan)
for t in np.arange(N):
    R_phi[:,t] = np.power(R_at_sites[:,t], phi_vec)

# %%
# ------- 5. Generate X and Y--------------------------------
X_star = R_phi * W

# Calculation of Y can(?) be parallelized by time(?)
Y = np.full(shape=(num_sites, N), fill_value = np.nan)
for t in np.arange(N):
    Y[:,t] = qgev(pRW(X_star[:,t], phi_vec, gamma), mu, tau, ksi)

###########################################################################################
# %%
# ------- 6. Other Preparational Stuff(?) --------------------------------

Loc_matrix = np.full(shape = Y.shape, fill_value = mu)
Scale_matrix = np.full(shape = Y.shape, fill_value = tau)
Shape_matrix = np.full(shape = Y.shape, fill_value = ksi)
R_matrix = R_at_sites
gamma_vec = np.repeat(gamma, num_sites)
cholesky_matrix = scipy.linalg.cholesky(K, lower=False)

###########################################################################################
# Metropolis Updates ######################################################################
###########################################################################################

# %%
# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

random_generator = np.random.RandomState((rank+1)*7) # use of this avoids impacting the global np state

if rank == 0:
    start_time = time.time()

# %%
# ------- Preparation for Adaptive Metropolis -----------------------------

# constant to control adaptive Metropolis updates
c_0 = 1
c_1 = 0.8
offset = 3 # the iteration offset?
# r_opt_1d = .41
# r_opt_2d = .35
# r_opt = 0.234 # asymptotically
r_opt = .35
# eps = 1e-6

# posterior covariance matrix from trial run
phi_post_cov = 0.1*np.array([[ 1.02091028e-02, -4.65416081e-03,  2.83121774e-03,
        -1.98334540e-03,  6.43880363e-05,  2.91578540e-03,
         8.30697725e-04,  1.51244094e-03, -1.69856434e-03],
       [-4.65416081e-03,  1.78529332e-02, -6.46019647e-03,
        -3.08993583e-03, -2.22328955e-03, -8.82329739e-03,
         9.67951943e-04,  5.10212543e-03, -2.79892766e-03],
       [ 2.83121774e-03, -6.46019647e-03,  1.69812250e-02,
         4.33404401e-03, -4.30488074e-03, -5.48951838e-03,
        -3.58441843e-03,  1.83117898e-03, -5.07705637e-03],
       [-1.98334540e-03, -3.08993583e-03,  4.33404401e-03,
         9.80545980e-03, -6.43336271e-03,  3.85552957e-03,
        -2.20377952e-03, -1.11189476e-03,  3.01694619e-04],
       [ 6.43880363e-05, -2.22328955e-03, -4.30488074e-03,
        -6.43336271e-03,  1.79583705e-02, -4.12401123e-03,
         2.36776379e-04, -3.74790010e-03, -1.45715836e-03],
       [ 2.91578540e-03, -8.82329739e-03, -5.48951838e-03,
         3.85552957e-03, -4.12401123e-03,  2.59920768e-02,
         4.76402918e-03, -9.31711711e-03,  9.97438067e-03],
       [ 8.30697725e-04,  9.67951943e-04, -3.58441843e-03,
        -2.20377952e-03,  2.36776379e-04,  4.76402918e-03,
         5.19253871e-03, -2.10837952e-03,  2.51398841e-03],
       [ 1.51244094e-03,  5.10212543e-03,  1.83117898e-03,
        -1.11189476e-03, -3.74790010e-03, -9.31711711e-03,
        -2.10837952e-03,  1.20696319e-02, -5.23805452e-03],
       [-1.69856434e-03, -2.79892766e-03, -5.07705637e-03,
         3.01694619e-04, -1.45715836e-03,  9.97438067e-03,
         2.51398841e-03, -5.23805452e-03,  1.13886174e-02]])

range_post_cov = 0.1*np.array([[ 0.03558729, -0.01194256,  0.01789309, -0.02617062,  0.03662592,
        -0.02371117,  0.00875232, -0.00807813, -0.01552671],
       [-0.01194256,  0.04356771, -0.03063381,  0.01660483, -0.04042018,
         0.02988818,  0.0138222 ,  0.00860835,  0.0198619 ],
       [ 0.01789309, -0.03063381,  0.05484638, -0.02905475,  0.04804216,
        -0.04557623, -0.00314149, -0.02029084, -0.02739919],
       [-0.02617062,  0.01660483, -0.02905475,  0.10147915, -0.09118452,
         0.13567129,  0.02498169, -0.04548769,  0.06119967],
       [ 0.03662592, -0.04042018,  0.04804216, -0.09118452,  0.19990843,
        -0.17453938, -0.04141167,  0.02987613, -0.08793701],
       [-0.02371117,  0.02988818, -0.04557623,  0.13567129, -0.17453938,
         0.25707198,  0.06007342, -0.0865937 ,  0.10147855],
       [ 0.00875232,  0.0138222 , -0.00314149,  0.02498169, -0.04141167,
         0.06007342,  0.0469336 , -0.03794883,  0.0275728 ],
       [-0.00807813,  0.00860835, -0.02029084, -0.04548769,  0.02987613,
        -0.0865937 , -0.03794883,  0.10611076, -0.02885252],
       [-0.01552671,  0.0198619 , -0.02739919,  0.06119967, -0.08793701,
         0.10147855,  0.0275728 , -0.02885252,  0.07082559]])

GEV_post_cov = np.array([[0.00290557, 0.00159124, 0],
                         [0.00159124, 0.0010267,  0],
                         [0         , 0         , 1]])

# Scalors for adaptive updates
# (phi, range, GEV) these parameters are only proposed on worker 0
if rank == 0: 
    sigma_m_sq = {}
    sigma_m_sq['phi_block1'] = (2.4**2)/3
    sigma_m_sq['phi_block2'] = (2.4**2)/3
    sigma_m_sq['phi_block3'] = (2.4**2)/3
    sigma_m_sq['range_block1'] = (2.4**2)/3
    sigma_m_sq['range_block2'] = (2.4**2)/3
    sigma_m_sq['range_block3'] = (2.4**2)/3
    sigma_m_sq['GEV'] = (2.4**2)/3

    # initialize them with posterior covariance matrix
    Sigma_0 = {}
    Sigma_0['phi_block1'] = phi_post_cov[0:3,0:3]
    Sigma_0['phi_block2'] = phi_post_cov[3:6,3:6]
    Sigma_0['phi_block3'] = phi_post_cov[6:9,6:9]
    Sigma_0['range_block1'] = range_post_cov[0:3,0:3]
    Sigma_0['range_block2'] = range_post_cov[3:6,3:6]
    Sigma_0['range_block3'] = range_post_cov[6:9,6:9]
    Sigma_0['GEV'] = GEV_post_cov

    num_accepted = {}
    num_accepted['phi'] = 0
    num_accepted['range'] = 0
    num_accepted['GEV'] = 0

# Rt: each worker t proposed Rt at k knots at time t
if rank == 0:
    sigma_m_sq_Rt_list = [(2.4**2)/k]*size # comm scatter and gather preserves order
    num_accepted_Rt_list = [0]*size # [0, 0, ... 0]
else:
    sigma_m_sq_Rt_list = None
    num_accepted_Rt_list = None
sigma_m_sq_Rt = comm.scatter(sigma_m_sq_Rt_list, root = 0)
num_accepted_Rt = comm.scatter(num_accepted_Rt_list, root = 0)

########## Storage Place ##################################################
# %%
# Storage Place
## ---- R, log scaled, at the knots ----
if rank == 0:
    R_trace_log = np.full(shape = (n_iters, k, N), fill_value = np.nan) # [n_iters, num_knots, n_t]
    R_trace_log[0,:,:] = np.log(R_at_knots) # initialize
    R_init_log = R_trace_log[0,:,:]
else:
    R_init_log = None
R_init_log = comm.bcast(R_init_log, root = 0) # vector

## ---- phi, at the knots ----
if rank == 0:
    phi_knots_trace = np.full(shape = (n_iters, k), fill_value = np.nan)
    phi_knots_trace[0,:] = phi_at_knots
    phi_knots_init = phi_knots_trace[0,:]
else:
    phi_knots_init = None
phi_knots_init = comm.bcast(phi_knots_init, root = 0)

## ---- range_vec (length_scale) ----
if rank == 0:
    range_knots_trace = np.full(shape = (n_iters, k), fill_value = np.nan)
    range_knots_trace[0,:] = range_at_knots # set to true value
    range_knots_init = range_knots_trace[0,:]
else:
    range_knots_init = None
range_knots_init = comm.bcast(range_knots_init, root = 0)

## ---- GEV mu tau ksi (location, scale, shape) together ----
if rank == 0:
    GEV_knots_trace = np.full(shape=(n_iters, 3, k), fill_value = np.nan) # [n_iters, n_GEV, num_knots]
    GEV_knots_trace[0,:,:] = np.tile(np.array([mu, tau, ksi]), (k,1)).T
    GEV_knots_init = GEV_knots_trace[0,:,:]
else:
    GEV_knots_init = None
GEV_knots_init = comm.bcast(GEV_knots_init, root = 0)

## ---- overal likelihood? -----
if rank == 0:
    loglik_trace = np.full(shape = (n_iters,1), fill_value = np.nan)
else:
    loglik_trace = None

## ---- detail likelihood ----
if rank == 0:
    loglik_detail_trace = np.full(shape = (n_iters, 5), fill_value = np.nan)
else:
    loglik_detail_trace = None

########## Initialize ##################################################
# %%
# Initialize
## ---- R ----
# log-scale number(s), at "rank time", at the knots
R_current_log = np.array(R_init_log[:,rank])

## ---- phi ----
phi_knots_current = phi_knots_init
phi_vec_current = gaussian_weight_matrix @ phi_knots_current

## ---- range_vec (length_scale) ----
range_knots_current = range_knots_init
range_vec_current = gaussian_weight_matrix @ range_knots_current
K_current = ns_cov(range_vec = range_vec_current,
                sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = "matern")
cholesky_matrix_current = scipy.linalg.cholesky(K_current, lower = False)

## ---- GEV mu tau ksi (location, scale, shape) together ----
GEV_knots_current = GEV_knots_init
# will(?) be changed into matrix multiplication w/ more knots:
Loc_matrix_current = np.full(shape = (num_sites,N), fill_value = GEV_knots_current[0,0])
Scale_matrix_current = np.full(shape = (num_sites,N), fill_value = GEV_knots_current[1,0])
Shape_matrix_current = np.full(shape = (num_sites,N), fill_value = GEV_knots_current[2,0])

## ---- X_star ----
X_star_1t_current = X_star[:,rank]

########## Updates ##################################################
# %%
# Metropolis Updates
for iter in range(1, n_iters):
    # printing and drawings
    if rank == 0:
        if iter == 1:
            print(iter)
        if iter % 10 == 0:
            print(iter)
            print(strftime('%Y-%m-%d %H:%M:%S', localtime(time.time())))
        if iter % 100 == 0 or iter == n_iters-1:
            # Save data every 1000 iterations
            end_time = time.time()
            print('elapsed: ', round(end_time - start_time, 1), ' seconds')
            np.save('R_trace_log', R_trace_log)
            np.save('phi_knots_trace', phi_knots_trace)
            np.save('range_knots_trace', range_knots_trace)
            np.save('GEV_knots_trace', GEV_knots_trace)
            np.save('loglik_trace', loglik_trace)
            np.save('loglik_detail_trace', loglik_detail_trace)

            # Print traceplot every 1000 iterations
            xs = np.arange(iter)
            xs_thin = xs[0::10] # index 1, 11, 21, ...
            xs_thin2 = np.arange(len(xs_thin)) # numbers 1, 2, 3, ...
            R_trace_log_thin = R_trace_log[0:iter:10,:,:]
            phi_knots_trace_thin = phi_knots_trace[0:iter:10,:]
            range_knots_trace_thin = range_knots_trace[0:iter:10,:]
            GEV_knots_trace_thin = GEV_knots_trace[0:iter:10,:,:]
            loglik_trace_thin = loglik_trace[0:iter:10,:]
            loglik_detail_trace_thin = loglik_detail_trace[0:iter:10,:]

            # ---- phi ----
            plt.subplots()
            for i in range(k):
                plt.plot(xs_thin2, phi_knots_trace_thin[:,i], label='knot ' + str(i))
                # plt.plot(xs_thin2, phi_knots_trace_thin[:,1], label='knot ' + i)
            plt.title('traceplot for phi')
            plt.xlabel('iter thinned by 10')
            plt.ylabel('phi')
            plt.legend()
            plt.savefig('phi.pdf')
            plt.close()

            # ---- R_t ----
            plt.subplots()
            plt.plot(xs_thin2, R_trace_log_thin[:,0,0], label='knot 0 time 0')
            plt.plot(xs_thin2, R_trace_log_thin[:,0,1], label='knot 0 time 1')
            plt.plot(xs_thin2, R_trace_log_thin[:,0,2], label='knot 0 time 2')
            plt.plot(xs_thin2, R_trace_log_thin[:,1,0], label='knot 1 time 0')
            plt.plot(xs_thin2, R_trace_log_thin[:,1,1], label='knot 1 time 1')
            plt.plot(xs_thin2, R_trace_log_thin[:,1,2], label='knot 1 time 2')
            plt.title('traceplot for some R_t')
            plt.xlabel('iter thinned by 10')
            plt.ylabel('R_ts')
            plt.legend()
            plt.savefig('R_t.pdf')
            plt.close()

            # ---- range ----
            plt.subplots()
            for i in range(k):
                plt.plot(xs_thin2, range_knots_trace_thin[:,i], label='knot ' + str(i))
            # plt.plot(xs_thin2, range_knots_trace_thin[:,1], label='knot 1')
            plt.title('traceplot for range')
            plt.xlabel('iter thinned by 10')
            plt.ylabel('range')
            plt.legend()
            plt.savefig('range.pdf')
            plt.close()

            # ---- GEV ----
            ## location mu
            plt.subplots()
            plt.plot(xs_thin2, GEV_knots_trace_thin[:,0,0], label = 'knot 0') # location
            # plt.plot(xs_thin2, GEV_knots_trace_thin[:,0,1], label = 'knot 1')
            plt.title('traceplot for location')
            plt.xlabel('iter thinned by 10')
            plt.ylabel('mu')
            plt.legend()
            plt.savefig('mu.pdf')
            plt.close()

            ## scale tau
            plt.subplots()
            plt.plot(xs_thin2, GEV_knots_trace_thin[:,1,0], label = 'knot 0') # scale
            # plt.plot(xs_thin2, GEV_knots_trace_thin[:,1,1], label = 'knot 1')
            plt.title('traceplot for scale')
            plt.xlabel('iter thinned by 10')
            plt.ylabel('tau')
            plt.legend()
            plt.savefig('tau.pdf')
            plt.close()

            ## shape ksi
            plt.subplots()
            plt.plot(xs_thin2, GEV_knots_trace_thin[:,2,0], label = 'knot 0') # shape
            # plt.plot(xs_thin2, GEV_knots_trace_thin[:,2,1], label = 'knot 1')
            plt.title('traceplot for shape')
            plt.xlabel('iter thinned by 10')
            plt.ylabel('ksi')
            plt.legend()
            plt.savefig('ksi.pdf')
            plt.close()

            # ## together
            # plt.subplots()
            # plt.plot(xs_thin2, phi_knots_trace_thin, label='phi')
            # plt.plot(xs_thin2, GEV_knots_trace_thin[:,0,0], label='mu knot 0') # location
            # plt.plot(xs_thin2, GEV_knots_trace_thin[:,1,0], label='tau knot 0') # scale
            # plt.plot(xs_thin2, GEV_knots_trace_thin[:,2,0], label='ksi knot 0') # shape
            # plt.plot(xs_thin2, GEV_knots_trace_thin[:,0,1], label='mu knot 1') # location
            # plt.plot(xs_thin2, GEV_knots_trace_thin[:,1,1], label='tau knot 1') # scale
            # plt.plot(xs_thin2, GEV_knots_trace_thin[:,2,1], label='ksi knot 1') # shape
            # plt.title('traceplot for phi and GEV')
            # plt.legend()
            # plt.savefig('phi_GEV.pdf')

            # log-likelihood
            plt.subplots()
            plt.plot(xs_thin2, loglik_trace_thin)
            plt.title('traceplot for log-likelihood')
            plt.xlabel('iter thinned by 10')
            plt.ylabel('loglikelihood')
            plt.savefig('loglik.pdf')
            plt.close()

            plt.subplots()
            for i in range(5):
                plt.plot(xs_thin2, loglik_detail_trace_thin[:,i],label = i)
            plt.title('traceplot for detail log likelihood')
            plt.xlabel('iter thinned by 10')
            plt.ylabel('log likelihood')
            plt.legend()
            plt.savefig('loglik_detail.pdf')
            plt.close()

    # Adaptive Update autotunings
    if iter % 25 == 0:
            
        gamma1 = 1 / ((iter/25 + offset) ** c_1)
        gamma2 = c_0 * gamma1

        # R_t
        sigma_m_sq_Rt_list = comm.gather(sigma_m_sq_Rt, root = 0)
        num_accepted_Rt_list = comm.gather(num_accepted_Rt, root = 0)
        if rank == 0:
            for i in range(size):
                r_hat = num_accepted_Rt_list[i]/25
                num_accepted_Rt_list[i] = 0
                log_sigma_m_sq_hat = np.log(sigma_m_sq_Rt_list[i]) + gamma2*(r_hat - r_opt)
                sigma_m_sq_Rt_list[i] = np.exp(log_sigma_m_sq_hat)
        sigma_m_sq_Rt = comm.scatter(sigma_m_sq_Rt_list, root = 0)
        num_accepted_Rt = comm.scatter(num_accepted_Rt_list, root = 0)

        # phi, range, and GEV
        if rank == 0:
            # phi
            r_hat = num_accepted['phi']/25
            num_accepted['phi'] = 0
            ## phi_block1
            Sigma_0_hat = np.cov(np.array([phi_knots_trace[iter-25:iter,i].ravel() for i in range(0,3)]))
            log_sigma_m_sq_hat = np.log(sigma_m_sq['phi_block1']) + gamma2*(r_hat - r_opt)
            sigma_m_sq['phi_block1'] = np.exp(log_sigma_m_sq_hat)
            Sigma_0['phi_block1'] = Sigma_0['phi_block1'] + gamma1*(Sigma_0_hat - Sigma_0['phi_block1'])
            ## phi_block2
            Sigma_0_hat = np.cov(np.array([phi_knots_trace[iter-25:iter,i].ravel() for i in range(3,6)]))
            log_sigma_m_sq_hat = np.log(sigma_m_sq['phi_block2']) + gamma2*(r_hat - r_opt)
            sigma_m_sq['phi_block2'] = np.exp(log_sigma_m_sq_hat)
            Sigma_0['phi_block2'] = Sigma_0['phi_block2'] + gamma1*(Sigma_0_hat - Sigma_0['phi_block2'])
            ## phi_block3
            Sigma_0_hat = np.cov(np.array([phi_knots_trace[iter-25:iter,i].ravel() for i in range(6,9)]))
            log_sigma_m_sq_hat = np.log(sigma_m_sq['phi_block3']) + gamma2*(r_hat - r_opt)
            sigma_m_sq['phi_block3'] = np.exp(log_sigma_m_sq_hat)
            Sigma_0['phi_block3'] = Sigma_0['phi_block3'] + gamma1*(Sigma_0_hat - Sigma_0['phi_block3'])

            # range
            r_hat = num_accepted['range']/25
            num_accepted['range'] = 0
            ## range_block1
            Sigma_0_hat = np.cov(np.array([range_knots_trace[iter-25:iter,i].ravel() for i in range(0,3)]))
            log_sigma_m_sq_hat = np.log(sigma_m_sq['range_block1']) + gamma2*(r_hat - r_opt)
            sigma_m_sq['range_block1'] = np.exp(log_sigma_m_sq_hat)
            Sigma_0['range_block1'] = Sigma_0['range_block1'] + gamma1*(Sigma_0_hat - Sigma_0['range_block1'])
            ## range_block2
            Sigma_0_hat = np.cov(np.array([range_knots_trace[iter-25:iter,i].ravel() for i in range(3,6)]))
            log_sigma_m_sq_hat = np.log(sigma_m_sq['range_block2']) + gamma2*(r_hat - r_opt)
            sigma_m_sq['range_block2'] = np.exp(log_sigma_m_sq_hat)
            Sigma_0['range_block2'] = Sigma_0['range_block2'] + gamma1*(Sigma_0_hat - Sigma_0['range_block2'])
            ## range_block3
            Sigma_0_hat = np.cov(np.array([range_knots_trace[iter-25:iter,i].ravel() for i in range(6,9)]))
            log_sigma_m_sq_hat = np.log(sigma_m_sq['range_block3']) + gamma2*(r_hat - r_opt)
            sigma_m_sq['range_block3'] = np.exp(log_sigma_m_sq_hat)
            Sigma_0['range_block3'] = Sigma_0['range_block3'] + gamma1*(Sigma_0_hat - Sigma_0['range_block3'])
            
            # GEV
            r_hat = num_accepted['GEV']/25
            num_accepted['GEV'] = 0
            sample_cov = np.cov(np.array([GEV_knots_trace[iter-25:iter,0,0].ravel(), # mu location
                                            GEV_knots_trace[iter-25:iter,1,0].ravel()])) # tau scale
            Sigma_0_hat = np.zeros((3,3)) # doing the hack because we are not updating ksi
            Sigma_0_hat[2,2] = 1
            Sigma_0_hat[0:2,0:2] += sample_cov
            log_sigma_m_sq_hat = np.log(sigma_m_sq['GEV']) + gamma2*(r_hat - r_opt)
            sigma_m_sq['GEV'] = np.exp(log_sigma_m_sq_hat)
            Sigma_0['GEV'] = Sigma_0['GEV'] + gamma1*(Sigma_0_hat - Sigma_0['GEV'])
        


#### ----- Update Rt ----- Parallelized Across N time
    # if rank == 0:
    #     print('Updating R')
    # Propose a R at time "rank", on log-scale
    # R_proposal_log = random_generator.normal(loc=0.0, scale=2.0, size=k) + R_current_log

    # Propose a R using adaptive update
    R_proposal_log = np.sqrt(sigma_m_sq_Rt)*random_generator.normal(loc = 0.0, scale = 1.0, size = k) + R_current_log

    # Conditional Likelihood at Current
    R_vec_current = wendland_weight_matrix @ np.exp(R_current_log)

    # log-likelihood:
    lik = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
                                                    Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank], 
                                                    phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
    # log-prior density
    prior = np.sum(scipy.stats.levy.logpdf(np.exp(R_current_log)) + R_current_log)

    # Conditional Likelihood at Proposal
    R_vec_proposal = wendland_weight_matrix @ np.exp(R_proposal_log)
    # if np.any(~np.isfinite(R_vec_proposal**phi_vec_current)): print("Negative or zero R, iter=", iter, ", rank=", rank, R_vec_proposal[0], phi_vec_current[0])
    lik_star = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
                                                    Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank], 
                                                    phi_vec_current, gamma_vec, R_vec_proposal, cholesky_matrix_current)
    prior_star = np.sum(scipy.stats.levy.logpdf(np.exp(R_proposal_log)) + R_proposal_log)

    # Accept or Reject
    u = random_generator.uniform()
    ratio = np.exp(lik_star + prior_star - lik - prior)
    if u > ratio: # Reject
        R_update_log = R_current_log
    else: # Accept, u <= ratio
        R_update_log = R_proposal_log
        num_accepted_Rt += 1
    
    R_current_log = R_update_log
    R_vec_current = wendland_weight_matrix @ np.exp(R_current_log)
    
    # Gather across N_t, store into trace matrix
    R_current_log_gathered = comm.gather(R_current_log, root=0)
    # print(R_current_log_gathered)
    if rank == 0:
        R_trace_log[iter,:,:] = np.vstack(R_current_log_gathered).T

#### ----- Update phi ----- parallelized likelihood calculation across N time
    # if rank == 0:
    #     print('Updating phi')
    # Propose new phi at the knots --> new phi vector
    if rank == 0:
        # random_walk_block1 = random_generator.multivariate_normal(np.zeros(3), phi_post_cov[0:3,0:3], size = None)
        # random_walk_block2 = random_generator.multivariate_normal(np.zeros(3), phi_post_cov[3:6,3:6], size = None)
        # random_walk_block3 = random_generator.multivariate_normal(np.zeros(3), phi_post_cov[6:9,6:9], size = None)
        random_walk_block1 = np.sqrt(sigma_m_sq['phi_block1'])*random_generator.multivariate_normal(np.zeros(3), Sigma_0['phi_block1'])
        random_walk_block2 = np.sqrt(sigma_m_sq['phi_block2'])*random_generator.multivariate_normal(np.zeros(3), Sigma_0['phi_block2'])
        random_walk_block3 = np.sqrt(sigma_m_sq['phi_block3'])*random_generator.multivariate_normal(np.zeros(3), Sigma_0['phi_block3'])        
        random_walk_perturb = np.hstack((random_walk_block1,random_walk_block2,random_walk_block3))
        phi_knots_proposal = phi_knots_current + random_walk_perturb
    else:
        phi_knots_proposal = None
    phi_knots_proposal = comm.bcast(phi_knots_proposal, root = 0)

    phi_vec_proposal = gaussian_weight_matrix @ phi_knots_proposal

    # Conditional Likelihood at Current
    # X_star_1t_current = qRW(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
    #                               phi_vec_current, gamma, 100)
    lik_1t = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
                                                    Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                    phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
    
    # Conditional Likelihood at Proposed
    phi_out_of_range = any(phi <= 0 for phi in phi_knots_proposal) or any(phi > 1 for phi in phi_knots_proposal) # U(0,1] prior

    if phi_out_of_range: #U(0,1] prior
        X_star_1t_proposal = np.NINF
        lik_1t_proposal = np.NINF
    else: # 0 < phi <= 1
        X_star_1t_proposal = qRW(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
                                      phi_vec_proposal, gamma)
        lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_proposal, 
                                                        Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                        phi_vec_proposal, gamma_vec, R_vec_current, cholesky_matrix_current)
    
    # Gather likelihood calculated across time
    lik_gathered = comm.gather(lik_1t, root = 0)
    lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

    # if rank == 0:
        # print("iter: ", iter, "lik after R: ", round(sum(lik_gathered),3))

    # Accept or Reject
    if rank == 0:
        phi_accepted = False
        lik = sum(lik_gathered)
        lik_proposal = sum(lik_proposal_gathered)

        u = random_generator.uniform()
        ratio = np.exp(lik_proposal - lik)
        if not np.isfinite(ratio):
            ratio = 0
        if u > ratio: # Reject
            phi_vec_update = phi_vec_current
            phi_knots_update = phi_knots_current
        else: # Accept, u <= ratio
            phi_vec_update = phi_vec_proposal
            phi_knots_update = phi_knots_proposal
            phi_accepted = True
            num_accepted['phi'] += 1
        
        # Store the result
        phi_knots_trace[iter,:] = phi_knots_update

        # Update the "current" value
        phi_vec_current = phi_vec_update
        phi_knots_current = phi_knots_update
    else:
        phi_accepted = False

    # Brodcast the updated values
    phi_vec_current = comm.bcast(phi_vec_current, root = 0)
    phi_knots_current = comm.bcast(phi_knots_current, root = 0)
    phi_accepted = comm.bcast(phi_accepted, root = 0)

    # Update X_star
    if phi_accepted:
        X_star_1t_current = qRW(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
                                        phi_vec_current, gamma)

#### ----- Update range_vec ----- parallelized likelihood calculation across N time
    # if rank == 0:
    #     print('Updating range')
    # Propose new range at the knots --> new range vector
    if rank == 0:
        random_walk_block1 = np.sqrt(sigma_m_sq['range_block1'])*random_generator.multivariate_normal(np.zeros(3), Sigma_0['range_block1'])
        random_walk_block2 = np.sqrt(sigma_m_sq['range_block2'])*random_generator.multivariate_normal(np.zeros(3), Sigma_0['range_block2'])
        random_walk_block3 = np.sqrt(sigma_m_sq['range_block3'])*random_generator.multivariate_normal(np.zeros(3), Sigma_0['range_block3'])    
        random_walk_perturb = np.hstack((random_walk_block1,random_walk_block2,random_walk_block3))
        range_knots_proposal = range_knots_current + random_walk_perturb
    else:
        range_knots_proposal = None
    range_knots_proposal = comm.bcast(range_knots_proposal, root = 0)

    range_vec_proposal = gaussian_weight_matrix @ range_knots_proposal

    # Conditional Likelihood at Current
    lik_1t = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
                                                    Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                    phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)

    # Conditional Likelihood at Proposed
    if any(range <= 0 for range in range_knots_proposal):
        lik_1t_proposal = np.NINF
    else:
        K_proposal = ns_cov(range_vec = range_vec_proposal,
                        sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = "matern")
        cholesky_matrix_proposal = scipy.linalg.cholesky(K_proposal, lower = False)

        lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
                                                    Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                    phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_proposal)

    # Gather likelihood calculated across time
    lik_gathered = comm.gather(lik_1t, root = 0)
    lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

    # Accept or Reject
    if rank == 0:
        range_accepted = False
        lik = sum(lik_gathered)
        lik_proposal = sum(lik_proposal_gathered)

        u = random_generator.uniform()
        ratio = np.exp(lik_proposal - lik)
        if not np.isfinite(ratio):
            ratio = 0 # Force a rejection
        if u > ratio: # Reject
            range_vec_update = range_vec_current
            range_knots_update = range_knots_current
        else: # Accept, u <= ratio
            range_vec_update = range_vec_proposal
            range_knots_update = range_knots_proposal
            range_accepted = True
            num_accepted['range'] += 1
        
        # Store the result
        range_knots_trace[iter,:] = range_knots_update

        # Update the "current" value
        range_vec_current = range_vec_update
        range_knots_current = range_knots_update
    else:
        range_accepted = False

    # Brodcast the updated values
    range_vec_current = comm.bcast(range_vec_current, root = 0)
    range_knots_current = comm.bcast(range_knots_current, root = 0)
    range_accepted = comm.bcast(range_accepted, root = 0)

    # Update the K
    if range_accepted:
        K_current = ns_cov(range_vec = range_vec_current,
                            sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = "matern")
        cholesky_matrix_current = scipy.linalg.cholesky(K_current, lower = False)

#### ----- Update GEV mu tau ksi (location, scale, shape) together ----
#### ----- Do not update ksi -----
    # if rank == 0:
    #     print('Updating GEV')
    # Propose new GEV params at the knots --> new GEV params vector
    if rank == 0:
        # random_walk_3x3 = random_generator.multivariate_normal(np.zeros(3), GEV_post_cov, size = k).T
        random_walk_3x3 = np.sqrt(sigma_m_sq['GEV'])*random_generator.multivariate_normal(np.zeros(3), Sigma_0['GEV'], size = k).T
        GEV_knots_proposal = GEV_knots_current + random_walk_3x3
        GEV_knots_proposal[:,1:] = np.vstack(GEV_knots_proposal[:,0]) # treat it as if it's only one knot, GEV params spatial stationary
        GEV_knots_proposal[2,:] = GEV_knots_current[2,:] # hold ksi constant
        # GEV_knots_proposal[0:2,:] = GEV_knots_current[0:2,:] # hold location and scale constant
    else:
        GEV_knots_proposal = None
    GEV_knots_proposal = comm.bcast(GEV_knots_proposal, root = 0)

    # will be changed into matrix multiplication w/ more knots
    Loc_matrix_proposal = np.full(shape = (num_sites,N), fill_value = GEV_knots_proposal[0,0])
    Scale_matrix_proposal = np.full(shape = (num_sites,N), fill_value = GEV_knots_proposal[1,0])
    Shape_matrix_proposal = np.full(shape = (num_sites,N), fill_value = GEV_knots_proposal[2,0])

    # Conditional Likelihodd at Current
    lik_1t = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
                                                    Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                    phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)

    # Conditional Likelihood at Proposed
    Scale_out_of_range = any(scale <= 0 for scale in GEV_knots_proposal[1,:])
    Shape_out_of_range = any(shape <= -0.5 for shape in GEV_knots_proposal[2,:]) or any(shape > 0.5 for shape in GEV_knots_proposal[2,:])
    if Scale_out_of_range or Shape_out_of_range:
        X_star_1t_proposal = np.NINF
        lik_1t_proposal = np.NINF
    else:
        X_star_1t_proposal = qRW(pgev(Y[:,rank], Loc_matrix_proposal[:,rank], Scale_matrix_proposal[:,rank], Shape_matrix_proposal[:,rank]),
                                      phi_vec_current, gamma)
        lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_proposal, 
                                                        Loc_matrix_proposal[:,rank], Scale_matrix_proposal[:,rank], Shape_matrix_proposal[:,rank],
                                                        phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
    
    # Gather likelihood calculated across time
    lik_gathered = comm.gather(lik_1t, root = 0)
    lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

    # Accept or Reject
    if rank == 0:

        # for now there is only one set of GEV parameters
        # (constant across all time and space)
        # log-prior density for scale as P(tau) = 1/tau
        prior_scale = -np.log(Scale_matrix_current[0][0])
        prior_scale_proposal = -np.log(Scale_matrix_proposal[0][0])

        prior_mu = scipy.stats.norm.logpdf(Loc_matrix_current[0][0], loc = 0, scale = 0.1)
        prior_mu_proposal = scipy.stats.norm.logpdf(Loc_matrix_current[0][0], loc = 0, scale = 0.1)
        
        GEV_accepted = False
        lik = sum(lik_gathered) + prior_scale + prior_mu
        lik_proposal = sum(lik_proposal_gathered) + prior_scale_proposal + prior_mu_proposal

        u = random_generator.uniform()
        ratio = np.exp(lik_proposal - lik)
        if not np.isfinite(ratio):
            ratio = 0
        if u > ratio: # Reject
            Loc_matrix_update = Loc_matrix_current
            Scale_matrix_update = Scale_matrix_current
            Shape_matrix_update = Shape_matrix_current
            GEV_knots_update = GEV_knots_current
        else: # Accept, u <= ratio
            Loc_matrix_update = Loc_matrix_proposal
            Scale_matrix_update = Scale_matrix_proposal
            Shape_matrix_update = Shape_matrix_proposal
            GEV_knots_update = GEV_knots_proposal
            GEV_accepted = True
            num_accepted['GEV'] += 1
        
        # Store the result
        GEV_knots_trace[iter,:,:] = GEV_knots_update

        # Update the "current" value
        Loc_matrix_current = Loc_matrix_update
        Scale_matrix_current = Scale_matrix_update
        Shape_matrix_current = Shape_matrix_update
        GEV_knots_current = GEV_knots_update
    else:
        GEV_accepted = False

    # Brodcast the updated values
    Loc_matrix_current = comm.bcast(Loc_matrix_current, root = 0)
    Scale_matrix_current = comm.bcast(Scale_matrix_current, root = 0)
    Shape_matrix_current = comm.bcast(Shape_matrix_current, root = 0)
    GEV_knots_current = comm.bcast(GEV_knots_current, root = 0)
    GEV_accepted = comm.bcast(GEV_accepted, root = 0)

    # Update X_star
    if GEV_accepted:
        X_star_1t_current = qRW(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
                                        phi_vec_current, gamma)
    

    # Keeping track of likelihood after this iteration
    lik_final_1t_detail = marg_transform_data_mixture_likelihood_1t_detail(Y[:,rank], X_star_1t_current, 
                                            Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                            phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
    lik_final_1t = sum(lik_final_1t_detail)
    lik_final_detail_gathered = comm.gather(lik_final_1t_detail, root = 0)
    lik_final_gathered = comm.gather(lik_final_1t, root = 0)
    if rank == 0:
        loglik_trace[iter,0] = round(sum(lik_final_gathered),3) # storing the overall log likelihood
        loglik_detail_trace[iter,:] = np.matrix(lik_final_detail_gathered).sum(axis=0) # storing the detail log likelihood


# End of MCMC
if rank == 0:
    end_time = time.time()
    print('total time: ', round(end_time - start_time, 1), ' seconds')
    print('true R: ', R_at_knots)
    np.save('R_trace_log', R_trace_log)
    np.save('phi_knots_trace', phi_knots_trace)
    np.save('range_knots_trace', range_knots_trace)
    np.save('GEV_knots_trace', GEV_knots_trace)
    np.save('loglik_trace', loglik_trace)
    np.save('loglik_detail_trace', loglik_detail_trace)

# # %%
# # Results

# folder = './data/20230904_5knots_10000_500_32_phi_range/'
# phi_knots_trace = np.load(folder + 'phi_knots_trace.npy')
# R_trace_log = np.load(folder + 'R_trace_log.npy')
# range_knots_trace = np.load(folder + 'range_knots_trace.npy')
# GEV_knots_trace = np.load(folder + 'GEV_knots_trace.npy')
# xs = np.arange(10000)

###########################################################################################
# Posterior Covariance Matrix
###########################################################################################

# GEV_post_cov = np.cov(np.array([GEV_knots_trace[:,0,0].ravel(), # mu location
#                                 GEV_knots_trace[:,1,0].ravel()])) # tau scale

# phi_post_cov = np.cov(np.array([phi_knots_trace[:,i].ravel() for i in range(k)]))

# range_post_cov = np.cov(np.array([range_knots_trace[:,i].ravel() for i in range(k)]))





###########################################################################################
# Plotting
###########################################################################################

# # %%
# # Plotting
# # Plot phi
# plt.plot(xs, phi_knots_trace) # 1 knot

# plt.plot(xs, phi_knots_trace[:,0], label = 'knot 0')
# plt.plot(xs, phi_knots_trace[:,1], label = 'knot 1')
# plt.legend()
# # %%
# # Plot R
# plt.plot(xs, R_trace_log[:,0]) # 1 knot
# plt.plot(xs, R_trace_log[:,1]) # 1 knot
# plt.plot(xs, R_trace_log[:,2]) # 1 knot
# plt.plot(xs, R_trace_log[:,3]) # 1 knot
# plt.plot(xs, R_trace_log[:,4]) # 1 knot

# plt.plot(xs, R_trace_log[:,0,0], label='knot 0 time 0')
# plt.plot(xs, R_trace_log[:,0,1], label='knot 0 time 1')
# plt.plot(xs, R_trace_log[:,0,2], label='knot 0 time 2')
# plt.plot(xs, R_trace_log[:,1,0], label='knot 1 time 0')
# plt.plot(xs, R_trace_log[:,1,1], label='knot 1 time 1')
# plt.plot(xs, R_trace_log[:,1,2], label='knot 1 time 2')
# plt.legend()

# # %%
# # Plot range
# plt.plot(xs, range_knots_trace) # 1 knot

# plt.plot(xs, range_knots_trace[:,0], label='knot 0')
# plt.plot(xs, range_knots_trace[:,1], label='knot 1')
# plt.plot(xs, range_knots_trace[:,2], label='knot 2')
# plt.plot(xs, range_knots_trace[:,3], label='knot 3')
# plt.plot(xs, range_knots_trace[:,4], label='knot 4')
# plt.legend()

# # %%
# # mu location
# plt.plot(xs, GEV_knots_trace[:,0]) # location

# plt.plot(xs, GEV_knots_trace[:,0,0], label = 'knot 0') # location
# plt.plot(xs, GEV_knots_trace[:,0,1], label = 'knot 1')
# plt.legend()

# # %%
# plt.plot(xs, GEV_knots_trace[:,1]) # scale

# plt.plot(xs, GEV_knots_trace[:,1,0], label = 'knot 0') # scale
# plt.plot(xs, GEV_knots_trace[:,1,1], label = 'knot 1')
# plt.legend()

# # %%
# plt.plot(xs, GEV_knots_trace[:,2]) # shape

# plt.plot(xs, GEV_knots_trace[:,2,0], label = 'knot 0') # shape
# plt.plot(xs, GEV_knots_trace[:,2,1], label = 'knot 1')
# plt.legend()

# # %%
# plt.plot(xs, phi_knots_trace)
# plt.plot(xs, GEV_knots_trace[:,0]) # location
# plt.plot(xs, GEV_knots_trace[:,1]) # scale
# plt.plot(xs, GEV_knots_trace[:,2]) # shape

# plt.plot(xs, phi_knots_trace[:, 0], label='phi knot 0')
# plt.plot(xs, GEV_knots_trace[:,0,0], label='mu knot 0') # location
# plt.plot(xs, GEV_knots_trace[:,1,0], label='tau knot 0') # scale
# plt.plot(xs, GEV_knots_trace[:,2,0], label='ksi knot 0') # shape
# plt.legend()

# plt.plot(xs, phi_knots_trace[:, 1], label='phi knot 1')
# plt.plot(xs, GEV_knots_trace[:,0,1], label='mu knot 1') # location
# plt.plot(xs, GEV_knots_trace[:,1,1], label='tau knot 1') # scale
# plt.plot(xs, GEV_knots_trace[:,2,1], label='ksi knot 1') # shape
# plt.legend()


# ###########################################################################################
# Evaluate Profile Likelihoods
# ###########################################################################################

# # %%
# # Evaluate Phi

# phis = np.linspace(0.1, 0.8, num = 50)
# lik = []
# for phi in phis:
#     phi_vec = np.repeat(phi, num_sites)
#     # X_star_tmp = X_star
#     X_star_tmp = qRW(pgev(Y, Loc_matrix, Scale_matrix, Shape_matrix), phi, gamma, 100)
#     lik.append(marg_transform_data_mixture_likelihood(Y, X_star_tmp, Loc_matrix, Scale_matrix, Shape_matrix, phi_vec, gamma_vec, R_matrix, cholesky_matrix))
#     # lik.append(marg_transform_data_mixture_likelihood(Y, X, Loc_matrix, Scale_matrix, Shape_matrix, phi_vec, gamma_vec, R_matrix, cholesky_matrix))

# plt.plot(phis, lik)
# phi = 0.33 # set it back
# phi_vec = np.repeat(phi, num_sites)

# # %%
# # Evaluate mu (location parameter)
# Locs = np.linspace(-1, 1, num = 20)
# lik = []
# for Loc in Locs:
#     Loc_matrix = np.full(shape = Y.shape, fill_value = Loc)
#     X_star_tmp = qRW(pgev(Y, Loc_matrix, Scale_matrix, Shape_matrix), phi, gamma, 100)
#     lik.append(marg_transform_data_mixture_likelihood(Y, X_star_tmp, Loc_matrix, Scale_matrix, Shape_matrix, phi_vec, gamma_vec, R_matrix, cholesky_matrix))
# plt.plot(Locs, lik)
# Loc_matrix = np.full(shape = Y.shape, fill_value = mu)

# # %%
# # Evaluate tau (scale parameter)
# Scales = np.linspace(0.8, 1.2, num = 20)
# lik = []
# for Scale in Scales:
#     Scale_matrix = np.full(shape = Y.shape, fill_value = Scale)
#     X_star_tmp = qRW(pgev(Y, Loc_matrix, Scale_matrix, Shape_matrix), phi, gamma, 100)
#     lik.append(marg_transform_data_mixture_likelihood(Y, X_star_tmp, Loc_matrix, Scale_matrix, Shape_matrix, phi_vec, gamma_vec, R_matrix, cholesky_matrix))
# plt.plot(Scales, lik)
# Scale_matrix = np.full(shape = Y.shape, fill_value = tau)

# # %%
# # Evaluate ksi (shape parameter)
# Shapes = np.linspace(0.1, 0.3, num = 20)
# lik = []
# for Shape in Shapes:
#     Shape_matrix = np.full(shape = Y.shape, fill_value = Shape)
#     X_star_tmp = qRW(pgev(Y, Loc_matrix, Scale_matrix, Shape_matrix), phi, gamma, 100)
#     lik.append(marg_transform_data_mixture_likelihood(Y, X_star_tmp, Loc_matrix, Scale_matrix, Shape_matrix, phi_vec, gamma_vec, R_matrix, cholesky_matrix))
# plt.plot(Shapes, lik)
# Shape_matrix = np.full(shape = Y.shape, fill_value = ksi)

# # %%
# # Evaluate length_scale (range parameter)
# Ranges = np.linspace(0.1, 1, num = 20)
# lik = []
# for Range in Ranges:
#     range_vec = np.repeat(Range, num_sites)
#     K = ns_cov(range_vec=range_vec,  sigsq_vec = sigsq_vec, 
#              coords = sites_xy, kappa = nu, cov_model = "matern")
#     cholesky_matrix = scipy.linalg.cholesky(K, lower=False)

#     lik.append(marg_transform_data_mixture_likelihood(Y, X_star, Loc_matrix, Scale_matrix, Shape_matrix, phi_vec, gamma_vec, R_matrix, cholesky_matrix))
# plt.plot(Ranges, lik)

# range_vec = np.repeat(length_scale, num_sites)
# K = ns_cov(range_vec=range_vec,  sigsq_vec = sigsq_vec, 
#              coords = sites_xy, kappa = nu, cov_model = "matern")
# cholesky_matrix = scipy.linalg.cholesky(K, lower=False)


# # %%
# # Evaluate R_t

# rank = 5
# R_tmp = R[rank]
# Rs = np.linspace(R[rank]*0.1, R[rank]*2)
# # Rs = np.linspace(0.1, 4, num = 100)
# lik = []
# dlevys = []

# lik1, lik21, lik22, lik23, lik24 = [],[],[],[],[]
# for r in Rs:
#     R_vec = np.repeat(r, num_sites)
#     lik.append(marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star[:,rank], 
#                                                          Loc_matrix[:,rank], Scale_matrix[:,rank], Shape_matrix[:,rank], 
#                                                          phi_vec, gamma_vec, R_vec, cholesky_matrix)
#                )
#     part1, part21, part22, part23, part24 = marg_transform_data_mixture_likelihood_1t_detail(Y[:,rank], X_star[:,rank],
#                                                                                              Loc_matrix[:,rank], Scale_matrix[:,rank], Shape_matrix[:,rank],
#                                                                                              phi_vec, gamma_vec, R_vec, cholesky_matrix)
#     lik1.append(part1)
#     lik21.append(part21)
#     lik22.append(part22)
#     lik23.append(part23)
#     lik24.append(part24)
#     dlevys.append(dlevy(r=r, m=delta, s=1, log=True))

# #%%
# plt.plot(Rs, lik)
# #%%
# plt.plot(Rs, dlevys)
# #%%
# print(R[rank])
# plt.plot(Rs, np.array(lik) + np.array(dlevys), '-bo')
# #%%
# plt.plot(Rs, lik1)
# plt.plot(Rs, lik21)
# plt.plot(Rs, lik22)

# # %%

# dlevy_vec = np.vectorize(dlevy)
# # sum(dlevy_vec(R, 0, 0.5))

# R1000 = scipy.stats.levy.rvs(loc=delta,scale=gamma,size=1000)

# gammas = np.linspace(0.05, 0.6, num = 10)
# lik = []
# for g in gammas:
#     # lik.append(sum(dlevy_vec(R, 0, g)))
#     lik.append(sum(scipy.stats.levy.pdf(R, 0, g)))
# plt.plot(gammas, lik)


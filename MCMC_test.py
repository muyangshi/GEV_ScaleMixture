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
from model_sim import *
from ns_cov import *
from mpi4py import MPI
# from multiprocessing import Pool
# from p_cubature import *

# %%
# ------- 0. Simulation Setting --------------------------------------

## space setting
np.random.seed(2345)
N = 5 # number of time replicates
num_sites = 50 # number of sites/stations
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
# n_iters

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

radius = 3.5 # from 6 to 4 to 3.5
radius_from_knots = np.repeat(radius, k) # ?influence radius from a knot?

# Plot the space
fig, ax = plt.subplots()
ax.plot(sites_x, sites_y, 'b.', alpha = 0.4)
ax.plot(knots_x, knots_y, 'r+')
space_rectangle = plt.Rectangle(xy = (0,0), width = 10, height = 10,
                                fill = False, color = 'black')
for i in range(k):
    circle_i = plt.Circle((knots_xy[i,0],knots_xy[i,1]), radius_from_knots[0], 
                     color='r', fill=True, fc='grey', ec = 'red', alpha = 0.2)
    ax.add_patch(circle_i)
# circle0 = plt.Circle((knots_xy[0,0],knots_xy[0,1]), radius_from_knots[0], 
#                      color='r', fill=True, fc='grey', ec = 'red', alpha = 0.2)
# circle1 = plt.Circle((knots_xy[1,0],knots_xy[1,1]), radius_from_knots[1], 
#                      color='r', fill=True, fc='grey', ec = 'red', alpha = 0.2)
# circle2 = plt.Circle((knots_xy[2,0],knots_xy[2,1]), radius_from_knots[1], 
#                      color='r', fill=True, fc='grey', ec = 'red', alpha = 0.2)
# circle3 = plt.Circle((knots_xy[3,0],knots_xy[3,1]), radius_from_knots[1], 
#                      color='r', fill=True, fc='grey', ec = 'red', alpha = 0.2)
# circle4 = plt.Circle((knots_xy[4,0],knots_xy[4,1]), radius_from_knots[1], 
#                      color='r', fill=True, fc='grey', ec = 'red', alpha = 0.2)
ax.add_patch(space_rectangle)
# ax.add_patch(circle0)
# ax.add_patch(circle1)
# ax.add_patch(circle2)
# ax.add_patch(circle3)
# ax.add_patch(circle4)
plt.xlim([-2,12])
plt.ylim([-2,12])
plt.show()

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

# %%
# ------- 3. Generate covariance matrix, Z, and W --------------------------------

## range_vec
rho = 2.0 # the rho in matern kernel exp(-rho * x)
length_scale = 1/rho # scikit/learn parameterization (length_scale)
# range_at_knots = np.full(shape = k, fill_value = length_scale) # array([0.5, 0.5])
range_at_knots = np.array([0.3,0.3,0.3,
                           0.3,0.3,0.3,
                           0.3,0.3,0.3])
range_vec = gaussian_weight_matrix @ range_at_knots

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
phi_at_knots = np.array([0.2,0.2,0.2,
                         0.2,0.8,0.2,
                         0.2,0.2,0.2])
phi_vec = gaussian_weight_matrix @ phi_at_knots

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
# R_matrix = np.tile(R, num_sites).reshape(Y.shape)
R_matrix = R_at_sites
# phi_vec = np.repeat(phi, num_sites)
gamma_vec = np.repeat(gamma, num_sites)
cholesky_matrix = scipy.linalg.cholesky(K, lower=False)

###########################################################################################
# Metropolis Updates ######################################################################
###########################################################################################

# %%
# MPI and setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

random_generator = np.random.RandomState()
n_iters = 100

# mvn_cov = 2*np.array([[ 1.51180152e-02, -3.53233442e-05,  6.96443508e-03, 7.08467852e-03],
#                     [-3.53233442e-05,  1.60576481e-03,  9.46786420e-04,-8.60876113e-05],
#                     [ 6.96443508e-03,  9.46786420e-04,  4.25227059e-03, 3.39474201e-03],
#                     [ 7.08467852e-03, -8.60876113e-05,  3.39474201e-03, 3.92065445e-03]])

# mvn_cov_3x3 = 2*np.array([[ 1.60576481e-03,  9.46786420e-04, -8.60876113e-05],
#                         [ 9.46786420e-04,  4.25227059e-03,  3.39474201e-03],
#                         [-8.60876113e-05,  3.39474201e-03,  3.92065445e-03]])

# phi_post_cov = np.array([[ 2.2348e-03, -6.4440e-04,  8.5400e-05,  3.6920e-04, -1.8740e-04],
#                         [-6.4440e-04,  2.8613e-03,  3.3760e-04,  1.2440e-04, -5.6460e-04],
#                         [ 8.5400e-05,  3.3760e-04,  2.0855e-03,  6.0150e-04,  4.8340e-04],
#                         [ 3.6920e-04,  1.2440e-04,  6.0150e-04,  2.0463e-03, -6.0290e-04],
#                         [-1.8740e-04, -5.6460e-04,  4.8340e-04, -6.0290e-04,  5.7503e-03]])

# range_post_cov = np.array([[ 0.00049806,  0.00017201,  0.00021982,  0.00057124, -0.00144159],
#                         [ 0.00017201,  0.00159722,  0.00050871,  0.00048282, -0.0017226 ],
#                         [ 0.00021982,  0.00050871,  0.00314174,  0.00149776, -0.00172151],
#                         [ 0.00057124,  0.00048282,  0.00149776,  0.00615477, -0.00169341],
#                         [-0.00144159, -0.0017226 , -0.00172151, -0.00169341,  0.01281447]])

# phi_post_cov = 0.001 * np.identity(k)

# range_post_cov = 0.001 * np.identity(k)

phi_post_cov = np.array([[ 9.58542418e-03, -6.01140297e-03,  1.06511988e-03,
        -7.01106204e-03,  5.07443726e-03, -4.07658075e-03,
         2.43384515e-04,  4.43488080e-04, -1.72051955e-04],
       [-6.01140297e-03,  2.29216490e-02, -1.08331037e-02,
         4.00797383e-03, -1.58483844e-02,  6.59980653e-03,
         1.75053208e-03,  4.53228164e-03, -6.90583213e-04],
       [ 1.06511988e-03, -1.08331037e-02,  2.01752882e-02,
        -7.67372584e-04,  9.22744820e-03, -1.13651380e-02,
        -4.18897332e-04, -1.60789731e-03,  9.24313673e-04],
       [-7.01106204e-03,  4.00797383e-03, -7.67372584e-04,
         1.79168903e-02, -1.65285992e-02,  8.39889941e-03,
        -1.51799364e-03, -4.17429455e-05,  9.21351991e-04],
       [ 5.07443726e-03, -1.58483844e-02,  9.22744820e-03,
        -1.65285992e-02,  4.68883346e-02, -2.03977996e-02,
        -1.32932964e-04, -7.18075752e-03,  1.84976077e-03],
       [-4.07658075e-03,  6.59980653e-03, -1.13651380e-02,
         8.39889941e-03, -2.03977996e-02,  2.27058260e-02,
         8.87141186e-04,  2.38616187e-03, -2.02146764e-03],
       [ 2.43384515e-04,  1.75053208e-03, -4.18897332e-04,
        -1.51799364e-03, -1.32932964e-04,  8.87141186e-04,
         3.87579111e-03, -1.48625496e-03, -1.14163720e-04],
       [ 4.43488080e-04,  4.53228164e-03, -1.60789731e-03,
        -4.17429455e-05, -7.18075752e-03,  2.38616187e-03,
        -1.48625496e-03,  1.20973384e-02, -3.31096423e-03],
       [-1.72051955e-04, -6.90583213e-04,  9.24313673e-04,
         9.21351991e-04,  1.84976077e-03, -2.02146764e-03,
        -1.14163720e-04, -3.31096423e-03,  3.95069513e-03]])

range_post_cov = np.array([[ 0.00389826, -0.00275459,  0.00131422, -0.00287918,  0.00224106,
        -0.00184821,  0.00015821,  0.0001398 ,  0.00017417],
       [-0.00275459,  0.00745223, -0.0036786 ,  0.00010056, -0.00165754,
         0.001429  ,  0.00114549,  0.00085336, -0.00027707],
       [ 0.00131422, -0.0036786 ,  0.00581071, -0.00039857,  0.00047248,
        -0.00327776,  0.0001384 , -0.00066359,  0.00064227],
       [-0.00287918,  0.00010056, -0.00039857,  0.01212723, -0.00988254,
         0.00303601, -0.00182846,  0.00019348,  0.00052403],
       [ 0.00224106, -0.00165754,  0.00047248, -0.00988254,  0.02122722,
        -0.00634579,  0.0001077 , -0.00272059,  0.00061086],
       [-0.00184821,  0.001429  , -0.00327776,  0.00303601, -0.00634579,
         0.01171366,  0.00070328,  0.00042865, -0.00224499],
       [ 0.00015821,  0.00114549,  0.0001384 , -0.00182846,  0.0001077 ,
         0.00070328,  0.00448887, -0.00225439,  0.00085958],
       [ 0.0001398 ,  0.00085336, -0.00066359,  0.00019348, -0.00272059,
         0.00042865, -0.00225439,  0.00751886, -0.00263526],
       [ 0.00017417, -0.00027707,  0.00064227,  0.00052403,  0.00061086,
        -0.00224499,  0.00085958, -0.00263526,  0.00326332]])

GEV_post_cov = np.array([[0.00093752, 0.00046485, 0],
                         [0.00046485, 0.00031506, 0],
                         [0         , 0         , 1]])

if rank == 0:
    start_time = time.time()
else:
    start_time = None

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
        if iter % 25 == 0:
            print(iter)
        if iter % 1000 == 0 or iter == n_iters-1:
            # Save data every 1000 iterations
            end_time = time.time()
            print('elapsed: ', round(end_time - start_time, 1), ' seconds')
            np.save('R_trace_log', R_trace_log)
            np.save('phi_knots_trace', phi_knots_trace)
            np.save('range_knots_trace', range_knots_trace)
            np.save('GEV_knots_trace', GEV_knots_trace)

            # Print traceplot every 1000 iterations
            xs = np.arange(iter)
            xs_thin = xs[0::10]
            xs_thin2 = np.arange(len(xs_thin))
            R_trace_log_thin = R_trace_log[0:iter:10,:,:]
            phi_knots_trace_thin = phi_knots_trace[0:iter:10,:]
            range_knots_trace_thin = range_knots_trace[0:iter:10,:]
            GEV_knots_trace_thin = GEV_knots_trace[0:iter:10,:,:]

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


#### ----- Update Rt ----- Parallelized Across N time

    # Propose a R at time "rank", on log-scale
    R_proposal_log = random_generator.normal(loc=0.0, scale=2.0, size=k) + R_current_log

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
    
    R_current_log = R_update_log
    R_vec_current = wendland_weight_matrix @ np.exp(R_current_log)
    
    # Gather across N_t, store into trace matrix
    R_current_log_gathered = comm.gather(R_current_log, root=0)
    # print(R_current_log_gathered)
    if rank == 0:
        R_trace_log[iter,:,:] = np.vstack(R_current_log_gathered).T

#### ----- Update phi ----- parallelized likelihood calculation across N time

    # Propose new phi at the knots --> new phi vector
    if rank == 0:
        # random_walk_kxk = random_generator.multivariate_normal(np.zeros(k), phi_post_cov, size = None) # size = None returns vector
        # phi_knots_proposal = phi_knots_current + random_walk_kxk
        # phi_knots_proposal = random_generator.normal(loc = 0.0, scale = 0.1, size = k) + phi_knots_current
        random_walk_block1 = random_generator.multivariate_normal(np.zeros(3), phi_post_cov[0:3,0:3], size = None)
        random_walk_block2 = random_generator.multivariate_normal(np.zeros(3), phi_post_cov[3:6,3:6], size = None)
        random_walk_block3 = random_generator.multivariate_normal(np.zeros(3), phi_post_cov[6:9,6:9], size = None)
        random_walk_perturb = np.hstack((random_walk_block1,random_walk_block2,random_walk_block3))
        phi_knots_proposal = phi_knots_current + random_walk_perturb
    else:
        phi_knots_proposal = None
    phi_knots_proposal = comm.bcast(phi_knots_proposal, root = 0)

    phi_vec_proposal = gaussian_weight_matrix @ phi_knots_proposal

    # Conditional Likelihood at Current
    # X_star_1t_current = qRW_Newton(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
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
        X_star_1t_proposal = qRW_Newton(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
                                      phi_vec_proposal, gamma, 100)
        lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_proposal, 
                                                        Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                        phi_vec_proposal, gamma_vec, R_vec_current, cholesky_matrix_current)
    
    # Gather likelihood calculated across time
    lik_gathered = comm.gather(lik_1t, root = 0)
    lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

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
        X_star_1t_current = qRW_Newton(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
                                        phi_vec_current, gamma, 100)

#### ----- Update range_vec ----- parallelized likelihood calculation across N time

    # Propose new range at the knots --> new range vector
    if rank == 0:
        # random_walk_kxk = random_generator.multivariate_normal(np.zeros(k), range_post_cov, size = None) # size = None so returns vector
        # range_knots_proposal = range_knots_current + random_walk_kxk
        # range_knots_proposal = random_generator.normal(loc = 0.0, scale = 0.1, size = k) + range_knots_current
        random_walk_block1 = random_generator.multivariate_normal(np.zeros(3), range_post_cov[0:3,0:3], size = None)
        random_walk_block2 = random_generator.multivariate_normal(np.zeros(3), range_post_cov[3:6,3:6], size = None)
        random_walk_block3 = random_generator.multivariate_normal(np.zeros(3), range_post_cov[6:9,6:9], size = None)
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

    # Propose new GEV params at the knots --> new GEV params vector
    if rank == 0:
        random_walk_3x3 = random_generator.multivariate_normal(np.zeros(3), GEV_post_cov, size = k).T
        GEV_knots_proposal = GEV_knots_current + random_walk_3x3
        GEV_knots_proposal[:,1:] = np.vstack(GEV_knots_proposal[:,0]) # treat it as if it's only one knot
        GEV_knots_proposal[2,:] = GEV_knots_current[2,:] # hold ksi constant
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
        X_star_1t_proposal = qRW_Newton(pgev(Y[:,rank], Loc_matrix_proposal[:,rank], Scale_matrix_proposal[:,rank], Shape_matrix_proposal[:,rank]),
                                      phi_vec_current, gamma, 100)
        lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_proposal, 
                                                        Loc_matrix_proposal[:,rank], Scale_matrix_proposal[:,rank], Shape_matrix_proposal[:,rank],
                                                        phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
    # Gather likelihood calculated across time
    lik_gathered = comm.gather(lik_1t, root = 0)
    lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

    # Accept or Reject
    if rank == 0:
        GEV_accepted = False
        lik = sum(lik_gathered)
        lik_proposal = sum(lik_proposal_gathered)

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
        X_star_1t_current = qRW_Newton(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
                                        phi_vec_current, gamma, 100)
    

# End of MCMC
if rank == 0:
    end_time = time.time()
    print('total time: ', round(end_time - start_time, 1), ' seconds')
    print('true R: ', R_at_knots)
    np.save('R_trace_log', R_trace_log)
    np.save('phi_knots_trace', phi_knots_trace)
    np.save('range_knots_trace', range_knots_trace)
    np.save('GEV_knots_trace', GEV_knots_trace)

# # %%
# # Plotting

# folder = './data/20230904_5knots_10000_500_32_phi_range/'
# phi_knots_trace = np.load(folder + 'phi_knots_trace.npy')
# R_trace_log = np.load(folder + 'R_trace_log.npy')
# range_knots_trace = np.load(folder + 'range_knots_trace.npy')
# GEV_knots_trace = np.load(folder + 'GEV_knots_trace.npy')
# xs = np.arange(10000)

# # %%
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

# # %%
# np.corrcoef(np.array([phi_knots_trace[:,0].ravel(), 
#                       GEV_knots_trace[:,0,0].ravel(), 
#                       GEV_knots_trace[:,1,0].ravel()]))

# np.corrcoef(np.array([phi_knots_trace[:,1].ravel(), 
#                       GEV_knots_trace[:,0,1].ravel(), 
#                       GEV_knots_trace[:,1,1].ravel()]))

# np.cov(np.array([phi_knots_trace[:,0].ravel(), 
#                 GEV_knots_trace[:,0,0].ravel(), 
#                 GEV_knots_trace[:,1,0].ravel()]))

# np.cov(np.array([phi_knots_trace[:,1].ravel(), 
#                 GEV_knots_trace[:,0,1].ravel(), 
#                 GEV_knots_trace[:,1,1].ravel()]))

# # %%
# phi_post_cov = np.cov(np.array([phi_knots_trace[:,0].ravel(), 
#                                 phi_knots_trace[:,1].ravel(),
#                                 phi_knots_trace[:,2].ravel(),
#                                 phi_knots_trace[:,3].ravel(),
#                                 phi_knots_trace[:,4].ravel()]))

# range_post_cov = np.cov(np.array([range_knots_trace[:,0].ravel(), 
#                                 range_knots_trace[:,1].ravel(),
#                                 range_knots_trace[:,2].ravel(),
#                                 range_knots_trace[:,3].ravel(),
#                                 range_knots_trace[:,4].ravel()]))

# GEV_post_cov = np.cov(np.array([GEV_knots_trace[:,0,0].ravel(), # mu location
#                                 GEV_knots_trace[:,1,0].ravel()])) # tau scale

# phi_post_cov = np.cov(np.array([phi_knots_trace[:,i].ravel() for i in range(k)]))

# range_post_cov = np.cov(np.array([range_knots_trace[:,i].ravel() for i in range(k)]))

# Evaluate Profile Likelihoods
# ###########################################################################################

# # %%
# # Evaluate Phi

# phis = np.linspace(0.1, 0.8, num = 50)
# lik = []
# for phi in phis:
#     phi_vec = np.repeat(phi, num_sites)
#     # X_star_tmp = X_star
#     X_star_tmp = qRW_Newton(pgev(Y, Loc_matrix, Scale_matrix, Shape_matrix), phi, gamma, 100)
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
#     X_star_tmp = qRW_Newton(pgev(Y, Loc_matrix, Scale_matrix, Shape_matrix), phi, gamma, 100)
#     lik.append(marg_transform_data_mixture_likelihood(Y, X_star_tmp, Loc_matrix, Scale_matrix, Shape_matrix, phi_vec, gamma_vec, R_matrix, cholesky_matrix))
# plt.plot(Locs, lik)
# Loc_matrix = np.full(shape = Y.shape, fill_value = mu)

# # %%
# # Evaluate tau (scale parameter)
# Scales = np.linspace(0.8, 1.2, num = 20)
# lik = []
# for Scale in Scales:
#     Scale_matrix = np.full(shape = Y.shape, fill_value = Scale)
#     X_star_tmp = qRW_Newton(pgev(Y, Loc_matrix, Scale_matrix, Shape_matrix), phi, gamma, 100)
#     lik.append(marg_transform_data_mixture_likelihood(Y, X_star_tmp, Loc_matrix, Scale_matrix, Shape_matrix, phi_vec, gamma_vec, R_matrix, cholesky_matrix))
# plt.plot(Scales, lik)
# Scale_matrix = np.full(shape = Y.shape, fill_value = tau)

# # %%
# # Evaluate ksi (shape parameter)
# Shapes = np.linspace(0.1, 0.3, num = 20)
# lik = []
# for Shape in Shapes:
#     Shape_matrix = np.full(shape = Y.shape, fill_value = Shape)
#     X_star_tmp = qRW_Newton(pgev(Y, Loc_matrix, Scale_matrix, Shape_matrix), phi, gamma, 100)
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


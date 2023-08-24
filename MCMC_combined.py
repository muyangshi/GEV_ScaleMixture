#%%
# Imports and Set Parameters
import numpy as np
import matplotlib.pyplot as plt
# from multiprocessing import Pool
# from p_cubature import *
from model_sim import *
from ns_cov import *
import scipy
from scipy.stats import uniform
from mpi4py import MPI
import time

# Sites Parameters
np.random.seed(2345)
# n_cores = 6
N = 32 # number of time replicates
num_sites = 50
k = 1 # number of knots
sites_xy = np.random.random((num_sites, 2)) * 10
sites_x = sites_xy[:,0]
sites_y = sites_xy[:,1]

# Z parameters
covariance_matrix = 1 # the Cov for multivariate gaussian Z(s)
rho = 2.0 # the rho in matern kernel exp(-rho * x)
length_scale = 1/rho # scikit/learn parameterization (length_scale)
nu = 0.5 # exponential kernel for matern with nu = 1/2
range_vec = np.repeat(length_scale, num_sites)
sigsq_vec = np.repeat(1, num_sites)

# GEV Parameters
mu = 0.0
tau = 1.0
ksi = 0.2

# Scale Mixture Parameters
phi = 0.33 # the phi in R^phi*W
gamma = 0.5 # thiis s the gamma that goes in rlevy
delta = 0.0

###########################################################################################

# %% 
# Simulation
K = ns_cov(range_vec=range_vec,  sigsq_vec = sigsq_vec, 
             coords = sites_xy, kappa = nu, cov_model = "matern")
Z = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(num_sites,)),cov=K,size=N).T
W = norm_to_Pareto1(Z)
R = rlevy(n=N, m=delta, s = gamma)
R = scipy.stats.levy.rvs(loc=delta, scale = gamma, size = N)


R_phi = pow(base=R,exp=phi)
X_star = R_phi * W
Y = qgev(pRW(X_star, phi, gamma), mu, tau, ksi)


Loc_matrix = np.full(shape = Y.shape, fill_value = mu)
Scale_matrix = np.full(shape = Y.shape, fill_value = tau)
Shape_matrix = np.full(shape = Y.shape, fill_value = ksi)
R_matrix = np.tile(R, num_sites).reshape(Y.shape)
phi_vec = np.repeat(phi, num_sites)
gamma_vec = np.repeat(gamma, num_sites)
cholesky_matrix = scipy.linalg.cholesky(K, lower=False)

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


###########################################################################################
# Metropolis Updates ######################################################################
###########################################################################################

# %%
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

random_generator = np.random.RandomState()
n_iters = 10000

mvn_cov = 2*np.array([[ 1.51180152e-02, -3.53233442e-05,  6.96443508e-03, 7.08467852e-03],
                    [-3.53233442e-05,  1.60576481e-03,  9.46786420e-04,-8.60876113e-05],
                    [ 6.96443508e-03,  9.46786420e-04,  4.25227059e-03, 3.39474201e-03],
                    [ 7.08467852e-03, -8.60876113e-05,  3.39474201e-03, 3.92065445e-03]])

mvn_cov_3x3 = 2*np.array([[ 1.60576481e-03,  9.46786420e-04, -8.60876113e-05],
                        [ 9.46786420e-04,  4.25227059e-03,  3.39474201e-03],
                        [-8.60876113e-05,  3.39474201e-03,  3.92065445e-03]])

if rank == 0:
    start_time = time.time()
else:
    start_time = None

########## Storage Place ##################################################

## ---- R log scaled ----
if rank == 0:
    R_trace_log = np.empty((n_iters, N)) # N is n_t
    R_trace_log[:,:] = np.nan
    R_trace_log[0,:] = np.log(R) # initialize
    R_init_log = R_trace_log[0,:]
else:
    R_init_log = None
R_init_log = comm.bcast(R_init_log, root = 0) # vector

## ---- phi ----
if rank == 0:
    phi_knots_trace = np.empty((n_iters,k))
    phi_knots_trace[:,:] = np.nan
    phi_knots_trace[0,:] = np.repeat(0.33, k)
    phi_knots_init = phi_knots_trace[0,:]
else:
    phi_knots_init = None
phi_knots_init = comm.bcast(phi_knots_init, root = 0)

## ---- range_vec (length_scale) ----
if rank == 0:
    range_knots_trace = np.empty((n_iters, k))
    range_knots_trace[:,:] = np.nan
    range_knots_trace[0,:] = np.repeat(0.5, k)
    range_knots_init = range_knots_trace[0,:]
else:
    range_knots_init = None
range_knots_init = comm.bcast(range_knots_init, root = 0)

## ---- GEV mu (location) ----

## ---- GEV tau (scale) ----

## ---- GEV ksi (shape) ----

## ---- GEV mu tau ksi (location, scale, shape) together ----
if rank == 0:
    GEV_knots_trace = np.empty((n_iters, 3, k)) # [n_iters, n_GEV, num_knots]
    GEV_knots_trace[:,:,:] = np.nan
    GEV_knots_trace[0,:,:] = np.tile(np.array([mu, tau, ksi]), (k,1)).T
    GEV_knots_init = GEV_knots_trace[0,:,:]
else:
    GEV_knots_init = None
GEV_knots_init = comm.bcast(GEV_knots_init, root = 0)

########## Initialize ##################################################

## ---- R ----
R_current_log = np.array(R_init_log[rank]) # log-scale number, initialize, worker specific

## ---- phi ----
phi_knots_current = phi_knots_init
phi_vec_current = np.repeat(1, num_sites) * phi_knots_current # will be changed into matrix multiplication w/ more knots

## ---- range_vec (length_scale) ----
range_knots_current = range_knots_init
range_vec_current = np.repeat(1, num_sites) * range_knots_current # will be changed into matrix multiplication w/ more knots
K_current = ns_cov(range_vec = range_vec_current,
                       sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = "matern")
cholesky_matrix_current = scipy.linalg.cholesky(K_current, lower = False)

## ---- GEV mu (location) ----

## ---- GEV tau (scale) ----

## --- GEV ksi (shape) ----

## ---- GEV mu tau ksi (location, scale, shape) together ----
GEV_knots_current = GEV_knots_init
    # will be changed into matrix multiplication w/ more knots:
Loc_matrix_current = GEV_knots_current[0,0] * np.full(shape = (num_sites,N), fill_value = 1)
Scale_matrix_current = GEV_knots_current[1,0] * np.full(shape = (num_sites,N), fill_value = 1)
Shape_matrix_current = GEV_knots_current[2,0] * np.full(shape = (num_sites,N), fill_value = 1)

########## Updates ##################################################

for iter in range(1, n_iters):
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
            R_trace_log_thin = R_trace_log[0:iter:10,:]
            phi_knots_trace_thin = phi_knots_trace[0:iter:10,:]
            range_knots_trace_thin = range_knots_trace[0:iter:10,:]
            GEV_knots_trace_thin = GEV_knots_trace[0:iter:10,:,:]

            # ---- phi ----
            plt.subplots()
            plt.plot(xs_thin2, phi_knots_trace_thin)
            plt.title('traceplot for phi')
            plt.xlabel('iter thinned by 10')
            plt.ylabel('phi')
            plt.savefig('phi.pdf')

            # ---- R_t ----
            plt.subplots()
            plt.plot(xs_thin2, R_trace_log_thin[:,0])
            plt.plot(xs_thin2, R_trace_log_thin[:,1])
            plt.plot(xs_thin2, R_trace_log_thin[:,2])
            plt.plot(xs_thin2, R_trace_log_thin[:,3])
            plt.plot(xs_thin2, R_trace_log_thin[:,4])
            plt.title('traceplot for some R_t')
            plt.xlabel('iter thinned by 10')
            plt.ylabel('R_ts')
            plt.savefig('R_t.pdf')

            # ---- range ----
            plt.subplots()
            plt.plot(xs_thin2, range_knots_trace_thin)
            plt.title('traceplot for range')
            plt.xlabel('iter thinned by 10')
            plt.ylabel('range')
            plt.savefig('range.pdf')

            # ---- GEV ----
            ## location mu
            plt.subplots()
            plt.plot(xs_thin2, GEV_knots_trace_thin[:,0]) # location
            plt.title('traceplot for location')
            plt.xlabel('iter thinned by 10')
            plt.ylabel('mu')
            plt.savefig('mu.pdf')
            ## scale tau
            plt.subplots()
            plt.plot(xs_thin2, GEV_knots_trace_thin[:,1]) # scale
            plt.title('traceplot for scale')
            plt.xlabel('iter thinned by 10')
            plt.ylabel('tau')
            plt.savefig('tau.pdf')
            ## shape ksi
            plt.subplots()
            plt.plot(xs_thin2, GEV_knots_trace_thin[:,2]) # shape
            plt.title('traceplot for shape')
            plt.xlabel('iter thinned by 10')
            plt.ylabel('ksi')
            plt.savefig('ksi.pdf')
            ## together
            plt.subplots()
            plt.plot(xs_thin2, phi_knots_trace_thin, label='phi')
            plt.plot(xs_thin2, GEV_knots_trace_thin[:,0], label='mu') # location
            plt.plot(xs_thin2, GEV_knots_trace_thin[:,1], label='tau') # scale
            plt.plot(xs_thin2, GEV_knots_trace_thin[:,2], label='ksi') # shape
            plt.title('traceplot for phi and GEV')
            plt.legend()
            plt.savefig('phi_GEV.pdf')


#### ----- Update Rt ----- Parallelized Across N time

    # Propose a R at time "rank", on log-scale
    R_proposal_log = random_generator.normal(loc=0.0, scale=3.0, size=1) + R_current_log

    # Conditional Likelihood at Current
    R_vec_current = np.repeat(np.exp(R_current_log), num_sites) # will be changed into matrix multiplication w/ more knots
    # if np.any(~np.isfinite(R_vec_current**phi_vec_current)): print("Negative or zero R, iter=", iter, ", rank=", rank, R_vec_current[0], phi_vec_current[0])
    lik = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star[:,rank], 
                                                    Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank], 
                                                    phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
    prior = scipy.stats.levy.logpdf(np.exp(R_current_log)) + R_current_log

    # Conditional Likelihood at Proposal
    R_vec_star = np.repeat(np.exp(R_proposal_log), num_sites) # will be changed into matrix multiplication w/ more knots
    # if np.any(~np.isfinite(R_vec_star**phi_vec_current)): print("Negative or zero R, iter=", iter, ", rank=", rank, R_vec_star[0], phi_vec_current[0])
    lik_star = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star[:,rank], 
                                                    Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank], 
                                                    phi_vec_current, gamma_vec, R_vec_star, cholesky_matrix_current)
    prior_star = scipy.stats.levy.logpdf(np.exp(R_proposal_log)) + R_proposal_log

    # Accept or Reject
    u = random_generator.uniform()
    ratio = np.exp(lik_star + prior_star - lik - prior)
    if u > ratio: # Reject
        R_update_log = R_current_log
    else: # Accept, u <= ratio
        R_update_log = R_proposal_log
    
    R_current_log = R_update_log
    R_vec_current = np.repeat(np.exp(R_current_log), num_sites)
    
    # Gather across N_t, store into trace matrix
    R_current_log_gathered = comm.gather(R_current_log, root=0)
    if rank == 0:
        R_trace_log[iter,:] = np.vstack(R_current_log_gathered).T.ravel()

#### ----- Update phi ----- parallelized likelihood calculation across N time

    # Propose new phi at the knots --> new phi vector
    if rank == 0:
        phi_knots_proposal = random_generator.normal(loc = 0.0, scale = 0.1, size = k) + phi_knots_current
    else:
        phi_knots_proposal = None
    phi_knots_proposal = comm.bcast(phi_knots_proposal, root = 0)

    # will be changed into matrix multiplication w/ more knots
    phi_vec_proposal = np.repeat(1, num_sites) * phi_knots_proposal

    # Conditional Likelihood at Current
    X_star_1t_current = qRW_Newton(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
                                  phi_vec_current, gamma, 100)
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
        # if np.isnan(lik_1t_proposal):
        #     print(np.exp(R_current_log)) 
        #     tmp = marg_transform_data_mixture_likelihood_1t_detail(Y[:,rank], X_star_1t_proposal, 
        #                                                 Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
        #                                                 phi_vec_proposal, gamma_vec, R_vec_current, cholesky_matrix_current)
        #     print(tmp)
    
    # Gather likelihood calculated across time
    lik_gathered = comm.gather(lik_1t, root = 0)
    lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

    # Accept or Reject
    if rank == 0:
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
        
        # Store the result
        phi_knots_trace[iter,:] = phi_knots_update

        # Update the "current" value
        phi_vec_current = phi_vec_update
        phi_knots_current = phi_knots_update

    # Brodcast the updated values
    phi_vec_current = comm.bcast(phi_vec_current, root = 0)
    phi_knots_current = comm.bcast(phi_knots_current, root = 0)

# #### ----- Update phi and GEV together ----- parallelized likelihood calculation across N time
    
#     # Propose new (phi, mu, tau, ksi) at the knots
#     if rank == 0:
#         random_walk_4x4 = random_generator.multivariate_normal(np.zeros(4), mvn_cov, size = k).T
#         phi_knots_proposal = phi_knots_current + random_walk_4x4[0,:]
#         GEV_knots_proposal = GEV_knots_current + random_walk_4x4[1:4, :]
#         GEV_knots_proposal[2,:] = GEV_knots_current[2,:]
#     else:
#         phi_knots_proposal = None
#         GEV_knots_proposal = None
#     phi_knots_proposal = comm.bcast(phi_knots_proposal, root = 0)
#     GEV_knots_proposal = comm.bcast(GEV_knots_proposal, root = 0)

#     # will be changed into matrix multiplication w/ more knots
#     phi_vec_proposal = np.repeat(1, num_sites) * phi_knots_proposal[0]
#     Loc_matrix_proposal = GEV_knots_proposal[0,0] * np.full(shape = (num_sites,N), fill_value = 1)
#     Scale_matrix_proposal = GEV_knots_proposal[1,0] * np.full(shape = (num_sites,N), fill_value = 1)
#     Shape_matrix_proposal = GEV_knots_proposal[2,0] * np.full(shape = (num_sites,N), fill_value = 1)

#     # Conditional Likelihood at Current
#     X_star_1t_current = qRW_Newton(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
#                                     phi_vec_current, gamma, 100)
#     lik_1t = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
#                                                     Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
#                                                     phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)

#     # Conditional Likelihood at Proposed
#     phi_out_of_range = any(phi <= 0 for phi in phi_knots_proposal) or any(phi > 1 for phi in phi_knots_proposal) # U(0,1] prior
#     Scale_out_of_range = any(scale <= 0 for scale in GEV_knots_proposal[1,:]) # U(0, inf) prior
#     Shape_out_of_range = any(shape <= -0.5 for shape in GEV_knots_proposal[2,:]) or any(shape > 0.5 for shape in GEV_knots_proposal[2,:]) # U(-0.5, 0.5] prior
#     if any((phi_out_of_range, Scale_out_of_range, Shape_out_of_range)):
#         X_star_1t_proposal = np.NINF
#         lik_1t_proposal = np.NINF
#     else:
#         X_star_1t_proposal = qRW_Newton(pgev(Y[:,rank], Loc_matrix_proposal[:,rank], Scale_matrix_proposal[:,rank], Shape_matrix_proposal[:,rank]),
#                                         phi_vec_proposal, gamma, 100)
#         lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_proposal, 
#                                                         Loc_matrix_proposal[:,rank], Scale_matrix_proposal[:,rank], Shape_matrix_proposal[:,rank],
#                                                         phi_vec_proposal, gamma_vec, R_vec_current, cholesky_matrix_current)

#     # Gather likelihood calculated across time
#     lik_gathered = comm.gather(lik_1t, root = 0)
#     lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

#     # Accept or Reject
#     if rank == 0:
#         lik = sum(lik_gathered)
#         lik_proposal = sum(lik_proposal_gathered)

#         u = random_generator.uniform()
#         ratio = np.exp(lik_proposal - lik)
#         if not np.isfinite(ratio):
#             ratio = 0
#         if u > ratio: # Reject
#             phi_vec_update = phi_vec_current
#             Loc_matrix_update = Loc_matrix_current
#             Scale_matrix_update = Scale_matrix_current
#             Shape_matrix_update = Shape_matrix_current
#             phi_knots_update = phi_knots_current
#             GEV_knots_update = GEV_knots_current
#         else: # Accept, u <= ratio
#             phi_vec_update = phi_vec_proposal
#             Loc_matrix_update = Loc_matrix_proposal
#             Scale_matrix_update = Scale_matrix_proposal
#             Shape_matrix_update = Shape_matrix_proposal
#             phi_knots_update = phi_knots_proposal
#             GEV_knots_update = GEV_knots_proposal

#         # Store the result
#         phi_knots_trace[iter,:] = phi_knots_update
#         GEV_knots_trace[iter,:,:] = GEV_knots_update

#         # Update the "current" values
#         phi_vec_current = phi_vec_update
#         phi_knots_current = phi_knots_update
#         Loc_matrix_current = Loc_matrix_update
#         Scale_matrix_current = Scale_matrix_update
#         Shape_matrix_current = Shape_matrix_update
#         GEV_knots_current = GEV_knots_update

#     # Brodcast the updated values
#     phi_vec_current = comm.bcast(phi_vec_current, root = 0)
#     phi_knots_current = comm.bcast(phi_knots_current, root = 0)
#     Loc_matrix_current = comm.bcast(Loc_matrix_current, root = 0)
#     Scale_matrix_current = comm.bcast(Scale_matrix_current, root = 0)
#     Shape_matrix_current = comm.bcast(Shape_matrix_current, root = 0)
#     GEV_knots_current = comm.bcast(GEV_knots_current, root = 0)

#### ----- Update range_vec ----- parallelized likelihood calculation across N time

    # Propose new range at the knots --> new range vector
    if rank == 0:
        range_knots_proposal = random_generator.normal(loc = 0.0, scale = 0.1, size = 1) + range_knots_current
    else:
        range_knots_proposal = None
    range_knots_proposal = comm.bcast(range_knots_proposal, root = 0)
    range_vec_proposal = np.repeat(1, num_sites) * range_knots_proposal # will be changed into matrix multiplication w/ more knots
    K_proposal = ns_cov(range_vec = range_vec_proposal,
                        sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = "matern")
    cholesky_matrix_proposal = scipy.linalg.cholesky(K_proposal, lower = False)

    # One calculation of X_star is enough, with the most current phi_vec
    X_star_1t_current = qRW_Newton(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
                                  phi_vec_current, gamma, 100)

    # Conditional Likelihood at Current
    lik_1t = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
                                                    Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                    phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)

    # Conditional Likelihood at Proposed
    if any(range <= 0 for range in range_knots_proposal):
        lik_1t_proposal = np.NINF
    else:
        lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
                                                    Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                    phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_proposal)

    # Gather likelihood calculated across time
    lik_gathered = comm.gather(lik_1t, root = 0)
    lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

    # Accept or Reject
    if rank == 0:
        lik = sum(lik_gathered)
        lik_proposal = sum(lik_proposal_gathered)

        u = random_generator.uniform()
        ratio = np.exp(lik_proposal - lik)
        if not np.isfinite(ratio):
            ratio = 0 # Force a rejection
        if u > ratio: # Reject
            range_vec_update = range_vec_current
        else: # Accept, u <= ratio
            range_vec_update = range_vec_proposal
        
        # Store the result
        range_knots_update = np.array(range_vec_update[0]) # will be changed into matrix multiplication w/ more knots
        range_knots_trace[iter,:] = range_knots_update

        # Update the "current" value
        range_vec_current = range_vec_update
        range_knots_current = range_knots_update

    # Brodcast the updated values
    range_vec_current = comm.bcast(range_vec_current, root = 0)
    range_knots_current = comm.bcast(range_knots_current, root = 0)
    # pass

#### ----- Update GEV mu tau ksi (location, scale, shape) together ----
#### ----- Do not update ksi -----

    # Propose new GEV params at the knots --> new GEV params vector
    if rank == 0:
        random_walk_3x3 = random_generator.multivariate_normal(np.zeros(3), mvn_cov_3x3, size = k).T
        GEV_knots_proposal = GEV_knots_current + random_walk_3x3
        GEV_knots_proposal[2,:] = GEV_knots_current[2,:] # hold ksi constant
    else:
        GEV_knots_proposal = None
    GEV_knots_proposal = comm.bcast(GEV_knots_proposal, root = 0)

    # will be changed into matrix multiplication w/ more knots
    Loc_matrix_proposal = GEV_knots_proposal[0,0] * np.full(shape = (num_sites,N), fill_value = 1)
    Scale_matrix_proposal = GEV_knots_proposal[1,0] * np.full(shape = (num_sites,N), fill_value = 1)
    Shape_matrix_proposal = GEV_knots_proposal[2,0] * np.full(shape = (num_sites,N), fill_value = 1)

    # Conditional Likelihodd at Current
    X_star_1t_current = qRW_Newton(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
                                  phi_vec_current, gamma, 100)
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
        
        # Store the result
        GEV_knots_trace[iter,:,:] = GEV_knots_update

        # Update the "current" value
        Loc_matrix_current = Loc_matrix_update
        Scale_matrix_current = Scale_matrix_update
        Shape_matrix_current = Shape_matrix_update
        GEV_knots_current = GEV_knots_update

    # Brodcast the updated values
    Loc_matrix_current = comm.bcast(Loc_matrix_current, root = 0)
    Scale_matrix_current = comm.bcast(Scale_matrix_current, root = 0)
    Shape_matrix_current = comm.bcast(Shape_matrix_current, root = 0)
    GEV_knots_current = comm.bcast(GEV_knots_current, root = 0)

# End of MCMC
if rank == 0:
    end_time = time.time()
    print('total time: ', round(end_time - start_time, 1), ' seconds')
    print('true R: ', R)
    np.save('R_trace_log', R_trace_log)
    np.save('phi_knots_trace', phi_knots_trace)
    np.save('range_knots_trace', range_knots_trace)
    np.save('GEV_knots_trace', GEV_knots_trace)

# %%
# Plotting
phi_knots_trace = np.load('phi_knots_trace.npy')
R_trace_log = np.load('R_trace_log.npy')
range_knots_trace = np.load('range_knots_trace.npy')
GEV_knots_trace = np.load('GEV_knots_trace.npy')
xs = np.arange(3000)
# %%
plt.plot(xs, phi_knots_trace)
# %%
plt.plot(xs, R_trace_log[:,0])
plt.plot(xs, R_trace_log[:,1])
plt.plot(xs, R_trace_log[:,2])
plt.plot(xs, R_trace_log[:,3])
plt.plot(xs, R_trace_log[:,4])
# %%
plt.plot(xs, range_knots_trace)
# %%
plt.plot(xs, GEV_knots_trace[:,0]) # location
# %%
plt.plot(xs, GEV_knots_trace[:,1]) # scale
# %%
plt.plot(xs, GEV_knots_trace[:,2]) # shape
# %%
plt.plot(xs, phi_knots_trace)
plt.plot(xs, GEV_knots_trace[:,0]) # location
plt.plot(xs, GEV_knots_trace[:,1]) # scale
plt.plot(xs, GEV_knots_trace[:,2]) # shape
# %%

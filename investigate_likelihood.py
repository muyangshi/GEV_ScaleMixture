# This is a MCMC sampler that constantly gets updated
# Scratch work and modifications are done in this file
#%%
# Imports
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
N = 32 # number of time replicates
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

radius = 4 # from 6 to 4 to 3.5
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
# range_at_knots = np.sqrt(0.3*knots_x + 0.4*knots_y)/2
range_at_knots = np.array([0.3]*9)
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
# R_matrix = np.tile(R, num_sites).reshape(Y.shape)
R_matrix = R_at_sites
# phi_vec = np.repeat(phi, num_sites)
gamma_vec = np.repeat(gamma, num_sites)
cholesky_matrix = scipy.linalg.cholesky(K, lower=False)


###########################################################
# Analyze Likelihood
###########################################################

# %%
# Load data

folder = './data/20231007_all_adaptive_seed2345_site625_length10000_fullthread/'
phi_knots_trace = np.load(folder + 'phi_knots_trace.npy')
R_trace_log = np.load(folder + 'R_trace_log.npy')
range_knots_trace = np.load(folder + 'range_knots_trace.npy')
GEV_knots_trace = np.load(folder + 'GEV_knots_trace.npy')
xs = np.arange(10000)

# %%
# likelihood at initial setting

marg_transform_data_mixture_likelihood(Y, X_star, Loc_matrix, Scale_matrix, Shape_matrix, phi_vec, gamma_vec, R_matrix, cholesky_matrix)
# -26356.58689759078

# %%
# iteration 870

## need to calculate the X
### The marginal parameters are not spatially varying
Loc_matrix_870 = np.full(shape = Y.shape, fill_value = GEV_knots_trace[870, 0, :][0])
Scale_matrix_870 = np.full(shape = Y.shape, fill_value = GEV_knots_trace[870, 1, :][0])
Shape_matrix_870 = np.full(shape = Y.shape, fill_value = GEV_knots_trace[870, 2, :][0])
phi_vec_870 = gaussian_weight_matrix @ phi_knots_trace[870,:]

X = np.full(shape = Y.shape, fill_value = np.nan)
rank = 0
for rank in np.arange(32): # takes 90 seconds
    start = time.time()
    X[:,rank] = qRW_Newton(pgev(Y[:,rank], Loc_matrix_870[:,rank], Scale_matrix_870[:,rank], Shape_matrix_870[:,rank]),
                                phi_vec_870[:], gamma, 100)
    end = time.time()
    print("t = ", rank, " takes ", round(end - start, 3))

## need to calculate the K from range for its cholesky decomp
range_vec_870 = gaussian_weight_matrix @ range_knots_trace[870,:]
K_870 = ns_cov(range_vec = range_vec_870,
                sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = "matern")
cholesky_matrix_870 = scipy.linalg.cholesky(K_870, lower=False)

R_matrix_870 = wendland_weight_matrix @ np.exp(R_trace_log[870, :, :])

marg_transform_data_mixture_likelihood(Y, X, Loc_matrix_870, Scale_matrix_870, Shape_matrix_870, 
                                       phi_vec_870, gamma_vec, R_matrix_870, cholesky_matrix_870)

detail_likelihoods = np.full(shape = (32,5), fill_value = np.nan)
for t in range(32):
    detail_likelihoods[t,:] = marg_transform_data_mixture_likelihood_1t_detail(
        Y[0:1,t], X[0:1,t], Loc_matrix_870[0:1,t], Scale_matrix_870[0:1,t], Shape_matrix_870[0:1,t], 
        phi_vec_870, gamma_vec, R_matrix_870[0:1,t], cholesky_matrix_870)

## Likelihood at the sites?
def marg_transform_data_mixture_likelihood_1t1s_detail(Y, X, Loc, Scale, Shape, phi_vec, gamma_vec, R_vec, cholesky_U):
    if(isinstance(Y, (int, np.int64, float))): Y=np.array([Y], dtype='float64')


    ## Initialize space to store the log-likelihoods for each observation:
    W_vec = X/R_vec**phi_vec

    Z_vec = pareto1_to_Norm(W_vec)
    # part1 = -0.5*eig2inv_quadform_vector(V, 1/d, Z_vec)-0.5*np.sum(np.log(d)) # multivariate density
    # cholesky_inv = lapack.dpotrs(cholesky_U,Z_vec)
    # part1 = -0.5*np.sum(Z_vec*cholesky_inv[0])-np.sum(np.log(np.diag(cholesky_U))) # multivariate density

    ## Jacobian determinant
    part21 = 0.5*np.sum(Z_vec**2) # 1/standard Normal densities of each Z_j
    part22 = np.sum(-phi_vec*np.log(R_vec)-2*np.log(W_vec+1)) # R_j^phi_j/X_j^2
    part23 = np.sum(dgev(Y, Loc=Loc, Scale=Scale, Shape=Shape, log=True))
    part24 = np.sum(-np.log(dRW(X, phi_vec, gamma_vec)))

    return np.array([part21 ,part22, part23, part24])


detail_likelihoods_by_sites = np.full(shape = (625, 32, 4), fill_value = np.nan)
for s in range(625):
    for t in range(32):
        detail_likelihoods_by_sites[s,t] = marg_transform_data_mixture_likelihood_1t1s_detail(
            Y[s,t],X[s,t],
            Loc_matrix_870[s,t],Scale_matrix_870[s,t],Shape_matrix_870[s,t],
            phi_vec_870[s],gamma_vec[s],R_matrix_870[s,t],cholesky_matrix_870
        )



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




###########################################################
# Integration with Scipy Quad
###########################################################

from scipy.integrate import quad

def dRW_integrand_scipy(r, x, phi, gamma):
    return (r**(phi-3/2)) / ((x+r**phi)**2) * np.exp(-(gamma/(2*r)))

def dRW_scipy(x, phi, gamma):
    return np.sqrt(gamma/(2*np.pi)) * quad(dRW_integrand_scipy, 0, np.inf, args=(x, phi, gamma))[0]


def dRW_integrand_transformed_scipy(t, x, phi, gamma):
    ratio_numerator = np.power((1-t)/t, phi-1.5)
    ratio_denominator = (x + np.power((1-t)/t, phi))**2
    exponential_term = np.exp(-gamma/(2*((1-t)/t)))
    jacobian = 1/(t**2)
    return (ratio_numerator/ratio_denominator) * exponential_term * jacobian
def dRW_transformed_scipy(x, phi, gamma):
    return np.sqrt(gamma/(2*np.pi)) * quad(dRW_integrand_transformed_scipy, 0, 1, args=(x, phi, gamma))[0]

dRW_transformed_scipy_vec = np.vectorize(dRW_transformed_scipy)

pgev(Y[148,8], GEV_knots_trace[870,0,:][0], GEV_knots_trace[870, 1, :][0], GEV_knots_trace[870, 2, :][0])

pgev(Y[148,8], 0, 1, 0.2)

lib.pmixture_C(X[148,8], phi_vec[148], gamma)


###########################################################
# Integration with mpmath
###########################################################

from mpmath import mp
mp.dps = 30

# for the shifted Pareto, no nugget

# mpmath dRW
def dRW_integrand_mpmath(r, x, phi, gamma):
    numerator = mp.power(r, phi-1.5)
    denominator = mp.power(x + mp.power(r, phi), 2)
    exp = mp.exp(-(gamma/(2*r)))
    return numerator / denominator * exp
def dRW_mpmath(x, phi, gamma, **kwargs):
    return mp.sqrt(gamma/(2 * mp.pi)) * mp.quad(lambda r : dRW_integrand_mpmath(r, x, phi, gamma), [0, mp.inf], method='tanh-sinh', **kwargs)
dRW_mpmath_vec = np.vectorize(dRW_mpmath)

# mpmath pRW
def pRW_integrand_mpmath(r, x, phi, gamma):
    numerator = mp.power(r, phi-1.5)
    denominator = x + mp.power(r, phi)
    exp = mp.exp(-(gamma/(2*r)))
    return numerator / denominator * exp
def pRW_mpmath(x, phi, gamma):
    return 1 - mp.sqrt(gamma/(2*mp.pi)) * mp.quad(lambda r : pRW_integrand_mpmath(r, x, phi, gamma), [0, mp.inf], method='tanh-sinh')
pRW_mpmath_vec = np.vectorize(pRW_mpmath)

# mpmath transform dRW -- no significant gain in terms of accuracy as compared to dRW_mpmath
# mpmath with high dps can handle integration from [0, mp.inf] well
def dRW_integrand_transformed_mpmath(t, x, phi, gamma):
    numerator = mp.power((1-t)/t, phi-1.5)
    denominator = mp.power(x + mp.power((1-t)/t, phi), 2)
    exp = mp.exp(-gamma/(2*(1-t)/t))
    jacobian = 1 / mp.power(t, 2)
    return (numerator / denominator) * exp * jacobian
def dRW_transformed_mpmath(x, phi, gamma):
    return mp.sqrt(gamma/(2 * mp.pi)) * mp.quad(lambda t : dRW_integrand_transformed_mpmath(t, x, phi, gamma), [0, 1], method='tanh-sinh')


# mpmath transform pRW -- no significant gain in terms of accuracy, as compared to pRW_mpmath
# mpmath with high dps can handle integration from [0, mp.inf] well
def pRW_integrand_transformed_mpmath(t, x, phi, gamma):
    numerator = mp.power((1-t)/t, phi-1.5)
    denominator = x + mp.power((1-t)/t, phi)
    exp = mp.exp(- gamma / (2 * (1-t)/t))
    jacobian = 1 / mp.power(t, 2)
    return numerator / denominator * exp * jacobian
def pRW_transformed_mpmath(x, phi, gamma):
    return 1 - mp.sqrt(gamma/(2*mp.pi)) * mp.quad(lambda t: pRW_integrand_transformed_mpmath(t, x, phi, gamma), [0, 1], method='tanh-sinh')

plt.plot(np.linspace(1e6,1e10, 500), pRW_mpmath_vec(np.linspace(1e6,1e10, 500), 0.5, 0.5))
plt.show()

def qRW_mpmath(p, phi, gamma):
    return mp.findroot(lambda x : pRW_mpmath(x, phi, gamma) - p,
                       [0,1e12],
                       solver='anderson')

# %%
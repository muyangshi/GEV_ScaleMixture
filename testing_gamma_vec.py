import sys

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

def MLE(seed):
    data_seed = seed

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

    ## Remember to change below
    # knots locations
    # radius
    # range at knots
    # phi_at_knots
    # phi_post_cov
    # range_post_cov
    n_iters = 15000

    
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

    assert k == len(knots_xy)

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

    constant_weight_matrix = np.full(shape = (num_sites, k), fill_value = np.nan)
    for site_id in np.arange(num_sites):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                        XB = knots_xy)
        # influence coming from each of the knots
        weight_from_knots = np.repeat(1, k)/k
        constant_weight_matrix[site_id, :] = weight_from_knots


    
    # ------- 3. Generate covariance matrix, Z, and W --------------------------------

    ## range_vec
    range_at_knots = np.sqrt(0.3*knots_x + 0.4*knots_y)/2 # scenario 2
    # range_at_knots = [0.3]*k
    range_vec = gaussian_weight_matrix @ range_at_knots
    # range_vec = one_weight_matrix @ range_at_knots

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

    # # heatplot of range surface
    # range_vec_for_plot = gaussian_weight_matrix_for_plot @ range_at_knots
    # graph, ax = plt.subplots()
    # heatmap = ax.imshow(range_vec_for_plot.reshape(25,25), cmap ='hot', interpolation='nearest')
    # ax.invert_yaxis()
    # graph.colorbar(heatmap)
    # plt.show()
    # plt.close()

    #####################################################################################################################
    # Write my own covariance function ################################################################################################
    #####################################################################################################################

    # def matern_correlation(d, range, nu):
    #     # using wikipedia definition
    #     part1 = 2**(1-nu)/scipy.special.gamma(nu)
    #     part2 = (np.sqrt(2*nu) * d / range)**nu
    #     part3 = scipy.special.kv(nu, np.sqrt(2*nu) * d / range)
    #     return(part1*part2*part3)
    # matern_correlation_vec = np.vectorize(matern_correlation, otypes=[float])

    # # pairwise_distance = scipy.spatial.distance.pdist(sites_xy)
    # # matern_correlation_vec(pairwise_distance, 1, nu) # gives same result as skMatern(sites_xy)

    # # tri = np.zeros((4,4))
    # # tri[np.triu_indices(4,1)] = matern_correlation_vec(pairwise_distance, 1, 1)
    # # tri + tri.T + np.identity(4)

    # matern_covariance_matrix = np.full(shape=(num_sites, num_sites), 
    #                                    fill_value = 0.0)
    # for i in range(num_sites):
    #     for j in range(i+1, num_sites):
    #         distance = scipy.spatial.distance.pdist(sites_xy[(i,j),])
    #         variance = np.sqrt(sigsq_vec[i] * sigsq_vec[j])
    #         avg_range = (range_vec[i] + range_vec[j])/2
    #         prod_range = np.sqrt(range_vec[i] * range_vec[j])
    #         C = variance * (prod_range / avg_range) * matern_correlation(distance/np.sqrt(avg_range), 1, nu)
    #         matern_covariance_matrix[i,j] = C[0]
    # matern_covariance_matrix += matern_covariance_matrix.T + sigsq * np.identity(num_sites)

    ## Covariance matrix K
    ## sigsq_vec
    sigsq_vec = np.repeat(sigsq, num_sites) # hold at 1
    K = ns_cov(range_vec = range_vec, sigsq_vec = sigsq_vec,
            coords = sites_xy, kappa = nu, cov_model = "matern")
    # K = np.identity(num_sites)
    Z = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(num_sites,)),cov=K,size=N).T
    W = norm_to_Pareto(Z) 

    
    # ------- 4. Generate Scaling Factor, R^phi --------------------------------

    ## phi_vec
    # phi_at_knots = 0.65-np.sqrt((knots_x-3)**2/4 + (knots_y-3)**2/3)/10 # scenario 1
    phi_at_knots = 0.65-np.sqrt((knots_x-5.1)**2/5 + (knots_y-5.3)**2/4)/11.6 # scenario 2
    # phi_at_knots = 10*(0.5*scipy.stats.multivariate_normal.pdf(knots_xy, 
    #                                                            mean = np.array([2.5,3]), 
    #                                                            cov = 2*np.matrix([[1,0.2],[0.2,1]])) + 
    #                     0.5*scipy.stats.multivariate_normal.pdf(knots_xy, 
    #                                                             mean = np.array([7,7.5]), 
    #                                                             cov = 2*np.matrix([[1,-0.2],[-0.2,1]]))) + \
    #                 0.37# scenario 3
    # phi_at_knots = np.array([0.3]*k)
    phi_vec = gaussian_weight_matrix @ phi_at_knots
    # phi_vec = one_weight_matrix @ phi_at_knots

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

    # # heatplot of phi surface
    # phi_vec_for_plot = gaussian_weight_matrix_for_plot @ phi_at_knots
    # graph, ax = plt.subplots()
    # heatmap = ax.imshow(phi_vec_for_plot.reshape(25,25), cmap ='hot', interpolation='nearest', extent = [0, 10, 10, 0])
    # ax.invert_yaxis()
    # graph.colorbar(heatmap)
    # plt.show()
    # plt.close()

    ## R
    ## Generate them at the knots
    R_at_knots = np.full(shape = (k, N), fill_value = np.nan)
    for t in np.arange(N):
        R_at_knots[:,t] = rlevy(n = k, m = delta, s = gamma) # generate R at time t, spatially varying k knots
        # R_at_knots[:,t] = scipy.stats.levy.rvs(delta, gamma, k)
        # R_at_knots[:,t] = np.repeat(rlevy(n = 1, m = delta, s = gamma), k) # generate R at time t, spatially constant k knots

    ## Matrix Multiply to the sites
    R_at_sites = wendland_weight_matrix @ R_at_knots
    # R_at_sites = constant_weight_matrix @ R_at_knots

    ## R^phi
    R_phi = np.full(shape = (num_sites, N), fill_value = np.nan)
    for t in np.arange(N):
        R_phi[:,t] = np.power(R_at_sites[:,t], phi_vec)

    # ------- 5. Generate mu(s) = C(s)Beta ----------------------

    # # C(s) is the covariate
    # Beta = 0.3
    # Loc_at_knots = np.tile(np.sqrt(0.5*knots_x + 0.2*knots_y),
    #                        (N, 1)).T * Beta
    # # Which basis should I use? Gaussian or Wendland?
    # Loc_matrix = gaussian_weight_matrix @ Loc_at_knots # shape (N, num_sites)

    # Loc_site_for_plot = gaussian_weight_matrix_for_plot @ Loc_at_knots
    # Loc_site_for_plot = Loc_site_for_plot[:,0] # look at time t=0
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_surface(plotgrid_X, plotgrid_Y, np.matrix(Loc_site_for_plot).reshape(25,25))
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('mu(s)')
    # ax.scatter(knots_x, knots_y, Loc_at_knots[:,0], c='red', marker='o', s=100)


    
    # ------- 6. Generate X and Y--------------------------------
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

    Y_cst_gamma = np.full(shape=(num_sites, N), fill_value = np.nan)
    for t in np.arange(N):
        Y_cst_gamma[:,t] = qgev(pRW(X_star[:,t], phi_vec, 0.5), mu, tau, ksi)

    return(scipy.stats.genextreme.fit(Y.ravel()), 
           scipy.stats.genextreme.fit(Y_cst_gamma.ravel()))


some_seeds = scipy.stats.randint.rvs(low = 1, high = 200, size = 30)

result_gamma_bar = np.full((30, 3), fill_value=np.nan)
result_gamma_cst = np.full((30, 3), fill_value=np.nan)
for i in range(len(some_seeds)):
    print(i)
    seed = some_seeds[i]
    result_gamma_bar, result_gamma_cst = MLE(seed)
    result_gamma_bar[i,:] = result_gamma_bar
    result_gamma_cst[i,:] = result_gamma_cst

np.mean(result_gamma_bar, axis = 0)
np.mean(result_gamma_cst, axis = 0)

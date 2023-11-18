# This is a MCMC sampler that constantly gets updated
# Scratch work and modifications are done in this file
# Require:
#   - utilities.py
if __name__ == "__main__":
    # %%
    import sys
    data_seed = int(sys.argv[1]) if len(sys.argv) == 2 else 2345
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
    k = 10 # number of knots

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

    # putting two knots in the middle
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
    # range_at_knots = [0.3]*k
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
    K = ns_cov(range_vec = range_vec, sigsq_vec = sigsq_vec,
            coords = sites_xy, kappa = nu, cov_model = "matern")
    # K = np.identity(num_sites)
    Z = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(num_sites,)),cov=K,size=N).T
    W = norm_to_Pareto(Z) 

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


    # %%
    # ------- 6. Generate X and Y--------------------------------
    X_star = R_phi * W

    # Calculation of Y can(?) be parallelized by time(?)
    Y = np.full(shape=(num_sites, N), fill_value = np.nan)
    for t in np.arange(N):
        Y[:,t] = qgev(pRW(X_star[:,t], phi_vec, gamma), mu, tau, ksi)

    # %%
    # ------- 7. Other Preparational Stuff(?) --------------------------------

    gamma_vec = np.repeat(gamma, num_sites)

    # theo_quantiles = qRW(np.linspace(1e-2,1-1e-2,num=500), phi_vec, gamma)
    # plt.plot(sorted(X_star[:,0].ravel()), theo_quantiles)
    # plt.hist(pRW(X_star[:,0], phi_vec, gamma))

    # # R_at_knots should look levy (those are S)
    # for i in range(k):
    #     scipy.stats.probplot(R_at_knots[i,:], dist='levy', fit=False, plot=plt)
    #     plt.axline((0,0), slope = 1, color='black')
    #     plt.show()

    # R_at_knots**(-1/2) should look halfnormal(0, 1/sqrt(scale))
    # for i in range(k):
    #     scipy.stats.probplot((gamma**(1/4))*R_at_knots[i,:]**(-1/2), dist=scipy.stats.halfnorm, fit = False, plot=plt)
    #     plt.axline((0,0),slope=1,color='black')
    #     plt.show()

    # # log(W + 1) should look exponential (at each time t with num_site spatial points?)
    # for i in range(N):
    #     expo = np.log(W[:,i] + 1)
    #     scipy.stats.probplot(expo, dist="expon", fit = False, plot=plt)
    #     plt.axline((0,0), slope=1, color='black')
    #     plt.show()

    # # log(W + 1) should look exponential (at each site with N time replicates?)
    # for i in range(num_sites):
    #     expo = np.log(W[i,:] + 1)
    #     scipy.stats.probplot(expo, dist="expon", fit = False, plot=plt)
    #     plt.axline((0,0), slope=1, color='black')
    #     plt.show()

    # # pRW(X_star) should look uniform (at each time t?)
    # for i in range(N):
    #     # fig, ax = plt.subplots()
    #     unif = pRW(X_star[:,i], phi_vec, gamma)
    #     scipy.stats.probplot(unif, dist="uniform", fit = False, plot=plt)
    #     # plt.plot([0,1],[0,1], transform=ax.transAxes, color = 'black')
    #     plt.axline((0,0), slope=1, color='black')
    #     plt.show()

    # # pRW(X_star) should look uniform (at each site with N time replicates?)
    # for i in range(num_sites):
    #     # fig, ax = plt.subplots()
    #     unif = pRW(X_star[i,:], phi_vec[i], gamma)
    #     scipy.stats.probplot(unif, dist="uniform", fit = False, plot=plt)
    #     # plt.plot([0,1],[0,1], transform=ax.transAxes, color = 'black')
    #     plt.axline((0,0), slope=1, color='black')
    #     plt.show()

    # %%
    #####################################################################################################################
    # Metropolis Updates ################################################################################################
    #####################################################################################################################

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
    phi_post_cov = np.array([[ 5.92557062e-03, -1.46353751e-03,  5.50501725e-04,
        -2.74496569e-03, -7.73730891e-04, -1.22641230e-03,
         9.09912111e-05,  1.83705829e-04,  1.07871476e-03,
        -8.91819088e-04],
       [-1.46353751e-03,  7.70804511e-03, -2.18722439e-03,
        -3.04537076e-04, -7.85931283e-04, -1.06956251e-03,
         1.02458674e-03,  5.55836013e-04, -3.72922091e-04,
        -1.81606879e-04],
       [ 5.50501725e-04, -2.18722439e-03,  4.88285247e-03,
        -2.36760693e-04,  4.11916544e-04, -1.59585922e-04,
        -1.48391375e-03,  4.05034185e-04, -4.26611198e-05,
         3.18853919e-04],
       [-2.74496569e-03, -3.04537076e-04, -2.36760693e-04,
         7.65198692e-03, -2.67332192e-03,  4.82715909e-03,
        -1.37644459e-03, -1.53101514e-03, -6.96455983e-05,
         6.77379484e-04],
       [-7.73730891e-04, -7.85931283e-04,  4.11916544e-04,
        -2.67332192e-03,  6.74406560e-03, -2.25444613e-03,
        -9.52365588e-04,  5.87172996e-04, -1.27228048e-03,
         5.62821289e-04],
       [-1.22641230e-03, -1.06956251e-03, -1.59585922e-04,
         4.82715909e-03, -2.25444613e-03,  5.93959614e-03,
        -2.48980543e-03, -1.41287864e-03,  8.07183330e-05,
         8.27977455e-04],
       [ 9.09912111e-05,  1.02458674e-03, -1.48391375e-03,
        -1.37644459e-03, -9.52365588e-04, -2.48980543e-03,
         6.40163719e-03,  4.59755172e-04,  7.07642009e-04,
        -2.00765603e-03],
       [ 1.83705829e-04,  5.55836013e-04,  4.05034185e-04,
        -1.53101514e-03,  5.87172996e-04, -1.41287864e-03,
         4.59755172e-04,  2.96899735e-03, -1.27747005e-03,
         1.09434952e-05],
       [ 1.07871476e-03, -3.72922091e-04, -4.26611198e-05,
        -6.96455983e-05, -1.27228048e-03,  8.07183330e-05,
         7.07642009e-04, -1.27747005e-03,  4.23894124e-03,
        -1.83733931e-03],
       [-8.91819088e-04, -1.81606879e-04,  3.18853919e-04,
         6.77379484e-04,  5.62821289e-04,  8.27977455e-04,
        -2.00765603e-03,  1.09434952e-05, -1.83733931e-03,
         3.12776791e-03]])

    # phi_post_cov = 1e-3 * np.identity(k)

    assert k == phi_post_cov.shape[0]

    range_post_cov = np.array([[ 0.02401395, -0.0179877 ,  0.00505275, -0.02762382,  0.01157263,
         0.00235135, -0.0049559 ,  0.00873347, -0.0067072 ,  0.00085849],
       [-0.0179877 ,  0.0787402 , -0.02462011,  0.02626455, -0.03715657,
        -0.03051822,  0.03079424, -0.00182821,  0.01591707, -0.0101003 ],
       [ 0.00505275, -0.02462011,  0.03240835, -0.00802728,  0.0143812 ,
         0.0005288 , -0.01733352,  0.00114741, -0.00416739,  0.00681568],
       [-0.02762382,  0.02626455, -0.00802728,  0.09249391, -0.06397261,
         0.00803622,  0.00811489, -0.02521591,  0.02935126, -0.00787506],
       [ 0.01157263, -0.03715657,  0.0143812 , -0.06397261,  0.14020956,
        -0.04828856,  0.00769365, -0.00188575, -0.0225929 ,  0.01074224],
       [ 0.00235135, -0.03051822,  0.0005288 ,  0.00803622, -0.04828856,
         0.13711145, -0.08655356,  0.00890924, -0.02423586,  0.01607728],
       [-0.0049559 ,  0.03079424, -0.01733352,  0.00811489,  0.00769365,
        -0.08655356,  0.1029441 , -0.00639869,  0.02712238, -0.02813725],
       [ 0.00873347, -0.00182821,  0.00114741, -0.02521591, -0.00188575,
         0.00890924, -0.00639869,  0.03124207, -0.02381514,  0.00860954],
       [-0.0067072 ,  0.01591707, -0.00416739,  0.02935126, -0.0225929 ,
        -0.02423586,  0.02712238, -0.02381514,  0.05904663, -0.02597645],
       [ 0.00085849, -0.0101003 ,  0.00681568, -0.00787506,  0.01074224,
         0.01607728, -0.02813725,  0.00860954, -0.02597645,  0.03293283]])

    # range_post_cov = 1e-2 * np.identity(k)

    assert k == range_post_cov.shape[0]

    GEV_post_cov = np.array([[0.00066527, 0.00027579, 0],
                            [0.00027579, 0.00017949,  0],
                            [0         , 0         , 1e-4]])

    # GEV_post_cov = 1e-4 * np.identity(3)

    ########## Adaptive Update Initialization ############################################
    # Scalors for adaptive updates
    # (phi, range, GEV) these parameters are only proposed on worker 0
    if rank == 0: 
        sigma_m_sq = {}
        sigma_m_sq['phi_block1'] = (2.4**2)/3
        sigma_m_sq['phi_block2'] = (2.4**2)/4
        sigma_m_sq['phi_block3'] = (2.4**2)/3
        sigma_m_sq['range_block1'] = (2.4**2)/3
        sigma_m_sq['range_block2'] = (2.4**2)/4
        sigma_m_sq['range_block3'] = (2.4**2)/3
        sigma_m_sq['GEV'] = (2.4**2)/3

        # initialize them with posterior covariance matrix
        Sigma_0 = {}
        Sigma_0['phi_block1'] = phi_post_cov[0:3,0:3]
        Sigma_0['phi_block2'] = phi_post_cov[3:7,3:7]
        Sigma_0['phi_block3'] = phi_post_cov[7:10,7:10]
        Sigma_0['range_block1'] = range_post_cov[0:3,0:3]
        Sigma_0['range_block2'] = range_post_cov[3:7,3:7]
        Sigma_0['range_block3'] = range_post_cov[7:10,7:10]
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
    R_vec_current = wendland_weight_matrix @ np.exp(R_current_log)

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
    # will(?) be changed into matrix multiplication w/ more knots or Covariate:
    Loc_matrix_current = np.full(shape = (num_sites,N), fill_value = GEV_knots_current[0,0])
    Scale_matrix_current = np.full(shape = (num_sites,N), fill_value = GEV_knots_current[1,0])
    Shape_matrix_current = np.full(shape = (num_sites,N), fill_value = GEV_knots_current[2,0])

    ## ---- X_star ----
    X_star_1t_current = X_star[:,rank]

    ########## Loops ##################################################
    # %%
    # Metropolis Updates
    for iter in range(1, n_iters):
        ###################################
        # Printing and Drawings
        ###################################
        if rank == 0:
            if iter == 1:
                print(iter)
            if iter % 50 == 0:
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
                plt.title('traceplot for location')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('mu')
                plt.legend()
                plt.savefig('mu.pdf')
                plt.close()

                ## scale tau
                plt.subplots()
                plt.plot(xs_thin2, GEV_knots_trace_thin[:,1,0], label = 'knot 0') # scale
                plt.title('traceplot for scale')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('tau')
                plt.legend()
                plt.savefig('tau.pdf')
                plt.close()

                ## shape ksi
                plt.subplots()
                plt.plot(xs_thin2, GEV_knots_trace_thin[:,2,0], label = 'knot 0') # shape
                plt.title('traceplot for shape')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('ksi')
                plt.legend()
                plt.savefig('ksi.pdf')
                plt.close()

                # log-likelihood
                plt.subplots()
                plt.plot(xs_thin2, loglik_trace_thin)
                plt.title('traceplot for log-likelihood')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('loglikelihood')
                plt.savefig('loglik.pdf')
                plt.close()

                # log-likelihood in details
                plt.subplots()
                for i in range(5):
                    plt.plot(xs_thin2, loglik_detail_trace_thin[:,i],label = i)
                plt.title('traceplot for detail log likelihood')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('log likelihood')
                plt.legend()
                plt.savefig('loglik_detail.pdf')
                plt.close()
        
        comm.Barrier() # block for drawing

        ###################################
        # Adaptive Update autotunings
        ###################################
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
                Sigma_0_hat = np.cov(np.array([phi_knots_trace[iter-25:iter,i].ravel() for i in range(3,7)]))
                log_sigma_m_sq_hat = np.log(sigma_m_sq['phi_block2']) + gamma2*(r_hat - r_opt)
                sigma_m_sq['phi_block2'] = np.exp(log_sigma_m_sq_hat)
                Sigma_0['phi_block2'] = Sigma_0['phi_block2'] + gamma1*(Sigma_0_hat - Sigma_0['phi_block2'])
                ## phi_block3
                Sigma_0_hat = np.cov(np.array([phi_knots_trace[iter-25:iter,i].ravel() for i in range(7,10)]))
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
                Sigma_0_hat = np.cov(np.array([range_knots_trace[iter-25:iter,i].ravel() for i in range(3,7)]))
                log_sigma_m_sq_hat = np.log(sigma_m_sq['range_block2']) + gamma2*(r_hat - r_opt)
                sigma_m_sq['range_block2'] = np.exp(log_sigma_m_sq_hat)
                Sigma_0['range_block2'] = Sigma_0['range_block2'] + gamma1*(Sigma_0_hat - Sigma_0['range_block2'])
                ## range_block3
                Sigma_0_hat = np.cov(np.array([range_knots_trace[iter-25:iter,i].ravel() for i in range(7,10)]))
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
        
        comm.Barrier() # block for adaptive update

    #####################################################################################################################
    # Actual Param Update ###############################################################################################
    #####################################################################################################################

    #### ----- Update Rt ----- Parallelized Across N time
        # if rank == 0:
        #     print('Updating R')
        # Propose a R at time "rank", on log-scale

        # Propose a R using adaptive update
        R_proposal_log = np.sqrt(sigma_m_sq_Rt)*random_generator.normal(loc = 0.0, scale = 1.0, size = k) + R_current_log
        # R_proposal_log = np.sqrt(sigma_m_sq_Rt)*np.repeat(random_generator.normal(loc = 0.0, scale = 1.0, size = 1), k) + R_current_log # spatially cst R(t)

        # Conditional Likelihood at Current
        R_vec_current = wendland_weight_matrix @ np.exp(R_current_log)

        # log-likelihood:
        lik = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
                                                        Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank], 
                                                        phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
        # log-prior density
        prior = np.sum(scipy.stats.levy.logpdf(np.exp(R_current_log), scale = gamma) + R_current_log)
        # prior = prior/k # if R(t) is spatially constant

        # Conditional Likelihood at Proposal
        R_vec_proposal = wendland_weight_matrix @ np.exp(R_proposal_log)
        # if np.any(~np.isfinite(R_vec_proposal**phi_vec_current)): print("Negative or zero R, iter=", iter, ", rank=", rank, R_vec_proposal[0], phi_vec_current[0])
        lik_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
                                                        Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank], 
                                                        phi_vec_current, gamma_vec, R_vec_proposal, cholesky_matrix_current)
        prior_proposal = np.sum(scipy.stats.levy.logpdf(np.exp(R_proposal_log), scale = gamma) + R_proposal_log)
        # prior_proposal = prior_proposal/k # if R(t) is spatially constant

        # Accept or Reject
        u = random_generator.uniform()
        ratio = np.exp(lik_proposal + prior_proposal - lik - prior)
        if u > ratio: # Reject
            R_update_log = R_current_log
        else: # Accept, u <= ratio
            R_update_log = R_proposal_log
            num_accepted_Rt += 1
        
        R_current_log = R_update_log
        R_vec_current = wendland_weight_matrix @ np.exp(R_current_log)
        
        # Gather across N_t, store into trace matrix
        R_current_log_gathered = comm.gather(R_current_log, root=0)
        
        if rank == 0:
            R_trace_log[iter,:,:] = np.vstack(R_current_log_gathered).T

        comm.Barrier() # block for R_t updates

    #### ----- Update phi ----- parallelized likelihood calculation across N time
        # if rank == 0:
        #     print('Updating phi')
        # Propose new phi at the knots --> new phi vector
        if rank == 0:
            random_walk_block1 = np.sqrt(sigma_m_sq['phi_block1'])*random_generator.multivariate_normal(np.zeros(3), Sigma_0['phi_block1'])
            random_walk_block2 = np.sqrt(sigma_m_sq['phi_block2'])*random_generator.multivariate_normal(np.zeros(4), Sigma_0['phi_block2'])
            random_walk_block3 = np.sqrt(sigma_m_sq['phi_block3'])*random_generator.multivariate_normal(np.zeros(3), Sigma_0['phi_block3'])        
            random_walk_perturb = np.hstack((random_walk_block1,random_walk_block2,random_walk_block3))
            # random_walk_perturb = np.repeat(random_walk_perturb[0], k) # keep phi spatially constant
            phi_knots_proposal = phi_knots_current + random_walk_perturb
        else:
            phi_knots_proposal = None
        phi_knots_proposal = comm.bcast(phi_knots_proposal, root = 0)

        phi_vec_proposal = gaussian_weight_matrix @ phi_knots_proposal

        # Conditional Likelihood at Current
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

        # Accept or Reject
        if rank == 0:
            phi_accepted = False

            # use Beta(5,5) prior on each one of the k range parameters
            lik = sum(lik_gathered) + np.sum(scipy.stats.beta.logpdf(phi_knots_current, a = 5, b = 5))
            lik_proposal = sum(lik_proposal_gathered) + np.sum(scipy.stats.beta.logpdf(phi_knots_proposal, a = 5, b = 5))

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
            # X_star_1t_current = qRW(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
            #                                 phi_vec_current, gamma)
            X_star_1t_current = X_star_1t_proposal

        comm.Barrier() # block for phi updates

    #### ----- Update range_vec ----- parallelized likelihood calculation across N time
        # if rank == 0:
        #     print('Updating range')
        # Propose new range at the knots --> new range vector
        if rank == 0:
            random_walk_block1 = np.sqrt(sigma_m_sq['range_block1'])*random_generator.multivariate_normal(np.zeros(3), Sigma_0['range_block1'])
            random_walk_block2 = np.sqrt(sigma_m_sq['range_block2'])*random_generator.multivariate_normal(np.zeros(4), Sigma_0['range_block2'])
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

            # use Half-Normal Prior on each one of the k range parameters
            lik = sum(lik_gathered) + np.sum(scipy.stats.halfnorm.logpdf(range_knots_current, loc = 0, scale = 2))
            lik_proposal = sum(lik_proposal_gathered) + np.sum(scipy.stats.halfnorm.logpdf(range_knots_proposal, loc = 0, scale = 2))

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
            # K_current = ns_cov(range_vec = range_vec_current,
            #                     sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = "matern")
            # cholesky_matrix_current = scipy.linalg.cholesky(K_current, lower = False)
            K_current = K_proposal
            cholesky_matrix_current = cholesky_matrix_proposal

        comm.Barrier() # block for range updates

    #### ----- Update GEV mu tau ksi (location, scale, shape) together ----
    #### ----- Do not update ksi -----
        # if rank == 0:
        #     print('Updating GEV')
        # Propose new GEV params at the knots --> new GEV params vector
        if rank == 0:
            # random_walk = random_generator.multivariate_normal(np.zeros(3), GEV_post_cov, size = k).T
            random_walk = np.sqrt(sigma_m_sq['GEV'])*random_generator.multivariate_normal(np.zeros(3), Sigma_0['GEV'], size = k).T
            GEV_knots_proposal = GEV_knots_current + random_walk
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
        
        # When fixing the GEV parameters
        # X_star_1t_proposal = np.NINF
        # lik_1t_proposal = np.NINF 

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

            prior_mu = scipy.stats.norm.logpdf(Loc_matrix_current[0][0])
            prior_mu_proposal = scipy.stats.norm.logpdf(Loc_matrix_current[0][0])
            

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
            # X_star_1t_current = qRW(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
            #                                 phi_vec_current, gamma)
            X_star_1t_current = X_star_1t_proposal
        
        comm.Barrier() # block for GEV updates

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

        comm.Barrier() # block for one iteration of update

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


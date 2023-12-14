"""
# This is a MCMC sampler that constantly gets updated
# Scratch work and modifications are done in this file

# Dec 6 2023, adding covariate for theta GEVs to the model
"""
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
    from time import strftime, localtime
    from utilities import *

    # %%
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #####################################################################################################################
    # Generating Dataset ################################################################################################
    #####################################################################################################################
    # %%
    # ------- 0. Simulation Setting --------------------------------------

    ## space setting
    np.random.seed(data_seed)
    Nt = 32 # number of time replicates
    Ns = 25 # number of sites/stations
    k = 9 # number of knots

    ## unchanged constants or parameters
    gamma = 0.5 # this is the gamma that goes in rlevy, gamma_at_knots
    delta = 0.0 # this is the delta in levy, stays 0
    mu = 0.0 # GEV location
    sigma = 1.0 # GEV scale
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

    sites_xy = np.random.random((Ns, 2)) * 10
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

    # %%
    # ------- 2. Generate the weight matrices ------------------------------------

    # Weight matrix generated using Gaussian Smoothing Kernel
    bandwidth = 4 # ?what is bandwidth?
    gaussian_weight_matrix = np.full(shape = (Ns, k), fill_value = np.nan)
    for site_id in np.arange(Ns):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                        XB = knots_xy)
        # influence coming from each of the knots
        weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
        gaussian_weight_matrix[site_id, :] = weight_from_knots

    # Weight matrix generated using wendland basis
    wendland_weight_matrix = np.full(shape = (Ns,k), fill_value = np.nan)
    for site_id in np.arange(Ns):
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
    
    constant_weight_matrix = np.full(shape = (Ns, k), fill_value = np.nan)
    for site_id in np.arange(Ns):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                        XB = knots_xy)
        # influence coming from each of the knots
        weight_from_knots = np.repeat(1, k)/k
        constant_weight_matrix[site_id, :] = weight_from_knots


    # %%
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

    ## Covariance matrix K
    ## sigsq_vec
    sigsq_vec = np.repeat(sigsq, Ns) # hold at 1
    K = ns_cov(range_vec = range_vec, sigsq_vec = sigsq_vec,
            coords = sites_xy, kappa = nu, cov_model = "matern")
    # K = np.identity(Ns)
    Z = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(Ns,)),cov=K,size=Nt).T
    W = norm_to_Pareto(Z) 

    # %%
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
    R_at_knots = np.full(shape = (k, Nt), fill_value = np.nan)
    for t in np.arange(Nt):
        R_at_knots[:,t] = rlevy(n = k, m = delta, s = gamma) # generate R at time t, spatially varying k knots
        # should need to vectorize rlevy so in future s = gamma_at_knots (k,) vector
        # R_at_knots[:,t] = scipy.stats.levy.rvs(delta, gamma, k)
        # R_at_knots[:,t] = np.repeat(rlevy(n = 1, m = delta, s = gamma), k) # generate R at time t, spatially constant k knots

    ## Matrix Multiply to the sites
    R_at_sites = wendland_weight_matrix @ R_at_knots
    # R_at_sites = constant_weight_matrix @ R_at_knots

    ## R^phi
    R_phi = np.full(shape = (Ns, Nt), fill_value = np.nan)
    for t in np.arange(Nt):
        R_phi[:,t] = np.power(R_at_sites[:,t], phi_vec)

    # %%
    # ------- 5. Generate mu(s) = C(s)Beta ----------------------

    #######
    # Loc #
    #######
    # C_mu(j,s,t) is the mu covariate at 
    # beta_j, site s and time t
    # simple case of mu(s) = x(s) + y(s)
    ## coefficients
    Beta_mu_0 = 0.0 # intercept for mu
    Beta_mu_1 = 1.0 # slope of beta 1 for mu, beta_1*X
    Beta_mu_2 = 1.0 # slope of beta 2 for mu, beta_2*X
    Beta_mu = np.array([Beta_mu_0, Beta_mu_1, Beta_mu_2])
    Beta_mu_m = len(Beta_mu) # number of betas for mu
    ## covariates, the "design matrix"
    C_mu = np.full(shape=(Beta_mu_m, Ns, Nt), fill_value = np.nan) # design matrix
    C_mu[0, :, :] = 1.0 # column of 1 for the intercept
    C_mu[1, :, :] = np.tile(sites_x, reps = (Nt, 1)).T # mu(s,t) = 0
    C_mu[2, :, :] = np.tile(sites_y, reps = (Nt, 1)).T
    ## actual surface for mu(s)
    # mu_matrix = np.full(shape=(Ns, Nt), fill_value = np.nan)
    # for t in range(Nt):
    #     mu_matrix[:,t] = C_mu.T[t,:,:] @ Beta_mu
    mu_matrix = (C_mu.T @ Beta_mu).T
    ## heatplot of mu surface
    C_mu_plot = np.full(shape = (Beta_mu_m, len(plotgrid_xy), Nt), fill_value = np.nan) # 
    C_mu_plot[0, :, :] = 1.0
    C_mu_plot[1, :, :] = np.tile(plotgrid_xy[:,0], reps=(Nt,1)).T
    C_mu_plot[2, :, :] = np.tile(plotgrid_xy[:,1], reps=(Nt,1)).T
    mu_surface_plot = (C_mu_plot.T @ Beta_mu).T
    graph, ax = plt.subplots()
    heatmap = ax.imshow(mu_surface_plot[:,0].reshape(25,25), cmap='hot',interpolation='nearest',extent=[0,10,10,0])
    ax.invert_yaxis()
    graph.colorbar(heatmap)
    plt.show()
    plt.close()
    ## 3d plot of mu surface
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_surface(plotgrid_X, plotgrid_Y, np.matrix(mu_surface_plot[:,0]).reshape(25,25))
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('mu(s)')

    #########
    # Scale #
    #########
    # C_sigma(j,s,t) is the scale covariates at beta_j, site s and time t
    # constant case of sigma(s) = sigma
    ## coefficients
    Beta_sigma_0 = sigma
    Beta_sigma = np.array([Beta_sigma_0])
    # Beta_sigma_1 = 0
    # Beta_sigma = np.array([Beta_sigma_0, Beta_sigma_1])
    Beta_sigma_m = len(Beta_sigma)
    ## covariates, the design matrix
    C_sigma = np.full(shape = (Beta_sigma_m, Ns, Nt), fill_value = np.nan)
    C_sigma[0,:,:] = 1.0
    # C_sigma[1,:,:] = np.tile(sites_x, reps = (Nt, 1)).T
    # C_sigma[2,:,:] = np.tile(sites_y, reps = (Nt, 1)).T
    ## actual surface for sigma(s)
    sigma_matrix = (C_sigma.T @ Beta_sigma).T
    ## heatplot of sigma(s) surface
    C_sigma_plot = np.full(shape = (Beta_sigma_m, len(plotgrid_xy), Nt), fill_value = np.nan) # 
    C_sigma_plot[0, :, :] = sigma
    # C_sigma_plot[1, :, :] = np.tile(plotgrid_xy[:,0], reps=(Nt,1)).T
    sigma_surface_plot = (C_sigma_plot.T @ Beta_sigma).T
    graph, ax = plt.subplots()
    heatmap = ax.imshow(sigma_surface_plot[:,0].reshape(25,25), cmap='hot',interpolation='nearest',extent=[0,10,10,0])
    ax.invert_yaxis()
    graph.colorbar(heatmap)
    plt.show()
    plt.close()

    #########
    # Shape #
    #########
    # C_ksi(j,s,t) is the shape covariates at beta_j, site s and time t
    # constant case of ksi(s) = ksi
    ## coefficients
    Beta_ksi_0 = ksi
    Beta_ksi = np.array([Beta_ksi_0])
    # Beta_ksi_1 = 0
    # Beta_ksi = np.array([Beta_ksi_0, Beta_ksi_1])
    Beta_ksi_m = len(Beta_ksi)
    ## covariates, the design matrix
    C_ksi = np.full(shape = (Beta_ksi_m, Ns, Nt), fill_value = np.nan)
    C_ksi[0,:,:] = 1.0
    # C_ksi[1,:,:] = np.tile(sites_x, reps = (Nt, 1)).T
    ## actual surface for ksi(s)
    ksi_matrix = (C_ksi.T @ Beta_ksi).T
    ## heatplot of ksi(s) surface
    C_ksi_plot = np.full(shape = (Beta_ksi_m, len(plotgrid_xy), Nt), fill_value = np.nan) # 
    C_ksi_plot[0, :, :] = sigma
    # C_ksi_plot[1, :, :] = np.tile(plotgrid_xy[:,0], reps=(Nt,1)).T
    ksi_surface_plot = (C_ksi_plot.T @ Beta_ksi).T
    graph, ax = plt.subplots()
    heatmap = ax.imshow(ksi_surface_plot[:,0].reshape(25,25), cmap='hot',interpolation='nearest',extent=[0,10,10,0])
    ax.invert_yaxis()
    graph.colorbar(heatmap)
    plt.show()
    plt.close()

    # %%
    # ------- 5.5 Generate hyperparameter, sigma_Beta_GEV ----------------------
    sigma_Beta_mu = 1.0
    sigma_Beta_sigma = 1.0
    sigma_Beta_ksi = 1.0

    # %%
    # ------- 6. Generate X and Y--------------------------------
    X_star = R_phi * W

    alpha = 0.5
    gamma_at_knots = np.repeat(gamma, k)
    gamma_vec = np.sum(np.multiply(wendland_weight_matrix, gamma_at_knots)**(alpha), 
                       axis = 1)**(1/alpha) # axis = 1 to sum over K knots
    # gamma_vec is the gamma bar in the overleaf document

    # Calculation of Y can(?) be parallelized by time(?)
    Y = np.full(shape=(Ns, Nt), fill_value = np.nan)
    for t in np.arange(Nt):
        # Y[:,t] = qgev(pRW(X_star[:,t], phi_vec, gamma_vec), mu, sigma, ksi)
        Y[:,t] = qgev(pRW(X_star[:,t], phi_vec, gamma_vec), mu_matrix[:,t], sigma_matrix[:,t], ksi_matrix[:,t])

    # %%
    # ------- 7. Other Preparational Stuff(?) --------------------------------

    # theo_quantiles = qRW(np.linspace(1e-2,1-1e-2,num=500), phi_vec, gamma_vec)
    # plt.plot(sorted(X_star[:,0].ravel()), theo_quantiles)
    # plt.hist(pRW(X_star[:,0], phi_vec, gamma_vec))

    # checking stable variables S

    # # levy.cdf(R_at_knots, loc = 0, scale = gamma) should look uniform
    # for i in range(k):
    #     scipy.stats.probplot(scipy.stats.levy.cdf(R_at_knots[i,:], scale=gamma), dist='uniform', fit=False, plot=plt)
    #     plt.axline((0,0), slope = 1, color = 'black')
    #     plt.show()

    # R_at_knots**(-1/2) should look halfnormal(0, 1/sqrt(scale))
    # for i in range(k):
    #     scipy.stats.probplot((gamma**(1/2))*R_at_knots[i,:]**(-1/2), dist=scipy.stats.halfnorm, fit = False, plot=plt)
    #     plt.axline((0,0),slope=1,color='black')
    #     plt.show()

    # checking Pareto distribution

    # # shifted pareto.cdf(W[i,:] + 1, b = 1, loc = 0, scale = 1) shoud look uniform
    # for i in range(Ns):
    #     scipy.stats.probplot(scipy.stats.pareto.cdf(W[i,:]+1, b = 1, loc = 0, scale = 1), dist='uniform', fit=False, plot=plt)
    #     plt.axline((0,0), slope = 1, color = 'black')
    #     plt.show()

    # # standard pareto.cdf(W[i,:], b = 1, loc = 0, scale = 1) shoud look uniform
    # for i in range(Ns):
    #     scipy.stats.probplot(scipy.stats.pareto.cdf(W[i,:], b = 1, loc = 0, scale = 1), dist='uniform', fit=False, plot=plt)
    #     plt.axline((0,0), slope = 1, color = 'black')
    #     plt.show()

    # # log(W + 1) should look exponential (at each time t with num_site spatial points?)
    # for i in range(Nt):
    #     expo = np.log(W[:,i] + 1)
    #     scipy.stats.probplot(expo, dist="expon", fit = False, plot=plt)
    #     plt.axline((0,0), slope=1, color='black')
    #     plt.show()

    # # log(W + 1) should look exponential (at each site with Nt time replicates?)
    # # for shifted Pareto
    # for i in range(Ns):
    #     expo = np.log(W[i,:] + 1)
    #     scipy.stats.probplot(expo, dist="expon", fit = False, plot=plt)
    #     plt.axline((0,0), slope=1, color='black')
    #     plt.show()

    # # log(W) should look exponential (at each site with Nt time replicates?)
    # # for standard Pareto
    # for i in range(Ns):
    #     expo = np.log(W[i,:])
    #     scipy.stats.probplot(expo, dist="expon", fit = False, plot=plt)
    #     plt.axline((0,0), slope=1, color='black')
    #     plt.show()

    # checking model cdf

    # # pRW(X_star) should look uniform (at each time t?)
    # for i in range(Nt):
    #     # fig, ax = plt.subplots()
    #     unif = pRW(X_star[:,i], phi_vec, gamma_vec[i])
    #     scipy.stats.probplot(unif, dist="uniform", fit = False, plot=plt)
    #     # plt.plot([0,1],[0,1], transform=ax.transAxes, color = 'black')
    #     plt.axline((0,0), slope=1, color='black')
    #     plt.show()

    # # pRW(X_star) should look uniform (at each site with Nt time replicates?)
    # for i in range(Ns):
    #     # fig, ax = plt.subplots()
    #     unif = pRW(X_star[i,:], phi_vec[i], gamma_vec[i])
    #     scipy.stats.probplot(unif, dist="uniform", fit = False, plot=plt)
    #     # plt.plot([0,1],[0,1], transform=ax.transAxes, color = 'black')
    #     plt.axline((0,0), slope=1, color='black')
    #     plt.show()

    # unifs = scipy.stats.uniform.rvs(0,1,size=10000)
    # Y_from_unifs = qgev(unifs, 0, 1, 0.2)
    # scipy.stats.genextreme.fit(Y_from_unifs) # this is unbiased

    # a = np.flip(sorted(X_star.ravel())) # check a from Jupyter variables

    # myfits = [scipy.stats.genextreme.fit(Y[site,:]) for site in range(500)]
    # plt.hist([fit[1] for fit in myfits]) # loc
    # plt.hist([fit[2] for fit in myfits]) # scale
    # plt.hist([fit[0] for fit in myfits]) # -shape

    # %%
    #####################################################################################################################
    # Metropolis Updates ################################################################################################
    #####################################################################################################################

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
    phi_post_cov = np.array([
       [ 1.71595567e-03, -1.62351108e-03,  5.40782727e-04,
        -7.39783709e-04,  5.18647363e-04, -3.04089297e-04,
        -5.71744286e-05,  3.09075985e-04,  4.29528231e-06],
       [-1.62351108e-03,  3.83498399e-03, -1.64905040e-03,
         2.81541085e-06, -1.48305211e-03,  7.70876687e-04,
         5.05809724e-04, -2.42279339e-04,  5.47733425e-05],
       [ 5.40782727e-04, -1.64905040e-03,  2.42768982e-03,
         7.89354829e-05,  3.38706927e-04, -1.33417236e-03,
        -5.88460771e-06, -4.15771322e-05,  3.26340045e-04],
       [-7.39783709e-04,  2.81541085e-06,  7.89354829e-05,
         3.10731257e-03, -1.33483891e-03,  3.93067423e-04,
        -1.40512231e-03,  3.86608462e-04,  8.15222055e-05],
       [ 5.18647363e-04, -1.48305211e-03,  3.38706927e-04,
        -1.33483891e-03,  5.82846826e-03, -2.28460694e-03,
         1.89505396e-04, -1.45725699e-03,  2.19050158e-04],
       [-3.04089297e-04,  7.70876687e-04, -1.33417236e-03,
         3.93067423e-04, -2.28460694e-03,  3.15293790e-03,
        -4.05295100e-05,  3.98273559e-04, -8.95240062e-04],
       [-5.71744286e-05,  5.05809724e-04, -5.88460771e-06,
        -1.40512231e-03,  1.89505396e-04, -4.05295100e-05,
         1.88765845e-03, -1.29365986e-03,  2.86677573e-04],
       [ 3.09075985e-04, -2.42279339e-04, -4.15771322e-05,
         3.86608462e-04, -1.45725699e-03,  3.98273559e-04,
        -1.29365986e-03,  3.79140159e-03, -1.17335363e-03],
       [ 4.29528231e-06,  5.47733425e-05,  3.26340045e-04,
         8.15222055e-05,  2.19050158e-04, -8.95240062e-04,
         2.86677573e-04, -1.17335363e-03,  1.74786663e-03]])

    # phi_post_cov = 1e-3 * np.identity(k)

    assert k == phi_post_cov.shape[0]

    range_post_cov = np.array([
       [ 0.00888606, -0.00964968,  0.00331823, -0.01147588,  0.01378476,
        -0.00456044,  0.00455141, -0.00561015,  0.0020646 ],
       [-0.00964968,  0.02704678, -0.01138214,  0.01338328, -0.04013097,
         0.01380413, -0.00591529,  0.01721602, -0.00600377],
       [ 0.00331823, -0.01138214,  0.01723129, -0.0043743 ,  0.01134919,
        -0.01592546,  0.00158623, -0.00530012,  0.00580562],
       [-0.01147588,  0.01338328, -0.0043743 ,  0.03540402, -0.04741295,
         0.01675298, -0.01613912,  0.02149959, -0.00803375],
       [ 0.01378476, -0.04013097,  0.01134919, -0.04741295,  0.14918746,
        -0.05188579,  0.02373275, -0.06965559,  0.0241972 ],
       [-0.00456044,  0.01380413, -0.01592546,  0.01675298, -0.05188579,
         0.04733445, -0.00731039,  0.02407662, -0.01946985],
       [ 0.00455141, -0.00591529,  0.00158623, -0.01613912,  0.02373275,
        -0.00731039,  0.01686881, -0.02343455,  0.00816378],
       [-0.00561015,  0.01721602, -0.00530012,  0.02149959, -0.06965559,
         0.02407662, -0.02343455,  0.06691174, -0.02429487],
       [ 0.0020646 , -0.00600377,  0.00580562, -0.00803375,  0.0241972 ,
        -0.01946985,  0.00816378, -0.02429487,  0.01848764]])

    # range_post_cov = 1e-2 * np.identity(k)

    assert k == range_post_cov.shape[0]

    # GEV_post_cov = np.array([[2.88511464e-04, 1.13560517e-04, 0],
    #                         [1.13560517e-04, 6.40933053e-05,  0],
    #                         [0         , 0         , 1e-4]])

    # GEV_post_cov = 1e-4 * np.identity(3)

    ########## Adaptive Update Initialization ############################################
    # Scalors for adaptive updates
    # (phi, range, GEV) these parameters are only proposed on worker 0
    if rank == 0: 
        sigma_m_sq = {
            'phi_block1'    : (2.4**2)/3,
            'phi_block2'    : (2.4**2)/3,
            'phi_block3'    : (2.4**2)/3,
            'range_block1'  : (2.4**2)/3,
            'range_block2'  : (2.4**2)/3,
            'range_block3'  : (2.4**2)/3,
            'Beta_mu'       : 0.5 * 1e-4,
            'Beta_sigma'    : 0.2 * 1e-4,
            'Beta_ksi'      : 0.15 * 1e-4
        }
        # sigma_m_sq['phi_block1'] = (2.4**2)/3
        # sigma_m_sq['phi_block2'] = (2.4**2)/3
        # sigma_m_sq['phi_block3'] = (2.4**2)/3
        # sigma_m_sq['range_block1'] = (2.4**2)/3
        # sigma_m_sq['range_block2'] = (2.4**2)/3
        # sigma_m_sq['range_block3'] = (2.4**2)/3
        # sigma_m_sq['GEV'] = (2.4**2)/3
        # sigma_m_sq['Beta_mu'] = 0.5
        # sigma_m_sq['Beta_sigma'] = 0.2
        # sigma_m_sq['Beta_ksi'] = 0.15

        # initialize them with posterior covariance matrix
        Sigma_0 = {
            'phi_block1'    : phi_post_cov[0:3,0:3],
            'phi_block2'    : phi_post_cov[3:6,3:6],
            'phi_block3'    : phi_post_cov[6:9,6:9],
            'range_block1'  : range_post_cov[0:3,0:3],
            'range_block2'  : range_post_cov[3:6,3:6],
            'range_block3'  : range_post_cov[6:9,6:9]
        }
        # Sigma_0['phi_block1'] = phi_post_cov[0:3,0:3]
        # Sigma_0['phi_block2'] = phi_post_cov[3:6,3:6]
        # Sigma_0['phi_block3'] = phi_post_cov[6:9,6:9]
        # Sigma_0['range_block1'] = range_post_cov[0:3,0:3]
        # Sigma_0['range_block2'] = range_post_cov[3:6,3:6]
        # Sigma_0['range_block3'] = range_post_cov[6:9,6:9]
        # Sigma_0['GEV'] = GEV_post_cov

        num_accepted = {
            'phi'         : 0,
            'range'       : 0,
            # 'GEV'         : 0,
            'Beta_mu'     : 0,
            'Beta_sigma'  : 0,
            'Beta_ksi'    : 0
        }
        # num_accepted['phi'] = 0
        # num_accepted['range'] = 0
        # # num_accepted['GEV'] = 0
        # num_accepted['Beta_mu'] = 0
        # num_accepted['Beta_sigma'] = 0
        # num_accepted['Beta_ksi'] = 0

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

    ## ---- R, log scaled, at the knots ----
    if rank == 0:
        R_trace_log = np.full(shape = (n_iters, k, Nt), fill_value = np.nan) # [n_iters, num_knots, n_t]
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

    ## ---- GEV mu sigma ksi (location, scale, shape) together ----
    # if rank == 0:
    #     GEV_knots_trace = np.full(shape=(n_iters, 3, k), fill_value = np.nan) # [n_iters, n_GEV, num_knots]
    #     GEV_knots_trace[0,:,:] = np.tile(np.array([mu, sigma, ksi]), (k,1)).T
    #     GEV_knots_init = GEV_knots_trace[0,:,:]
    # else:
    #     GEV_knots_init = None
    # GEV_knots_init = comm.bcast(GEV_knots_init, root = 0)

    ## ---- GEV covariate coefficients ----
    if rank == 0:
        Beta_mu_trace = np.full(shape=(n_iters, Beta_mu_m), fill_value = np.nan)
        Beta_mu_trace[0,:] = Beta_mu
        Beta_mu_init = Beta_mu_trace[0,:]

        Beta_sigma_trace = np.full(shape=(n_iters, Beta_sigma_m), fill_value = np.nan)
        Beta_sigma_trace[0,:] = Beta_sigma
        Beta_sigma_init = Beta_sigma_trace[0,:]

        Beta_ksi_trace = np.full(shape=(n_iters, Beta_ksi_m), fill_value = np.nan)
        Beta_ksi_trace[0,:] = Beta_ksi
        Beta_ksi_init = Beta_ksi_trace[0,:]
    else:
        Beta_mu_init = None
        Beta_sigma_init = None
        Beta_ksi_init = None
    Beta_mu_init = comm.bcast(Beta_mu_init, root = 0)
    Beta_sigma_init = comm.bcast(Beta_sigma_init, root = 0)
    Beta_ksi_init = comm.bcast(Beta_ksi_init, root = 0)

    ## ---- GEV covariate coefficient prior variance ----
    if rank == 0:
        sigma_Beta_mu_trace      = np.full(shape=(n_iters, 1), fill_value = np.nan)
        sigma_Beta_mu_trace[0,:] = sigma_Beta_mu
        sigma_Beta_mu_init       = sigma_Beta_mu_trace[0,:]

        sigma_Beta_sigma_trace      = np.full(shape=(n_iters, 1), fill_value = np.nan)
        sigma_Beta_sigma_trace[0,:] = sigma_Beta_sigma
        sigma_Beta_sigma_init       = sigma_Beta_sigma_trace[0,:]

        sigma_Beta_ksi_trace      = np.full(shape=(n_iters, 1), fill_value = np.nan)
        sigma_Beta_ksi_trace[0,:] = sigma_Beta_ksi
        sigma_Beta_ksi_init       = sigma_Beta_ksi_trace[0,:]
    else:
        sigma_Beta_mu_init     = None
        sigma_Beta_sigma_init  = None
        sigma_Beta_ksi_init    = None
    sigma_Beta_mu_init    = comm.bcast(sigma_Beta_mu_init, root = 0)
    sigma_Beta_sigma_init = comm.bcast(sigma_Beta_sigma_init, root = 0)
    sigma_Beta_ksi_init   = comm.bcast(sigma_Beta_ksi_init, root = 0)

    ########## Initialize ##################################################
    # %%
    # Initialize

    ## ---- X_star ----
    X_star_1t_current = X_star[:,rank]

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

    ## ---- GEV covariate coefficients ----
    Beta_mu_current     = Beta_mu_init
    Beta_sigma_current  = Beta_sigma_init
    Beta_ksi_current    = Beta_ksi_init
    Loc_matrix_current   = (C_mu.T @ Beta_mu_current).T
    Scale_matrix_current = (C_sigma.T @ Beta_sigma_current).T
    Shape_matrix_current = (C_ksi.T @ Beta_ksi_current).T

    ## ---- GEV covariate coefficients prior variance ----
    sigma_Beta_mu_current    = sigma_Beta_mu_init
    sigma_Beta_sigma_current = sigma_Beta_sigma_init
    sigma_Beta_ksi_current   = sigma_Beta_ksi_init

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
                # np.save('GEV_knots_trace', GEV_knots_trace)
                np.save('Beta_mu_trace', Beta_mu_trace)
                np.save('Beta_sigma_trace', Beta_sigma_trace)
                np.save('Beta_ksi_trace', Beta_ksi_trace)
                np.save('loglik_trace', loglik_trace)
                np.save('loglik_detail_trace', loglik_detail_trace)

                # Print traceplot every 1000 iterations
                xs = np.arange(iter)
                xs_thin = xs[0::10] # index 1, 11, 21, ...
                xs_thin2 = np.arange(len(xs_thin)) # numbers 1, 2, 3, ...
                R_trace_log_thin = R_trace_log[0:iter:10,:,:]
                phi_knots_trace_thin = phi_knots_trace[0:iter:10,:]
                range_knots_trace_thin = range_knots_trace[0:iter:10,:]
                # GEV_knots_trace_thin = GEV_knots_trace[0:iter:10,:,:]
                Beta_mu_trace_thin = Beta_mu_trace[0:iter:10,:]
                Beta_sigma_trace_thin = Beta_sigma_trace[0:iter:10,:]
                Beta_ksi_trace_thin = Beta_ksi_trace[0:iter:10,:]
                loglik_trace_thin = loglik_trace[0:iter:10,:]
                loglik_detail_trace_thin = loglik_detail_trace[0:iter:10,:]

                # ---- phi ----
                plt.subplots()
                for i in range(k):
                    plt.plot(xs_thin2, phi_knots_trace_thin[:,i], label='knot ' + str(i))
                    # plt.plot(xs_thin2, phi_knots_trace_thin[:,1], label='knot ' + i)
                    plt.annotate('knot ' + str(i), xy=(xs_thin2[-1], phi_knots_trace_thin[:,i][-1]))
                plt.title('traceplot for phi')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('phi')
                plt.legend()
                plt.savefig('phi.pdf')
                plt.close()

                # ---- R_t ----
                plt.subplots()
                # plt.plot(xs_thin2, R_trace_log_thin[:,0,0], label='knot 0 time 0')
                # plt.plot(xs_thin2, R_trace_log_thin[:,0,1], label='knot 0 time 1')
                # plt.plot(xs_thin2, R_trace_log_thin[:,0,2], label='knot 0 time 2')
                # plt.plot(xs_thin2, R_trace_log_thin[:,1,0], label='knot 1 time 0')
                # plt.plot(xs_thin2, R_trace_log_thin[:,1,1], label='knot 1 time 1')
                # plt.plot(xs_thin2, R_trace_log_thin[:,1,2], label='knot 1 time 2')
                for i in [0,4,8]:
                    for t in np.arange(Nt)[np.arange(Nt) % 15 == 0]:
                        plt.plot(xs_thin2, R_trace_log_thin[:,i,t], label = 'knot '+str(i) + ' time ' + str(t))
                        plt.annotate('knot ' + str(i) + ' time ' + str(t), xy=(xs_thin2[-1], R_trace_log_thin[:,i,t][-1]))
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
                    plt.annotate('knot ' + str(i), xy=(xs_thin2[-1], range_knots_trace_thin[:,i][-1]))
                plt.title('traceplot for phi')
                plt.title('traceplot for range')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('range')
                plt.legend()
                plt.savefig('range.pdf')
                plt.close()

                # ---- GEV ----
                ## location coefficients
                plt.subplots()
                for j in range(Beta_mu_m):
                    plt.plot(xs_thin2, Beta_mu_trace_thin[:,j], label = 'Beta_' + str(j))
                    plt.annotate('Beta_' + str(j), xy=(xs_thin2[-1], Beta_mu_trace_thin[:,j][-1]))
                plt.title('traceplot for Beta_mu s')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('Beta_mu')
                plt.legend()
                plt.savefig('Beta_mu.pdf')
                plt.close()
                ## scale coefficients
                plt.subplots()
                for j in range(Beta_sigma_m):
                    plt.plot(xs_thin2, Beta_sigma_trace_thin[:,j], label = 'Beta_' + str(j))
                    plt.annotate('Beta_' + str(j), xy=(xs_thin2[-1], Beta_sigma_trace_thin[:,j][-1]))
                plt.title('traceplot for Beta_sigma s')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('Beta_sigma')
                plt.legend()
                plt.savefig('Beta_sigma.pdf')
                plt.close()
                ## shape coefficients
                plt.subplots()
                for j in range(Beta_ksi_m):
                    plt.plot(xs_thin2, Beta_ksi_trace_thin[:,j], label = 'Beta_' + str(j))
                    plt.annotate('Beta_' + str(j), xy=(xs_thin2[-1], Beta_ksi_trace_thin[:,j][-1]))
                plt.title('traceplot for Beta_ksi s')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('Beta_ksi')
                plt.legend()
                plt.savefig('Beta_ksi.pdf')
                plt.close()
                # ## location mu
                # plt.subplots()
                # plt.plot(xs_thin2, GEV_knots_trace_thin[:,0,0], label = 'knot 0') # location
                # plt.title('traceplot for location')
                # plt.xlabel('iter thinned by 10')
                # plt.ylabel('mu')
                # plt.legend()
                # plt.savefig('mu.pdf')
                # plt.close()

                # ## scale sigma
                # plt.subplots()
                # plt.plot(xs_thin2, GEV_knots_trace_thin[:,1,0], label = 'knot 0') # scale
                # plt.title('traceplot for scale')
                # plt.xlabel('iter thinned by 10')
                # plt.ylabel('sigma')
                # plt.legend()
                # plt.savefig('sigma.pdf')
                # plt.close()

                # ## shape ksi
                # plt.subplots()
                # plt.plot(xs_thin2, GEV_knots_trace_thin[:,2,0], label = 'knot 0') # shape
                # plt.title('traceplot for shape')
                # plt.xlabel('iter thinned by 10')
                # plt.ylabel('ksi')
                # plt.legend()
                # plt.savefig('ksi.pdf')
                # plt.close()

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
                    plt.annotate('piece ' + str(i), xy=(xs_thin2[-1], loglik_detail_trace_thin[:,i][-1]))
                plt.title('traceplot for phi')
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
                
                # # GEV
                # r_hat = num_accepted['GEV']/25
                # num_accepted['GEV'] = 0
                # sample_cov = np.cov(np.array([GEV_knots_trace[iter-25:iter,0,0].ravel(), # mu location
                #                                 GEV_knots_trace[iter-25:iter,1,0].ravel()])) # sigma scale
                # Sigma_0_hat = np.zeros((3,3)) # doing the hack because we are not updating ksi
                # Sigma_0_hat[2,2] = 1
                # Sigma_0_hat[0:2,0:2] += sample_cov
                # log_sigma_m_sq_hat = np.log(sigma_m_sq['GEV']) + gamma2*(r_hat - r_opt)
                # sigma_m_sq['GEV'] = np.exp(log_sigma_m_sq_hat)
                # Sigma_0['GEV'] = Sigma_0['GEV'] + gamma1*(Sigma_0_hat - Sigma_0['GEV'])

                # GEV coefficients
                ## Beta_mu
                r_hat = num_accepted['Beta_mu']/25
                num_accepted['Beta_mu'] = 0
                log_sigma_m_sq_hat    = np.log(sigma_m_sq['Beta_mu']) + gamma2*(r_hat - r_opt)
                sigma_m_sq['Beta_mu'] = np.exp(log_sigma_m_sq_hat)
                ## Beta_sigma
                r_hat = num_accepted['Beta_sigma']/25
                num_accepted['Beta_sigma'] = 0
                log_sigma_m_sq_hat       = np.log(sigma_m_sq['Beta_sigma']) + gamma2*(r_hat - r_opt)
                sigma_m_sq['Beta_sigma'] = np.exp(log_sigma_m_sq_hat)
                ## Beta_ksi
                r_hat = num_accepted['Beta_ksi']/25
                num_accepted['Beta_ksi'] = 0
                log_sigma_m_sq_hat     = np.log(sigma_m_sq['Beta_ksi']) + gamma2*(r_hat - r_opt)
                sigma_m_sq['Beta_ksi'] = np.exp(log_sigma_m_sq_hat)
        
        comm.Barrier() # block for adaptive update

    #####################################################################################################################
    # Actual Param Update ###############################################################################################
    #####################################################################################################################

        #### ----- Update Rt ----- Parallelized Across Nt time
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

        #### ----- Update phi ----- parallelized likelihood calculation across Nt time
        # if rank == 0:
        #     print('Updating phi')
        # Propose new phi at the knots --> new phi vector
        if rank == 0:
            random_walk_block1 = np.sqrt(sigma_m_sq['phi_block1'])*random_generator.multivariate_normal(np.zeros(3), Sigma_0['phi_block1'])
            random_walk_block2 = np.sqrt(sigma_m_sq['phi_block2'])*random_generator.multivariate_normal(np.zeros(3), Sigma_0['phi_block2'])
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
                                        phi_vec_proposal, gamma_vec)
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
            X_star_1t_current = X_star_1t_proposal

        comm.Barrier() # block for phi updates

        #### ----- Update range_vec ----- parallelized likelihood calculation across Nt time
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

        #### ----- Update covariate coefficients -----
        if rank == 0:
            # make proposal variance scalar adaptive
            Beta_mu_proposal    = Beta_mu_current    + np.sqrt(sigma_m_sq['Beta_mu'])*random_generator.normal(np.zeros(1), 1, size = Beta_mu_m)
            Beta_sigma_proposal = Beta_sigma_current + np.sqrt(sigma_m_sq['Beta_sigma'])*random_generator.normal(np.zeros(1), 1, size = Beta_sigma_m)
            Beta_ksi_proposal   = Beta_ksi_current   + np.sqrt(sigma_m_sq['Beta_ksi'])*random_generator.normal(np.zeros(1), 1, size = Beta_ksi_m)
        else:
            Beta_mu_proposal    = None
            Beta_sigma_proposal = None
            Beta_ksi_proposal   = None
        Beta_mu_proposal    = comm.bcast(Beta_mu_proposal, root = 0)
        Beta_sigma_proposal = comm.bcast(Beta_sigma_proposal, root = 0)
        Beta_ksi_proposal   = comm.bcast(Beta_ksi_proposal, root = 0)

        Loc_matrix_proposal   = (C_mu.T @ Beta_mu_proposal).T
        Scale_matrix_proposal = (C_sigma.T @ Beta_sigma_proposal).T
        Shape_matrix_proposal = (C_ksi.T @ Beta_ksi_proposal).T

        # Conditional log likelihood at Current
        lik_1t = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
                                                            Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                            phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)

        # Conditional log likelihood at proposal
        Scale_out_of_range = np.any(scale <= 0 for scale in Scale_matrix_proposal)
        Shape_out_of_range = np.any([shape <= -0.5 for shape in Shape_matrix_proposal]) or np.any([shape > 0.5 for shape in Shape_matrix_proposal])
        if Scale_out_of_range or Shape_out_of_range:
            X_star_1t_proposal = np.NINF
            lik_1t_proposal = np.NINF
        else:
            X_star_1t_proposal = qRW(pgev(Y[:,rank], Loc_matrix_proposal[:,rank], Scale_matrix_proposal[:,rank], Shape_matrix_proposal[:,rank]),
                                        phi_vec_current, gamma_vec)
            lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_proposal, 
                                                            Loc_matrix_proposal[:,rank], Scale_matrix_proposal[:,rank], Shape_matrix_proposal[:,rank],
                                                            phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
        
        # Gather likelihood calculated across time
        lik_gathered = comm.gather(lik_1t, root = 0)
        lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

        if rank == 0:
            prior_Beta_mu_current    = scipy.stats.norm.logpdf(Beta_mu_current, loc=0, scale=sigma_Beta_mu_current)
            prior_Beta_sigma_current = scipy.stats.norm.logpdf(Beta_sigma_current, loc=0, scale=sigma_Beta_sigma_current)
            prior_Beta_ksi_current   = scipy.stats.norm.logpdf(Beta_ksi_current, loc=0, scale=sigma_Beta_ksi_current)

            prior_Beta_mu_proposal    = scipy.stats.norm.logpdf(Beta_mu_proposal, loc=0, scale=sigma_Beta_mu_current)
            prior_Beta_sigma_proposal = scipy.stats.norm.logpdf(Beta_sigma_proposal, loc=0, scale=sigma_Beta_sigma_current)
            prior_Beta_ksi_proposal   = scipy.stats.norm.logpdf(Beta_ksi_proposal, loc=0, scale=sigma_Beta_ksi_current)
            
            lik_current = sum(lik_gathered) + \
                sum(prior_Beta_mu_current) + sum(prior_Beta_sigma_current) + sum(prior_Beta_ksi_current)
            lik_proposal = sum(lik_proposal_gathered) + \
                sum(prior_Beta_mu_proposal) + sum(prior_Beta_sigma_proposal) + sum(prior_Beta_ksi_proposal)

            u = random_generator.uniform()
            ratio = np.exp(lik_proposal - lik_current)
            if not np.isfinite(ratio):
                ratio = 0
            if u > ratio: # Reject
                Beta_mu_accepted    = False
                Beta_sigma_accepted = False
                Beta_ksi_accepted   = False
                Beta_mu_update      = Beta_mu_current
                Beta_sigma_update   = Beta_sigma_current
                Beta_ksi_update     = Beta_ksi_current
            else: # Accept
                Beta_mu_accepted    = True
                Beta_sigma_accepted = True
                Beta_ksi_accepted   = True
                Beta_mu_update      = Beta_mu_proposal
                Beta_sigma_update   = Beta_sigma_proposal
                Beta_ksi_update     = Beta_ksi_proposal
                num_accepted['Beta_mu']    += 1
                num_accepted['Beta_sigma'] += 1
                num_accepted['Beta_ksi']   += 1

            # Store the result
            Beta_mu_trace[iter, :]    = Beta_mu_update
            Beta_sigma_trace[iter, :] = Beta_sigma_update
            Beta_ksi_trace[iter, :]   = Beta_ksi_update

            # Update the current value
            Beta_mu_current     = Beta_mu_update
            Beta_sigma_current  = Beta_sigma_update
            Beta_ksi_current    = Beta_ksi_update
        else: # not rank = 1
            # other workers need to know acceptance, 
            # b/c although adaptive MH is calculated under worker0, need to know if X_star changed
            Beta_mu_accepted    = None
            Beta_sigma_accepted = None
            Beta_ksi_accepted   = None
        
        # Broadcast the update values
        Beta_mu_current     = comm.bcast(Beta_mu_current, root = 0)
        Beta_sigma_current  = comm.bcast(Beta_sigma_current, root = 0)
        Beta_ksi_current    = comm.bcast(Beta_ksi_current, root = 0)
        Beta_mu_accepted    = comm.bcast(Beta_mu_accepted, root = 0)
        Beta_sigma_accepted = comm.bcast(Beta_sigma_accepted, root = 0)
        Beta_ksi_accepted   = comm.bcast(Beta_ksi_accepted, root = 0)

        Loc_matrix_current   = (C_mu.T @ Beta_mu_current).T
        Scale_matrix_current = (C_sigma.T @ Beta_sigma_current).T
        Shape_matrix_current = (C_ksi.T @ Beta_ksi_current).T

        # Update X_star
        if Beta_mu_accepted:
            X_star_1t_current = X_star_1t_proposal
        
        comm.Barrier() # block for beta updates

    #### ---- Update sigma_beta, the hyper priors -----

        pass

    # #### ----- Update GEV mu sigma ksi (location, scale, shape) together ----
    # #### ----- Do not update ksi -----
    #     # if rank == 0:
    #     #     print('Updating GEV')
    #     # Propose new GEV params at the knots --> new GEV params vector
    #     if rank == 0:
    #         # random_walk = random_generator.multivariate_normal(np.zeros(3), GEV_post_cov, size = k).T
    #         random_walk = np.sqrt(sigma_m_sq['GEV'])*random_generator.multivariate_normal(np.zeros(3), Sigma_0['GEV'], size = k).T
    #         GEV_knots_proposal = GEV_knots_current + random_walk
    #         GEV_knots_proposal[:,1:] = np.vstack(GEV_knots_proposal[:,0]) # treat it as if it's only one knot, GEV params spatial stationary
    #         GEV_knots_proposal[2,:] = GEV_knots_current[2,:] # hold ksi constant
    #         # GEV_knots_proposal[0:2,:] = GEV_knots_current[0:2,:] # hold location and scale constant
    #     else:
    #         GEV_knots_proposal = None
    #     GEV_knots_proposal = comm.bcast(GEV_knots_proposal, root = 0)

    #     # will be changed into matrix multiplication w/ more knots
    #     Loc_matrix_proposal = np.full(shape = (Ns,Nt), fill_value = GEV_knots_proposal[0,0])
    #     Scale_matrix_proposal = np.full(shape = (Ns,Nt), fill_value = GEV_knots_proposal[1,0])
    #     Shape_matrix_proposal = np.full(shape = (Ns,Nt), fill_value = GEV_knots_proposal[2,0])

    #     # Conditional Likelihodd at Current
    #     lik_1t = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
    #                                                     Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
    #                                                     phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)

    #     # Conditional Likelihood at Proposed
    #     Scale_out_of_range = any(scale <= 0 for scale in GEV_knots_proposal[1,:])
    #     Shape_out_of_range = any(shape <= -0.5 for shape in GEV_knots_proposal[2,:]) or any(shape > 0.5 for shape in GEV_knots_proposal[2,:])
    #     if Scale_out_of_range or Shape_out_of_range:
    #         X_star_1t_proposal = np.NINF
    #         lik_1t_proposal = np.NINF
    #     else:
    #         X_star_1t_proposal = qRW(pgev(Y[:,rank], Loc_matrix_proposal[:,rank], Scale_matrix_proposal[:,rank], Shape_matrix_proposal[:,rank]),
    #                                     phi_vec_current, gamma_vec)
    #         lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_proposal, 
    #                                                         Loc_matrix_proposal[:,rank], Scale_matrix_proposal[:,rank], Shape_matrix_proposal[:,rank],
    #                                                         phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
        
    #     # When fixing the GEV parameters
    #     # X_star_1t_proposal = np.NINF
    #     # lik_1t_proposal = np.NINF 

    #     # Gather likelihood calculated across time
    #     lik_gathered = comm.gather(lik_1t, root = 0)
    #     lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

    #     # Accept or Reject
    #     if rank == 0:

    #         # for now there is only one set of GEV parameters
    #         # (constant across all time and space)
    #         # log-prior density for scale as P(sigma) = 1/sigma
    #         prior_scale = -np.log(Scale_matrix_current[0][0])
    #         prior_scale_proposal = -np.log(Scale_matrix_proposal[0][0])

    #         prior_mu = scipy.stats.norm.logpdf(Loc_matrix_current[0][0])
    #         prior_mu_proposal = scipy.stats.norm.logpdf(Loc_matrix_current[0][0])
            

    #         GEV_accepted = False
    #         lik = sum(lik_gathered) + prior_scale + prior_mu
    #         lik_proposal = sum(lik_proposal_gathered) + prior_scale_proposal + prior_mu_proposal

    #         u = random_generator.uniform()
    #         ratio = np.exp(lik_proposal - lik)
    #         if not np.isfinite(ratio):
    #             ratio = 0
    #         if u > ratio: # Reject
    #             Loc_matrix_update = Loc_matrix_current
    #             Scale_matrix_update = Scale_matrix_current
    #             Shape_matrix_update = Shape_matrix_current
    #             GEV_knots_update = GEV_knots_current
    #         else: # Accept, u <= ratio
    #             Loc_matrix_update = Loc_matrix_proposal
    #             Scale_matrix_update = Scale_matrix_proposal
    #             Shape_matrix_update = Shape_matrix_proposal
    #             GEV_knots_update = GEV_knots_proposal
    #             GEV_accepted = True
    #             num_accepted['GEV'] += 1
            
    #         # Store the result
    #         GEV_knots_trace[iter,:,:] = GEV_knots_update

    #         # Update the "current" value
    #         Loc_matrix_current = Loc_matrix_update
    #         Scale_matrix_current = Scale_matrix_update
    #         Shape_matrix_current = Shape_matrix_update
    #         GEV_knots_current = GEV_knots_update
    #     else:
    #         GEV_accepted = False

    #     # Brodcast the updated values
    #     Loc_matrix_current = comm.bcast(Loc_matrix_current, root = 0)
    #     Scale_matrix_current = comm.bcast(Scale_matrix_current, root = 0)
    #     Shape_matrix_current = comm.bcast(Shape_matrix_current, root = 0)
    #     GEV_knots_current = comm.bcast(GEV_knots_current, root = 0)
    #     GEV_accepted = comm.bcast(GEV_accepted, root = 0)

    #     # Update X_star
    #     if GEV_accepted:
    #         X_star_1t_current = X_star_1t_proposal
        
    #     comm.Barrier() # block for GEV updates

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
        # np.save('GEV_knots_trace', GEV_knots_trace)
        np.save('Beta_mu_trace', Beta_mu_trace)
        np.save('Beta_sigma_trace', Beta_sigma_trace)
        np.save('Beta_ksi_trace', Beta_ksi_trace)
        np.save('loglik_trace', loglik_trace)
        np.save('loglik_detail_trace', loglik_detail_trace)


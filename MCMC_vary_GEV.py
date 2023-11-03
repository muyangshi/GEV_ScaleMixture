# This is a MCMC sampler that constantly gets updated
# Scratch work and modifications are done in this file
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
    n_iters = 5000

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
    W = norm_to_Pareto1(Z) 
    # W = norm_to_std_Pareto(Z)

    # %%
    # ------- 4. Generate Scaling Factor, R^phi --------------------------------

    ## phi_vec
    phi_at_knots = 0.65-np.sqrt((knots_x-5.1)**2/5 + (knots_y-5.3)**2/4)/11.6 # scenario 2
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

    # # %%
    # # ------- 5. Generate mu(s) = C(s)Beta ----------------------

    # C(s) is the covariate
    Beta = 0.3
    Loc_at_knots = np.tile(np.sqrt(0.5*knots_x + 0.2*knots_y),
                           (N, 1)).T * Beta
    # Which basis should I use? Gaussian or Wendland?
    Loc_matrix = gaussian_weight_matrix @ Loc_at_knots # shape (N, num_sites)

    Loc_site_for_plot = gaussian_weight_matrix_for_plot @ Loc_at_knots
    Loc_site_for_plot = Loc_site_for_plot[:,0] # look at time t=0
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(plotgrid_X, plotgrid_Y, np.matrix(Loc_site_for_plot).reshape(25,25))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('mu(s)')
    ax.scatter(knots_x, knots_y, Loc_at_knots[:,0], c='red', marker='o', s=100)


    # %%
    # ------- 5. Generate X and Y--------------------------------
    X_star = R_phi * W

    # Calculation of Y can(?) be parallelized by time(?)
    Y = np.full(shape=(num_sites, N), fill_value = np.nan)
    for t in np.arange(N):
        Y[:,t] = qgev(pRW(X_star[:,t], phi_vec, gamma), mu, tau, ksi)

    # %%
    # ------- 6. Other Preparational Stuff(?) --------------------------------

    Loc_matrix = np.full(shape = Y.shape, fill_value = mu)
    Scale_matrix = np.full(shape = Y.shape, fill_value = tau)
    Shape_matrix = np.full(shape = Y.shape, fill_value = ksi)
    R_matrix = R_at_sites
    gamma_vec = np.repeat(gamma, num_sites)
    cholesky_matrix = scipy.linalg.cholesky(K, lower=False)

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
    phi_post_cov = np.array([
       [ 0.00260663, -0.00265617,  0.00096525, -0.00238565,  0.00204624,
        -0.00067302,  0.00088654, -0.00042376,  0.00015664],
       [-0.00265617,  0.00685654, -0.00312096,  0.00172352, -0.00529544,
         0.00227863, -0.00089019,  0.00225528, -0.0010919 ],
       [ 0.00096525, -0.00312096,  0.00301097, -0.00053988,  0.00284049,
        -0.00245542,  0.00037857, -0.00139068,  0.00083743],
       [-0.00238565,  0.00172352, -0.00053988,  0.00643117, -0.00518007,
         0.00237855, -0.00306443,  0.00204393, -0.00085607],
       [ 0.00204624, -0.00529544,  0.00284049, -0.00518007,  0.01390194,
        -0.00813707,  0.00302358, -0.00724033,  0.00364708],
       [-0.00067302,  0.00227863, -0.00245542,  0.00237855, -0.00813707,
         0.00714062, -0.00137521,  0.00406788, -0.00293598],
       [ 0.00088654, -0.00089019,  0.00037857, -0.00306443,  0.00302358,
        -0.00137521,  0.00318241, -0.00323841,  0.00123624],
       [-0.00042376,  0.00225528, -0.00139068,  0.00204393, -0.00724033,
         0.00406788, -0.00323841,  0.0086864 , -0.00364338],
       [ 0.00015664, -0.0010919 ,  0.00083743, -0.00085607,  0.00364708,
        -0.00293598,  0.00123624, -0.00364338,  0.0028415 ]])

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

    GEV_post_cov = np.array([[2.88511464e-04, 1.13560517e-04, 0],
                            [1.13560517e-04, 6.40933053e-05,  0],
                            [0         , 0         , 1e-4]])

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
        # printing and drawings
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
            
    #####################################################################################################################
    # Actual Param Update ###############################################################################################
    #####################################################################################################################

    #### ----- Update Rt ----- Parallelized Across N time
        # if rank == 0:
        #     print('Updating R')
        # Propose a R at time "rank", on log-scale

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
        lik_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
                                                        Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank], 
                                                        phi_vec_current, gamma_vec, R_vec_proposal, cholesky_matrix_current)
        prior_proposal = np.sum(scipy.stats.levy.logpdf(np.exp(R_proposal_log)) + R_proposal_log)

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

    #### ----- Update phi ----- parallelized likelihood calculation across N time
        # if rank == 0:
        #     print('Updating phi')
        # Propose new phi at the knots --> new phi vector
        if rank == 0:
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


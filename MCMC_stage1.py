# Stage 1 of the Two-Stage Sampler
# Assume independent GEV response to get posterior draws
# Use these posteriors for stage 2
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
    N = 4 # number of time replicates
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
    GEV_post_cov = np.array([[1e-04, 0, 0],
                            [0, 1e-04,  0],
                            [0, 0      , 1e-4]])

    # Scalors for adaptive updates
    # (phi, range, GEV) these parameters are only proposed on worker 0
    if rank == 0: 
        sigma_m_sq = {}
        # sigma_m_sq['GEV'] = (2.4**2)/3
        sigma_m_sq['GEV'] = (2.4**2)/3

        # initialize them with posterior covariance matrix
        Sigma_0 = {}
        Sigma_0['GEV'] = GEV_post_cov

        num_accepted = {}
        num_accepted['GEV'] = 0

    ########## Storage Place ##################################################
    # %%
    # Storage Place

    ## ---- GEV mu tau ksi (location, scale, shape) together ----
    if rank == 0:
        GEV_knots_trace_stage1 = np.full(shape=(n_iters, 3, k), fill_value = np.nan) # [n_iters, n_GEV, num_knots]
        GEV_knots_trace_stage1[0,:,:] = np.tile(np.array([mu, tau, ksi]), (k,1)).T
        GEV_knots_init = GEV_knots_trace_stage1[0,:,:]
    else:
        GEV_knots_init = None
    GEV_knots_init = comm.bcast(GEV_knots_init, root = 0)

    ## ---- overal likelihood? -----
    if rank == 0:
        loglik_trace = np.full(shape = (n_iters,1), fill_value = np.nan)
    else:
        loglik_trace = None

    ########## Initialize ##################################################
    # %%
    # Initialize
    ## ---- GEV mu tau ksi (location, scale, shape) together ----
    GEV_knots_current = GEV_knots_init
    # will(?) be changed into matrix multiplication w/ more knots or Covariate:
    Loc_matrix_current = np.full(shape = (num_sites,N), fill_value = GEV_knots_current[0,0])
    Scale_matrix_current = np.full(shape = (num_sites,N), fill_value = GEV_knots_current[1,0])
    Shape_matrix_current = np.full(shape = (num_sites,N), fill_value = GEV_knots_current[2,0])

    ########## Loops ##################################################
    # %%
    # Metropolis Updates
    for iter in range(1, n_iters):
        # printing and drawings
        if rank == 0:
            # print(num_accepted['GEV'])
            # print(sigma_m_sq['GEV'])
            if iter == 1:
                print(iter)
            if iter % 50 == 0:
                print(iter)
                print(strftime('%Y-%m-%d %H:%M:%S', localtime(time.time())))
            if iter % 1000 == 0 or iter == n_iters-1:
                # Save data every 1000 iterations
                end_time = time.time()
                print('elapsed: ', round(end_time - start_time, 1), ' seconds')
                np.save('GEV_knots_trace_stage1', GEV_knots_trace_stage1)
                np.save('loglik_trace', loglik_trace)

                # Print traceplot every 1000 iterations
                xs = np.arange(iter)
                xs_thin = xs[0::10] # index 1, 11, 21, ...
                xs_thin2 = np.arange(len(xs_thin)) # numbers 1, 2, 3, ...
                GEV_knots_trace_thin = GEV_knots_trace_stage1[0:iter:10,:,:]
                loglik_trace_thin = loglik_trace[0:iter:10,:]

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
        
        comm.Barrier() # block for drawing

        # Adaptive Update autotunings
        if iter % 25 == 0:
                
            gamma1 = 1 / ((iter/25 + offset) ** c_1)
            gamma2 = c_0 * gamma1

            # phi, range, and GEV
            if rank == 0:
                # GEV
                r_hat = num_accepted['GEV']/25
                num_accepted['GEV'] = 0
                sample_cov = np.cov(np.array([GEV_knots_trace_stage1[iter-25:iter,0,0].ravel(), # mu location
                                                GEV_knots_trace_stage1[iter-25:iter,1,0].ravel(), # tau scale
                                                GEV_knots_trace_stage1[iter-25:iter,2,0]])) # ksi shape
                Sigma_0_hat = sample_cov
                log_sigma_m_sq_hat = np.log(sigma_m_sq['GEV']) + gamma2*(r_hat - r_opt)
                sigma_m_sq['GEV'] = np.exp(log_sigma_m_sq_hat)
                Sigma_0['GEV'] = Sigma_0['GEV'] + gamma1*(Sigma_0_hat - Sigma_0['GEV'])
        
        comm.Barrier() # block for adaptive update

    #####################################################################################################################
    # Actual Param Update ###############################################################################################
    #####################################################################################################################

    #### ----- Update GEV mu tau ksi (location, scale, shape) together ----
        if rank == 0:
            random_walk = np.sqrt(sigma_m_sq['GEV']) * random_generator.multivariate_normal(np.zeros(3), Sigma_0['GEV'], size = k).T
            GEV_knots_proposal = GEV_knots_current + random_walk
            GEV_knots_proposal[:,1:] = np.vstack(GEV_knots_proposal[:,0])
        else:
            GEV_knots_proposal = None
        GEV_knots_proposal = comm.bcast(GEV_knots_proposal, root = 0)

        Loc_matrix_proposal = np.full(shape = (num_sites,N), fill_value = GEV_knots_proposal[0,0])
        Scale_matrix_proposal = np.full(shape = (num_sites,N), fill_value = GEV_knots_proposal[1,0])
        Shape_matrix_proposal = np.full(shape = (num_sites,N), fill_value = GEV_knots_proposal[2,0])

        # GEV log likelihood at Current
        lik_1t = np.sum(dgev(Y[:,rank], 
                             Loc_matrix_current[:,rank], 
                             Scale_matrix_current[:,rank], 
                             Shape_matrix_current[:,rank],
                             log = True))

        # GEV log likelihood at Proposal
        Scale_out_of_range = any(scale <= 0 for scale in GEV_knots_proposal[1,:])
        Shape_out_of_range = any(shape <= -0.5 for shape in GEV_knots_proposal[2,:]) or any(shape > 0.5 for shape in GEV_knots_proposal[2,:])
        if Scale_out_of_range or Shape_out_of_range:
            lik_1t_proposal = np.NINF
        else:
            lik_1t_proposal = np.sum(dgev(Y[:,rank],
                                          Loc_matrix_proposal[:,rank],
                                          Scale_matrix_proposal[:,rank],
                                          Shape_matrix_proposal[:,rank],
                                          log = True))

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
            GEV_knots_trace_stage1[iter,:,:] = GEV_knots_update

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
        
        comm.Barrier() # block for GEV updates

        # Keeping track of likelihood after this iteration
        lik_final_1t = lik_1t_proposal if GEV_accepted else lik_1t
        lik_final_gathered = comm.gather(lik_final_1t, root = 0)
        if rank == 0:
            loglik_trace[iter,0] = round(sum(lik_final_gathered),3) # storing the overall log likelihood

        comm.Barrier() # block for one iteration of update

    # End of MCMC
    if rank == 0:
        end_time = time.time()
        print('total time: ', round(end_time - start_time, 1), ' seconds')
        print('true R: ', R_at_knots)
        # np.save('R_trace_log', R_trace_log)
        # np.save('phi_knots_trace', phi_knots_trace)
        # np.save('range_knots_trace', range_knots_trace)
        np.save('GEV_knots_trace_stage1', GEV_knots_trace_stage1)
        np.save('loglik_trace', loglik_trace)
        # np.save('loglik_detail_trace', loglik_detail_trace)


# %%
# MLE results
# from scipy.stats import genextreme
# genextreme.fit(Y) # params order is (shape, loc, scale), and shape is scipy's negated
# (-0.17235585204019163, 0.2633514101978924, 1.188742072832001)
if __name__ == "__main__":
    # %% for reading seed from bash
    import sys
    data_seed = int(sys.argv[1]) if len(sys.argv) == 2 else 2345

    # %% imports
    # imports
    import os
    os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
    os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    # from matplotlib import colormaps
    import scipy
    import time
    from mpi4py import MPI
    from time import strftime, localtime
    from utilities import *
    import gstools as gs
    import rpy2.robjects as robjects
    from rpy2.robjects import r 
    from rpy2.robjects.numpy2ri import numpy2rpy
    from rpy2.robjects.packages import importr
    import pickle

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    random_generator = np.random.RandomState((rank+1)*7) # use of this avoids impacting the global np state

    try: # data_seed is defined when python MCMC.py
        data_seed
    except: # when running on local machine interactively
        data_seed = 2345 # default seed
    finally:
        if rank == 0: print('data_seed: ', data_seed)
    np.random.seed(data_seed)

    if rank == 0: print('Pareto: ', norm_pareto)

    # %% Simulation Setup

    # Spatial Domain Setup --------------------------------------------------------------------------------------------

    # Numbers - Ns, Nt --------------------------------------------------------
    
    np.random.seed(data_seed)
    Nt = 24 # number of time replicates
    Ns = 300 # number of sites/stations
    Time = np.linspace(-Nt/2, Nt/2-1, Nt)/np.std(np.linspace(-Nt/2, Nt/2-1, Nt), ddof=1)

    # missing indicator matrix ------------------------------------------------
    
    ## random missing
    miss_matrix = np.full(shape = (Ns, Nt), fill_value = 0)
    for t in range(Nt):
        miss_matrix[:,t] = np.random.choice([0, 1], size=(Ns,), p=[0.9, 0.1])
    miss_matrix = miss_matrix.astype(bool) # matrix of True/False indicating missing, True means missing
    
    # Sites - random uniformly (x,y) generate site locations ------------------
    
    sites_xy = np.random.random((Ns, 2)) * 10
    sites_x = sites_xy[:,0]
    sites_y = sites_xy[:,1]

    # # define the lower and upper limits for x and y
    minX, maxX = np.floor(np.min(sites_x)), np.ceil(np.max(sites_x))
    minY, maxY = np.floor(np.min(sites_y)), np.ceil(np.max(sites_y))

    # Elevation Function ------------------------------------------------------

    elev_surf_generator = gs.SRF(gs.Gaussian(dim=2, var = 1, len_scale = 2), seed=data_seed)
    elevations = elev_surf_generator((sites_x, sites_y))

    # Knots locations w/ isometric grid ---------------------------------------

    N_outer_grid = 9
    h_dist_between_knots     = (maxX - minX) / (int(2*np.sqrt(N_outer_grid))-1)
    v_dist_between_knots     = (maxY - minY) / (int(2*np.sqrt(N_outer_grid))-1)
    x_pos                    = np.linspace(minX + h_dist_between_knots/2, maxX + h_dist_between_knots/2, 
                                           num = int(2*np.sqrt(N_outer_grid)))
    y_pos                    = np.linspace(minY + v_dist_between_knots/2, maxY + v_dist_between_knots/2, 
                                           num = int(2*np.sqrt(N_outer_grid)))
    x_outer_pos              = x_pos[0::2]
    x_inner_pos              = x_pos[1::2]
    y_outer_pos              = y_pos[0::2]
    y_inner_pos              = y_pos[1::2]
    X_outer_pos, Y_outer_pos = np.meshgrid(x_outer_pos, y_outer_pos)
    X_inner_pos, Y_inner_pos = np.meshgrid(x_inner_pos, y_inner_pos)
    knots_outer_xy           = np.vstack([X_outer_pos.ravel(), Y_outer_pos.ravel()]).T
    knots_inner_xy           = np.vstack([X_inner_pos.ravel(), Y_inner_pos.ravel()]).T
    knots_xy                 = np.vstack((knots_outer_xy, knots_inner_xy))
    knots_id_in_domain       = [row for row in range(len(knots_xy)) if (minX < knots_xy[row,0] < maxX and minY < knots_xy[row,1] < maxY)]
    knots_xy                 = knots_xy[knots_id_in_domain]
    knots_x                  = knots_xy[:,0]
    knots_y                  = knots_xy[:,1]
    k                        = len(knots_id_in_domain)

    # Copula/Data Model Setup - X_star = R^phi * g(Z) -----------------------------------------------------------------

    # Splines -----------------------------------------------------------------

    radius = 4
    radius_from_knots = np.repeat(radius, k) # Wendland kernel influence radius from a knot
    effective_range = radius # Gaussian kernel effective range: exp(-3) = 0.05
    bandwidth = effective_range**2/6 # range for the gaussian kernel
    assert k == len(knots_xy)
    
    # Weight matrix generated using Gaussian Smoothing Kernel
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
    
    # # constant weight matrix
    # constant_weight_matrix = np.full(shape = (Ns, k), fill_value = np.nan)
    # for site_id in np.arange(Ns):
    #     # Compute distance between each pair of the two collections of inputs
    #     d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
    #                                     XB = knots_xy)
    #     # influence coming from each of the knots
    #     weight_from_knots = np.repeat(1, k)/k
    #     constant_weight_matrix[site_id, :] = weight_from_knots

    # Covariance K for Gaussian Field g(Z) ------------------------------------

    nu = 0.5 # exponential kernel for matern with nu = 1/2
    sigsq = 1.0 # sill for Z
    sigsq_vec = np.repeat(sigsq, Ns) # hold at 1

    # Scale Mixture R^phi -----------------------------------------------------

    gamma = 0.5 # this is the gamma that goes in rlevy, gamma_at_knots
    delta = 0.0 # this is the delta in levy, stays 0
    alpha = 0.5
    gamma_at_knots = np.repeat(gamma, k)
    gamma_vec = np.sum(np.multiply(wendland_weight_matrix, gamma_at_knots)**(alpha), 
                       axis = 1)**(1/alpha) # bar{gamma}, axis = 1 to sum over K knots

    # Marginal GEV Model Setup ----------------------------------------------------------------------------------------

    # Splines -----------------------------------------------------------------

    # "knots" and prediction sites for splines 
    gs_x        = np.linspace(minX, maxX, 50)
    gs_y        = np.linspace(minY, maxY, 50)
    gs_xy       = np.vstack([coords.ravel() for coords in np.meshgrid(gs_x, gs_y, indexing='ij')]).T # indexing='ij' fill vertically, need .T in imshow

    gs_x_ro     = numpy2rpy(gs_x)        # Convert to R object
    gs_y_ro     = numpy2rpy(gs_y)        # Convert to R object
    gs_xy_ro    = numpy2rpy(gs_xy)       # Convert to R object
    sites_xy_ro = numpy2rpy(sites_xy)    # Convert to R object

    r.assign("gs_x_ro", gs_x_ro)         # Note: this is a matrix in R, not df
    r.assign("gs_y_ro", gs_y_ro)         # Note: this is a matrix in R, not df
    r.assign("gs_xy_ro", gs_xy_ro)       # Note: this is a matrix in R, not df
    r.assign('sites_xy_ro', sites_xy_ro) # Note: this is a matrix in R, not df

    mgcv = importr('mgcv')
    r('''
        gs_xy_df <- as.data.frame(gs_xy_ro)
        colnames(gs_xy_df) <- c('x','y')
        sites_xy_df <- as.data.frame(sites_xy_ro)
        colnames(sites_xy_df) <- c('x','y')
        ''')
    
    # r("save(gs_x_ro, file='gs_x_ro.gzip', compress=TRUE)")
    # r("save(gs_y_ro, file='gs_y_ro.gzip', compress=TRUE)")
    # r("save(gs_xy_df, file='gs_xy_df.gzip', compress=TRUE)")
    # r("save(sites_xy_df, file='sites_xy_df.gzip',compress=TRUE)")

    # Location mu_0(s) --------------------------------------------------------

    Beta_mu0_splines_m = 12 - 1 # number of splines basis, -1 b/c drop constant column
    Beta_mu0_m         = Beta_mu0_splines_m + 2 # adding intercept and elevation
    C_mu0_splines      = np.array(r('''
                                    basis      <- smoothCon(s(x, y, k = {Beta_mu0_splines_m}, fx = TRUE), data = gs_xy_df)[[1]]
                                    basis_site <- PredictMat(basis, data = sites_xy_df)
                                    # basis_site
                                    basis_site[,c(-(ncol(basis_site)-2))] # dropped the 3rd to last column of constant
                                    '''.format(Beta_mu0_splines_m = Beta_mu0_splines_m+1))) # shaped(Ns, Beta_mu0_splines_m)
    C_mu0_1t           = np.column_stack((np.ones(Ns),  # intercept
                                        elevations,     # elevation
                                        C_mu0_splines)) # splines (excluding intercept)
    C_mu0              = np.tile(C_mu0_1t.T[:,:,None], reps = (1, 1, Nt))

    # Location mu_1(s) --------------------------------------------------------
    
    Beta_mu1_splines_m = 12 - 1 # drop the 3rd to last column of constant
    Beta_mu1_m         = Beta_mu1_splines_m + 2 # adding intercept and elevation
    C_mu1_splines      = np.array(r('''
                                    basis      <- smoothCon(s(x, y, k = {Beta_mu1_splines_m}, fx = TRUE), data = gs_xy_df)[[1]]
                                    basis_site <- PredictMat(basis, data = sites_xy_df)
                                    # basis_site
                                    basis_site[,c(-(ncol(basis_site)-2))] # drop the 3rd to last column of constant
                                    '''.format(Beta_mu1_splines_m = Beta_mu1_splines_m+1))) # shaped(Ns, Beta_mu1_splines_m)
    C_mu1_1t           = np.column_stack((np.ones(Ns),  # intercept
                                        elevations,     # elevation
                                        C_mu1_splines)) # splines (excluding intercept)
    C_mu1              = np.tile(C_mu1_1t.T[:,:,None], reps = (1, 1, Nt))

    # Scale logsigma(s) -------------------------------------------------------
    
    Beta_logsigma_m   = 2 # just intercept and elevation
    C_logsigma        = np.full(shape = (Beta_logsigma_m, Ns, Nt), fill_value = np.nan)
    C_logsigma[0,:,:] = 1.0 
    C_logsigma[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T

    # Shape ksi(s) ------------------------------------------------------------
    
    Beta_ksi_m   = 2 # just intercept and elevation
    C_ksi        = np.full(shape = (Beta_ksi_m, Ns, Nt), fill_value = np.nan) # ksi design matrix
    C_ksi[0,:,:] = 1.0
    C_ksi[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T


    # Model Parameter Setup (Truth) -----------------------------------------------------------------------------------

    # Data Model Parameters - X_star = R^phi * g(Z) ---------------------------

    range_at_knots = np.sqrt(0.3*knots_x + 0.4*knots_y)/2 # range for spatial Matern Z

    ### scenario 1
    # sim_case     = 1
    # phi_at_knots = 0.65-np.sqrt((knots_x-3)**2/4 + (knots_y-3)**2/3)/10
    ### scenario 2
    sim_case   = 2
    phi_at_knots = 0.65-np.sqrt((knots_x-5.1)**2/5 + (knots_y-5.3)**2/4)/11.6
    ### scenario 3
    # sim_case     = 3
    # phi_at_knots = 0.37 + 5*(scipy.stats.multivariate_normal.pdf(knots_xy, mean = np.array([2.5,3]), cov = 2*np.matrix([[1,0.2],[0.2,1]])) + 
    #                          scipy.stats.multivariate_normal.pdf(knots_xy, mean = np.array([7,7.5]), cov = 2*np.matrix([[1,-0.2],[-0.2,1]])))

    # Marginal Parameters - GEV(mu, sigma, ksi) -------------------------------

    Beta_mu0            = np.concatenate(([0], [0.1], np.array([0.05]*Beta_mu0_splines_m)))
    Beta_mu1            = np.concatenate(([0], [0.01], np.array([0.01] * Beta_mu1_splines_m)))
    Beta_logsigma       = np.array([0.0, 0.01])
    Beta_ksi            = np.array([0.2, 0.05])
    sigma_Beta_mu0      = 1
    sigma_Beta_mu1      = 1
    sigma_Beta_logsigma = 1
    sigma_Beta_ksi      = 1

    mu0_estimates = (C_mu0.T @ Beta_mu0).T[:,0]
    mu1_estimates = (C_mu1.T @ Beta_mu1).T[:,0]
    logsigma_estimates = (C_logsigma.T @ Beta_logsigma).T[:,0]
    ksi_estimates = (C_ksi.T @ Beta_ksi).T[:,0]


    # %% Generate Simulation Data

    # Transformed Gaussian Process - W = g(Z), Z ~ MVN(0, K) ------------------

    range_vec = gaussian_weight_matrix @ range_at_knots
    K         = ns_cov(range_vec = range_vec, sigsq_vec = sigsq_vec,
                        coords = sites_xy, kappa = nu, cov_model = "matern")
    Z         = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(Ns,)),cov=K,size=Nt).T
    W         = norm_to_Pareto(Z) 

    # Random Scaling Factor - R^phi -------------------------------------------

    phi_vec    = gaussian_weight_matrix @ phi_at_knots
    R_at_knots = np.full(shape = (k, Nt), fill_value = np.nan)
    for t in np.arange(Nt):
        R_at_knots[:,t] = rlevy(n = k, m = delta, s = gamma) # generate R at time t, spatially varying k knots
        # should need to vectorize rlevy so in future s = gamma_at_knots (k,) vector
        # R_at_knots[:,t] = scipy.stats.levy.rvs(delta, gamma, k)
        # R_at_knots[:,t] = np.repeat(rlevy(n = 1, m = delta, s = gamma), k) # generate R at time t, spatially constant k knots
    R_at_sites = wendland_weight_matrix @ R_at_knots
    R_phi      = np.full(shape = (Ns, Nt), fill_value = np.nan)
    for t in np.arange(Nt):
        R_phi[:,t] = np.power(R_at_sites[:,t], phi_vec)

    # Marginal Transform to GEV F_Y(y) = F_Xstar(Xstar = R^phi * g(Z)) --------

    mu_matrix    = (C_mu0.T @ Beta_mu0).T + (C_mu1.T @ Beta_mu1).T * Time
    sigma_matrix = np.exp((C_logsigma.T @ Beta_logsigma).T)
    ksi_matrix   = (C_ksi.T @ Beta_ksi).T
    X_star       = R_phi * W
    Y            = np.full(shape = (Ns, Nt), fill_value = np.nan)
    for t in np.arange(Nt):
        Y[:,t] = qgev(pRW(X_star[:,t], phi_vec, gamma_vec), mu_matrix[:,t], sigma_matrix[:,t], ksi_matrix[:,t])


    # %% Save Simulated Dataset ---------------------------------------------------------------------------------------
    
    # original dataset
    
    if rank == 0: np.save('Y_truth_full_sim_sc'+str(sim_case)+'_t'+str(Nt)+'_s'+str(Ns), Y)
   
    # missing at random indicator matrix
    if rank == 0: np.save('miss_matrix_bool', miss_matrix)
    
    # dataset with NAs

    for t in range(Nt):
        Y[:,t][miss_matrix[:,t]] = np.nan
    if rank == 0: np.save('Y_truth_NA_sim_sc'+str(sim_case)+'_t'+str(Nt)+'_s'+str(Ns), Y)
    
    # complete dataset in .RData format
    #   - JJA_maxima_nonimputed
    #   - GEV_estimates
    #   - stations
    #   - elev

    JJA_maxima_nonimputed_ro = numpy2rpy(Y)
    GEV_estimates            = np.column_stack(((C_mu0.T @ Beta_mu0).T[:,0],
                                                (C_mu1.T @ Beta_mu1).T[:,0],
                                                (C_logsigma.T @ Beta_logsigma).T[:,0],
                                                (C_ksi.T @ Beta_ksi).T[:,0]))
    GEV_estimates_ro         = numpy2rpy(GEV_estimates)
    stations_ro              = numpy2rpy(sites_xy)
    elev_ro                  = numpy2rpy(elevations)

    r.assign('JJA_maxima_nonimputed', JJA_maxima_nonimputed_ro)
    r.assign('GEV_estimates', GEV_estimates_ro)
    r.assign('stations', stations_ro)
    r.assign('elev', elev_ro)

    r('''
      GEV_estimates <- as.data.frame(GEV_estimates)
      colnames(GEV_estimates) <- c('mu0','mu1','logsigma','xi')

      stations <- as.data.frame(stations)
      colnames(stations) <- c('x','y')

      elev <- c(elev)

      save(JJA_maxima_nonimputed, GEV_estimates, stations, elev,
           file = 'simulated_data.RData')
      ''')


    # %% Checks on Data Generation ------------------------------------------------------------------------------------

    # Check stable variables S ------------------------------------------------

    # levy.cdf(R_at_knots, loc = 0, scale = gamma) should look uniform
    
    for i in range(k):
        scipy.stats.probplot(scipy.stats.levy.cdf(R_at_knots[i,:], scale=gamma), dist='uniform', fit=False, plot=plt)
        plt.axline((0,0), slope = 1, color = 'black')
        plt.savefig(f'QQPlot_levy_knot_{i}.png')
        plt.show()
        plt.close()

    # Check Pareto distribution -----------------------------------------------

    # shifted pareto.cdf(W[site_i,:] + 1, b = 1, loc = 0, scale = 1) shoud look uniform
    
    if norm_pareto == 'shifted':
        for site_i in range(Ns):
            if site_i % 10 == 0: # don't print all sites
                scipy.stats.probplot(scipy.stats.pareto.cdf(W[site_i,:]+1, b = 1, loc = 0, scale = 1), dist='uniform', fit=False, plot=plt)
                plt.axline((0,0), slope = 1, color = 'black')
                plt.savefig(f'QQPlot_Pareto_site_{site_i}.png')
                plt.show()
                plt.close()

    # standard pareto.cdf(W[site_i,:], b = 1, loc = 0, scale = 1) shoud look uniform
    if norm_pareto == 'standard':
        for site_i in range(Ns):
            if site_i % 10 == 0:
                scipy.stats.probplot(scipy.stats.pareto.cdf(W[site_i,:], b = 1, loc = 0, scale = 1), dist='uniform', fit=False, plot=plt)
                plt.axline((0,0), slope = 1, color = 'black')
                plt.savefig(f'QQPlot_Pareto_site_{site_i}.png')
                plt.show()
                plt.close()

    # Check model X_star ------------------------------------------------------

    # pRW(X_star) should look uniform (at each site with Nt time replicates)
    for site_i in range(Ns):
        if site_i % 20 == 0:
            unif = pRW(X_star[site_i,:], phi_vec[site_i], gamma_vec[site_i])
            scipy.stats.probplot(unif, dist="uniform", fit = False, plot=plt)
            plt.axline((0,0), slope=1, color='black')
            plt.savefig(f'QQPlot_Xstar_site_{site_i}.png')
            plt.show()
            plt.close()
            
    # pRW(X_star) at each time t should deviates from uniform b/c spatial correlation
    for t in range(Nt):
        if t % 5 == 0:
            unif = pRW(X_star[:,t], phi_vec, gamma_vec[t])
            scipy.stats.probplot(unif, dist="uniform", fit = False, plot=plt)
            plt.axline((0,0), slope=1, color='black')
            plt.savefig(f'QQPlot_Xstar_time_{t}.png')
            plt.show()
            plt.close()

    # Check Marginal Y --------------------------------------------------------

    # A simple GEV MLE-fit should roughly reflects values around truth

    myfits = [scipy.stats.genextreme.fit(Y[site,:][~np.isnan(Y[site,:])]) for site in range(300)]

    color_loc = 'blue'
    color_scale = 'green'
    color_shape = 'red'

    plt.hist([fit[1] for fit in myfits], bins=15, alpha=0.7, color=color_loc, label='Location')
    plt.hist([fit[2] for fit in myfits], bins=15, alpha=0.7, color=color_scale, label='Scale')
    plt.hist([fit[0] for fit in myfits], bins=15, alpha=0.7, color=color_shape, label='Shape')
    plt.hist([fit[1] for fit in myfits], bins=15, alpha=1.0, histtype='step', edgecolor='black')
    plt.hist([fit[2] for fit in myfits], bins=15, alpha=1.0, histtype='step', edgecolor='black')
    plt.hist([fit[0] for fit in myfits], bins=15, alpha=1.0, histtype='step', edgecolor='black')

    legend_handles = [
        matplotlib.patches.Patch(facecolor=color_loc, edgecolor='black', label='Location'),
        matplotlib.patches.Patch(facecolor=color_scale, edgecolor='black', label='Scale'),
        matplotlib.patches.Patch(facecolor=color_shape, edgecolor='black', label='Shape')
    ]

    plt.legend(handles=legend_handles)
    plt.title('MLE-fitted GEV')
    plt.savefig('Histogram_MLE_fitted_GEV.png')
    plt.show()
    plt.close()

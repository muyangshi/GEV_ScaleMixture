"""
This is a MCMC sampler

# Require:
#   - utilities_pureAI.py
#   - proposal_cov.py
# Example Usage:
# mpirun -n 2 --output-filename <put here: folder_name/filename> python3 MCMC.py > combinedoutput.txt &
# mpirun -n 2 python3 MCMC.py > output.txt 2>&1 &

Notes on (installation of) Rpy2
    - Work around of conda+rpy2: conda install rpy2 also installs an r-base;
      if using the conda installed R, don't use the default R on machine to avoid issue
      (i.e. change the default R path to the end the $PATH)
    - Alternatively, split the MCMC into three pieces: Python generate data --> R generate X design --> Python run MCMC
    - In importr, lib_loc specifies where an R package is installed

Note on plotting heatmap:
plotgrid_xy is meshgrid(order='xy') fills horizontally (x changes first, then y changes), so no need tranpose in imshow
gs_xy is meshgrid(order='ij') fills vertically (y changes first, then x changes), so NEED tranpose in imshow

Note on mgcv
- It appears the basis matrix produced by smoothCon is slightly (~ 3 to 4 decimal places) different between machines
- Note that in the splines constructed by mgcv, the 3rd to last column is a flat plane (which duplicate the intercept term)
    so remember to remove it!

Note on the marginal model
    for mu:        theta(s) = Beta_0 + Beta_1 * Elev(s) + splines(s) @ Beta_splines
    for sigma/ksi: theta(s) = Beta_0 + Beta_1 * Elev(s)
More specifically,
    mu(s,t)       = mu_0(s) + mu_1(s) * t
    logsigma(s,t) = logsigma(s)
    ksi(s,t)      = ksi(s)
where
    t           = - Nt/2, -Nt/2 + 1, ..., 0, 1, ..., Nt/2 - 1
    mu_0(s)     = Beta_mu0_0 + Beta_mu0_1 * Elev(s) + splines(s) @ Beta_mu0_splines
    mu_1(s)     = Beta_mu1_0 + Beta_mu1_1 * Elev(s) + splines(s) @ Beta_mu1_splines
    logsigma(s) = Beta_logsigma_0 + Beta_logsimga_1 * elev(s)                      ............ still the very simple linear surface
    ksi(s)      = Beta_ksi_0      + Beta_ksi_1      * elev(s)                      ............ still the very simple linear surface
so, for example, we have
    Beta_mu0    = (Beta_mu0_0, Beta_mu0_1, Beta_mu0_splines)
    C_mu0(s)    = (1, Elev(s), splines(s))
and
    Beta_ksi    = (Beta_ksi_0, Beta_ksi_1)
    C_ksi(s)    = (1, elev(s))

    20250520: fixed like_1t_proposal bug --> lik_1t_proposal
"""

if __name__ == "__main__":
    # %% for reading seed from bash
    import sys
    data_seed = int(sys.argv[1]) if len(sys.argv) == 2 else 2345

    # %% imports
    # imports
    import os
    os.environ["OMP_NUM_THREADS"]        = "1"  # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"]   = "1"  # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"]        = "1"  # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
    os.environ["NUMEXPR_NUM_THREADS"]    = "1"  # export NUMEXPR_NUM_THREADS=1
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import scipy
    import time
    from mpi4py import MPI
    from time import strftime, localtime
    from utilities_pureAI import *
    import proposal_cov
    import gstools as gs
    import rpy2.robjects as robjects
    from rpy2.robjects import r
    from rpy2.robjects.numpy2ri import numpy2rpy
    from rpy2.robjects.packages import importr
    import pickle
    mgcv = importr('mgcv')
    import geopandas as gpd
    state_map = gpd.read_file('./cb_2018_us_state_20m/cb_2018_us_state_20m.shp').to_crs(epsg=4326)

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    random_generator = np.random.RandomState((rank+1)*7) # use of this avoids impacting the global np state

    try: # data_seed is defined when python MCMC.py
        data_seed
    except: # when running on local machine interactively
        data_seed = 2345
    finally:
        if rank == 0: print('data_seed: ', data_seed)
    np.random.seed(data_seed)

    if rank == 0: print('Pareto: ', norm_pareto)

    try:
        with open('iter.pkl','rb') as file:
            start_iter = pickle.load(file) + 1
            if rank == 0: print('start_iter loaded from pickle, set to be:', start_iter)
    except Exception as e:
        if rank == 0:
            print('Exception loading iter.pkl:', e)
            print('Setting start_iter to 1')
        start_iter = 1

    if norm_pareto == 'shifted':  n_iters = 20000
    if norm_pareto == 'standard': n_iters = 200000

    # %% Load Dataset -----------------------------------------------------------------------------------------------
    # Load Dataset    -----------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------------------------
    # data

    r('''load('JJA_precip_maxima_nonimputed.RData')''')
    GEV_estimates      = np.array(r('GEV_estimates')).T
    mu0_estimates      = GEV_estimates[:,0]
    mu1_estimates      = GEV_estimates[:,1]
    logsigma_estimates = GEV_estimates[:,2]
    ksi_estimates      = GEV_estimates[:,3]
    JJA_maxima         = np.array(r('JJA_maxima_nonimputed'))
    stations           = np.array(r('stations')).T
    elevations         = np.array(r('elev')).T/200

    # # truncate if only running a random subset
    # Nt                 = 24
    # Ns                 = 125
    # times_subset       = np.arange(Nt)
    # sites_subset       = np.random.default_rng(data_seed).choice(JJA_maxima.shape[0],size=Ns,replace=False,shuffle=False)
    # GEV_estimates      = GEV_estimates[sites_subset,:]
    # mu0_estimates      = GEV_estimates[:,0]
    # mu1_estimates      = GEV_estimates[:,1]
    # logsigma_estimates = GEV_estimates[:,2]
    # ksi_estimates      = GEV_estimates[:,3]
    # JJA_maxima         = JJA_maxima[sites_subset,:][:,times_subset]
    # stations           = stations[sites_subset]
    # elevations         = elevations[sites_subset]

    Y           = JJA_maxima.copy()
    miss_matrix = np.isnan(Y)


    # %% Setup (Covariates and Constants) ----------------------------------------------------------------------------
    # Setup (Covariates and Constants)    ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------
    # Ns, Nt

    Nt = JJA_maxima.shape[1] # number of time replicates
    Ns = JJA_maxima.shape[0] # number of sites/stations
    start_year = 1949
    end_year   = 2023
    all_years  = np.linspace(start_year, end_year, Nt)
    # Note, to use the mu1 estimates from Likun, the `Time` must be standardized the same way
    Time       = (all_years - np.mean(all_years))/np.std(all_years, ddof=1) # delta degress of freedom, to match the n-1 in R
    Time       = Time[0:Nt] # if there is any truncation specified above
    assert len(all_years) == Nt

    # ----------------------------------------------------------------------------------------------------------------
    # Sites

    sites_xy = stations
    sites_x = sites_xy[:,0]
    sites_y = sites_xy[:,1]

    # define the lower and upper limits for x and y
    minX, maxX = np.floor(np.min(sites_x)), np.ceil(np.max(sites_x))
    minY, maxY = np.floor(np.min(sites_y)), np.ceil(np.max(sites_y))

    # ----------------------------------------------------------------------------------------------------------------
    # Knots

    # isometric knot grid - for R and phi
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

    # isometric knot grid - for rho (de-coupled from phi)
    N_outer_grid_rho = 9
    h_dist_between_knots_rho     = (maxX - minX) / (int(2*np.sqrt(N_outer_grid_rho))-1)
    v_dist_between_knots_rho     = (maxY - minY) / (int(2*np.sqrt(N_outer_grid_rho))-1)
    x_pos_rho                    = np.linspace(minX + h_dist_between_knots_rho/2, maxX + h_dist_between_knots_rho/2,
                                           num = int(2*np.sqrt(N_outer_grid_rho)))
    y_pos_rho                    = np.linspace(minY + v_dist_between_knots_rho/2, maxY + v_dist_between_knots_rho/2,
                                           num = int(2*np.sqrt(N_outer_grid_rho)))
    x_outer_pos_rho              = x_pos_rho[0::2]
    x_inner_pos_rho              = x_pos_rho[1::2]
    y_outer_pos_rho              = y_pos_rho[0::2]
    y_inner_pos_rho              = y_pos_rho[1::2]
    X_outer_pos_rho, Y_outer_pos_rho = np.meshgrid(x_outer_pos_rho, y_outer_pos_rho)
    X_inner_pos_rho, Y_inner_pos_rho = np.meshgrid(x_inner_pos_rho, y_inner_pos_rho)
    knots_outer_xy_rho           = np.vstack([X_outer_pos_rho.ravel(), Y_outer_pos_rho.ravel()]).T
    knots_inner_xy_rho           = np.vstack([X_inner_pos_rho.ravel(), Y_inner_pos_rho.ravel()]).T
    knots_xy_rho                 = np.vstack((knots_outer_xy_rho, knots_inner_xy_rho))
    knots_id_in_domain_rho       = [row for row in range(len(knots_xy_rho)) if (minX < knots_xy_rho[row,0] < maxX and minY < knots_xy_rho[row,1] < maxY)]
    knots_xy_rho                 = knots_xy_rho[knots_id_in_domain_rho]
    knots_x_rho                  = knots_xy_rho[:,0]
    knots_y_rho                  = knots_xy_rho[:,1]
    k_rho                        = len(knots_id_in_domain_rho)

    # ----------------------------------------------------------------------------------------------------------------
    # Copula Splines

    # Basis Parameters - for the Gaussian and Wendland Basis

    radius            = 4                    # radius of Wendland Basis for R
    radius_from_knots = np.repeat(radius, k) # influence radius from a knot

    bandwidth         = radius               # range for the gaussian basis for phi
    # eff_range       = 4                    # range for the gaussian basis s.t. effective range is `radius`: exp(-3) = 0.05
    # bandwidth       = eff_range**2/6

    bandwidth_rho     = 4                    # range for the gaussian basis for rho
    # eff_range_rho   = 4                    # range for the gaussian basis for rho s.t. effective range is `radius`
    # bandwidth_rho   = eff_range_rho**2/6

    # Generate the weight matrices
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

    # Gaussian weight matrix specific to the rho/range surface
    gaussian_weight_matrix_rho = np.full(shape = (Ns, k_rho), fill_value = np.nan)
    for site_id in np.arange(Ns):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)),
                                                    XB = knots_xy_rho)
        # influence coming from each of the knots
        weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_rho, cutoff = False)
        gaussian_weight_matrix_rho[site_id, :] = weight_from_knots

    # # constant weight matrix
    # constant_weight_matrix = np.full(shape = (Ns, k), fill_value = np.nan)
    # for site_id in np.arange(Ns):
    #     # Compute distance between each pair of the two collections of inputs
    #     d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)),
    #                                     XB = knots_xy)
    #     # influence coming from each of the knots
    #     weight_from_knots = np.repeat(1, k)/k
    #     constant_weight_matrix[site_id, :] = weight_from_knots

    # ----------------------------------------------------------------------------------------------------------------
    # Setup For the Marginal Model - GEV(mu, sigma, ksi)

    # ----- using splines for mu0 and mu1 ---------------------------------------------------------------------------
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

    r('''
        gs_xy_df <- as.data.frame(gs_xy_ro)
        colnames(gs_xy_df) <- c('x','y')
        sites_xy_df <- as.data.frame(sites_xy_ro)
        colnames(sites_xy_df) <- c('x','y')
        ''')

    # Location mu_0(s) ----------------------------------------------------------------------------------------------

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

    # Location mu_1(s) ----------------------------------------------------------------------------------------------

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

    # Scale logsigma(s) ----------------------------------------------------------------------------------------------

    Beta_logsigma_m   = 2 # just intercept and elevation
    C_logsigma        = np.full(shape = (Beta_logsigma_m, Ns, Nt), fill_value = np.nan)
    C_logsigma[0,:,:] = 1.0
    C_logsigma[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T

    # Shape ksi(s) ----------------------------------------------------------------------------------------------

    Beta_ksi_m   = 2 # just intercept and elevation
    C_ksi        = np.full(shape = (Beta_ksi_m, Ns, Nt), fill_value = np.nan) # ksi design matrix
    C_ksi[0,:,:] = 1.0
    C_ksi[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T

    # ----------------------------------------------------------------------------------------------------------------
    # Setup For the Copula/Data Model - X_star = R^phi * g(Z)

    # Covariance K for Gaussian Field g(Z) --------------------------------------------------------------------------
    nu = 0.5 # exponential kernel for matern with nu = 1/2
    sigsq = 1.0 # sill for Z
    sigsq_vec = np.repeat(sigsq, Ns) # hold at 1

    # Scale Mixture R^phi --------------------------------------------------------------------------------------------
    ## phi and gamma
    gamma = 0.5 # this is the gamma that goes in rlevy, gamma_at_knots
    delta = 0.0 # this is the delta in levy, stays 0
    alpha = 0.5
    gamma_at_knots = np.repeat(gamma, k)
    gamma_vec = np.sum(np.multiply(wendland_weight_matrix, gamma_at_knots)**(alpha),
                       axis = 1)**(1/alpha) # bar{gamma}, axis = 1 to sum over K knots


    # %% Estimate Parameter -----------------------------------------------------------------------------------------------
    # Estimate Parameter    -----------------------------------------------------------------------------------------------

    if start_iter == 1:
        # ----------------------------------------------------------------------------------------------------------------
        # Marginal Parameters - GEV(mu, sigma, ksi)

        Beta_mu0 = np.linalg.lstsq(a=C_mu0[:,:,0].T, b=mu0_estimates,rcond=None)[0]
        Beta_mu1 = np.linalg.lstsq(a=C_mu1[:,:,0].T, b=mu1_estimates,rcond=None)[0]
        Beta_logsigma = np.linalg.lstsq(a=C_logsigma[:,:,0].T, b=logsigma_estimates,rcond=None)[0]
        Beta_ksi = np.linalg.lstsq(a=C_ksi[:,:,0].T, b=ksi_estimates,rcond=None)[0]

        ## Note: these sigma_Beta_xx must be values, can't be arrays
        sigma_Beta_mu0      = 9.62944645
        sigma_Beta_mu1      = 0.22947093
        sigma_Beta_logsigma = 1.79421561
        sigma_Beta_ksi      = 0.13111096

        mu_matrix    = (C_mu0.T @ Beta_mu0).T + (C_mu1.T @ Beta_mu1).T * Time
        sigma_matrix = np.exp((C_logsigma.T @ Beta_logsigma).T)
        ksi_matrix   = (C_ksi.T @ Beta_ksi).T

        # ----------------------------------------------------------------------------------------------------------------
        # Data Model Parameters - X_star = R^phi * g(Z)

        # Covariance K for Gaussian Field g(Z) --------------------------------------------------------------------------------------------

        # Estimate range: using sites within the radius of each knot
        range_at_knots = np.array([])
        distance_matrix = np.full(shape=(Ns, k_rho), fill_value=np.nan)
        # distance from knots
        for site_id in np.arange(Ns):
            d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), XB = knots_xy_rho)
            distance_matrix[site_id,:] = d_from_knots
        # each knot's "own" sites
        sites_within_knots = {}
        for knot_id in np.arange(k_rho):
            knot_name = 'knot_' + str(knot_id)
            sites_within_knots[knot_name] = np.where(distance_matrix[:,knot_id] <= bandwidth_rho)[0]

        # empirical variogram estimates
        for key in sites_within_knots.keys():
            selected_sites           = sites_within_knots[key]
            # demeaned_Y               = Y - mu_matrix
            demeaned_Y               = np.full_like(Y, np.nan)
            demeaned_Y[~miss_matrix] = Y[~miss_matrix] - mu_matrix[~miss_matrix]
            bin_center, gamma_variog = gs.vario_estimate((sites_x[selected_sites], sites_y[selected_sites]),
                                                        np.nanmean(demeaned_Y[selected_sites], axis=1))
            fit_model = gs.Exponential(dim=2)
            fit_model.fit_variogram(bin_center, gamma_variog, nugget=False)
            # ax = fit_model.plot(x_max = 4)
            # ax.scatter(bin_center, gamma_variog)
            range_at_knots = np.append(range_at_knots, fit_model.len_scale)
        if rank == 0:
            print('estimated range:',range_at_knots)

        range_at_knots = np.array([0.2] * k_rho)
        if rank == 0: print('range_at_knots set to:', range_at_knots)

        # check for unreasonably large values, intialize at some smaller ones
        range_upper_bound = 4
        if len(np.where(range_at_knots > range_upper_bound)[0]) > 0:
            if rank == 0: print('estimated range >', range_upper_bound, ' at:', np.where(range_at_knots > range_upper_bound)[0])
            if rank == 0: print('range at those knots set to be at', range_upper_bound)
            range_at_knots[np.where(range_at_knots > range_upper_bound)[0]] = range_upper_bound

        # check for unreasonably small values, initialize at some larger ones
        range_lower_bound = 0.01
        if len(np.where(range_at_knots < range_lower_bound)[0]) > 0:
            if rank == 0: print('estimated range <', range_lower_bound, ' at:', np.where(range_at_knots < range_lower_bound)[0])
            if rank == 0: print('range at those knots set to be at', range_lower_bound)
            range_at_knots[np.where(range_at_knots < range_lower_bound)[0]] = range_lower_bound

        # range_vec = gaussian_weight_matrix @ range_at_knots

        # Scale Mixture R^phi --------------------------------------------------------------------------------------------

        # phi_at_knots = np.array([0.4] * k)
        # phi_vec = gaussian_weight_matrix @ phi_at_knots

        # if norm_pareto == 'standard':
        #     R_at_knots = np.full(shape = (k, Nt), fill_value = np.nan)
        #     for t in np.arange(Nt):
        #         # R_at_knots[:,t] = (np.min(qRW(pgev(Y[:,t], mu_matrix[:,t], sigma_matrix[:,t], ksi_matrix[:,t]),
        #         #                         phi_vec, gamma_vec))/1.5)**2

        #         # only use non-missing values
        #         miss_index_1t = np.where(miss_matrix[:,t] == True)[0]
        #         obs_index_1t  = np.where(miss_matrix[:,t] == False)[0]
        #         R_at_knots[:,t] = (np.min(qRW(pgev(Y[obs_index_1t,t],
        #                                            mu_matrix[obs_index_1t,t], sigma_matrix[obs_index_1t,t], ksi_matrix[obs_index_1t,t]),
        #                                     phi_vec[obs_index_1t], gamma_vec[obs_index_1t]))/1.5)**(1/phi_at_knots)
        # else: # norm_pareto == 'shifted':
        #     # Calculate Rt in Parallel, only use non-missing values
        #     comm.Barrier()
        #     miss_index_1t = np.where(miss_matrix[:,rank] == True)[0]
        #     obs_index_1t  = np.where(miss_matrix[:,rank] == False)[0]
        #     X_1t       = qRW(pgev(Y[obs_index_1t,rank], mu_matrix[obs_index_1t,rank], sigma_matrix[obs_index_1t,rank], ksi_matrix[obs_index_1t,rank]),
        #                         phi_vec[obs_index_1t], gamma_vec[obs_index_1t])
        #     R_1t       = np.array([np.median(X_1t)**2] * k)
        #     R_gathered = comm.gather(R_1t, root = 0)
        #     R_at_knots = np.array(R_gathered).T if rank == 0 else None
        #     R_at_knots = comm.bcast(R_at_knots, root = 0)

    # %% Load Parameter
    # Load Parameter

    # # ----------------------------------------------------------------------------------------------------------------
    # # Marginal Parameters - GEV(mu, sigma, ksi)

    # Beta_mu0            = np.array([ 3.33548229e+01, -5.15749144e-03, -1.35501856e+01, -2.79089002e+00,
    #     1.31831870e+00,  2.43392492e+00,  2.33852452e+00,  1.76254921e+00,
    #    -1.39179393e+00, -9.16697094e-01, -3.06398612e-01, -4.64149124e+00,
    #    -3.68110772e-01])
    # Beta_mu1            = np.array([ 5.36116734e-01,  8.02941602e-04, -2.98349165e-01, -2.64425731e-01,
    #    -6.63459968e-01,  1.09443470e+00, -2.14915139e-01, -9.93815491e-02,
    #    -8.19032201e-02,  7.04091330e-01, -3.78407454e-02, -2.66154474e-01,
    #     3.05301913e-01,  7.92966770e-01,  2.08138130e-01,  6.02364824e-01,
    #    -1.62533200e-01,  8.86602422e-02, -8.07930545e-01])
    # Beta_logsigma       = np.array([3.07567306e+00, 1.64081493e-05])
    # Beta_ksi            = np.array([0.0879018 , 0.00036929])
    # sigma_Beta_mu0      = 9.68703611 # these must be values, can't be arrays!
    # sigma_Beta_mu1      = 0.47164467
    # sigma_Beta_logsigma = 1.1662009
    # sigma_Beta_ksi      = 0.20657275

    # # ----------------------------------------------------------------------------------------------------------------
    # # Data Model Parameters - X_star = R^phi * g(Z)

    # range_at_knots      = np.array([1.24636076, 0.53974012, 0.87688557, 4.31146549, 0.56016386,
    #    4.12702834, 5.2169322 , 3.25579305, 7.55346101])
    # phi_at_knots        = np.array([0.26050357, 0.21347445, 0.49264611, 0.16703945, 0.4541109 ,
    #    0.1351857 , 0.88671958, 0.20855159, 0.20105651])
    # R_at_knots          = np.exp(np.array([[ 2.65846867,  4.23794112,  4.40162513, -1.00510114,  3.85693745,
    #      3.19606955,  0.04695516, -1.02203322,  6.08541252, -0.22421416,
    #      0.16634089,  4.37316881,  3.5761424 , -0.04023258, -0.01507836,
    #      2.89043391,  0.18032337, -1.62838632, -0.70321109,  1.68320423,
    #      1.17207671,  0.35745514, -0.7294368 ,  1.93253295, -0.18396605,
    #     -1.33264493, -0.3568662 , -1.83521041,  0.04887536, -0.05989468,
    #     -1.19153176,  1.42597755,  0.17858646, -0.3083675 ,  1.02834152,
    #      1.16056064, -1.57043418, -0.12018036, -2.41627665,  0.93725518,
    #     -1.90715797, -1.62213589, -0.76970134,  0.39655864, -1.55908599,
    #     -0.91468297,  2.05533046, -0.125236  ,  0.86058794,  3.9774375 ,
    #      0.93157582,  0.05335151, -1.13838187,  1.2675335 ,  1.64588939,
    #      2.26176912,  1.20289763, -1.46425137,  3.17301376, -1.51065717,
    #     -0.68858224,  3.45099882,  0.14597834,  2.07299684,  0.65106834,
    #     -1.3734546 ,  4.36935242,  2.6865186 ],
    #    [-2.96531989,  0.21181869,  0.26725579,  4.85078149,  1.43013605,
    #      3.02290791, -1.25002938,  1.48349832,  8.32745066,  2.73449334,
    #     -2.07853954,  1.94730814,  1.38162062, -0.09164674, -0.79250504,
    #     -2.404911  ,  1.19328541,  0.71326039,  2.4779161 ,  1.82565686,
    #     -1.76673691,  2.22046094, -3.14252534, -1.09913093,  0.25067704,
    #      0.89835373, -2.10573933, -0.85228206,  0.31770357,  3.00466048,
    #     -1.2869365 ,  0.85172078, -1.51971738,  3.43001539,  0.55318098,
    #      5.23375572,  0.21495927, -1.63935957,  0.83550282,  5.22447765,
    #      1.05350475,  1.23883639,  0.23542387, -1.52075471, -0.09015844,
    #      0.83991323,  0.30176794, -1.09689097, -0.59744381, -0.76924567,
    #      0.48433218, -1.05010214,  3.77446646, -0.486379  ,  4.0288744 ,
    #      2.95186008, -0.649789  ,  4.1923073 , -1.29122141,  4.46742733,
    #      3.58666719, -1.83189044, -1.39326156, -0.94715639, -1.10395708,
    #      6.00803829, -1.55091387,  6.2561323 ],
    #    [ 5.46077887,  3.60186892,  3.27120629,  0.4772092 ,  1.24085218,
    #     -2.02461438,  1.23343831, -2.22517587,  2.2783368 ,  2.66539594,
    #      1.68480866,  2.10836077,  2.14371583,  0.8252471 ,  2.80758908,
    #      0.96904255,  0.28199001,  4.1854125 ,  2.55399306,  0.06208338,
    #     -0.59149094,  3.37503793,  2.54835601,  3.76419656,  3.17398049,
    #     -0.83492182, -1.09395239,  0.10192679,  1.89442199,  1.69356824,
    #     -2.15313573,  4.12459514,  2.606451  ,  3.11819902,  0.22303512,
    #      5.39451443, -2.2464116 ,  1.25126982,  1.03058118,  1.94620968,
    #      1.24855552,  1.21143258,  2.53548681,  2.9463441 , -0.13724797,
    #      1.13366249,  2.63807482,  2.01144315,  0.7329417 , -0.48277038,
    #      4.00503905,  1.28124316,  1.18753945,  0.65430138,  4.62589005,
    #     -0.54030765, -0.40850011,  2.75639653,  7.31456157,  1.03672077,
    #     -0.46272319,  2.65789674,  4.43135979,  5.04114008,  5.60809149,
    #     -0.15921768,  0.42493613, -0.05348762],
    #    [-0.40774592,  2.7703567 ,  0.58001045,  1.61240176, -0.48418521,
    #     -1.63276229, -0.45583018, -2.34352353, -0.12216726, -1.12450502,
    #     -0.82361434, -0.73389033,  2.80016795, -1.12719208, -1.34656297,
    #      0.32373245, -0.19564385, -0.05687316, -0.87928196, -0.07599409,
    #     -0.07686009, -1.2781721 , -2.1395268 , -0.48329389,  0.93628477,
    #     -1.07526141, -2.15091535, -0.76667039, -1.80380847, -1.41429247,
    #     -0.51020046, -1.26483766,  0.60877512, -0.81193612, -2.37209715,
    #     -1.93173691, -1.20144978, -0.5227675 , -2.02727475, -1.78468163,
    #      0.51844098,  0.24877497, -0.4794689 , -0.80361148, -1.43963403,
    #     -1.32230398,  2.39418178,  1.42786665,  0.83919015,  3.39892615,
    #     -1.20625313, -1.9783033 , -1.23441638, -1.8744995 ,  2.85307009,
    #      1.23171289,  0.4353684 , -1.39856516,  0.17199494, -1.93006367,
    #     -2.22825375, -0.48608374, -1.70029272,  1.61428732, -0.11195462,
    #      2.86391181,  2.79556424,  6.86778372],
    #    [-1.86522729, -1.1909772 ,  2.25247274,  1.19830195, -0.76498812,
    #     -0.96817767, -0.88371626,  0.5669276 , -1.1660425 ,  1.3521487 ,
    #      1.90895985, -1.8877353 ,  0.04930002, -1.25220285, -0.05369206,
    #     -1.40085219, -0.05006659, -0.8544703 , -0.89736363, -0.74020823,
    #     -1.00051625, -1.78101423, -2.37604261,  2.05046544, -0.36763347,
    #      0.95692993, -0.58699828,  0.33279392, -1.74892268, -2.60956999,
    #     -1.57998038,  0.200277  , -1.05410989, -1.91753646, -1.31659458,
    #     -0.54499406, -2.60886925, -2.15997333, -1.5965256 , -1.89627198,
    #     -1.45002378, -1.23818628, -1.50073375,  1.69191823, -1.76862568,
    #     -1.89840318, -0.2115662 ,  0.86096516,  0.51965756, -1.82043885,
    #     -2.26100958, -0.51922099, -0.50401991,  0.39482463,  0.89493331,
    #      4.50555734, -0.57733684, -1.19823096, -1.09224942, -1.7198214 ,
    #     -0.78393383, -1.04576052,  2.54365678, -0.8528608 ,  4.9497378 ,
    #     -3.0438423 , -0.78477474,  0.79344726],
    #    [-1.28942697,  8.7668832 ,  1.45176966, -0.27085911,  4.95782377,
    #      1.00607288, -2.45055329,  0.37009243,  1.28074429, -2.1113845 ,
    #     -1.12430065, -0.11838937, -0.54356165, -1.1672409 , -0.43516211,
    #     -0.63188361, -2.63141214, -0.9904584 , -0.99135499, -0.73046198,
    #      3.8958755 , -0.45603958, -0.82806559,  0.51003481, -2.62785624,
    #     -0.10264157, -0.53024033,  1.18084798,  1.58764388, -2.12015134,
    #     -1.75987616,  0.37725701, -0.52580438, -1.52765258, -2.42625327,
    #      0.42250217, -0.28935379,  0.18976528, -0.24337855,  4.85761925,
    #     -0.35119483, -0.4159018 ,  2.36474922, -1.33815169, -1.68752179,
    #      3.49484797, -2.01742449, -1.61463686,  0.33324212, -1.07863462,
    #     -0.55365371,  1.05933451, -2.39095   ,  1.94589941, -1.83960344,
    #     -0.1603993 , -0.15900495, -0.91536571, -1.21733674,  0.85969458,
    #      0.19360514,  2.07484102, -0.83107221,  0.02082449,  0.59845075,
    #      6.12358828, -1.30383076, -2.24260112],
    #    [ 0.82239498,  1.55553541,  1.51880678, -0.20077946,  0.73454979,
    #     -0.44351965,  2.27930069, -0.03244853,  0.37178893, -0.77653266,
    #      1.73268792, -1.34025747,  0.05068475, -2.66271775,  0.34217625,
    #     -0.92567121, -0.53292493,  0.5164409 ,  0.8983977 ,  0.08426838,
    #      2.55307107, -2.32693568,  0.5807171 ,  0.6124164 ,  1.31807011,
    #      0.76721538, -0.82885039, -1.29580868, -1.46147271, -0.37510141,
    #     -0.06237743,  1.03087723,  1.25028123, -0.87733851, -0.47755207,
    #     -0.8092368 , -0.10257399, -1.43215951, -0.77598542,  0.14778991,
    #     -1.45389258, -1.20059179,  2.3620157 ,  3.50739156,  1.27139245,
    #      0.82812871, -0.73326603, -0.03520199,  0.44580981, -1.79418331,
    #     -1.75243415,  0.22622009, -0.75282877,  1.10369068, -0.2905909 ,
    #      1.48765135,  2.34850898, -0.2543541 ,  1.68614499,  2.30186135,
    #     -0.09707889,  3.30872478, -0.93185567,  0.98172183,  4.43406561,
    #      0.91198708, -0.12804793,  1.78017998],
    #    [-1.31432828,  4.77751341,  0.3576654 , -1.4688279 , -2.36865667,
    #     -0.27339917, -1.21549209, -0.10389791, -1.76767502, -0.64920485,
    #      0.65567656,  0.36319555,  0.7318987 ,  0.50933672,  0.99240593,
    #     -1.80587741,  0.99900202,  0.95989635, -1.46415509,  0.84957646,
    #      0.8428495 , -0.13596185, -1.37908831, -0.3696574 ,  0.18548149,
    #     -2.37078367, -1.37511376, -1.19361057,  1.4201955 ,  2.19255656,
    #     -0.42057256, -2.52047116, -0.29092934,  1.04721867, -0.36839008,
    #      0.20642679,  0.88329431,  0.46731264, -2.68067035,  0.65179712,
    #      0.40124649, -2.67207964,  2.13411937, -1.22944423,  3.05643611,
    #     -1.93975478, -0.44856279, -1.6702242 , -2.11776917,  0.17171288,
    #     -2.1021769 ,  1.02959567, -2.68735833, -0.06901532,  0.31673787,
    #      1.92285418, -0.16574996, -1.13721145, -1.60427144, -2.18999318,
    #      3.37872964,  2.41866263, -0.94054476,  4.47735977,  3.20561404,
    #      5.82141384,  4.0989914 , -1.24645039],
    #    [ 0.18312756, -1.55254102,  2.48196592, -2.49641654,  4.65463114,
    #     -0.37385087, -1.12359576,  0.23481077, -3.07582371, -0.3820585 ,
    #      3.4604728 , -0.58940107, -2.27985788, -0.9325758 , -1.22865035,
    #     -0.22485722, -0.06592722,  0.1460683 , -2.45876709,  1.34277407,
    #     -0.61234517, -0.05976425,  0.77339817, -0.86829045,  0.8517377 ,
    #     -0.2926204 , -2.30401851, -1.01615711, -2.29761036, -0.79891994,
    #     -0.9644573 , -1.07693444, -1.15671632, -1.400673  , -0.90097705,
    #     -0.65697   ,  1.59380095, -0.42236117,  2.07149074, -0.16859658,
    #      0.30016854, -0.20752207, -0.70473439, -0.87194097,  0.77336122,
    #     -0.11742935,  0.36134095, -2.98921181, -1.08992179, -1.19727863,
    #     -0.94175959,  3.69773   , -0.34159897, -1.91022441,  0.71933433,
    #      0.13747757, -1.87765751, -1.87946638, -0.66058947,  1.07280658,
    #      2.24293671,  0.60606451,  0.71507063,  1.79901319,  4.95974394,
    #      3.60572766,  0.92684843, -1.45688259]]))

    # # ----------------------------------------------------------------------------------------------------------------
    # # Basis Expand to Sites

    # phi_vec      = gaussian_weight_matrix @ phi_at_knots
    # range_vec    = gaussian_weight_matrix @ range_at_knots
    # R_matrix        = wendland_weight_matrix @ R_at_knots
    # mu_matrix    = (C_mu0.T @ Beta_mu0).T + (C_mu1.T @ Beta_mu1).T * Time
    # sigma_matrix = np.exp((C_logsigma.T @ Beta_logsigma).T)
    # ksi_matrix   = (C_ksi.T @ Beta_ksi).T


    # %% Plot Parameter Surface
    # Plot Parameter Surface

    if rank == 0 and start_iter == 1:
        # 0. Grids for plots
        plotgrid_res_x = 150
        plotgrid_res_y = 175
        plotgrid_res_xy = plotgrid_res_x * plotgrid_res_y
        plotgrid_x = np.linspace(minX,maxX,plotgrid_res_x)
        plotgrid_y = np.linspace(minY,maxY,plotgrid_res_y)
        plotgrid_X, plotgrid_Y = np.meshgrid(plotgrid_x, plotgrid_y)
        plotgrid_xy = np.vstack([plotgrid_X.ravel(), plotgrid_Y.ravel()]).T

        gaussian_weight_matrix_for_plot = np.full(shape = (plotgrid_res_xy, k), fill_value = np.nan)
        for site_id in np.arange(plotgrid_res_xy):
            # Compute distance between each pair of the two collections of inputs
            d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)),
                                            XB = knots_xy)
            # influence coming from each of the knots
            weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
            gaussian_weight_matrix_for_plot[site_id, :] = weight_from_knots

        gaussian_weight_matrix_for_plot_rho = np.full(shape = (plotgrid_res_xy, k_rho), fill_value = np.nan)
        for site_id in np.arange(plotgrid_res_xy):
            # Compute distance between each pair of the two collections of inputs
            d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)),
                                                        XB = knots_xy_rho)
            # influence coming from each of the knots
            weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_rho, cutoff = False)
            gaussian_weight_matrix_for_plot_rho[site_id, :] = weight_from_knots

        wendland_weight_matrix_for_plot = np.full(shape = (plotgrid_res_xy,k), fill_value = np.nan)
        for site_id in np.arange(plotgrid_res_xy):
            # Compute distance between each pair of the two collections of inputs
            d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)),
                                            XB = knots_xy)
            # influence coming from each of the knots
            weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
            wendland_weight_matrix_for_plot[site_id, :] = weight_from_knots

        # # weight from knot plots --------------------------------------------------------------------------------------
        # # visualize the weights coming from a knot

        # # Define the colors for the colormap (white to red)
        # # Create a LinearSegmentedColormap
        # colors = ["#ffffff", "#ff0000"]
        # n_bins = 50  # Number of discrete color bins
        # cmap_name = "white_to_red"
        # colormap = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        # min_w = 0
        # max_w = 1
        # n_ticks = 11  # (total) number of ticks
        # ticks = np.linspace(min_w, max_w, n_ticks).round(3)

        # idx = 5
        # gaussian_weights_for_plot = gaussian_weight_matrix_for_plot[:,idx]
        # wendland_weights_for_plot = wendland_weight_matrix_for_plot[:,idx]

        # fig, axes = plt.subplots(1,2)
        # state_map.boundary.plot(ax=axes[0], color = 'black', linewidth = 0.5)
        # heatmap = axes[0].imshow(wendland_weights_for_plot.reshape(plotgrid_res_y,plotgrid_res_x),
        #                     cmap = colormap, vmin = min_w, vmax = max_w,
        #                     interpolation='nearest',
        #                     extent = [minX, maxX, maxY, minY])
        # axes[0].invert_yaxis()
        # # axes[0].scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.8)
        # axes[0].scatter(knots_x, knots_y, s = 30, color = 'white', marker = '+')
        # axes[0].set_xlim(minX, maxX)
        # axes[0].set_ylim(minY, maxY)
        # axes[0].set_aspect('equal', 'box')
        # axes[0].title.set_text('wendland weights knot ' + str(idx))

        # state_map.boundary.plot(ax=axes[1], color = 'black', linewidth = 0.5)
        # heatmap = axes[1].imshow(gaussian_weights_for_plot.reshape(plotgrid_res_y,plotgrid_res_x),
        #                     cmap = colormap, vmin = min_w, vmax = max_w,
        #                     interpolation='nearest',
        #                     extent = [minX, maxX, maxY, minY])
        # axes[1].invert_yaxis()
        # # axes[1].scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.8)
        # axes[1].scatter(knots_x, knots_y, s = 30, color = 'white', marker = '+')
        # axes[1].set_xlim(minX, maxX)
        # axes[1].set_ylim(minY, maxY)
        # axes[1].set_aspect('equal', 'box')
        # axes[1].title.set_text('gaussian weights knot ' + str(idx))

        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
        # fig.colorbar(heatmap, cax = cbar_ax, ticks = ticks)
        # plt.savefig('Plot_knot_weights.pdf')
        # plt.show()
        # plt.close()
        # # -------------------------------------------------------------------------------------------------------------

        # 0. USMap
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 8)

        # Plot boundary
        state_map.boundary.plot(ax=ax, color='lightgrey', zorder = 1)

        # Scatter plot for sites and knots
        ax.scatter(sites_x, sites_y, marker='.', c='blue', edgecolor = 'white', label='sites', zorder = 2)

        # Plot spatial domain rectangle
        space_rectangle = plt.Rectangle(xy=(minX, minY), width=maxX-minX, height=maxY-minY,
                                        fill=False, color='black')
        ax.add_patch(space_rectangle)

        # Set ticks and labels
        # ax.set_xticks(np.linspace(minX, maxX, num=3))
        # ax.set_yticks(np.linspace(minY, maxY, num=5))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Longitude', fontsize=20)
        plt.ylabel('Latitude', fontsize=20)

        # Set limits to match the exact range of your data
        plt.xlim([-130, -65])
        plt.ylim([25,50])

        # Save or show plot
        plt.savefig('Plot_US.pdf', bbox_inches="tight")
        # plt.show()
        plt.close()

        # 1. Station, Knots
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 8)

        # Plot knots and circles
        for i in range(k):
            circle_i = plt.Circle((knots_xy[i, 0], knots_xy[i, 1]), radius_from_knots[i],
                                color='r', fill=True, fc='grey', ec='None', alpha=0.2)
            ax.add_patch(circle_i)

        # Scatter plot for sites and knots
        ax.scatter(sites_x, sites_y, marker='.', c='blue', label='sites')
        ax.scatter(knots_x, knots_y, marker='+', c='red', label='knot', s=300)

        # Plot spatial domain rectangle
        space_rectangle = plt.Rectangle(xy=(minX, minY), width=maxX-minX, height=maxY-minY,
                                        fill=False, color='black')
        ax.add_patch(space_rectangle)

        # Set ticks and labels
        ax.set_xticks(np.linspace(minX, maxX, num=3))
        ax.set_yticks(np.linspace(minY, maxY, num=5))
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel('Longitude', fontsize=40)
        plt.ylabel('Latitude', fontsize=40)

        # Set limits to match the exact range of your data
        plt.xlim([-106, -88])
        plt.ylim([28,48])

        # Plot boundary
        state_map.boundary.plot(ax=ax, color='black')
        ax.set_aspect('equal', 'box')  # Ensures 1:1 ratio for data units

        # Adjust the position of the legend to avoid overlap with the plot
        box = ax.get_position()
        legend_elements = [matplotlib.lines.Line2D([0], [0], marker= '.', linestyle='None', color='b', label='Site'),
                        matplotlib.lines.Line2D([0], [0], marker='+', linestyle = "None", color='red', label='Knot Center',  markersize=30),
                        matplotlib.lines.Line2D([0], [0], marker = 'o', linestyle = 'None', label = 'Knot Radius', markerfacecolor = 'grey', markersize = 30, alpha = 0.2),
                        matplotlib.lines.Line2D([], [], color='None', marker='s', linestyle='None', markeredgecolor = 'black', markersize=30, label='Spatial Domain')]
        plt.legend(handles = legend_elements, bbox_to_anchor=(1.01,1.01), fontsize = 40)

        # Save or show plot
        plt.savefig('Plot_stations.pdf', bbox_inches="tight")
        plt.show()
        plt.close()


        # 2. Elevation
        fig, ax = plt.subplots()
        elev_scatter = ax.scatter(sites_x, sites_y, s=10, c = elevations,
                                  cmap = 'bwr')
        ax.set_aspect('equal', 'box')
        plt.colorbar(elev_scatter)
        plt.savefig('Plot_station_elevation.pdf')
        # plt.show()
        plt.close()


        # # 3. phi surface
        # # heatplot of phi surface
        # phi_vec_for_plot = (gaussian_weight_matrix_for_plot @ phi_at_knots).round(3)
        # graph, ax = plt.subplots()
        # heatmap = ax.imshow(phi_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x),
        #                     cmap ='bwr', interpolation='nearest', extent = [minX, maxX, maxY, minY])
        # ax.invert_yaxis()
        # graph.colorbar(heatmap)
        # plt.savefig('Plot_initial_heatmap_phi_surface.pdf')
        # # plt.show()
        # plt.close()


        # 4. Plot range surface
        # heatplot of range surface
        range_vec_for_plot = gaussian_weight_matrix_for_plot_rho @ range_at_knots
        graph, ax = plt.subplots()
        heatmap = ax.imshow(range_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x),
                            cmap ='bwr', interpolation='nearest', extent = [minX, maxX, maxY, minY])
        ax.invert_yaxis()
        graph.colorbar(heatmap)
        # plt.show()
        plt.savefig('Plot_initial_heatmap_rho_surface.pdf')
        # plt.show()
        plt.close()


        # 5. GEV Surfaces
        mu0_matrix      = (C_mu0.T @ Beta_mu0).T
        mu1_matrix      = (C_mu1.T @ Beta_mu1).T
        mu_matrix       = mu0_matrix + mu1_matrix * Time
        logsigma_matrix = (C_logsigma.T @ Beta_logsigma).T
        sigma_matrix    = np.exp(logsigma_matrix)
        ksi_matrix      = (C_ksi.T @ Beta_ksi).T

        def my_ceil(a, precision=0):
            return np.true_divide(np.ceil(a * 10**precision), 10**precision)

        def my_floor(a, precision=0):
            return np.true_divide(np.floor(a * 10**precision), 10**precision)

        # Location # -------------------------------------------------------------------------------------
        ## mu0(s) plot stations
        vmin = min(np.floor(min(mu0_estimates)), np.floor(min(mu0_matrix[:,0])))
        vmax = max(np.ceil([max(mu0_estimates), max(mu0_matrix[:,0])]))
        divnorm = matplotlib.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)
        fig, ax = plt.subplots(1,2)
        mu0_scatter = ax[0].scatter(sites_x, sites_y, s = 10, cmap = 'bwr', c = mu0_estimates, norm = divnorm)
        ax[0].set_aspect('equal', 'box')
        ax[0].title.set_text('GEV mu0 estimates')
        mu0_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, cmap = 'bwr', c = mu0_matrix[:,0], norm = divnorm)
        ax[1].set_aspect('equal','box')
        ax[1].title.set_text('spline mu0 fit')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(mu0_est_scatter, cax = cbar_ax)
        plt.savefig('Plot_initial_mu0_estimates.pdf')
        plt.close()

        ## mu1(s) plot stations
        vmin = min(np.floor(min(mu1_estimates)), np.floor(min(mu1_matrix[:,0])))
        vmax = max(np.ceil([max(mu1_estimates), max(mu1_matrix[:,0])]))
        divnorm = matplotlib.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)
        fig, ax = plt.subplots(1,2)
        mu1_scatter = ax[0].scatter(sites_x, sites_y, s = 10, cmap = 'bwr', c = mu1_estimates, norm = divnorm)
        ax[0].set_aspect('equal', 'box')
        ax[0].title.set_text('GEV mu1 estimates')
        mu1_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, cmap = 'bwr', c = mu1_matrix[:,0], norm = divnorm)
        ax[1].set_aspect('equal','box')
        ax[1].title.set_text('spline mu1 fit')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(mu1_est_scatter, cax = cbar_ax)
        plt.savefig('Plot_initial_mu1_estimates.pdf')
        plt.close()

        # Scale # -------------------------------------------------------------------------------------
        ## logsigma(s) plot stations
        vmin = min(my_floor(min(logsigma_estimates), 1), my_floor(min(logsigma_matrix[:,0]), 1))
        vmax = max(my_ceil(max(logsigma_estimates), 1), my_ceil(max(logsigma_matrix[:,0]), 1))
        divnorm = matplotlib.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)
        fig, ax = plt.subplots(1,2)
        logsigma_scatter = ax[0].scatter(sites_x, sites_y, s = 10, cmap = 'bwr', c = logsigma_estimates, norm = divnorm)
        ax[0].set_aspect('equal', 'box')
        ax[0].title.set_text('GEV logsigma estimates')
        logsigma_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, cmap = 'bwr', c = logsigma_matrix[:,0], norm = divnorm)
        ax[1].set_aspect('equal','box')
        ax[1].title.set_text('spline logsigma fit')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(logsigma_est_scatter, cax = cbar_ax)
        plt.savefig('Plot_initial_logsigma_estimates.pdf')
        plt.close()

        # Shape # -------------------------------------------------------------------------------------
        # ksi(s) plot stations
        vmin = min(my_floor(min(ksi_estimates), 1), my_floor(min(ksi_matrix[:,0]), 1))
        vmax = max(my_ceil(max(ksi_estimates), 1), my_ceil(max(ksi_matrix[:,0]), 1))
        divnorm = matplotlib.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)
        fig, ax = plt.subplots(1,2)
        ksi_scatter = ax[0].scatter(sites_x, sites_y, s = 10, cmap = 'bwr', c = ksi_estimates, norm = divnorm)
        ax[0].set_aspect('equal', 'box')
        ax[0].title.set_text('GEV ksi estimates')
        ksi_est_scatter = ax[1].scatter(sites_x, sites_y, s = 10, cmap = 'bwr', c = ksi_matrix[:,0], norm = divnorm)
        ax[1].set_aspect('equal','box')
        ax[1].title.set_text('spline ksi fit')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(ksi_est_scatter, cax = cbar_ax)
        plt.savefig('Plot_initial_ksi_estimates.pdf')
        plt.close()

    # %% MCMC Parameters
    # MCMC Parameters
    ##########################################################################################################
    ########### MCMC Parameters ##############################################################################
    ##########################################################################################################

    # ----------------------------------------------------------------------------------------------------------------
    # Block Update Specification

    if norm_pareto == 'standard':
        phi_block_idx_size   = 1
        range_block_idx_size = 1

    if norm_pareto == 'shifted':
        phi_block_idx_size = 4
        range_block_idx_size = 1

    Beta_mu0_block_idx_size = 4
    Beta_mu1_block_idx_size = 4

    # Create Coefficient Index Blocks - each block size does not exceed size specified above

    # ## phi
    # phi_block_idx_dict = {}
    # lst = list(range(k))
    # for i in range(0, k, phi_block_idx_size):
    #     start_index = i
    #     end_index   = i + phi_block_idx_size
    #     key         = 'phi_block_idx_'+str(i//phi_block_idx_size+1)
    #     phi_block_idx_dict[key] = lst[start_index:end_index]

    ## range
    range_block_idx_dict = {}
    lst = list(range(k_rho))
    for i in range(0, k_rho, range_block_idx_size):
        start_index = i
        end_index   = i + range_block_idx_size
        key         = 'range_block_idx_'+str(i//range_block_idx_size+1)
        range_block_idx_dict[key] = lst[start_index:end_index]

    ## Beta_mu0
    Beta_mu0_block_idx_dict = {}
    lst = list(range(Beta_mu0_m))
    for i in range(0, Beta_mu0_m, Beta_mu0_block_idx_size):
        start_index = i
        end_index   = i + Beta_mu0_block_idx_size
        key         = 'Beta_mu0_block_idx_'+str(i//Beta_mu0_block_idx_size+1)
        Beta_mu0_block_idx_dict[key] = lst[start_index:end_index]

    ## Beta_mu1
    Beta_mu1_block_idx_dict = {}
    lst = list(range(Beta_mu1_m))
    for i in range(0, Beta_mu1_m, Beta_mu1_block_idx_size):
        start_index = i
        end_index   = i + Beta_mu1_block_idx_size
        key         = 'Beta_mu1_block_idx_'+str(i//Beta_mu1_block_idx_size + 1)
        Beta_mu1_block_idx_dict[key] = lst[start_index:end_index]

    # ----------------------------------------------------------------------------------------------------------------
    # Adaptive Update: tuning constants

    c_0 = 1
    c_1 = 0.8
    offset = 3 # the iteration offset: trick the updater thinking chain is longer
    # r_opt_1d = .41
    # r_opt_2d = .35
    # r_opt = 0.234 # asymptotically
    r_opt = .35
    adapt_size = 10

    # ----------------------------------------------------------------------------------------------------------------
    # Adaptive Update: Proposal Variance Scalar and Covariance Matrix

    if start_iter == 1: # initialize the adaptive update necessities
        # with no trial run
        # phi_cov                 = 1e-2 * np.identity(k)
        range_cov               = 0.5  * np.identity(k)
        Beta_mu0_cov            = 1e-2 * np.identity(Beta_mu0_m)
        Beta_mu1_cov            = 1e-2 * np.identity(Beta_mu1_m)
        Beta_logsigma_cov       = 1e-6 * np.identity(Beta_logsigma_m)
        Beta_ksi_cov            = 1e-7 * np.identity(Beta_ksi_m)
        sigma_Beta_mu0_cov      = 1
        sigma_Beta_mu1_cov      = 1
        sigma_Beta_logsigma_cov = 1
        sigma_Beta_ksi_cov      = 1
        # R_log_cov               = np.tile(((2.4**2)/k)*np.eye(k)[:,:,None], reps = (1,1,Nt))

        # with trial run
        # if proposal_cov.phi_cov is not None:                 phi_cov                 = proposal_cov.phi_cov
        if proposal_cov.range_cov is not None:               range_cov               = proposal_cov.range_cov
        if proposal_cov.Beta_mu0_cov is not None:            Beta_mu0_cov            = proposal_cov.Beta_mu0_cov
        if proposal_cov.Beta_mu1_cov is not None:            Beta_mu1_cov            = proposal_cov.Beta_mu1_cov
        if proposal_cov.Beta_logsigma_cov is not None:       Beta_logsigma_cov       = proposal_cov.Beta_logsigma_cov
        if proposal_cov.Beta_ksi_cov is not None:            Beta_ksi_cov            = proposal_cov.Beta_ksi_cov
        if proposal_cov.sigma_Beta_mu0_cov is not None:      sigma_Beta_mu0_cov      = proposal_cov.sigma_Beta_mu0_cov
        if proposal_cov.sigma_Beta_mu1_cov is not None:      sigma_Beta_mu1_cov      = proposal_cov.sigma_Beta_mu1_cov
        if proposal_cov.sigma_Beta_logsigma_cov is not None: sigma_Beta_logsigma_cov = proposal_cov.sigma_Beta_logsigma_cov
        if proposal_cov.sigma_Beta_ksi_cov is not None:      sigma_Beta_ksi_cov      = proposal_cov.sigma_Beta_ksi_cov
        # if proposal_cov.R_log_cov is not None:               R_log_cov               = proposal_cov.R_log_cov

        # assert k               == phi_cov.shape[0]
        assert k               == range_cov.shape[0]
        # assert k               == R_log_cov.shape[0]
        # assert Nt              == R_log_cov.shape[2]
        assert Beta_mu0_m      == Beta_mu0_cov.shape[0]
        assert Beta_mu1_m      == Beta_mu1_cov.shape[0]
        assert Beta_logsigma_m == Beta_logsigma_cov.shape[0]
        assert Beta_ksi_m      == Beta_ksi_cov.shape[0]

        # make parameter block (for block updates)
        # ## phi
        # phi_block_cov_dict = {}
        # for key in phi_block_idx_dict.keys():
        #     start_idx                    = phi_block_idx_dict[key][0]
        #     end_idx                      = phi_block_idx_dict[key][-1]+1
        #     phi_block_cov_dict[key] = phi_cov[start_idx:end_idx, start_idx:end_idx]

        ## range rho
        range_block_cov_dict = {}
        for key in range_block_idx_dict.keys():
            start_idx                      = range_block_idx_dict[key][0]
            end_idx                        = range_block_idx_dict[key][-1]+1
            range_block_cov_dict[key] = range_cov[start_idx:end_idx, start_idx:end_idx]

        ## Beta_mu0
        Beta_mu0_block_cov_dict = {}
        for key in Beta_mu0_block_idx_dict.keys():
            start_idx                         = Beta_mu0_block_idx_dict[key][0]
            end_idx                           = Beta_mu0_block_idx_dict[key][-1]+1
            Beta_mu0_block_cov_dict[key] = Beta_mu0_cov[start_idx:end_idx, start_idx:end_idx]

        ## Beta_mu1
        Beta_mu1_block_cov_dict          = {}
        for key in Beta_mu1_block_idx_dict.keys():
            start_idx                         = Beta_mu1_block_idx_dict[key][0]
            end_idx                           = Beta_mu1_block_idx_dict[key][-1]+1
            Beta_mu1_block_cov_dict[key] = Beta_mu1_cov[start_idx:end_idx, start_idx:end_idx]


        if rank == 0: # Handle phi, range, GEV on Worker 0
            # proposal variance scalar
            sigma_m_sq = {
                'Beta_logsigma'       : (2.4**2)/Beta_logsigma_m,
                'Beta_ksi'            : (2.4**2)/Beta_ksi_m,
                'sigma_Beta_mu0'      : sigma_Beta_mu0_cov,
                'sigma_Beta_mu1'      : sigma_Beta_mu1_cov,
                'sigma_Beta_logsigma' : sigma_Beta_logsigma_cov,
                'sigma_Beta_ksi'      : sigma_Beta_ksi_cov
            }
            # for key in phi_block_idx_dict.keys():
                # sigma_m_sq[key] = (2.4**2)/len(phi_block_idx_dict[key])
            for key in range_block_idx_dict.keys():
                sigma_m_sq[key] = (2.4**2)/len(range_block_idx_dict[key])
            for key in Beta_mu0_block_idx_dict.keys():
                sigma_m_sq[key] = (2.4**2)/len(Beta_mu0_block_idx_dict[key])
            for key in Beta_mu1_block_idx_dict.keys():
                sigma_m_sq[key] = (2.4**2)/len(Beta_mu1_block_idx_dict[key])

            # proposal covariance matrix
            Sigma_0 = {
                'Beta_logsigma' : Beta_logsigma_cov,
                'Beta_ksi'      : Beta_ksi_cov
            }
            # Sigma_0.update(phi_block_cov_dict)
            Sigma_0.update(range_block_cov_dict)
            Sigma_0.update(Beta_mu0_block_cov_dict)
            Sigma_0.update(Beta_mu1_block_cov_dict)

        # Rt: each Worker_t propose k-R(t)s at time t
        # if rank == 0:
        #     if norm_pareto == 'shifted':
        #         sigma_m_sq_Rt_list = [np.mean(np.diag(R_log_cov[:,:,t])) for t in range(Nt)]
        #     if norm_pareto == 'standard':
        #         sigma_m_sq_Rt_list = [(np.diag(R_log_cov[:,:,t])) for t in range(Nt)]
        # else:
        #     sigma_m_sq_Rt_list = None
        # sigma_m_sq_Rt = comm.scatter(sigma_m_sq_Rt_list, root = 0) if size>1 else sigma_m_sq_Rt_list[0]
    else:
        # # start_iter != 1, pickle load the Proposal Variance Scalar, Covariance Matrix

        # ## Proposal Variance Scalar for Rt
        # if rank == 0:
        #     with open('sigma_m_sq_Rt_list.pkl', 'rb') as file:
        #         sigma_m_sq_Rt_list = pickle.load(file)
        # else:
        #     sigma_m_sq_Rt_list = None
        # if size != 1: sigma_m_sq_Rt = comm.scatter(sigma_m_sq_Rt_list, root = 0)

        ## Proposal Variance Scalar and Covariance Matrix for other variables
        if rank == 0:
            with open('sigma_m_sq.pkl','rb') as file:
                sigma_m_sq = pickle.load(file)
            with open('Sigma_0.pkl', 'rb') as file:
                Sigma_0    = pickle.load(file)

    # ----------------------------------------------------------------------------------------------------------------
    # Adaptive Update: Counter

    # ## Counter for Rt
    # if norm_pareto == 'shifted':
    #     num_accepted_Rt_list = [0] * size if rank == 0 else None
    #     if size != 1: num_accepted_Rt = comm.scatter(num_accepted_Rt_list, root = 0)

    # if norm_pareto == 'standard':
    #     num_accepted_Rt_list = [[0] * k] * size if rank == 0 else None
    #     if size > 1: num_accepted_Rt = comm.scatter(num_accepted_Rt_list, root = 0)
    #     if size == 1: num_accepted_Rt = num_accepted_Rt_list[0]

    ## Counter for other variables
    if rank == 0:
        num_accepted = { # acceptance counter
            'Beta_logsigma'       : 0,
            'Beta_ksi'            : 0,
            'sigma_Beta_mu0'      : 0,
            'sigma_Beta_mu1'      : 0,
            'sigma_Beta_logsigma' : 0,
            'sigma_Beta_ksi'      : 0
        }
        # for key in phi_block_idx_dict.keys():
        #     num_accepted[key] = 0
        for key in range_block_idx_dict.keys():
            num_accepted[key] = 0
        for key in Beta_mu0_block_idx_dict.keys():
            num_accepted[key] = 0
        for key in Beta_mu1_block_idx_dict.keys():
            num_accepted[key] = 0

    # 8. Storage for Traceplots -----------------------------------------------

    if start_iter == 1:
        if rank == 0:
            loglik_trace              = np.full(shape = (n_iters, 1), fill_value = np.nan) # overall likelihood
            loglik_detail_trace       = np.full(shape = (n_iters, 2), fill_value = np.nan) # detail likelihood
            # R_trace_log               = np.full(shape = (n_iters, k, Nt), fill_value = np.nan) # log(R)
            # phi_knots_trace           = np.full(shape = (n_iters, k), fill_value = np.nan) # phi_at_knots
            range_knots_trace         = np.full(shape = (n_iters, k_rho), fill_value = np.nan) # range_at_knots
            Beta_mu0_trace            = np.full(shape = (n_iters, Beta_mu0_m), fill_value = np.nan) # mu0 Covariate Coefficients
            Beta_mu1_trace            = np.full(shape = (n_iters, Beta_mu1_m), fill_value = np.nan) # mu1 covariate Coefficients
            Beta_logsigma_trace       = np.full(shape = (n_iters, Beta_logsigma_m), fill_value = np.nan) # logsigma Covariate Coefficients
            Beta_ksi_trace            = np.full(shape=(n_iters, Beta_ksi_m), fill_value = np.nan) # ksi Covariate Coefficients
            sigma_Beta_mu0_trace      = np.full(shape=(n_iters, 1), fill_value = np.nan) # prior sd for beta_mu0's
            sigma_Beta_mu1_trace      = np.full(shape=(n_iters, 1), fill_value = np.nan) # prior sd for beta_mu1's
            sigma_Beta_logsigma_trace = np.full(shape=(n_iters, 1), fill_value = np.nan) # prior sd for beta_logsigma's
            sigma_Beta_ksi_trace      = np.full(shape = (n_iters, 1), fill_value = np.nan) # prior sd for beta_ksi's
            Y_trace                   = np.full(shape = (n_iters, Ns, Nt), fill_value = np.nan)
        else:
            loglik_trace              = None
            loglik_detail_trace       = None
            # R_trace_log               = None
            # phi_knots_trace           = None
            range_knots_trace         = None
            Beta_mu0_trace            = None
            Beta_mu1_trace            = None
            Beta_logsigma_trace       = None
            Beta_ksi_trace            = None
            sigma_Beta_mu0_trace      = None
            sigma_Beta_mu1_trace      = None
            sigma_Beta_logsigma_trace = None
            sigma_Beta_ksi_trace      = None
            Y_trace                   = None
    else:
        if rank == 0:
            loglik_trace              = np.load('loglik_trace.npy')
            loglik_detail_trace       = np.load('loglik_detail_trace.npy')
            # R_trace_log               = np.load('R_trace_log.npy')
            # phi_knots_trace           = np.load('phi_knots_trace.npy')
            range_knots_trace         = np.load('range_knots_trace.npy')
            Beta_mu0_trace            = np.load('Beta_mu0_trace.npy')
            Beta_mu1_trace            = np.load('Beta_mu1_trace.npy')
            Beta_logsigma_trace       = np.load('Beta_logsigma_trace.npy')
            Beta_ksi_trace            = np.load('Beta_ksi_trace.npy')
            sigma_Beta_mu0_trace      = np.load('sigma_Beta_mu0_trace.npy')
            sigma_Beta_mu1_trace      = np.load('sigma_Beta_mu1_trace.npy')
            sigma_Beta_logsigma_trace = np.load('sigma_Beta_logsigma_trace.npy')
            sigma_Beta_ksi_trace      = np.load('sigma_Beta_ksi_trace.npy')
            Y_trace                   = np.load('Y_trace.npy')
        else:
            loglik_trace              = None
            loglik_detail_trace       = None
            # R_trace_log               = None
            # phi_knots_trace           = None
            range_knots_trace         = None
            Beta_mu0_trace            = None
            Beta_mu1_trace            = None
            Beta_logsigma_trace       = None
            Beta_ksi_trace            = None
            sigma_Beta_mu0_trace      = None
            sigma_Beta_mu1_trace      = None
            sigma_Beta_logsigma_trace = None
            sigma_Beta_ksi_trace      = None
            Y_trace                   = None
    # Initialize Parameters -------------------------------------------------------------------------------------

    if start_iter == 1:
        # Initialize at the truth/at other values
        # R_matrix_init_log        = np.log(R_at_knots)  if rank == 0 else None
        # phi_knots_init           = phi_at_knots        if rank == 0 else None
        range_knots_init         = range_at_knots      if rank == 0 else None
        Beta_mu0_init            = Beta_mu0            if rank == 0 else None
        Beta_mu1_init            = Beta_mu1            if rank == 0 else None
        Beta_logsigma_init       = Beta_logsigma       if rank == 0 else None
        Beta_ksi_init            = Beta_ksi            if rank == 0 else None
        sigma_Beta_mu0_init      = sigma_Beta_mu0      if rank == 0 else None
        sigma_Beta_mu1_init      = sigma_Beta_mu1      if rank == 0 else None
        sigma_Beta_logsigma_init = sigma_Beta_logsigma if rank == 0 else None
        sigma_Beta_ksi_init      = sigma_Beta_ksi      if rank == 0 else None
        Y_init                   = Y                   if rank == 0 else None
        if rank == 0: # store initial value into first row of traceplot
            # R_trace_log[0,:,:]             = R_matrix_init_log # matrix (k, Nt)
            # phi_knots_trace[0,:]           = phi_knots_init
            range_knots_trace[0,:]         = range_knots_init
            Beta_mu0_trace[0,:]            = Beta_mu0_init
            Beta_mu1_trace[0,:]            = Beta_mu1_init
            Beta_logsigma_trace[0,:]       = Beta_logsigma_init
            Beta_ksi_trace[0,:]            = Beta_ksi_init
            sigma_Beta_mu0_trace[0,:]      = sigma_Beta_mu0_init
            sigma_Beta_mu1_trace[0,:]      = sigma_Beta_mu1_init
            sigma_Beta_logsigma_trace[0,:] = sigma_Beta_logsigma_init
            sigma_Beta_ksi_trace[0,:]      = sigma_Beta_ksi_init
            Y_trace[0,:,:]                 = Y_init
    else:
        last_iter = start_iter - 1
        # R_matrix_init_log        = R_trace_log[last_iter,:,:]             if rank == 0 else None
        # phi_knots_init           = phi_knots_trace[last_iter,:]           if rank == 0 else None
        range_knots_init         = range_knots_trace[last_iter,:]         if rank == 0 else None
        Beta_mu0_init            = Beta_mu0_trace[last_iter,:]            if rank == 0 else None
        Beta_mu1_init            = Beta_mu1_trace[last_iter,:]            if rank == 0 else None
        Beta_logsigma_init       = Beta_logsigma_trace[last_iter,:]       if rank == 0 else None
        Beta_ksi_init            = Beta_ksi_trace[last_iter,:]            if rank == 0 else None
        ## Note: these sigma_Beta_xx must be values, can't be arrays
        sigma_Beta_mu0_init      = sigma_Beta_mu0_trace[last_iter,0]      if rank == 0 else None
        sigma_Beta_mu1_init      = sigma_Beta_mu1_trace[last_iter,0]      if rank == 0 else None
        sigma_Beta_logsigma_init = sigma_Beta_logsigma_trace[last_iter,0] if rank == 0 else None
        sigma_Beta_ksi_init      = sigma_Beta_ksi_trace[last_iter,0]      if rank == 0 else None
        Y_init                   = Y_trace[last_iter,:,:]                 if rank == 0 else None


    # Set Current Values
    ## ---- log(R) --------------------------------------------------------------------------------------------
    # note: directly comm.scatter an numpy nd array along an axis is tricky,
    #       hence we first "redundantly" broadcast an entire R_matrix then split
    # R_matrix_init_log = comm.bcast(R_matrix_init_log, root = 0) # matrix (k, Nt)
    # R_current_log     = np.array(R_matrix_init_log[:,rank]) # vector (k,)
    # R_vec_current     = wendland_weight_matrix @ np.exp(R_current_log)

    ## ---- phi ------------------------------------------------------------------------------------------------
    # phi_knots_current = comm.bcast(phi_knots_init, root = 0)
    # phi_vec_current   = gaussian_weight_matrix @ phi_knots_current

    ## ---- range_vec (length_scale) ---------------------------------------------------------------------------
    range_knots_current = comm.bcast(range_knots_init, root = 0)
    range_vec_current   = gaussian_weight_matrix_rho @ range_knots_current
    K_current           = ns_cov(range_vec = range_vec_current,
                                 sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = "matern")
    cholesky_matrix_current = scipy.linalg.cholesky(K_current, lower = False)

    ## ---- GEV covariate coefficients --> GEV surface ----------------------------------------------------------
    Beta_mu0_current      = comm.bcast(Beta_mu0_init, root = 0)
    Beta_mu1_current      = comm.bcast(Beta_mu1_init, root = 0)
    Beta_logsigma_current = comm.bcast(Beta_logsigma_init, root = 0)
    Beta_ksi_current      = comm.bcast(Beta_ksi_init, root = 0)
    Loc_matrix_current    = (C_mu0.T @ Beta_mu0_current).T + (C_mu1.T @ Beta_mu1_current).T * Time
    Scale_matrix_current  = np.exp((C_logsigma.T @ Beta_logsigma_current).T)
    Shape_matrix_current  = (C_ksi.T @ Beta_ksi_current).T

    ## ---- GEV covariate coefficients prior variance -----------------------------------------------------------
    sigma_Beta_mu0_current      = comm.bcast(sigma_Beta_mu0_init, root = 0)
    sigma_Beta_mu1_current      = comm.bcast(sigma_Beta_mu1_init, root = 0)
    sigma_Beta_logsigma_current = comm.bcast(sigma_Beta_logsigma_init, root = 0)
    sigma_Beta_ksi_current      = comm.bcast(sigma_Beta_ksi_init, root = 0)

    ## ---- Y (Ns, Nt) ------------------------------------------------------------------------------------------------

    Y = comm.bcast(Y_init, root = 0)

    ## ---- X_star ----------------------------------------------------------------------------------------------------

    # miss_vec_1t   = miss_matrix[:,rank]
    # miss_index_1t = np.where(miss_vec_1t == True)[0]
    # obs_index_1t  = np.where(miss_vec_1t == False)[0]

    # if start_iter == 1: # initial imputation
    #     X_star_1t_current                = np.full(shape = (Ns,), fill_value = np.nan) # contain missing values
    #     X_star_1t_current[obs_index_1t]  = qRW(pgev(Y[:,rank][obs_index_1t],
    #                                             Loc_matrix_current[obs_index_1t,rank],
    #                                             Scale_matrix_current[obs_index_1t,rank],
    #                                             Shape_matrix_current[obs_index_1t,rank]),
    #                                         phi_vec_current[obs_index_1t], gamma_vec[obs_index_1t])
    #     X_star_1t_miss, Y_1t_miss        = impute_1t(miss_index_1t, obs_index_1t,
    #                                                 Y[:,rank], X_star_1t_current,
    #                                                 Loc_matrix_current[:, rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
    #                                                 phi_vec_current, gamma_vec, R_vec_current, K_current)
    #     X_star_1t_current[miss_index_1t] = X_star_1t_miss
    #     Y[:,rank][miss_index_1t]         = Y_1t_miss # this will modify Y[:,rank] because Y_1t_current shallow copy

    #     Y_gathered = comm.gather(Y[:,rank], root = 0)
    #     if rank == 0:
    #         Y_trace[0,:,:] = np.array(Y_gathered).T

    #     assert len(np.where(np.isnan(Y[:,rank]))[0]) == 0
    # else:
    #     X_star_1t_current = qRW(pgev(Y[:,rank],
    #                                     Loc_matrix_current[:,rank],
    #                                     Scale_matrix_current[:,rank],
    #                                     Shape_matrix_current[:,rank]),
    #                                 phi_vec_current, gamma_vec)

    ## ---- initial imputation ----------------------------------------------------------------------------------------

    miss_vec_1t   = miss_matrix[:,rank]
    miss_index_1t = np.where(miss_vec_1t == True)[0]
    obs_index_1t  = np.where(miss_vec_1t == False)[0]

    if start_iter == 1:
        Z_1t_current = np.full(shape = (Ns, ), fill_value = np.nan) # contain missing values
        Z_1t_current[obs_index_1t] = qZ(pgev(Y[:,rank][obs_index_1t],
                                            Loc_matrix_current[obs_index_1t,rank],
                                            Scale_matrix_current[obs_index_1t,rank],
                                            Shape_matrix_current[obs_index_1t,rank]))
        Z_1t_miss, Y_1t_miss = impute_ZY_1t(miss_index_1t, obs_index_1t, Z_1t_current,
                                            Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank], K_current)
        Z_1t_current[miss_index_1t] = Z_1t_miss
        Y[:,rank][miss_index_1t]    = Y_1t_miss
        assert len(np.where(np.isnan(Y[:,rank]))[0]) == 0
        if len(np.where(np.isinf(Z_1t_current))[0]) > 0:
            print('Z_1t_current contains infs, rank:', rank)
            print(np.where(np.isinf(Z_1t_current))[0])

        Y_gathered = comm.gather(Y[:,rank], root = 0)
        if rank == 0:
            Y_trace[0,:,:] = np.array(Y_gathered).T
    else:
        Z_1t_current = qZ(pgev(Y[:,rank],
                                Loc_matrix_current[:,rank],
                                Scale_matrix_current[:,rank],
                                Shape_matrix_current[:,rank]))



    # %% Metropolis-Hasting Updates
    # Metropolis-Hasting Updates
    #####################################################################################################################
    ########### Metropolis-Hasting Updates ##############################################################################
    #####################################################################################################################

    comm.Barrier() # Blocking before the update starts

    if rank == 0:
        start_time = time.time()
        print('started on:', strftime('%Y-%m-%d %H:%M:%S', localtime(time.time())))
    # lik_1t_current = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current,
    #                                                             Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
    #                                                             phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
    # prior_1t_current = np.sum(scipy.stats.levy.logpdf(np.exp(R_current_log), scale = gamma) + R_current_log)
    # if not np.isfinite(lik_1t_current) or not np.isfinite(prior_1t_current):
    #     print('initial values lead to none finite likelihood')
    #     print('rank:',rank)
    #     print('lik_1t_current:',lik_1t_current)
    #     print('prior_1t_current of R:',prior_1t_current)

    lik_1t_current = likelihood_1t(Y[:,rank], Z_1t_current,
                                    Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                    cholesky_matrix_current)
    if not np.isfinite(lik_1t_current):
        print('initial values lead to none finite likelihood')
        print('rank:',rank)
        print('lik_1t_current:',lik_1t_current)

    for iter in range(start_iter, n_iters):
        # %% Update range
        #########################################################################################
        ####  Update range_at_knots  ############################################################
        #########################################################################################

        # Update range ACTUALLY in blocks
        for key in range_block_idx_dict.keys():
            change_indices = np.array(range_block_idx_dict[key])

            # Propose new range_block at the change indices
            if rank == 0:
                range_knots_proposal                  = range_knots_current.copy()
                range_knots_proposal[change_indices] += np.sqrt(sigma_m_sq[key])* \
                                                        random_generator.multivariate_normal(np.zeros(len(change_indices)), Sigma_0[key])
            else:
                range_knots_proposal = None
            range_knots_proposal     = comm.bcast(range_knots_proposal, root = 0)
            range_vec_proposal       = gaussian_weight_matrix_rho @ range_knots_proposal

            # Conditional log likelihood at proposal
            if any(range <= 0 for range in range_knots_proposal):
                lik_1t_proposal = np.NINF
            else:
                K_proposal = ns_cov(range_vec = range_vec_proposal,
                                    sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = "matern")
                cholesky_matrix_proposal = scipy.linalg.cholesky(K_proposal, lower = False)
                lik_1t_proposal = likelihood_1t(Y[:,rank], Z_1t_current,
                                                Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                cholesky_matrix_proposal)

            # Gather likelihood calculated across time (no prior yet)
            lik_current_gathered  = comm.gather(lik_1t_current, root = 0)
            lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

            # Handle prior and Accept/Reject on worker 0
            if rank == 0:
                # use Half-Normal prior on each one of the range parameters in the block
                lik_current  = sum(lik_current_gathered)  + np.sum(scipy.stats.halfnorm.logpdf(range_knots_current, loc = 0, scale = 2))
                lik_proposal = sum(lik_proposal_gathered) + np.sum(scipy.stats.halfnorm.logpdf(range_knots_proposal, loc = 0, scale = 2))

                # Accept or Reject
                u     = random_generator.uniform()
                ratio = np.exp(lik_proposal - lik_current)
                if not np.isfinite(ratio):
                    ratio = 0 # Force a rejection
                if u > ratio: # Reject
                    range_accepted     = False
                    range_vec_update   = range_vec_current
                    range_knots_update = range_knots_current
                else:
                    range_accepted     = True
                    range_vec_update   = range_vec_proposal
                    range_knots_update = range_knots_proposal
                    num_accepted[key] += 1

                # Store the result
                range_knots_trace[iter,:] = range_knots_update

                # Update the current value
                range_vec_current   = range_vec_update
                range_knots_current = range_knots_update
            else: # Broadcast the update values
                range_accepted  = None
            range_vec_current   = comm.bcast(range_vec_current, root = 0)
            range_knots_current = comm.bcast(range_knots_current, root = 0)
            range_accepted      = comm.bcast(range_accepted, root = 0)

            # Update the K, cholesky_matrix, and likelihood
            if range_accepted:
                K_current               = K_proposal.copy()
                cholesky_matrix_current = cholesky_matrix_proposal.copy()
                lik_1t_current          = lik_1t_proposal

            comm.Barrier() # block for range_block updates

        # %% Update Missing Values
        #########################################################################################
        ####  Update missing values  ############################################################
        #########################################################################################

        # draw new Z_miss, Y_miss
        Z_1t_miss, Y_1t_miss = impute_ZY_1t(miss_index_1t, obs_index_1t, Z_1t_current,
                                            Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank], K_current)
        Z_1t_current[miss_index_1t] = Z_1t_miss
        Y[:,rank][miss_index_1t]    = Y_1t_miss
        if len(np.where(np.isinf(Z_1t_current))[0]) > 0:
            print('Z_1t_current contains infs, rank:', rank)
            print(np.where(np.isinf(Z_1t_current))[0])

        lik_1t_current = likelihood_1t(Y[:,rank], Z_1t_current,
                                        Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                        cholesky_matrix_current)

        Y_gathered = comm.gather(Y[:,rank], root = 0)
        if rank == 0:
            Y_trace[iter,:,:] = np.array(Y_gathered).T

        comm.Barrier()

        # %% Update Beta_mu0
        #############################################################
        #### ----- Update covariate coefficients, Beta_mu0 ----- ####
        #############################################################

        # Update by blocks
        for key in Beta_mu0_block_idx_dict.keys():
            change_indices   = np.array(Beta_mu0_block_idx_dict[key])
            unchange_indices = np.array([x for x in range(Beta_mu0_m) if x not in change_indices])

            # Propose new Beta_mu0 at the change_indices
            if rank == 0:
                # Beta_mu0 in this block share a same proposal scale, has proposal variance matrix
                Beta_mu0_proposal                 = Beta_mu0_current.copy()
                Beta_mu0_proposal[change_indices] += np.sqrt(sigma_m_sq[key]) * \
                                                    random_generator.multivariate_normal(np.zeros(len(change_indices)), Sigma_0[key])
            else:
                Beta_mu0_proposal = None
            Beta_mu0_proposal     = comm.bcast(Beta_mu0_proposal, root = 0)
            # Loc_matrix_proposal   = (C_mu0.T @ Beta_mu0_proposal).T
            Loc_matrix_proposal   = (C_mu0.T @ Beta_mu0_proposal).T + (C_mu1.T @ Beta_mu1_current).T * Time

            # Conditional log likelihood at proposal
            Z_1t_proposal = qZ(pgev(Y[:,rank], Loc_matrix_proposal[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]))
            if any(np.isinf(Z_1t_proposal)) or any(np.isnan(Z_1t_proposal)):
                lik_1t_proposal = np.NINF
            else:
                lik_1t_proposal = likelihood_1t(Y[:,rank], Z_1t_proposal,
                                                Loc_matrix_proposal[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                cholesky_matrix_current)

            # Gather likelihood calculated across time
            lik_current_gathered  = comm.gather(lik_1t_current, root = 0)
            lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

            # Handle prior and (Accept or Reject) on worker 0
            if rank == 0:
                # use Norm(0, sigma_Beta_mu0) prior on each Beta_mu0, like "shrinkage"
                prior_Beta_mu0_current  = scipy.stats.norm.logpdf(Beta_mu0_current, loc=0, scale=sigma_Beta_mu0_current)
                prior_Beta_mu0_proposal = scipy.stats.norm.logpdf(Beta_mu0_proposal, loc=0, scale=sigma_Beta_mu0_current)

                lik_current  = sum(lik_current_gathered)  + sum(prior_Beta_mu0_current)
                lik_proposal = sum(lik_proposal_gathered) + sum(prior_Beta_mu0_proposal)

                # Accept or Reject
                u = random_generator.uniform()
                ratio = np.exp(lik_proposal - lik_current)
                if not np.isfinite(ratio):
                    ratio = 0
                if u > ratio: # Reject
                    Beta_mu0_accepted = False
                    Beta_mu0_update   = Beta_mu0_current
                else: # Accept
                    Beta_mu0_accepted = True
                    Beta_mu0_update   = Beta_mu0_proposal
                    # num_accepted['Beta_mu0'] += 1
                    num_accepted[key] += 1

                # Store the result
                Beta_mu0_trace[iter,:] = Beta_mu0_update

                # Update the current value
                Beta_mu0_current = Beta_mu0_update
            else: # other workers
                Beta_mu0_accepted = None

            # Broadcast the updated values
            Beta_mu0_accepted = comm.bcast(Beta_mu0_accepted, root = 0)
            Beta_mu0_current  = comm.bcast(Beta_mu0_current, root = 0)

            # Update X_star, mu0 surface, and likelihood
            if Beta_mu0_accepted:
                Z_1t_current = Z_1t_proposal.copy()
                # Loc_matrix_current = (C_mu0.T @ Beta_mu0_current).T
                Loc_matrix_current = (C_mu0.T @ Beta_mu0_current).T + (C_mu1.T @ Beta_mu1_current).T * Time
                lik_1t_current     = lik_1t_proposal

            comm.Barrier() # block for beta_mu0 updates

        # %% Update Beta_mu1
        #############################################################
        #### ----- Update covariate coefficients, Beta_mu1 ----- ####
        #############################################################

        # Update by blocks
        for key in Beta_mu1_block_idx_dict.keys():
            change_indices = np.array(Beta_mu1_block_idx_dict[key])

            # Propose new Beta_mu1 at the change_indices
            if rank == 0:
                # Beta_mu1 in this block share a sample proposal scale, has proposal variance matrix
                Beta_mu1_proposal                  = Beta_mu1_current.copy()
                Beta_mu1_proposal[change_indices] += np.sqrt(sigma_m_sq[key]) * random_generator.multivariate_normal(np.zeros(len(change_indices)), Sigma_0[key])
            else:
                Beta_mu1_proposal                  = None
            Beta_mu1_proposal                      = comm.bcast(Beta_mu1_proposal, root = 0)
            Loc_matrix_proposal                    = (C_mu0.T @ Beta_mu0_current).T + (C_mu1.T @ Beta_mu1_proposal).T * Time

            # Conditional log likelihood at proposal
            Z_1t_proposal = qZ(pgev(Y[:,rank], Loc_matrix_proposal[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]))
            if any(np.isinf(Z_1t_proposal)) or any(np.isnan(Z_1t_proposal)):
                lik_1t_proposal = np.NINF
            else:
                lik_1t_proposal = likelihood_1t(Y[:,rank], Z_1t_proposal,
                                                Loc_matrix_proposal[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                cholesky_matrix_current)
            # Gather likelihood calculated across time
            lik_current_gathered  = comm.gather(lik_1t_current, root = 0)
            lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

            # Handle prior and (Accept or Reject) on worker 0
            if rank == 0:
                # use Norm(0, sigma_Beta_mu1) prior on each Beta_mu1, like "shrinkage"
                prior_Beta_mu1_current  = scipy.stats.norm.logpdf(Beta_mu1_current, loc = 0, scale = sigma_Beta_mu1_current)
                prior_Beta_mu1_proposal = scipy.stats.norm.logpdf(Beta_mu1_proposal,loc = 0, scale = sigma_Beta_mu1_current)

                lik_current  = sum(lik_current_gathered)  + sum(prior_Beta_mu1_current)
                lik_proposal = sum(lik_proposal_gathered) + sum(prior_Beta_mu1_proposal)

                # Accept or Reject
                u = random_generator.uniform()
                ratio = np.exp(lik_proposal - lik_current)
                if not np.isfinite(ratio):
                    ratio = 0
                if u > ratio: # Reject
                    Beta_mu1_accepted = False
                    Beta_mu1_update   = Beta_mu1_current
                else: # Accept
                    Beta_mu1_accepted = True
                    Beta_mu1_update   = Beta_mu1_proposal
                    num_accepted[key] += 1

                # Store the result
                Beta_mu1_trace[iter,:] = Beta_mu1_update

                # Update the "current" value
                Beta_mu1_current = Beta_mu1_update
            else: # other workers
                Beta_mu1_accepted = None
            Beta_mu1_accepted     = comm.bcast(Beta_mu1_accepted, root = 0)
            Beta_mu1_current      = comm.bcast(Beta_mu1_current, root = 0)

            # Update X_star, mu surface, and likelihood
            if Beta_mu1_accepted:
                Z_1t_current = Z_1t_proposal.copy()
                Loc_matrix_current = (C_mu0.T @ Beta_mu0_current).T + (C_mu1.T @ Beta_mu1_current).T * Time
                lik_1t_current     = lik_1t_proposal

            comm.Barrier() # block for Beta_mu1 updates

        # %% Update Beta_logsigma
        ##################################################################
        #### ----- Update covariate coefficients, Beta_logsigma ----- ####
        ##################################################################

        # Propose new Beta_logsigma --> new sigma surface
        if rank == 0:
            # Beta_logsigma's share a same proposal scale, no proposal matrix
            # Beta_logsigma_proposal = Beta_logsigma_current + np.sqrt(sigma_m_sq['Beta_logsigma'])*random_generator.normal(np.zeros(1), 1, size = Beta_logsigma_m)

            # Beta_logsigma's share a smae proposal scale, ALSO HAS proposal matrix
            Beta_logsigma_proposal = Beta_logsigma_current + np.sqrt(sigma_m_sq['Beta_logsigma']) * \
                                                random_generator.multivariate_normal(np.zeros(Beta_logsigma_m), Sigma_0['Beta_logsigma'])
        else:
            Beta_logsigma_proposal = None
        Beta_logsigma_proposal   = comm.bcast(Beta_logsigma_proposal, root = 0)
        Scale_matrix_proposal = np.exp((C_logsigma.T @ Beta_logsigma_proposal).T)

        # Conditional log likelihood at Current
        # No need to re-calculate, inherit from above
        # lik_1t_current = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current,
        #                                                     Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
        #                                                     phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
        # Conditional log likelihood at proposal
        # if np.any([scale <= 0 for scale in Scale_matrix_proposal]):
        lik_1t_proposal = np.NINF
        if np.all(Scale_matrix_proposal > 0):
            Z_1t_proposal = qZ(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_proposal[:,rank], Shape_matrix_current[:,rank]))
            if all(~np.isinf(Z_1t_proposal)) and all(~np.isnan(Z_1t_proposal)):
                lik_1t_proposal = likelihood_1t(Y[:,rank], Z_1t_proposal,
                                                Loc_matrix_current[:,rank], Scale_matrix_proposal[:,rank], Shape_matrix_current[:,rank],
                                                cholesky_matrix_current)

        # Gather likelihood calculated across time
        lik_current_gathered  = comm.gather(lik_1t_current, root = 0)
        lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

        # Handle prior and (Accept or Reject) on worker 0
        if rank == 0:
            # use Norm(0, sigma_Beta_sigma) prior on each Beta_sigma, like "shrinkage"
            prior_Beta_logsigma_current  = scipy.stats.norm.logpdf(Beta_logsigma_current, loc=0, scale=sigma_Beta_logsigma_current)
            prior_Beta_logsigma_proposal = scipy.stats.norm.logpdf(Beta_logsigma_proposal, loc=0, scale=sigma_Beta_logsigma_current)

            lik_current  = sum(lik_current_gathered)  + sum(prior_Beta_logsigma_current)
            lik_proposal = sum(lik_proposal_gathered) + sum(prior_Beta_logsigma_proposal)

            # Accept or Reject
            u = random_generator.uniform()
            ratio = np.exp(lik_proposal - lik_current)
            if not np.isfinite(ratio):
                ratio = 0
            if u > ratio: # Reject
                Beta_logsigma_accepted = False
                Beta_logsigma_update   = Beta_logsigma_current
            else: # Accept
                Beta_logsigma_accepted = True
                Beta_logsigma_update   = Beta_logsigma_proposal
                num_accepted['Beta_logsigma'] += 1

            # Store the result
            Beta_logsigma_trace[iter, :] = Beta_logsigma_update

            # Update the current value
            Beta_logsigma_current  = Beta_logsigma_update
        else:
            Beta_logsigma_accepted = None

        # Broadcast the udpated values
        Beta_logsigma_accepted = comm.bcast(Beta_logsigma_accepted, root = 0)
        Beta_logsigma_current  = comm.bcast(Beta_logsigma_current, root = 0)

        # Update X_star, sigma surface, and likelihood
        if Beta_logsigma_accepted:
            Z_1t_current = Z_1t_proposal.copy()
            Scale_matrix_current = np.exp((C_logsigma.T @ Beta_logsigma_current).T)
            lik_1t_current = lik_1t_proposal

        comm.Barrier() # block for beta_logsigma updates

        # %% Update Beta_ksi
        #############################################################
        #### ----- Update covariate coefficients, Beta_ksi ----- ####
        #############################################################

        # Propose new Beta_ksi --> new ksi surface
        if rank == 0:
            # Beta_kis's share a same proposal scale, no proposal matrix
            # Beta_ksi_proposal = Beta_ksi_current + np.sqrt(sigma_m_sq['Beta_ksi'])*random_generator.normal(np.zeros(1), 1, size = Beta_ksi_m)

            # Beta_ksi's share a same proposal scale, ALSO HAS proposal matrix
            Beta_ksi_proposal = Beta_ksi_current + np.sqrt(sigma_m_sq['Beta_ksi']) * \
                                                random_generator.multivariate_normal(np.zeros(Beta_ksi_m), Sigma_0['Beta_ksi'])
        else:
            Beta_ksi_proposal = None
        Beta_ksi_proposal     = comm.bcast(Beta_ksi_proposal, root = 0)
        Shape_matrix_proposal = (C_ksi.T @ Beta_ksi_proposal).T

        # Conditional log likelihood at Current
        # No need to re-calculate, inherit from above
        # lik_1t_current = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current,
        #                                                     Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
        #                                                     phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)

        # Conditional log likelihood at proposal
        # Shape_out_of_range = np.any([shape <= -0.5 for shape in Shape_matrix_proposal]) or np.any([shape > 0.5 for shape in Shape_matrix_proposal])
        Shape_out_of_range = np.any(Shape_matrix_proposal <= -0.5) or np.any(Shape_matrix_proposal > 0.5)

        lik_1t_proposal = np.NINF

        if not Shape_out_of_range:
            Z_1t_proposal = qZ(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_proposal[:,rank]))
            if all(~np.isinf(Z_1t_proposal)) and all(~np.isnan(Z_1t_proposal)):
                lik_1t_proposal = likelihood_1t(Y[:,rank], Z_1t_proposal,
                                                Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_proposal[:,rank],
                                                cholesky_matrix_current)

        # Gather likelihood calculated across time
        lik_current_gathered  = comm.gather(lik_1t_current, root = 0)
        lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

        # Handle prior and (Accept or Reject) on worker 0
        if rank == 0:
            # use Norm(0, sigma_Beta_ksi) prior on each Beta_ksi, like "shrinkage"
            prior_Beta_ksi_current  = scipy.stats.norm.logpdf(Beta_ksi_current, loc=0, scale=sigma_Beta_ksi_current)
            prior_Beta_ksi_proposal = scipy.stats.norm.logpdf(Beta_ksi_proposal, loc=0, scale=sigma_Beta_ksi_current)

            lik_current  = sum(lik_current_gathered)  + sum(prior_Beta_ksi_current)
            lik_proposal = sum(lik_proposal_gathered) + sum(prior_Beta_ksi_proposal)

            # Accept or Reject
            u = random_generator.uniform()
            ratio = np.exp(lik_proposal - lik_current)
            if not np.isfinite(ratio):
                ratio = 0
            if u > ratio: # Reject
                Beta_ksi_accepted = False
                Beta_ksi_update   = Beta_ksi_current
            else: # Accept
                Beta_ksi_accepted = True
                Beta_ksi_update   = Beta_ksi_proposal
                num_accepted['Beta_ksi'] += 1

            # Store the result
            Beta_ksi_trace[iter, :] = Beta_ksi_update

            # Update the current value
            Beta_ksi_current = Beta_ksi_update
        else: # not rank = 1
            # other workers need to know acceptance,
            # b/c although adaptive MH is calculated under worker0, need to know if X_star changed
            Beta_ksi_accepted   = None

        # Broadcast the update values
        Beta_ksi_current  = comm.bcast(Beta_ksi_current, root = 0)
        Beta_ksi_accepted = comm.bcast(Beta_ksi_accepted, root = 0)

        if Beta_ksi_accepted:
            Z_1t_current = Z_1t_proposal.copy()
            Shape_matrix_current = (C_ksi.T @ Beta_ksi_current).T
            lik_1t_current = lik_1t_proposal

        comm.Barrier() # block for beta updates

        # %% Update sigma_beta_xx
        #################################################################
        ## ---- Update sigma_beta_xx, priors variance for Beta_xx ---- ##
        #################################################################

        # Update sigma_Beta_xx separately -- poor mixing in combined update
        if rank == 0:
            ## sigma_Beta_mu0  ----------------------------------------------------------------------------------------------------------

            sigma_Beta_mu0_proposal = sigma_Beta_mu0_current + np.sqrt(sigma_m_sq['sigma_Beta_mu0']) * random_generator.standard_normal()

            # use Half-t(4) hyperprior on the sigma_Beta_xx priors
            lik_sigma_Beta_mu0_current       = np.log(dhalft(sigma_Beta_mu0_current, nu = 4))
            lik_sigma_Beta_mu0_proposal      = np.log(dhalft(sigma_Beta_mu0_proposal, nu = 4)) if sigma_Beta_mu0_proposal > 0 else np.NINF

            # Beta_mu_xx at current/proposal prior
            lik_Beta_mu0_prior_current       = scipy.stats.norm.logpdf(Beta_mu0_current, scale = sigma_Beta_mu0_current)
            lik_Beta_mu0_prior_proposal      = scipy.stats.norm.logpdf(Beta_mu0_current, scale = sigma_Beta_mu0_proposal)

            # Beta_xx not changed, so no need to calculate the data likelihood
            lik_current  = lik_sigma_Beta_mu0_current  + sum(lik_Beta_mu0_prior_current)
            lik_proposal = lik_sigma_Beta_mu0_proposal + sum(lik_Beta_mu0_prior_proposal)

            # Accept or Reject
            u = random_generator.uniform()
            if not all(np.isfinite([lik_proposal,lik_current])): # the likelihood values are not finite
                # print('likelihood values not finite in iter', iter, 'updating sigma_beta_mu0')
                # print('lik_proposal:', lik_proposal, 'lik_current:', lik_current)
                ratio = 0
            else: # the likelihood values are finite
                ratio = np.exp(lik_proposal - lik_current)
            if not np.isfinite(ratio): # likelihood fine, but ratio not finite
                # print('np.exp overflow in iter', iter, 'updating sigma_beta_mu0')
                # print('lik_proposal:', lik_proposal, 'lik_current:', lik_current)
                ratio = 0
            if u > ratio: # Reject
                sigma_Beta_mu0_accepted = False
                sigma_Beta_mu0_update   = sigma_Beta_mu0_current
            else: # Accept
                sigma_Beta_mu0_accepted         = True
                sigma_Beta_mu0_update           = sigma_Beta_mu0_proposal
                num_accepted['sigma_Beta_mu0'] += 1

            # Store the result
            sigma_Beta_mu0_trace[iter,:] = sigma_Beta_mu0_update
            # Update the current value
            sigma_Beta_mu0_current       = sigma_Beta_mu0_update

            ## sigma_Beta_mu1 ----------------------------------------------------------------------------------------------------------

            sigma_Beta_mu1_proposal = sigma_Beta_mu1_current + np.sqrt(sigma_m_sq['sigma_Beta_mu1']) * random_generator.standard_normal()

            # use Half-t(4) hyperprior on the sigma_Beta_xx priors
            lik_sigma_Beta_mu1_current       = np.log(dhalft(sigma_Beta_mu1_current, nu = 4))
            lik_sigma_Beta_mu1_proposal      = np.log(dhalft(sigma_Beta_mu1_proposal, nu = 4)) if sigma_Beta_mu1_proposal > 0 else np.NINF

            # Beta_mu_xx at current/proposal prior
            lik_Beta_mu1_prior_current       = scipy.stats.norm.logpdf(Beta_mu1_current, scale = sigma_Beta_mu1_current)
            lik_Beta_mu1_prior_proposal      = scipy.stats.norm.logpdf(Beta_mu1_current, scale = sigma_Beta_mu1_proposal)

            # Beta_xx not changed, so no need to calculate the data likelihood
            lik_current  = lik_sigma_Beta_mu1_current  + sum(lik_Beta_mu1_prior_current)
            lik_proposal = lik_sigma_Beta_mu1_proposal + sum(lik_Beta_mu1_prior_proposal)

            # Accept or Reject
            u     = random_generator.uniform()
            if not all(np.isfinite([lik_proposal,lik_current])): # the likelihood values are not finite
                # print('likelihood values not finite in iter', iter, 'updating sigma_beta_mu1')
                # print('lik_proposal:', lik_proposal, 'lik_current:', lik_current)
                ratio = 0
            else: # the likelihood values are finite
                ratio = np.exp(lik_proposal - lik_current)
            if not np.isfinite(ratio): # likelihood fine, but ratio not finite
                # print('np.exp overflow in iter', iter, 'updating sigma_beta_mu1')
                # print('lik_proposal:', lik_proposal, 'lik_current:', lik_current)
                ratio = 0
            if u > ratio: # Reject
                sigma_Beta_mu1_accepted = False
                sigma_Beta_mu1_update   = sigma_Beta_mu1_current
            else: # Accept
                sigma_Beta_mu1_accepted         = True
                sigma_Beta_mu1_update           = sigma_Beta_mu1_proposal
                num_accepted['sigma_Beta_mu1'] += 1

            # Store the result
            sigma_Beta_mu1_trace[iter,:] = sigma_Beta_mu1_update
            # Update the current value
            sigma_Beta_mu1_current       = sigma_Beta_mu1_update

            ## sigma_Beta_logsigma ----------------------------------------------------------------------------------------------------------

            sigma_Beta_logsigma_proposal = sigma_Beta_logsigma_current + np.sqrt(sigma_m_sq['sigma_Beta_logsigma']) * random_generator.standard_normal()

            # use Half-t(4) hyperprior on the sigma_Beta_xx priors
            lik_sigma_Beta_logsigma_current       = np.log(dhalft(sigma_Beta_logsigma_current, nu = 4))
            lik_sigma_Beta_logsigma_proposal      = np.log(dhalft(sigma_Beta_logsigma_proposal, nu = 4)) if sigma_Beta_logsigma_proposal > 0 else np.NINF

            # Beta_mu_xx at current/proposal prior
            lik_Beta_logsigma_prior_current       = scipy.stats.norm.logpdf(Beta_logsigma_current, scale = sigma_Beta_logsigma_current)
            lik_Beta_logsigma_prior_proposal      = scipy.stats.norm.logpdf(Beta_logsigma_current, scale = sigma_Beta_logsigma_proposal)

            # Beta_xx not changed, so no need to calculate the data likelihood
            lik_current  = lik_sigma_Beta_logsigma_current  + sum(lik_Beta_logsigma_prior_current)
            lik_proposal = lik_sigma_Beta_logsigma_proposal + sum(lik_Beta_logsigma_prior_proposal)

            # Accept or Reject
            u     = random_generator.uniform()
            if not all(np.isfinite([lik_proposal,lik_current])): # the likelihood values are not finite
                # print('likelihood values not finite in iter', iter, 'updating sigma_beta_logsigma')
                # print('lik_proposal:', lik_proposal, 'lik_current:', lik_current)
                ratio = 0
            else: # the likelihood values are finite
                ratio = np.exp(lik_proposal - lik_current)
            if not np.isfinite(ratio): # likelihood fine, but ratio not finite
                # print('np.exp overflow in iter', iter, 'updating sigma_beta_logsigma')
                # print('lik_proposal:', lik_proposal, 'lik_current:', lik_current)
                ratio = 0
            if u > ratio: # Reject
                sigma_Beta_logsigma_accepted = False
                sigma_Beta_logsigma_update   = sigma_Beta_logsigma_current
            else: # Accept
                sigma_Beta_logsigma_accepted         = True
                sigma_Beta_logsigma_update           = sigma_Beta_logsigma_proposal
                num_accepted['sigma_Beta_logsigma'] += 1

            # Store the result
            sigma_Beta_logsigma_trace[iter,:] = sigma_Beta_logsigma_update
            # Update the current value
            sigma_Beta_logsigma_current       = sigma_Beta_logsigma_update

            ## sigma_Beta_ksi ----------------------------------------------------------------------------------------------------------

            sigma_Beta_ksi_proposal = sigma_Beta_ksi_current + np.sqrt(sigma_m_sq['sigma_Beta_ksi']) * random_generator.standard_normal()

            # use Half-t(4) hyperprior on the sigma_Beta_xx priors
            lik_sigma_Beta_ksi_current       = np.log(dhalft(sigma_Beta_ksi_current, nu = 4))
            lik_sigma_Beta_ksi_proposal      = np.log(dhalft(sigma_Beta_ksi_proposal, nu = 4)) if sigma_Beta_ksi_proposal > 0 else np.NINF

            # Beta_mu_xx at current/proposal prior
            lik_Beta_ksi_prior_current       = scipy.stats.norm.logpdf(Beta_ksi_current, scale = sigma_Beta_ksi_current)
            lik_Beta_ksi_prior_proposal      = scipy.stats.norm.logpdf(Beta_ksi_current, scale = sigma_Beta_ksi_proposal)

            # Beta_xx not changed, so no need to calculate the data likelihood
            lik_current  = lik_sigma_Beta_ksi_current  + sum(lik_Beta_ksi_prior_current)
            lik_proposal = lik_sigma_Beta_ksi_proposal + sum(lik_Beta_ksi_prior_proposal)

            # Accept or Reject
            u     = random_generator.uniform()
            if not all(np.isfinite([lik_proposal,lik_current])): # the likelihood values are not finite
                # print('likelihood values not finite in iter', iter, 'updating sigma_beta_ksi')
                # print('lik_proposal:', lik_proposal, 'lik_current:', lik_current)
                ratio = 0
            else: # the likelihood values are finite
                ratio = np.exp(lik_proposal - lik_current)
            if not np.isfinite(ratio): # likelihood fine, but ratio not finite
                # print('np.exp overflow in iter', iter, 'updating sigma_beta_ksi')
                # print('lik_proposal:', lik_proposal, 'lik_current:', lik_current)
                ratio = 0
            if u > ratio: # Reject
                sigma_Beta_ksi_accepted = False
                sigma_Beta_ksi_update   = sigma_Beta_ksi_current
            else: # Accept
                sigma_Beta_ksi_accepted         = True
                sigma_Beta_ksi_update           = sigma_Beta_ksi_proposal
                num_accepted['sigma_Beta_ksi'] += 1

            # Store the result
            sigma_Beta_ksi_trace[iter,:] = sigma_Beta_ksi_update
            # Update the current value
            sigma_Beta_ksi_current       = sigma_Beta_ksi_update

        # Broadcast the updated values
        sigma_Beta_mu0_current      = comm.bcast(sigma_Beta_mu0_current, root = 0)
        sigma_Beta_mu1_current      = comm.bcast(sigma_Beta_mu1_current, root = 0)
        sigma_Beta_logsigma_current = comm.bcast(sigma_Beta_logsigma_current, root = 0)
        sigma_Beta_ksi_current      = comm.bcast(sigma_Beta_ksi_current, root = 0)

        comm.Barrier() # for updating prior variance for Beta_xx

        # %% After iteration likelihood
        ######################################################################
        #### ----- Keeping track of likelihood after this iteration ----- ####
        ######################################################################



        lik_final_1t_detail = likelihood_1t_detail(Y[:,rank], Z_1t_current,
                                                Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                cholesky_matrix_current)
        lik_final_1t = sum(lik_final_1t_detail)
        lik_final_detail_gathered = comm.gather(lik_final_1t_detail, root = 0)
        lik_final_gathered = comm.gather(lik_final_1t, root = 0)
        if rank == 0:
            loglik_trace[iter,0] = round(sum(lik_final_gathered),3) # storing the overall log likelihood
            loglik_detail_trace[iter,:] = np.matrix(lik_final_detail_gathered).sum(axis=0) # storing the detail log likelihood

        comm.Barrier() # block for one iteration of update

        # %% Adaptive Update tunings
        #####################################################
        ###### ----- Adaptive Update autotunings ----- ######
        #####################################################

        if iter % adapt_size == 0:

            gamma1 = 1 / ((iter/adapt_size + offset) ** c_1)
            gamma2 = c_0 * gamma1

            # phi, range, and GEV
            if rank == 0:
                # range block update
                for key in range_block_idx_dict.keys():
                    start_idx          = range_block_idx_dict[key][0]
                    end_idx            = range_block_idx_dict[key][-1]+1
                    r_hat              = num_accepted[key]/adapt_size
                    num_accepted[key]  = 0
                    log_sigma_m_sq_hat = np.log(sigma_m_sq[key]) + gamma2 * (r_hat - r_opt)
                    sigma_m_sq[key]    = np.exp(log_sigma_m_sq_hat)
                    Sigma_0_hat        = np.array(np.cov(range_knots_trace[iter-adapt_size:iter, start_idx:end_idx].T))
                    Sigma_0[key]       = Sigma_0[key] + gamma1 * (Sigma_0_hat - Sigma_0[key])

                # GEV coefficients

                ## Beta_logsigma
                r_hat = num_accepted['Beta_logsigma']/adapt_size
                num_accepted['Beta_logsigma'] = 0
                log_sigma_m_sq_hat          = np.log(sigma_m_sq['Beta_logsigma']) + gamma2*(r_hat - r_opt)
                sigma_m_sq['Beta_logsigma'] = np.exp(log_sigma_m_sq_hat)
                Sigma_0_hat                 = np.array(np.matrix(np.cov(Beta_logsigma_trace[iter-adapt_size:iter].T)))
                Sigma_0['Beta_logsigma']    = Sigma_0['Beta_logsigma'] + gamma1*(Sigma_0_hat - Sigma_0['Beta_logsigma'])
                ## Beta_ksi
                r_hat = num_accepted['Beta_ksi']/adapt_size
                num_accepted['Beta_ksi'] = 0
                log_sigma_m_sq_hat     = np.log(sigma_m_sq['Beta_ksi']) + gamma2*(r_hat - r_opt)
                sigma_m_sq['Beta_ksi'] = np.exp(log_sigma_m_sq_hat)
                Sigma_0_hat            = np.array(np.matrix(np.cov(Beta_ksi_trace[iter-adapt_size:iter].T)))
                Sigma_0['Beta_ksi']    = Sigma_0['Beta_ksi'] + gamma1*(Sigma_0_hat - Sigma_0['Beta_ksi'])

                ## Beta_mu0 Block Update
                for key in Beta_mu0_block_idx_dict.keys():
                    start_idx = Beta_mu0_block_idx_dict[key][0]
                    end_idx   = Beta_mu0_block_idx_dict[key][-1]+1
                    r_hat              = num_accepted[key]/adapt_size
                    num_accepted[key]  = 0
                    log_sigma_m_sq_hat = np.log(sigma_m_sq[key]) + gamma2*(r_hat - r_opt)
                    sigma_m_sq[key]    = np.exp(log_sigma_m_sq_hat)
                    Sigma_0_hat        = np.array(np.cov(Beta_mu0_trace[iter-adapt_size:iter, start_idx:end_idx].T))
                    Sigma_0[key]       = Sigma_0[key] + gamma1*(Sigma_0_hat - Sigma_0[key])

                ## Beta_mu1 Block Update
                for key in Beta_mu1_block_idx_dict.keys():
                    start_idx = Beta_mu1_block_idx_dict[key][0]
                    end_idx   = Beta_mu1_block_idx_dict[key][-1]+1
                    r_hat              = num_accepted[key]/adapt_size
                    num_accepted[key]  = 0
                    log_sigma_m_sq_hat = np.log(sigma_m_sq[key]) + gamma2 * (r_hat - r_opt)
                    sigma_m_sq[key]    = np.exp(log_sigma_m_sq_hat)
                    Sigma_0_hat        = np.array(np.cov(Beta_mu1_trace[iter-adapt_size:iter, start_idx:end_idx].T))
                    Sigma_0[key]       = Sigma_0[key] + gamma1 * (Sigma_0_hat - Sigma_0[key])

                # Prior variance for GEV Coefficients
                ## sigma_Beta_mu0
                r_hat                          = num_accepted['sigma_Beta_mu0']/adapt_size
                num_accepted['sigma_Beta_mu0'] = 0
                log_sigma_m_sq_hat             = np.log(sigma_m_sq['sigma_Beta_mu0']) + gamma2*(r_hat - r_opt)
                sigma_m_sq['sigma_Beta_mu0']   = np.exp(log_sigma_m_sq_hat)
                ## sigma_Beta_mu1
                r_hat                          = num_accepted['sigma_Beta_mu1']/adapt_size
                num_accepted['sigma_Beta_mu1'] = 0
                log_sigma_m_sq_hat             = np.log(sigma_m_sq['sigma_Beta_mu1']) + gamma2*(r_hat - r_opt)
                sigma_m_sq['sigma_Beta_mu1']   = np.exp(log_sigma_m_sq_hat)
                ## sigma_Beta_logsigma
                r_hat                               = num_accepted['sigma_Beta_logsigma']/adapt_size
                num_accepted['sigma_Beta_logsigma'] = 0
                log_sigma_m_sq_hat                  = np.log(sigma_m_sq['sigma_Beta_logsigma']) + gamma2*(r_hat - r_opt)
                sigma_m_sq['sigma_Beta_logsigma']   = np.exp(log_sigma_m_sq_hat)
                ## sigma_Beta_ksi
                r_hat                          = num_accepted['sigma_Beta_ksi']/adapt_size
                num_accepted['sigma_Beta_ksi'] = 0
                log_sigma_m_sq_hat             = np.log(sigma_m_sq['sigma_Beta_ksi']) + gamma2*(r_hat - r_opt)
                sigma_m_sq['sigma_Beta_ksi']   = np.exp(log_sigma_m_sq_hat)

        comm.Barrier() # block for adaptive update

        # %% Midway Printing, Drawings, and Savings
        ##############################################
        ###    Printing, Drawings, Savings       #####
        ##############################################

        if rank == 0: # Handle Drawing at worker 0
            # print(iter)
            if iter % 10 == 0:
                print(iter)
                # print(strftime('%Y-%m-%d %H:%M:%S', localtime(time.time())))
                end_time = time.time()
                print('elapsed: ', round(end_time - start_time, 1), ' seconds')
            if iter % 50 == 0 or iter == n_iters-1: # Save and print data every 50 iterations
                # %% Saving -------------------------------------------------------------------------------------------
                # Saving ----------------------------------------------------------------------------------------------

                ## Traceplots of parameters
                np.save('loglik_trace', loglik_trace)
                np.save('loglik_detail_trace', loglik_detail_trace)
                np.save('range_knots_trace', range_knots_trace)
                np.save('Beta_mu0_trace', Beta_mu0_trace)
                np.save('Beta_mu1_trace', Beta_mu1_trace)
                np.save('Beta_logsigma_trace', Beta_logsigma_trace)
                np.save('Beta_ksi_trace', Beta_ksi_trace)
                np.save('sigma_Beta_mu0_trace', sigma_Beta_mu0_trace)
                np.save('sigma_Beta_mu1_trace', sigma_Beta_mu1_trace)
                np.save('sigma_Beta_logsigma_trace', sigma_Beta_logsigma_trace)
                np.save('sigma_Beta_ksi_trace', sigma_Beta_ksi_trace)
                np.save('Y_trace', Y_trace)

                ## Adaptive Tuning "Parameter" for daisy-chainning the runs
                with open('iter.pkl', 'wb') as file:
                    pickle.dump(iter, file)

                with open('sigma_m_sq.pkl', 'wb') as file:
                    pickle.dump(sigma_m_sq, file)

                with open('Sigma_0.pkl', 'wb') as file:
                    pickle.dump(Sigma_0, file)


                # %% Printing -----------------------------------------------------------------------------------------
                # Printing --------------------------------------------------------------------------------------------
                # Print traceplot thinned by 10
                xs       = np.arange(iter)
                xs_thin  = xs[0::10] # index 1, 11, 21, ...
                xs_thin2 = np.arange(len(xs_thin)) # index 1, 2, 3, ...

                loglik_trace_thin              = loglik_trace[0:iter:10,:]
                loglik_detail_trace_thin       = loglik_detail_trace[0:iter:10,:]
                range_knots_trace_thin         = range_knots_trace[0:iter:10,:]
                Beta_mu0_trace_thin            = Beta_mu0_trace[0:iter:10,:]
                Beta_mu1_trace_thin            = Beta_mu1_trace[0:iter:10,:]
                Beta_logsigma_trace_thin       = Beta_logsigma_trace[0:iter:10,:]
                Beta_ksi_trace_thin            = Beta_ksi_trace[0:iter:10,:]
                sigma_Beta_mu0_trace_thin      = sigma_Beta_mu0_trace[0:iter:10,:]
                sigma_Beta_mu1_trace_thin      = sigma_Beta_mu1_trace[0:iter:10,:]
                sigma_Beta_logsigma_trace_thin = sigma_Beta_logsigma_trace[0:iter:10,:]
                sigma_Beta_ksi_trace_thin      = sigma_Beta_ksi_trace[0:iter:10,:]

                # ---- log-likelihood ----

                plt.subplots()
                plt.plot(xs_thin2, loglik_trace_thin)
                plt.title('traceplot for log-likelihood')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('loglikelihood')
                plt.savefig('Traceplot_loglik.pdf')
                plt.close()

                # ---- log-likelihood in details ----

                plt.subplots()
                for i in range(2):
                    plt.plot(xs_thin2, loglik_detail_trace_thin[:,i],label = i)
                    plt.annotate('piece ' + str(i), xy=(xs_thin2[-1], loglik_detail_trace_thin[:,i][-1]))
                plt.title('traceplot for detail log likelihood')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('log likelihood')
                plt.legend()
                plt.savefig('Traceplot_loglik_detail.pdf')
                plt.close()

                # ---- range ----

                plt.subplots()
                for i in range(k_rho):
                    plt.plot(xs_thin2, range_knots_trace_thin[:,i], label='knot ' + str(i))
                    plt.annotate('knot ' + str(i), xy=(xs_thin2[-1], range_knots_trace_thin[:,i][-1]))
                plt.title('traceplot for range')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('range')
                plt.legend()
                plt.savefig('Traceplot_range.pdf')
                plt.close()

                # ---- GEV ----

                ## location mu0 coefficients in blocks

                for key in Beta_mu0_block_idx_dict.keys():
                    plt.subplots()
                    for j in Beta_mu0_block_idx_dict[key]:
                        plt.plot(xs_thin2, Beta_mu0_trace_thin[:,j], label = 'Beta_' + str(j))
                        plt.annotate('Beta_' + str(j), xy=(xs_thin2[-1], Beta_mu0_trace_thin[:,j][-1]))
                    plt.title('traceplot for Beta_mu0' + str(Beta_mu0_block_idx_dict[key]))
                    plt.xlabel('iter thinned by 10')
                    plt.ylabel('Beta_mu0')
                    plt.legend()
                    plt.savefig('Traceplot_'+str(key)+'.pdf')
                    plt.close()

                ## location Beta_mu1 in blocks:

                for key in Beta_mu1_block_idx_dict.keys():
                    plt.subplots()
                    for j in Beta_mu1_block_idx_dict[key]:
                        plt.plot(xs_thin2, Beta_mu1_trace_thin[:,j], label = 'Beta_' + str(j))
                        plt.annotate('Beta_' + str(j), xy=(xs_thin2[-1], Beta_mu1_trace_thin[:,j][-1]))
                    plt.title('traceplot for Beta_mu1' + str(Beta_mu1_block_idx_dict[key]))
                    plt.xlabel('iter thinned by 10')
                    plt.ylabel('Beta_mu1')
                    plt.legend()
                    plt.savefig('Traceplot_'+str(key) + '.pdf')
                    plt.close()

                ## scale coefficients

                plt.subplots()
                for j in range(Beta_logsigma_m):
                    plt.plot(xs_thin2, Beta_logsigma_trace_thin[:,j], label = 'Beta_' + str(j))
                    plt.annotate('Beta_' + str(j), xy=(xs_thin2[-1], Beta_logsigma_trace_thin[:,j][-1]))
                plt.title('traceplot for Beta_logsigma s')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('Beta_logsigma')
                plt.legend()
                plt.savefig('Traceplot_Beta_logsigma.pdf')
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
                plt.savefig('Traceplot_Beta_ksi.pdf')
                plt.close()

                ## location Beta_xx prior variances combined on one plot (since they're updated togeter)

                plt.subplots()
                plt.plot(xs_thin2, sigma_Beta_mu0_trace_thin,      label = 'sigma_Beta_mu0')
                plt.plot(xs_thin2, sigma_Beta_mu1_trace_thin,      label = 'sigma_Beta_mu1')
                plt.plot(xs_thin2, sigma_Beta_logsigma_trace_thin, label = 'sigma_Beta_logsigma')
                plt.plot(xs_thin2, sigma_Beta_ksi_trace_thin,      label = 'sigma_Beta_ksi')
                plt.annotate('sigma_Beta_mu0',      xy=(xs_thin2[-1], sigma_Beta_mu0_trace_thin[:,0][-1]))
                plt.annotate('sigma_Beta_mu1',      xy=(xs_thin2[-1], sigma_Beta_mu1_trace_thin[:,0][-1]))
                plt.annotate('sigma_Beta_logsigma', xy=(xs_thin2[-1], sigma_Beta_logsigma_trace_thin[:,0][-1]))
                plt.annotate('sigma_Beta_ksi',      xy=(xs_thin2[-1], sigma_Beta_ksi_trace_thin[:,0][-1]))
                plt.title('sigma in Beta_xx ~ N(0, sigma^2)')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('sigma')
                plt.legend()
                plt.savefig('Traceplot_sigma_Beta_xx.pdf')
                plt.close()

        comm.Barrier() # block for drawing


    # %% 11. End of MCMC Saving Traceplot
    # 11. End of MCMC Saving Traceplot ------------------------------------------------------------------------------
    if rank == 0:
        end_time = time.time()
        print('total time: ', round(end_time - start_time, 1), ' seconds')
        # print('true R: ', R_at_knots)
        np.save('loglik_trace', loglik_trace)
        np.save('loglik_detail_trace', loglik_detail_trace)
        np.save('range_knots_trace', range_knots_trace)
        np.save('Beta_mu0_trace', Beta_mu0_trace)
        np.save('Beta_mu1_trace', Beta_mu1_trace)
        np.save('Beta_logsigma_trace', Beta_logsigma_trace)
        np.save('Beta_ksi_trace', Beta_ksi_trace)
        np.save('sigma_Beta_mu0_trace', sigma_Beta_mu0_trace)
        np.save('sigma_Beta_mu1_trace', sigma_Beta_mu1_trace)
        np.save('sigma_Beta_logsigma_trace', sigma_Beta_logsigma_trace)
        np.save('sigma_Beta_ksi_trace', sigma_Beta_ksi_trace)
        np.save('Y_trace', Y_trace)
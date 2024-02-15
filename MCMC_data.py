"""
Feb 11, 2024

MCMC Sampler that takes in real data

Jan 23 2024, theta(s) = Beta_0 + Beta_1 * Elev(s) + splines(s) @ Beta_splines
More specifically,
    mu(s,t) = mu_0(s) + mu_1(s) * t 
    logsigma(s,t) = logsigma(s)
    ksi(s,t) = ksi(s)
where 
    t           = - Nt/2, -Nt/2 + 1, ..., 0, 1, ..., Nt/2 - 1
    mu_0(s)     = Beta_mu0_0 + Beta_mu0_1 * Elev(s) + splines(s) @ Beta_mu0_splines
    mu_1(s)     = Beta_mu1_0 + Beta_mu1_1 * Elev(s) + splines(s) @ Beta_mu1_splines
(Feb 14, for the real data, use:)
    logsigma(s) = Beta_logsigma_0 + Beta_logsimga_1 * elev(s)
    ksi(s)      = Beta_ksi_0      + Beta_ksi_1      * elev(s)
so we have
    Beta_mu0    = (Beta_mu0_0, Beta_mu0_1, Beta_mu0_splines)
    C_mu0(s)    = (1, Elev(s), splines(s))
Note on mgcv
- It appears the basis matrix produced by smoothCon is slightly (~ 3 to 4 decimal places) different between machines
- Note that in the splines constructed by mgcv, the 3rd to last column is a flat plane (which duplicate the intercept term) 
    so remember to remove it!
"""
# Require:
#   - utilities.py
# Example Usage:
# mpirun -n 2 -output-filename folder_name python3 MCMC.py > pyout.txt &
# mpirun -n 2 python3 MCMC.py > output.txt 2>&1 &
if __name__ == "__main__":
    # %%
    # seed (maybe not necessary now)
    import sys
    data_seed = int(sys.argv[1]) if len(sys.argv) == 2 else 2345
    try: # data_seed is defined when python MCMC.py
        data_seed
    except: # when running on local machine interactively
        data_seed = 2345

    # %%
    # imports
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
    import gstools as gs
    import rpy2.robjects as robjects
    from rpy2.robjects import r
    from rpy2.robjects.numpy2ri import numpy2rpy
    from rpy2.robjects.packages import importr

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    random_generator = np.random.RandomState((rank+1)*7) # use of this avoids impacting the global np state

    #####################################################################################################################
    # Loading Dataset and Setting Parameters ############################################################################
    #####################################################################################################################
    # %% 0. Loading Dataset and Setting Parameters
    # 0. Loading Dataset and Setting Parameters -------------------------------------------------------------------------
    mgcv = importr('mgcv')
    r('''load('JJA_precip_maxima.RData')''')
    GEV_estimates      = np.array(r('GEV_estimates')).T
    mu0_estimates      = GEV_estimates[:,0]
    mu1_estimates      = GEV_estimates[:,1]
    logsigma_estimates = GEV_estimates[:,2]
    ksi_estimates      = GEV_estimates[:,3]
    JJA_maxima         = np.array(r('JJA_maxima')).T
    stations           = np.array(r('stations')).T
    elevations         = np.array(r('elev')).T

    # truncate for easier run on misspiggy
    JJA_maxima = JJA_maxima[:,0:32]

    # ----------------------------------------------------------------------------------------------------------------
    # Numbers - Ns, Nt, n_iters
    
    np.random.seed(data_seed)
    Nt = JJA_maxima.shape[1] # number of time replicates
    Ns = JJA_maxima.shape[0] # number of sites/stations
    n_iters = 5000
    # Time = np.linspace(-Nt/2, Nt/2-1, Nt)
    # Note, to use the mu1 estimates from Likun, the `Time`` must be standardized the same way
    start_year = 1950
    end_year   = 2017
    all_years  = np.linspace(start_year, end_year, Nt)
    Time       = (all_years - np.mean(all_years))/np.std(all_years, ddof=1) # delta degress of freedom, to match the n-1 in R

    Time = Time[0:JJA_maxima.shape[1]]

    # ----------------------------------------------------------------------------------------------------------------
    # Sites - random uniformly (x,y) generate site locations
    
    sites_xy = stations
    sites_x = sites_xy[:,0]
    sites_y = sites_xy[:,1]

    # ----------------------------------------------------------------------------------------------------------------
    # Knots - uniform grid of 9 knots, should do this programatically...

    res_x = 3
    res_y = 3
    k = res_x * res_y # number of knots
    # define the lower and upper limits for x and y
    minX, maxX = np.floor(np.min(sites_x)), np.ceil(np.max(sites_x))
    minY, maxY = np.floor(np.min(sites_y)), np.ceil(np.max(sites_y))
    # create one-dimensional arrays for x and y
    x_pos = np.linspace(minX, maxX, res_x+2)[1:-1]
    y_pos = np.linspace(minY, maxY, res_y+2)[1:-1]
    # create the mesh based on these arrays
    X_pos, Y_pos = np.meshgrid(x_pos,y_pos)
    knots_xy = np.vstack([X_pos.ravel(), Y_pos.ravel()]).T
    knots_x = knots_xy[:,0]
    knots_y = knots_xy[:,1]

    # ----------------------------------------------------------------------------------------------------------------
    # Basis Parameters - for the Gaussian and Wendland Basis

    bandwidth = 4.2 # range for the gaussian kernel
    radius = 4.2 # radius of infuence for basis, 3.5 might make some points closer to the edge of circle, might lead to numerical issues
    radius_from_knots = np.repeat(radius, k) # influence radius from a knot

    # ----------------------------------------------------------------------------------------------------------------
    # Data Model Parameters - X_star = R^phi * g(Z)

    ## Stable S_t --> R_t
    gamma = 0.5 # this is the gamma that goes in rlevy, gamma_at_knots
    delta = 0.0 # this is the delta in levy, stays 0
    alpha = 0.5

    ## scaling parameter
    phi_at_knots = np.array([0.5] * k)

    ## g(Z)
    nu = 0.5 # exponential kernel for matern with nu = 1/2
    sigsq = 1.0 # sill for Z
    # range_at_knots = np.array([2] * k) initial values will be estimated in section 3 below

    # ----------------------------------------------------------------------------------------------------------------
    # Marginal Parameters - GEV(mu, sigma, ksi)

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
    
    ## covariate
    Beta_mu0_splines_m = 12 - 1 # number of splines basis, -1 b/c drop constant column
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
    
    ## coefficients
    lm                      = np.linalg.lstsq(a=C_mu0[:,:,0].T, b=mu0_estimates,rcond=None)
    Beta_mu0                = lm[0]
    Beta_mu0_m              = len(Beta_mu0)
    Beta_mu0_block_idx_size = 4
    
    ## mu0(s,t)
    mu0_matrix = (C_mu0.T @ Beta_mu0).T      

    # Location mu_1(s) ----------------------------------------------------------------------------------------------
    
    ## covariates
    Beta_mu1_splines_m = 18 - 1 # drop the 3rd to last column of constant
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
    
    ## coefficients
    lm                      = np.linalg.lstsq(a=C_mu1[:,:,0].T, b=mu1_estimates,rcond=None)
    Beta_mu1                = lm[0]
    Beta_mu1_m              = len(Beta_mu1)
    Beta_mu1_block_idx_size = 4

    ## mu1(s,t)
    mu1_matrix = (C_mu1.T @ Beta_mu1).T

    # Location mu(s,t) -----------------------------------------------------------------------------------------------
    mu_matrix = mu0_matrix + mu1_matrix * Time

    # Create Coefficient Blocks for Beta_mu0 and Beta_mu1-----------------------------------------------------------

    ## Beta_mu0
    Beta_mu0_block_idx_dict = {}  # dictionary that stores the index of Beta_mu0 in each block
    Beta_mu0_nblock         = int(Beta_mu0_m/Beta_mu0_block_idx_size)
    for i in range(Beta_mu0_nblock):
        start_index = Beta_mu0_block_idx_size*i
        end_index   = start_index + Beta_mu0_block_idx_size
        if i+1 < Beta_mu0_nblock:
            Beta_mu0_block_idx_dict['Beta_mu0_block_idx_'+str(i+1)] = [index for index in range(start_index, end_index)]
        else: # last block
            Beta_mu0_block_idx_dict['Beta_mu0_block_idx_'+str(i+1)] = [index for index in range(start_index, Beta_mu0_m)]

    ## Beta_mu1
    Beta_mu1_block_idx_dict = {} # dictionary that stores the index of Beta_mu1 in each block
    Beta_mu1_nblock         = int(Beta_mu1_m/Beta_mu1_block_idx_size)
    for i in range(Beta_mu1_nblock):
        start_index = Beta_mu1_block_idx_size*i
        end_index   = start_index + Beta_mu1_block_idx_size
        if i + 1 < Beta_mu1_nblock:
            Beta_mu1_block_idx_dict['Beta_mu1_block_idx_'+str(i+1)] = [index for index in range(start_index, end_index)]
        else:
            Beta_mu1_block_idx_dict['Beta_mu1_block_idx_'+str(i+1)] = [index for index in range(start_index, Beta_mu1_m)]

    # ---- Linear surfaces for logsigma and ksi ---------------------------------------------------------------------------
    
    ## Scale sigma - logscale ##
    Beta_logsigma_dict = {
        'Beta_logsigma_0' : None, # intercept for logsimga
        'Beta_logsigma_1' : None # slope of elevation for logsigma
    }
    Beta_logsigma_m   = len(Beta_logsigma_dict)
    C_logsigma        = np.full(shape = (Beta_logsigma_m, Ns, Nt), fill_value = np.nan)
    C_logsigma[0,:,:] = 1.0 
    C_logsigma[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T
    lm                = np.linalg.lstsq(a=C_logsigma[:,:,0].T, b=logsigma_estimates,rcond=None)
    for key, value in zip(Beta_logsigma_dict.keys(), lm[0]):
        Beta_logsigma_dict[key] = value
    Beta_logsigma     = np.array(list(Beta_logsigma_dict.values()))
    logsigma_matrix   = (C_logsigma.T @ Beta_logsigma).T
    sigma_matrix = np.exp(logsigma_matrix)

    ## Shape ksi ##
    Beta_ksi_dict = {
        'Beta_ksi_0' : None, # intercept for ksi
        'Beta_ksi_1' : None # slope of elevation for ksi
    }
    Beta_ksi_m   = len(Beta_ksi_dict)
    C_ksi        = np.full(shape = (Beta_ksi_m, Ns, Nt), fill_value = np.nan) # ksi design matrix
    C_ksi[0,:,:] = 1.0
    C_ksi[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T
    lm           = np.linalg.lstsq(a=C_ksi[:,:,0].T, b=ksi_estimates,rcond=None)
    for key, value in zip(Beta_ksi_dict.keys(), lm[0]):
        Beta_ksi_dict[key] = value
    Beta_ksi     = np.array(list(Beta_ksi_dict.values()))
    ksi_matrix   = (C_ksi.T @ Beta_ksi).T

    # ----------------------------------------------------------------------------------------------------------------
    # Beta Coefficient Prior Parameter - sigma_Beta_xx ~ Halt-t(4)

    ## just initial values, right? Theses are not "truth"
    sigma_Beta_mu0      = 0.2
    sigma_Beta_mu1      = 0.2
    sigma_Beta_logsigma = 0.2
    sigma_Beta_ksi      = 0.2

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
    # Adaptive Update: TRIAL RUN Posterior Covariance Matrix

    # trial run posterior variance matrix for phi
    phi_post_cov = 1e-3 * np.identity(k)
    assert k == phi_post_cov.shape[0]

    # trial run posterior variance matrix for range rho
    range_post_cov = 1e-2 * np.identity(k)
    assert k == range_post_cov.shape[0]

    # posterior/proposal variance matrix for linear surface logsigma
    Beta_logsigma_post_cov = 1e-4 * np.identity(Beta_logsigma_m)
    assert Beta_logsigma_m == Beta_logsigma_post_cov.shape[0]

    # posterior/proposal variance matrix for linear surface ksi
    Beta_ksi_post_cov = 1e-4 * np.identity(Beta_ksi_m)
    assert Beta_ksi_m == Beta_ksi_post_cov.shape[0]

    # trial run posterior variance matrix for Beta_mu0
    Beta_mu0_all_post_cov = 1e-5 * np.identity(Beta_mu0_m)
    assert Beta_mu0_all_post_cov.shape[0] == Beta_mu0_m
    Beta_mu0_block_post_cov_dict = {}
    for key in Beta_mu0_block_idx_dict.keys():
        start_idx                         = Beta_mu0_block_idx_dict[key][0]
        end_idx                           = Beta_mu0_block_idx_dict[key][-1]+1
        Beta_mu0_block_post_cov_dict[key] = Beta_mu0_all_post_cov[start_idx:end_idx, start_idx:end_idx]

    # trial run posterior variance matrix for Beta_mu1
    Beta_mu1_all_post_cov = 1e-5 * np.identity(Beta_mu1_m)
    assert Beta_mu1_all_post_cov.shape[0] == Beta_mu1_m
    Beta_mu1_block_post_cov_dict          = {}
    for key in Beta_mu1_block_idx_dict.keys():
        start_idx                         = Beta_mu1_block_idx_dict[key][0]
        end_idx                           = Beta_mu1_block_idx_dict[key][-1]+1
        Beta_mu1_block_post_cov_dict[key] = Beta_mu1_all_post_cov[start_idx:end_idx, start_idx:end_idx]

    # ----------------------------------------------------------------------------------------------------------------
    # Adaptive Update: Proposal Variance Scalar, Covariance Matrix, and Counter
    if rank == 0: # Handle phi, range, GEV on Worker 0
        # proposal variance scalar
        sigma_m_sq = {
            'phi_block1'          : (2.4**2)/3,
            'phi_block2'          : (2.4**2)/3,
            'phi_block3'          : (2.4**2)/3,
            'range_block1'        : (2.4**2)/3,
            'range_block2'        : (2.4**2)/3,
            'range_block3'        : (2.4**2)/3,
            # 'GEV'                 : (2.4**2)/3
            # 'Beta_mu0'            : (2.4**2)/Beta_mu0_m,
            'Beta_logsigma'       : (2.4**2)/Beta_logsigma_m,
            'Beta_ksi'            : (2.4**2)/Beta_ksi_m,
            'sigma_Beta_mu0'      : 0.03749589, # from trial run
            'sigma_Beta_mu1'      : 0.01,
            'sigma_Beta_logsigma' : 0.24878523, # from trial run
            'sigma_Beta_ksi'      : 0.44929566  # from trial run
        }
        for key in Beta_mu0_block_idx_dict.keys():
            sigma_m_sq[key] = (2.4**2)/len(Beta_mu0_block_idx_dict[key])
        for key in Beta_mu1_block_idx_dict.keys():
            sigma_m_sq[key] = (2.4**2)/len(Beta_mu1_block_idx_dict[key])

        # proposal covariance matrix
        Sigma_0 = {
            'phi_block1'    : phi_post_cov[0:3,0:3],
            'phi_block2'    : phi_post_cov[3:6,3:6],
            'phi_block3'    : phi_post_cov[6:9,6:9],
            'range_block1'  : range_post_cov[0:3,0:3],
            'range_block2'  : range_post_cov[3:6,3:6],
            'range_block3'  : range_post_cov[6:9,6:9],
            # 'GEV'           : GEV_post_cov,
            # 'Beta_mu0'      : Beta_mu0_post_cov,
            'Beta_logsigma' : Beta_logsigma_post_cov,
            'Beta_ksi'      : Beta_ksi_post_cov
        }
        Sigma_0.update(Beta_mu0_block_post_cov_dict)
        Sigma_0.update(Beta_mu1_block_post_cov_dict)

        num_accepted = { # acceptance counter
            'phi'                 : 0,
            'range'               : 0,
            'phi_block1'          : 0,
            'phi_block2'          : 0,
            'phi_block3'          : 0,
            'range_block1'        : 0,
            'range_block2'        : 0,
            'range_block3'        : 0,
            # 'GEV'                 : 0,
            # 'Beta_mu0'            : 0,
            'Beta_logsigma'       : 0,
            'Beta_ksi'            : 0,
            'sigma_Beta_mu0'      : 0,
            'sigma_Beta_mu1'      : 0,
            'sigma_Beta_logsigma' : 0,
            'sigma_Beta_ksi'      : 0
        }
        for key in Beta_mu0_block_idx_dict.keys():
            num_accepted[key] = 0
        for key in Beta_mu1_block_idx_dict.keys():
            num_accepted[key] = 0

    # Rt: each Worker_t propose k-R(t)s at time t
    if rank == 0:
        sigma_m_sq_Rt_list = [(2.4**2)/k]*size # comm scatter and gather preserves order
        num_accepted_Rt_list = [0]*size # [0, 0, ... 0]
    else:
        sigma_m_sq_Rt_list = None
        num_accepted_Rt_list = None
    sigma_m_sq_Rt = comm.scatter(sigma_m_sq_Rt_list, root = 0)
    num_accepted_Rt = comm.scatter(num_accepted_Rt_list, root = 0)

    # %% 1. Plot Station, Knots, and Elevations
    # 1. Plot Station, Knots, and Elevations -------------------------------------------------------------------------------------
    if rank == 0: # plot the stations and knots
        fig, ax = plt.subplots()
        ax.plot(sites_x, sites_y, 'b.', alpha = 0.4)
        ax.plot(knots_x, knots_y, 'r+')
        space_rectangle = plt.Rectangle(xy=(minX, minY), width=maxX-minX, height=maxY-minY,
                                        fill = False, color = 'black')
        for i in range(k):
            circle_i = plt.Circle((knots_xy[i,0], knots_xy[i,1]), radius_from_knots[0],
                                    color='r', fill=True, fc='grey', ec='red', alpha = 0.2)
            ax.add_patch(circle_i)
        ax.add_patch(space_rectangle)
        ax.set_aspect('equal', 'box')
        # plt.show()
        plt.savefig('stations.pdf')
        plt.close()
    
    if rank == 0: # Plot the elevation
        fig, ax = plt.subplots()
        elev_scatter = ax.scatter(sites_x, sites_y, s=10, alpha = 0.7, c = elevations)
        ax.set_aspect('equal', 'box')
        plt.colorbar(elev_scatter)
        # plt.show()
        plt.savefig('station_elevation.pdf')
        plt.close()       
    
    if rank == 0: # create a plotting grid for weight matrices
        plotgrid_res_x = 50
        plotgrid_res_y = 75
        plotgrid_res_xy = plotgrid_res_x * plotgrid_res_y
        plotgrid_x = np.linspace(minX,maxX,plotgrid_res_x)
        plotgrid_y = np.linspace(minY,maxY,plotgrid_res_y)
        plotgrid_X, plotgrid_Y = np.meshgrid(plotgrid_x, plotgrid_y)
        plotgrid_xy = np.vstack([plotgrid_X.ravel(), plotgrid_Y.ravel()]).T


    # %% 2. Generate the Weight Matrices
    # 2. Generate the weight matrices -------------------------------------------------------------------------------------

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

    if rank == 0: # weight matrices for plotting
        gaussian_weight_matrix_for_plot = np.full(shape = (plotgrid_res_xy, k), fill_value = np.nan)
        for site_id in np.arange(plotgrid_res_xy):
            # Compute distance between each pair of the two collections of inputs
            d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)), 
                                            XB = knots_xy)
            # influence coming from each of the knots
            weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
            gaussian_weight_matrix_for_plot[site_id, :] = weight_from_knots

        wendland_weight_matrix_for_plot = np.full(shape = (plotgrid_res_xy,k), fill_value = np.nan)
        for site_id in np.arange(plotgrid_res_xy):
            # Compute distance between each pair of the two collections of inputs
            d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)), 
                                            XB = knots_xy)
            # influence coming from each of the knots
            weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
            wendland_weight_matrix_for_plot[site_id, :] = weight_from_knots


    # %% 3. Specify Covariance K for Z, generate W = g(Z)
    # 3. Specify Covariance K for Z, generate W = g(Z) ----------------------------------------------------------------

    # using sites within the radius of each knot to estimate (inital) range
    distance_matrix = np.full(shape=(Ns, k), fill_value=np.nan)
    for site_id in np.arange(Ns):
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), XB = knots_xy)
        distance_matrix[site_id,:] = d_from_knots
    sites_within_knots = {}
    for knot_id in np.arange(k):
        knot_name = 'knot_' + str(knot_id)
        sites_within_knots[knot_name] = np.where(distance_matrix[:,knot_id] <= radius_from_knots[knot_id])[0]

    range_at_knots = np.array([])
    for key in sites_within_knots.keys():
        selected_sites = sites_within_knots[key]
        demeaned_Y     = JJA_maxima - mu_matrix
        bin_center, gamma_variog = gs.vario_estimate((sites_x[selected_sites], sites_y[selected_sites]), 
                                            np.mean(demeaned_Y[selected_sites], axis=1))
        fit_model = gs.Exponential(dim=2)
        fit_model.fit_variogram(bin_center, gamma_variog, nugget=False)
        # ax = fit_model.plot(x_max = 4)
        # ax.scatter(bin_center, gamma_variog)
        range_at_knots = np.append(range_at_knots, fit_model.len_scale)


    ## range_vec
    range_vec = gaussian_weight_matrix @ range_at_knots
    # range_vec = one_weight_matrix @ range_at_knots

    ## Covariance matrix K
    ## sigsq_vec
    sigsq_vec = np.repeat(sigsq, Ns) # hold at 1
    K = ns_cov(range_vec = range_vec, sigsq_vec = sigsq_vec,
            coords = sites_xy, kappa = nu, cov_model = "matern")
    Z = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(Ns,)),cov=K,size=Nt).T
    W = norm_to_Pareto(Z) 

    if rank == 0: # plotting the range surface
        # heatplot of range surface
        range_vec_for_plot = gaussian_weight_matrix_for_plot @ range_at_knots
        graph, ax = plt.subplots()
        heatmap = ax.imshow(range_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x), 
                            cmap ='hot', interpolation='nearest', extent = [minX, maxX, maxY, minY])
        ax.invert_yaxis()
        graph.colorbar(heatmap)
        # plt.show()
        plt.savefig('heatmap range surface.pdf')
        plt.close()


    # %% 4. Generate R^phi Scaling Factor
    # 4. Generate R^phi Scaling Factor -------------------------------------------------------------------------------------

    ## phi_vec
    phi_vec = gaussian_weight_matrix @ phi_at_knots
    # phi_vec = one_weight_matrix @ phi_at_knots

    ## R
    ## Generate them at the knots
    R_at_knots = np.full(shape = (k, Nt), fill_value = np.nan)
    for t in np.arange(Nt):
        # R_at_knots[:,t] = rlevy(n = k, m = delta, s = gamma) # generate R at time t, spatially varying k knots
        R_at_knots[:,t] = scipy.stats.multivariate_normal.rvs(mean = 10*np.ones(k),cov=2*np.eye(k)) # for initial values for data analysis
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

    if rank == 0: # plotting the phi surface
        # heatplot of phi surface
        phi_vec_for_plot = gaussian_weight_matrix_for_plot @ phi_at_knots
        graph, ax = plt.subplots()
        heatmap = ax.imshow(phi_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x), 
                            cmap ='hot', interpolation='nearest', extent = [minX, maxX, maxY, minY])
        ax.invert_yaxis()
        graph.colorbar(heatmap)
        # plt.show()
        plt.savefig('heatmap phi surface.pdf')
        plt.close()

    # %% 5. Plot GEV Surfaces
    # 5. Plot GEV Surfaces -------------------------------------------------------------------------------------

    if rank == 0:
        # Location # -------------------------------------------------------------------------------------
        ## mu0(s) plot stations
        fig, ax     = plt.subplots()
        mu0_scatter = ax.scatter(sites_x, sites_y, s = 10, alpha = 0.7, c = mu0_estimates)
        ax.set_aspect('equal', 'box')
        plt.colorbar(mu0_scatter)
        plt.title('data: mu0_estimates')
        # plt.show()
        plt.savefig('data_mu0_estimates.pdf')
        plt.close()

        fig, ax     = plt.subplots()
        mu0_scatter = ax.scatter(sites_x, sites_y, s = 10, alpha = 0.7, c = mu0_matrix[:,0])
        ax.set_aspect('equal', 'box')
        plt.colorbar(mu0_scatter)
        plt.title('fitted: mu0 splines')
        # plt.show()
        plt.savefig('fitted_mu0_splines.pdf')
        plt.close()

        ## mu1(s) plot stations
        fig, ax     = plt.subplots()
        mu1_scatter = ax.scatter(sites_x, sites_y, s = 10, alpha = 0.7, c = mu1_estimates)
        ax.set_aspect('equal', 'box')
        plt.colorbar(mu1_scatter)
        plt.title('data: mu1_estimates')
        # plt.show()
        plt.savefig('data_mu1_estimates.pdf')
        plt.close()

        fig, ax     = plt.subplots()
        mu1_scatter = ax.scatter(sites_x, sites_y, s = 10, alpha = 0.7, c = mu1_matrix[:,0])
        ax.set_aspect('equal', 'box')
        plt.colorbar(mu1_scatter)
        plt.title('fitted: mu1 splines')
        # plt.show()
        plt.savefig('fitted_mu1_splines.pdf')
        plt.close()

        # Scale # -------------------------------------------------------------------------------------
        ## logsigma(s) plot stations
        fig, ax     = plt.subplots()
        logsigma_scatter = ax.scatter(sites_x, sites_y, s = 10, alpha = 0.7, c = logsigma_estimates)
        ax.set_aspect('equal', 'box')
        plt.colorbar(logsigma_scatter)
        plt.title('data: logsigma_estimates')
        # plt.show()
        plt.savefig('data_logsigma_estimates.pdf')
        plt.close()

        fig, ax     = plt.subplots()
        logsigma_scatter = ax.scatter(sites_x, sites_y, s = 10, alpha = 0.7, c = logsigma_matrix[:,0])
        ax.set_aspect('equal', 'box')
        plt.colorbar(logsigma_scatter)
        plt.title('fitted: logsigma')
        # plt.show()
        plt.savefig('fitted_logsigma.pdf')
        plt.close()

        # Shape # -------------------------------------------------------------------------------------
        # ksi(s) plot stations
        fig, ax     = plt.subplots()
        ksi_scatter = ax.scatter(sites_x, sites_y, s = 10, alpha = 0.7, c = ksi_estimates)
        ax.set_aspect('equal', 'box')
        plt.colorbar(ksi_scatter)
        plt.title('data: ksi_estimates')
        # plt.show()
        plt.savefig('data_ksi_estimates.pdf')
        plt.close()

        fig, ax     = plt.subplots()
        ksi_scatter = ax.scatter(sites_x, sites_y, s = 10, alpha = 0.7, c = ksi_matrix[:,0])
        ax.set_aspect('equal', 'box')
        plt.colorbar(ksi_scatter)
        plt.title('fitted: ksi')
        # plt.show()
        plt.savefig('fitted_ksi.pdf')
        plt.close()

    # %% 6. Generate X_star and Y
    # 6. Generate X and Y -------------------------------------------------------------------------------------
    
    # gamma_vec is the gamma bar in the overleaf document
    gamma_at_knots = np.repeat(gamma, k)
    gamma_vec = np.sum(np.multiply(wendland_weight_matrix, gamma_at_knots)**(alpha), 
                       axis = 1)**(1/alpha) # axis = 1 to sum over K knots

    # X_star = R_phi * W

    # Calculation of Y can(?) be parallelized by time(?)
    # Y = np.full(shape=(Ns, Nt), fill_value = np.nan)
    # for t in np.arange(Nt):
    #     # Y[:,t] = qgev(pRW(X_star[:,t], phi_vec, gamma_vec), mu, sigma, ksi)
    #     Y[:,t] = qgev(pRW(X_star[:,t], phi_vec, gamma_vec), mu_matrix[:,t], sigma_matrix[:,t], ksi_matrix[:,t])

    Y = JJA_maxima
    comm.Barrier()

    X_1t            = qRW(pgev(Y[:,rank], mu_matrix[:,rank], sigma_matrix[:,rank], ksi_matrix[:,rank]),
                            phi_vec, gamma_vec)
    X_star_gathered = comm.gather(X_1t, root = 0)
    X_star          = np.array(X_star_gathered).T if rank == 0 else None
    X_star          = comm.bcast(X_star, root = 0)


    # %% Metropolis-Hasting Updates
    
    #####################################################################################################################
    ###########                Metropolis-Hasting Updates                ################################################
    #####################################################################################################################

    comm.Barrier() # Blocking before the update starts

    # %% 8. Storage for Traceplots
    # 8. Storage for Traceplots -----------------------------------------------

    if rank == 0:
        loglik_trace              = np.full(shape = (n_iters, 1), fill_value = np.nan) # overall likelihood
        loglik_detail_trace       = np.full(shape = (n_iters, 5), fill_value = np.nan) # detail likelihood
        R_trace_log               = np.full(shape = (n_iters, k, Nt), fill_value = np.nan) # log(R)
        phi_knots_trace           = np.full(shape = (n_iters, k), fill_value = np.nan) # phi_at_knots
        range_knots_trace         = np.full(shape = (n_iters, k), fill_value = np.nan) # range_at_knots
        Beta_mu0_trace            = np.full(shape = (n_iters, Beta_mu0_m), fill_value = np.nan) # mu0 Covariate Coefficients
        Beta_mu1_trace            = np.full(shape = (n_iters, Beta_mu1_m), fill_value = np.nan) # mu1 covariate Coefficients
        Beta_logsigma_trace       = np.full(shape = (n_iters, Beta_logsigma_m), fill_value = np.nan) # logsigma Covariate Coefficients
        Beta_ksi_trace            = np.full(shape=(n_iters, Beta_ksi_m), fill_value = np.nan) # ksi Covariate Coefficients
        sigma_Beta_mu0_trace      = np.full(shape=(n_iters, 1), fill_value = np.nan) # prior sd for beta_mu0's
        sigma_Beta_mu1_trace      = np.full(shape=(n_iters, 1), fill_value = np.nan) # prior sd for beta_mu1's
        sigma_Beta_logsigma_trace = np.full(shape=(n_iters, 1), fill_value = np.nan) # prior sd for beta_logsigma's
        sigma_Beta_ksi_trace      = np.full(shape = (n_iters, 1), fill_value = np.nan) # prior sd for beta_ksi's
    else:
        loglik_trace              = None
        loglik_detail_trace       = None
        R_trace_log               = None
        phi_knots_trace           = None
        range_knots_trace         = None
        Beta_mu0_trace            = None
        Beta_mu1_trace            = None
        Beta_logsigma_trace       = None
        Beta_ksi_trace            = None
        sigma_Beta_mu0_trace      = None
        sigma_Beta_mu1_trace      = None
        sigma_Beta_logsigma_trace = None
        sigma_Beta_ksi_trace      = None

    # %% 9. Initialize
    # 9. Initialize -------------------------------------------------------------------------------------

    # Initialize at the truth/at other values
    R_matrix_init_log        = np.log(R_at_knots)  if rank == 0 else None
    phi_knots_init           = phi_at_knots        if rank == 0 else None
    range_knots_init         = range_at_knots      if rank == 0 else None
    Beta_mu0_init            = Beta_mu0            if rank == 0 else None
    Beta_mu1_init            = Beta_mu1            if rank == 0 else None
    Beta_logsigma_init       = Beta_logsigma       if rank == 0 else None
    Beta_ksi_init            = Beta_ksi            if rank == 0 else None
    sigma_Beta_mu0_init      = sigma_Beta_mu0      if rank == 0 else None
    sigma_Beta_mu1_init      = sigma_Beta_mu1      if rank == 0 else None
    sigma_Beta_logsigma_init = sigma_Beta_logsigma if rank == 0 else None
    sigma_Beta_ksi_init      = sigma_Beta_ksi      if rank == 0 else None
    if rank == 0: # store initial value into first row of traceplot
        R_trace_log[0,:,:]             = R_matrix_init_log # matrix (k, Nt)
        phi_knots_trace[0,:]           = phi_knots_init
        range_knots_trace[0,:]         = range_knots_init
        Beta_mu0_trace[0,:]            = Beta_mu0_init
        Beta_mu1_trace[0,:]            = Beta_mu1_init
        Beta_logsigma_trace[0,:]       = Beta_logsigma_init
        Beta_ksi_trace[0,:]            = Beta_ksi_init
        sigma_Beta_mu0_trace[0,:]      = sigma_Beta_mu0_init
        sigma_Beta_mu1_trace[0,:]      = sigma_Beta_mu1_init
        sigma_Beta_logsigma_trace[0,:] = sigma_Beta_logsigma_init
        sigma_Beta_ksi_trace[0,:]      = sigma_Beta_ksi_init

    # Set Current Values
    ## ---- X_star --------------------------------------------------------------------------------------------
    X_star_1t_current = X_star[:,rank]

    ## ---- log(R) --------------------------------------------------------------------------------------------
    # note: directly comm.scatter an numpy nd array along an axis is tricky,
    #       hence we first "redundantly" broadcast an entire R_matrix then split
    R_matrix_init_log = comm.bcast(R_matrix_init_log, root = 0) # matrix (k, Nt)
    R_current_log     = np.array(R_matrix_init_log[:,rank]) # vector (k,)
    R_vec_current     = wendland_weight_matrix @ np.exp(R_current_log)

    ## ---- phi ------------------------------------------------------------------------------------------------
    phi_knots_current = comm.bcast(phi_knots_init, root = 0)
    phi_vec_current   = gaussian_weight_matrix @ phi_knots_current

    ## ---- range_vec (length_scale) ---------------------------------------------------------------------------
    range_knots_current = comm.bcast(range_knots_init, root = 0)
    range_vec_current   = gaussian_weight_matrix @ range_knots_current
    K_current           = ns_cov(range_vec = range_vec_current,
                                 sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = "matern")
    cholesky_matrix_current = scipy.linalg.cholesky(K_current, lower = False)

    ## ---- GEV covariate coefficients --> GEV surface ----------------------------------------------------------
    Beta_mu0_current      = comm.bcast(Beta_mu0_init, root = 0)
    Beta_mu1_current      = comm.bcast(Beta_mu1_init, root = 0)
    Beta_logsigma_current = comm.bcast(Beta_logsigma_init, root = 0)
    Beta_ksi_current      = comm.bcast(Beta_ksi_init, root = 0)
    # Loc_matrix_current    = (C_mu0.T @ Beta_mu0_current).T
    Loc_matrix_current    = (C_mu0.T @ Beta_mu0_current).T + (C_mu1.T @ Beta_mu1_current).T * Time
    Scale_matrix_current  = np.exp((C_logsigma.T @ Beta_logsigma_current).T)
    Shape_matrix_current  = (C_ksi.T @ Beta_ksi_current).T

    ## ---- GEV covariate coefficients prior variance -----------------------------------------------------------
    sigma_Beta_mu0_current      = comm.bcast(sigma_Beta_mu0_init, root = 0)
    sigma_Beta_mu1_current      = comm.bcast(sigma_Beta_mu1_init, root = 0)
    sigma_Beta_logsigma_current = comm.bcast(sigma_Beta_logsigma_init, root = 0)
    sigma_Beta_ksi_current      = comm.bcast(sigma_Beta_ksi_init, root = 0)

    # %% 10. Metropolis Update Loops -------------------------------------------------------------------------------------
    # 10. Metropolis Update Loops
    if rank == 0:
        start_time = time.time()
        print('started on:', strftime('%Y-%m-%d %H:%M:%S', localtime(time.time())))

    for iter in range(1, n_iters):
        # %% Update Rt
        ###########################################################
        #### ----- Update Rt ----- Parallelized Across Nt time ####
        ###########################################################

        # Propose a R at time "rank", on log-scale
        # Propose a R using adaptive update
        R_proposal_log = np.sqrt(sigma_m_sq_Rt)*random_generator.normal(loc = 0.0, scale = 1.0, size = k) + R_current_log
        # R_proposal_log = np.sqrt(sigma_m_sq_Rt)*np.repeat(random_generator.normal(loc = 0.0, scale = 1.0, size = 1), k) + R_current_log # spatially cst R(t)

        # Conditional log likelihood at Current
        R_vec_current = wendland_weight_matrix @ np.exp(R_current_log)
        if iter == 1: # otherwise lik_1t_current will be inherited
            lik_1t_current = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
                                                        Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank], 
                                                        phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
        # log-prior density
        prior_1t_current = np.sum(scipy.stats.levy.logpdf(np.exp(R_current_log), scale = gamma) + R_current_log)
        # prior_1t_current = prior_1t_current/k # if R(t) is spatially constant

        # Conditional log likelihood at Proposal
        R_vec_proposal = wendland_weight_matrix @ np.exp(R_proposal_log)
        # if np.any(~np.isfinite(R_vec_proposal**phi_vec_current)): print("Negative or zero R, iter=", iter, ", rank=", rank, R_vec_proposal[0], phi_vec_current[0])
        lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
                                                        Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank], 
                                                        phi_vec_current, gamma_vec, R_vec_proposal, cholesky_matrix_current)
        prior_1t_proposal = np.sum(scipy.stats.levy.logpdf(np.exp(R_proposal_log), scale = gamma) + R_proposal_log)
        # prior_1t_proposal = prior_1t_proposal/k # if R(t) is spatially constant

        # Gather likelihood calculated across time
        # no need of R(t) because each worker takes care of one

        # Accept or Reject
        u = random_generator.uniform()
        ratio = np.exp(lik_1t_proposal + prior_1t_proposal - lik_1t_current - prior_1t_current)
        if not np.isfinite(ratio):
            ratio = 0 # Force a rejection
        if u > ratio: # Reject
            Rt_accepted = False
            R_update_log = R_current_log
        else: # Accept, u <= ratio
            Rt_accepted = True
            R_update_log = R_proposal_log
            num_accepted_Rt += 1
        
        # Store the result
        R_update_log_gathered = comm.gather(R_update_log, root=0)
        if rank == 0:
            R_trace_log[iter,:,:] = np.vstack(R_update_log_gathered).T

        # Update the current values
        R_current_log = R_update_log
        R_vec_current = wendland_weight_matrix @ np.exp(R_current_log)
        
        # Update the likelihood (to ease computation below)
        if Rt_accepted:
            lik_1t_current = lik_1t_proposal
        
        comm.Barrier() # block for R_t updates

        # %% Update phi
        ###################################################################################
        ####   Update phi_at_knots   ######################################################
        ###################################################################################

        # Update phi ACTUALLY in blocks
        for i in range(3):
            change_indices = np.array([i*3, i*3+1, i*3+2])
            block_name     = 'phi_block' + str(i+1)

            # Propose new phi_block at the change_indices
            if rank == 0:
                phi_knots_proposal                  = phi_knots_current.copy()
                phi_knots_proposal[change_indices] += np.sqrt(sigma_m_sq[block_name]) * \
                                                        random_generator.multivariate_normal(np.zeros(3), Sigma_0[block_name])
            else:
                phi_knots_proposal = None
            phi_knots_proposal     = comm.bcast(phi_knots_proposal, root = 0)
            phi_vec_proposal       = gaussian_weight_matrix @ phi_knots_proposal

            # Conditional log likelihood at proposal
            phi_out_of_range = any(phi <= 0 for phi in phi_knots_proposal) or any(phi > 1 for phi in phi_knots_proposal) # U(0,1] prior
            if phi_out_of_range:
                lik_1t_proposal = np.NINF
            else:
                X_star_1t_proposal = qRW(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
                                        phi_vec_proposal, gamma_vec)
                lik_1t_proposal    = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_proposal, 
                                        Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                        phi_vec_proposal, gamma_vec, R_vec_current, cholesky_matrix_current)
            
            # Gather likelihood calculated across time
            lik_current_gathered  = comm.gather(lik_1t_current, root = 0)
            lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

            # Handle prior and Accept/Reject on worker 0
            if rank == 0:
                # use Beta(5,5) prior on each one of the 3 parameters
                lik_current  = sum(lik_current_gathered)  + np.sum(scipy.stats.beta.logpdf(phi_knots_current,  a = 5, b = 5))
                lik_proposal = sum(lik_proposal_gathered) + np.sum(scipy.stats.beta.logpdf(phi_knots_proposal, a = 5, b = 5))
                
                # Accept or Reject
                u     = random_generator.uniform()
                ratio = np.exp(lik_proposal - lik_current)
                if not np.isfinite(ratio):
                    ratio = 0
                if u > ratio: # Reject
                    phi_accepted     = False
                    phi_vec_update   = phi_vec_current
                    phi_knots_update = phi_knots_current
                else: # Accept, u <= ratio
                    phi_accepted              = True
                    phi_vec_update            = phi_vec_proposal
                    phi_knots_update          = phi_knots_proposal
                    num_accepted[block_name] += 1
                
                # Store the result
                phi_knots_trace[iter,:] = phi_knots_update
                
                # Update the current value
                phi_vec_current   = phi_vec_update
                phi_knots_current = phi_knots_update
            else: # broadcast to other workers
                phi_accepted  = None
            phi_vec_current   = comm.bcast(phi_vec_current, root = 0)
            phi_knots_current = comm.bcast(phi_knots_current, root = 0)
            phi_accepted      = comm.bcast(phi_accepted, root = 0)
            
            # Update X_star and likelihood if accepted
            if phi_accepted:
                X_star_1t_current = X_star_1t_proposal
                lik_1t_current    = lik_1t_proposal

            comm.Barrier() # block for phi update

        # %% Update range
        #########################################################################################
        ####  Update range_at_knots  ############################################################
        #########################################################################################
        
        # # Propose new range at the knots --> new range vector
        # if rank == 0:
        #     random_walk_block1 = np.sqrt(sigma_m_sq['range_block1'])*random_generator.multivariate_normal(np.zeros(3), Sigma_0['range_block1'])
        #     random_walk_block2 = np.sqrt(sigma_m_sq['range_block2'])*random_generator.multivariate_normal(np.zeros(3), Sigma_0['range_block2'])
        #     random_walk_block3 = np.sqrt(sigma_m_sq['range_block3'])*random_generator.multivariate_normal(np.zeros(3), Sigma_0['range_block3'])    
        #     random_walk_kx1 = np.hstack((random_walk_block1,random_walk_block2,random_walk_block3))
        #     range_knots_proposal = range_knots_current + random_walk_kx1
        # else:
        #     range_knots_proposal = None
        # range_knots_proposal = comm.bcast(range_knots_proposal, root = 0)
        # range_vec_proposal = gaussian_weight_matrix @ range_knots_proposal

        # # Conditional log Likelihood at Current
        # # No need to re-calculate because likelihood inherit from above
        # # lik_1t_current = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
        # #                                                 Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
        # #                                                 phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)

        # # Conditional log Likelihood at Proposed
        # if any(range <= 0 for range in range_knots_proposal):
        #     lik_1t_proposal = np.NINF
        # else:
        #     K_proposal = ns_cov(range_vec = range_vec_proposal,
        #                     sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = "matern")
        #     cholesky_matrix_proposal = scipy.linalg.cholesky(K_proposal, lower = False)

        #     lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
        #                                                 Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
        #                                                 phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_proposal)

        # # Gather likelihood calculated across time (no prior yet)
        # lik_current_gathered  = comm.gather(lik_1t_current, root = 0)
        # lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

        # # Handle prior and (Accept or Reject) on worker 0
        # if rank == 0:
        #     # use Half-Normal Prior on each one of the k range parameters
        #     lik_current  = sum(lik_current_gathered)  + np.sum(scipy.stats.halfnorm.logpdf(range_knots_current, loc = 0, scale = 2))
        #     lik_proposal = sum(lik_proposal_gathered) + np.sum(scipy.stats.halfnorm.logpdf(range_knots_proposal, loc = 0, scale = 2))

        #     # Accept or Reject
        #     u = random_generator.uniform()
        #     ratio = np.exp(lik_proposal - lik_current)
        #     if not np.isfinite(ratio):
        #         ratio = 0 # Force a rejection
        #     if u > ratio: # Reject
        #         range_accepted     = False
        #         range_vec_update   = range_vec_current
        #         range_knots_update = range_knots_current
        #     else: # Accept, u <= ratio
        #         range_accepted     = True
        #         range_vec_update   = range_vec_proposal
        #         range_knots_update = range_knots_proposal
        #         num_accepted['range'] += 1
            
        #     # Store the result
        #     range_knots_trace[iter,:] = range_knots_update

        #     # Update the "current" value
        #     range_vec_current = range_vec_update
        #     range_knots_current = range_knots_update
        # else:
        #     range_accepted = None

        # # Brodcast the updated values
        # range_vec_current   = comm.bcast(range_vec_current, root = 0)
        # range_knots_current = comm.bcast(range_knots_current, root = 0)
        # range_accepted      = comm.bcast(range_accepted, root = 0)

        # # Update the K, cholesky_matrix, and likelihood
        # if range_accepted:
        #     K_current = K_proposal
        #     cholesky_matrix_current = cholesky_matrix_proposal
        #     lik_1t_current = lik_1t_proposal

        # comm.Barrier() # block for range updates

        # Update range ACTUALLY in blocks
        for i in range(3):
            change_indices = np.array([i*3, i*3+1, i*3+2])
            block_name     = 'range_block' + str(i+1)

            # Propose new range_block at the change indices
            if rank == 0:
                range_knots_proposal                  = range_knots_current.copy()
                range_knots_proposal[change_indices] += np.sqrt(sigma_m_sq[block_name])* \
                                                        random_generator.multivariate_normal(np.zeros(3), Sigma_0[block_name])
            else:
                range_knots_proposal = None
            range_knots_proposal     = comm.bcast(range_knots_proposal, root = 0)
            range_vec_proposal       = gaussian_weight_matrix @ range_knots_proposal

            # Conditional log likelihood at proposal
            if any(range <= 0 for range in range_knots_proposal):
                lik_1t_proposal = np.NINF
            else:
                K_proposal = ns_cov(range_vec = range_vec_proposal,
                                    sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = "matern")
                cholesky_matrix_proposal = scipy.linalg.cholesky(K_proposal, lower = False)
                lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
                                    Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                    phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_proposal)
            
            # Gather likelihood calculated across time (no prior yet)
            lik_current_gathered   = comm.gather(lik_1t_current, root = 0)
            like_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

            # Handle prior and Accept/Reject on worker 0
            if rank == 0:
                # use Half-Normal prior on each one of the 3 range parameters
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
                    range_accepted            = True
                    range_vec_update          = range_vec_proposal
                    range_knots_update        = range_knots_proposal
                    num_accepted[block_name] += 1
                
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
                K_current               = K_proposal
                cholesky_matrix_current = cholesky_matrix_proposal
                lik_1t_current          = lik_1t_proposal

            comm.Barrier() # block for range_block updates

        # %% Update Beta_mu0
        #############################################################
        #### ----- Update covariate coefficients, Beta_mu0 ----- ####
        #############################################################

        # # Propose new Beta_mu0 --> new mu0 surface
        # if rank == 0:
        #     # Beta_mu0's share a same proposal scale, no proposal matrix
        #     # Beta_mu0_proposal = Beta_mu0_current + np.sqrt(sigma_m_sq['Beta_mu0'])*random_generator.normal(0, 1, size = Beta_mu0_m)
            
        #     # Beta_mu0's share a smae proposal scale, ALSO HAS proposal matrix
        #     Beta_mu0_proposal = Beta_mu0_current + np.sqrt(sigma_m_sq['Beta_mu0']) * \
        #                                         random_generator.multivariate_normal(np.zeros(Beta_mu0_m), Sigma_0['Beta_mu0'])
            
        #     # Beta_mu0's have individual proposal scale, no proposal matrix
        #     # Beta_mu0_proposal = Beta_mu0_current + np.array([np.sqrt(sigma_m_sq['Beta_mu0_'+str(j)])*random_generator.normal(0,1) for j in range(Beta_mu0_m)])
        # else:    
        #     Beta_mu0_proposal = None
        # Beta_mu0_proposal    = comm.bcast(Beta_mu0_proposal, root = 0)
        # Loc_matrix_proposal = (C_mu0.T @ Beta_mu0_proposal).T

        # # Conditional log likelihood at current
        # # No need to re-calculate because likelihood inherit from above
        # # lik_1t_current = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_current, 
        # #                                                     Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
        # #                                                     phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
        
        # # Conditional log likelihood at proposal
        # X_star_1t_proposal = qRW(pgev(Y[:,rank], Loc_matrix_proposal[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
        #                             phi_vec_current, gamma_vec)
        # lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_proposal, 
        #                                                 Loc_matrix_proposal[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
        #                                                 phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)

        # # Gather likelihood calculated across time
        # lik_current_gathered  = comm.gather(lik_1t_current, root = 0)
        # lik_proposal_gathered = comm.gather(lik_1t_proposal, root = 0)

        # # Handle prior and (Accept or Reject) on worker 0
        # if rank == 0:
        #     # use Norm(0, sigma_Beta_mu0) prior on each Beta_mu0, like "shrinkage"
        #     prior_Beta_mu0_current  = scipy.stats.norm.logpdf(Beta_mu0_current, loc=0, scale=sigma_Beta_mu0_current)
        #     prior_Beta_mu0_proposal = scipy.stats.norm.logpdf(Beta_mu0_proposal, loc=0, scale=sigma_Beta_mu0_current)

        #     lik_current  = sum(lik_current_gathered)  + sum(prior_Beta_mu0_current)
        #     lik_proposal = sum(lik_proposal_gathered) + sum(prior_Beta_mu0_proposal)

        #     # Accept or Reject
        #     u = random_generator.uniform()
        #     ratio = np.exp(lik_proposal - lik_current)
        #     if not np.isfinite(ratio):
        #         ratio = 0
        #     if u > ratio: # Reject
        #         Beta_mu0_accepted = False
        #         Beta_mu0_update   = Beta_mu0_current
        #     else: # Accept
        #         Beta_mu0_accepted = True
        #         Beta_mu0_update   = Beta_mu0_proposal
        #         num_accepted['Beta_mu0'] += 1
            
        #     # Store the result
        #     Beta_mu0_trace[iter,:] = Beta_mu0_update

        #     # Update the current value
        #     Beta_mu0_current = Beta_mu0_update
        # else: # other workers
        #     Beta_mu0_accepted = None

        # # Broadcast the updated values
        # Beta_mu0_accepted = comm.bcast(Beta_mu0_accepted, root = 0)
        # Beta_mu0_current  = comm.bcast(Beta_mu0_current, root = 0)

        # # Update X_star, mu0 surface, and likelihood
        # if Beta_mu0_accepted:
        #     X_star_1t_current  = X_star_1t_proposal
        #     Loc_matrix_current = (C_mu0.T @ Beta_mu0_current).T
        #     lik_1t_current     = lik_1t_proposal
        
        # comm.Barrier() # block for beta_mu0 updates

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
            X_star_1t_proposal = qRW(pgev(Y[:,rank], Loc_matrix_proposal[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
                                        phi_vec_current, gamma_vec)
            lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_proposal, 
                                                            Loc_matrix_proposal[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                            phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
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
                X_star_1t_current  = X_star_1t_proposal
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
            X_star_1t_proposal = qRW(pgev(Y[:,rank], Loc_matrix_proposal[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank]),
                                        phi_vec_current, gamma_vec)
            lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_proposal, 
                                                            Loc_matrix_proposal[:,rank], Scale_matrix_current[:,rank], Shape_matrix_current[:,rank],
                                                            phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
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
                X_star_1t_current  = X_star_1t_proposal
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
        if np.any(Scale_matrix_proposal <= 0):
            # X_star_1t_proposal = np.NINF
            lik_1t_proposal = np.NINF
        else:
            X_star_1t_proposal = qRW(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_proposal[:,rank], Shape_matrix_current[:,rank]),
                                        phi_vec_current, gamma_vec)
            lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_proposal, 
                                                            Loc_matrix_current[:,rank], Scale_matrix_proposal[:,rank], Shape_matrix_current[:,rank],
                                                            phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
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
            X_star_1t_current = X_star_1t_proposal
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
        if Shape_out_of_range:
            # X_star_1t_proposal = np.NINF
            lik_1t_proposal = np.NINF
        else:
            X_star_1t_proposal = qRW(pgev(Y[:,rank], Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_proposal[:,rank]),
                                        phi_vec_current, gamma_vec)
            lik_1t_proposal = marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star_1t_proposal, 
                                                            Loc_matrix_current[:,rank], Scale_matrix_current[:,rank], Shape_matrix_proposal[:,rank],
                                                            phi_vec_current, gamma_vec, R_vec_current, cholesky_matrix_current)
        
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
            X_star_1t_current = X_star_1t_proposal
            Shape_matrix_current = (C_ksi.T @ Beta_ksi_current).T
            lik_1t_current = lik_1t_proposal

        comm.Barrier() # block for beta updates

        # %% Update sigma_beta_xx
        #################################################################
        ## ---- Update sigma_beta_xx, priors variance for Beta_xx ---- ##
        #################################################################

        # # Propose new sigma_beta_xx
        # if rank == 0:
        #     sigma_Beta_mu0_proposal      = sigma_Beta_mu0_current      + np.sqrt(sigma_m_sq['sigma_Beta_mu0']) * random_generator.standard_normal()
        #     sigma_Beta_mu1_proposal      = sigma_Beta_mu1_current      + np.sqrt(sigma_m_sq['sigma_Beta_mu1']) * random_generator.standard_normal()
        #     sigma_Beta_logsigma_proposal = sigma_Beta_logsigma_current + np.sqrt(sigma_m_sq['sigma_Beta_logsigma']) * random_generator.standard_normal()
        #     sigma_Beta_ksi_proposal      = sigma_Beta_ksi_current      + np.sqrt(sigma_m_sq['sigma_Beta_ksi']) * random_generator.standard_normal()
        # # Handle accept or reject on worker0
        #     # use Half-t(4) hyperprior on the sigma_Beta_xx priors
        #     lik_sigma_Beta_mu0_current       = np.log(dhalft(sigma_Beta_mu0_current, nu = 4))
        #     lik_sigma_Beta_mu0_proposal      = np.log(dhalft(sigma_Beta_mu0_proposal, nu = 4)) if sigma_Beta_mu0_proposal > 0 else np.NINF

        #     lik_sigma_Beta_mu1_current       = np.log(dhalft(sigma_Beta_mu1_current, nu = 4))
        #     lik_sigma_Beta_mu1_proposal      = np.log(dhalft(sigma_Beta_mu1_proposal, nu = 4)) if sigma_Beta_mu1_proposal > 0 else np.NINF

        #     lik_sigma_Beta_logsigma_current  = np.log(dhalft(sigma_Beta_logsigma_current, nu = 4))
        #     lik_sigma_Beta_logsigma_proposal = np.log(dhalft(sigma_Beta_logsigma_proposal, nu = 4)) if sigma_Beta_logsigma_proposal > 0 else np.NINF

        #     lik_sigma_Beta_ksi_current       = np.log(dhalft(sigma_Beta_ksi_current, nu = 4))
        #     lik_sigma_Beta_ksi_proposal      = np.log(dhalft(sigma_Beta_ksi_proposal, nu = 4)) if sigma_Beta_ksi_proposal > 0 else np.NINF

        #     # Beta_mu_xx at current/proposal prior
        #     lik_Beta_mu0_prior_current       = scipy.stats.norm.logpdf(Beta_mu0_current, scale = sigma_Beta_mu0_current)
        #     lik_Beta_mu0_prior_proposal      = scipy.stats.norm.logpdf(Beta_mu0_current, scale = sigma_Beta_mu0_proposal)

        #     lik_Beta_mu1_prior_current       = scipy.stats.norm.logpdf(Beta_mu1_current, scale = sigma_Beta_mu1_current)
        #     lik_Beta_mu1_prior_proposal      = scipy.stats.norm.logpdf(Beta_mu1_current, scale = sigma_Beta_mu1_proposal)

        #     lik_Beta_logsigma_prior_current  = scipy.stats.norm.logpdf(Beta_logsigma_current, scale = sigma_Beta_logsigma_current)
        #     lik_Beta_logsigma_prior_proposal = scipy.stats.norm.logpdf(Beta_logsigma_current, scale = sigma_Beta_logsigma_proposal)
            
        #     lik_Beta_ksi_prior_current       = scipy.stats.norm.logpdf(Beta_ksi_current, scale = sigma_Beta_ksi_current)
        #     lik_Beta_ksi_prior_proposal      = scipy.stats.norm.logpdf(Beta_ksi_current, scale = sigma_Beta_ksi_proposal)

        #     # Beta_xx not changed, so no need to calculate the data likelihood
        #     lik_current = lik_sigma_Beta_mu0_current + \
        #                   lik_sigma_Beta_mu1_current + \
        #                   lik_sigma_Beta_logsigma_current + \
        #                   lik_sigma_Beta_ksi_current + \
        #                   sum(lik_Beta_mu0_prior_current) + \
        #                   sum(lik_Beta_mu1_prior_current) + \
        #                   sum(lik_Beta_logsigma_prior_current) + \
        #                   sum(lik_Beta_ksi_prior_current)
        #     lik_proposal = lik_sigma_Beta_mu0_proposal + \
        #                    lik_sigma_Beta_mu1_proposal + \
        #                    lik_sigma_Beta_logsigma_proposal + \
        #                    lik_sigma_Beta_ksi_proposal + \
        #                    sum(lik_Beta_mu0_prior_proposal) + \
        #                    sum(lik_Beta_mu1_prior_proposal) + \
        #                    sum(lik_Beta_logsigma_prior_proposal) + \
        #                    sum(lik_Beta_ksi_prior_proposal)

        #     # Accept or Reject
        #     u = random_generator.uniform()
        #     ratio = np.exp(lik_proposal - lik_current)
        #     if not np.isfinite(ratio):
        #         ratio = 0
        #     if u > ratio: # Reject
        #         sigma_Beta_mu0_accepted      = False
        #         sigma_Beta_mu0_update        = sigma_Beta_mu0_current

        #         sigma_Beta_mu1_accepted      = False
        #         sigma_Beta_mu1_update        = sigma_Beta_mu1_current

        #         sigma_Beta_logsigma_accepted = False
        #         sigma_Beta_logsigma_update   = sigma_Beta_logsigma_current

        #         sigma_Beta_ksi_accepted      = False
        #         sigma_Beta_ksi_update        = sigma_Beta_ksi_current
        #     else: # Accept
        #         sigma_Beta_mu0_accepted             = True
        #         sigma_Beta_mu0_update               = sigma_Beta_mu0_proposal
        #         num_accepted['sigma_Beta_mu0']      += 1

        #         sigma_Beta_mu1_accepted             = True
        #         sigma_Beta_mu1_update               = sigma_Beta_mu1_proposal
        #         num_accepted['sigma_Beta_mu1']      += 1

        #         sigma_Beta_logsigma_accepted        = True
        #         sigma_Beta_logsigma_update          = sigma_Beta_logsigma_proposal
        #         num_accepted['sigma_Beta_logsigma'] += 1

        #         sigma_Beta_ksi_accepted             = True
        #         sigma_Beta_ksi_update               = sigma_Beta_ksi_proposal
        #         num_accepted['sigma_Beta_ksi']      += 1

        #     # Store the result
        #     sigma_Beta_mu0_trace[iter,:]      = sigma_Beta_mu0_update
        #     sigma_Beta_mu1_trace[iter,:]      = sigma_Beta_mu1_update
        #     sigma_Beta_logsigma_trace[iter,:] = sigma_Beta_logsigma_update
        #     sigma_Beta_ksi_trace[iter,:]      = sigma_Beta_ksi_update

        #     # Update the current value
        #     sigma_Beta_mu0_current      = sigma_Beta_mu0_update
        #     sigma_Beta_mu1_current      = sigma_Beta_mu1_update
        #     sigma_Beta_logsigma_current = sigma_Beta_logsigma_update
        #     sigma_Beta_ksi_current      = sigma_Beta_ksi_update
        
        # # Boradcast the updated values (actually no need because only involves worker0)
        # sigma_Beta_mu0_current      = comm.bcast(sigma_Beta_mu0_current, root = 0)
        # sigma_Beta_mu1_current      = comm.bcast(sigma_Beta_mu1_current, root = 0)
        # sigma_Beta_logsigma_current = comm.bcast(sigma_Beta_logsigma_current, root = 0)
        # sigma_Beta_ksi_current      = comm.bcast(sigma_Beta_ksi_current, root = 0)

        # comm.Barrier() # for updating prior variance for Beta_xx

        # Update sigma_Beta_xx separately -- poor mixing in combined update
        if rank == 0:
            ## sigma_Beta_mu0
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
            u     = random_generator.uniform()
            ratio = np.exp(lik_proposal - lik_current)
            if not np.isfinite(ratio):
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
        
            ## sigma_Beta_mu1
            sigma_Beta_mu1_proposal = sigma_Beta_mu1_current + np.sqrt(sigma_m_sq['sigma_Beta_mu1']) * random_generator.standard_normal()
            
            # use Half-t(4) hyperprior on the sigma_Beta_xx priors
            lik_sigma_Beta_mu1_current       = np.log(dhalft(sigma_Beta_mu1_current, nu = 4))
            lik_sigma_Beta_mu1_proposal      = np.log(dhalft(sigma_Beta_mu1_proposal, nu = 4)) if sigma_Beta_mu1_proposal > 0 else np.NINF
            
            # Beta_mu_xx at current/proposal prior
            lik_Beta_mu0_prior_current       = scipy.stats.norm.logpdf(Beta_mu0_current, scale = sigma_Beta_mu1_current)
            lik_Beta_mu0_prior_proposal      = scipy.stats.norm.logpdf(Beta_mu0_current, scale = sigma_Beta_mu1_proposal)

            # Beta_xx not changed, so no need to calculate the data likelihood
            lik_current  = lik_sigma_Beta_mu1_current  + sum(lik_Beta_mu0_prior_current)
            lik_proposal = lik_sigma_Beta_mu1_proposal + sum(lik_Beta_mu0_prior_proposal)

            # Accept or Reject
            u     = random_generator.uniform()
            ratio = np.exp(lik_proposal - lik_current)
            if not np.isfinite(ratio):
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

            ## sigma_Beta_logsigma
            sigma_Beta_logsigma_proposal = sigma_Beta_logsigma_current + np.sqrt(sigma_m_sq['sigma_Beta_logsigma']) * random_generator.standard_normal()
            
            # use Half-t(4) hyperprior on the sigma_Beta_xx priors
            lik_sigma_Beta_logsigma_current       = np.log(dhalft(sigma_Beta_logsigma_current, nu = 4))
            lik_sigma_Beta_logsigma_proposal      = np.log(dhalft(sigma_Beta_logsigma_proposal, nu = 4)) if sigma_Beta_logsigma_proposal > 0 else np.NINF
            
            # Beta_mu_xx at current/proposal prior
            lik_Beta_mu0_prior_current       = scipy.stats.norm.logpdf(Beta_mu0_current, scale = sigma_Beta_logsigma_current)
            lik_Beta_mu0_prior_proposal      = scipy.stats.norm.logpdf(Beta_mu0_current, scale = sigma_Beta_logsigma_proposal)

            # Beta_xx not changed, so no need to calculate the data likelihood
            lik_current  = lik_sigma_Beta_logsigma_current  + sum(lik_Beta_mu0_prior_current)
            lik_proposal = lik_sigma_Beta_logsigma_proposal + sum(lik_Beta_mu0_prior_proposal)

            # Accept or Reject
            u     = random_generator.uniform()
            ratio = np.exp(lik_proposal - lik_current)
            if not np.isfinite(ratio):
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

            ## sigma_Beta_ksi
            sigma_Beta_ksi_proposal = sigma_Beta_ksi_current + np.sqrt(sigma_m_sq['sigma_Beta_ksi']) * random_generator.standard_normal()
            
            # use Half-t(4) hyperprior on the sigma_Beta_xx priors
            lik_sigma_Beta_ksi_current       = np.log(dhalft(sigma_Beta_ksi_current, nu = 4))
            lik_sigma_Beta_ksi_proposal      = np.log(dhalft(sigma_Beta_ksi_proposal, nu = 4)) if sigma_Beta_ksi_proposal > 0 else np.NINF
            
            # Beta_mu_xx at current/proposal prior
            lik_Beta_mu0_prior_current       = scipy.stats.norm.logpdf(Beta_mu0_current, scale = sigma_Beta_ksi_current)
            lik_Beta_mu0_prior_proposal      = scipy.stats.norm.logpdf(Beta_mu0_current, scale = sigma_Beta_ksi_proposal)

            # Beta_xx not changed, so no need to calculate the data likelihood
            lik_current  = lik_sigma_Beta_ksi_current  + sum(lik_Beta_mu0_prior_current)
            lik_proposal = lik_sigma_Beta_ksi_proposal + sum(lik_Beta_mu0_prior_proposal)

            # Accept or Reject
            u     = random_generator.uniform()
            ratio = np.exp(lik_proposal - lik_current)
            if not np.isfinite(ratio):
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
        sigma_Beta_mu0_current = comm.bcast(sigma_Beta_mu0_current, root = 0)
        sigma_Beta_mu1_current      = comm.bcast(sigma_Beta_mu1_current, root = 0)
        sigma_Beta_logsigma_current = comm.bcast(sigma_Beta_logsigma_current, root = 0)
        sigma_Beta_ksi_current      = comm.bcast(sigma_Beta_ksi_current, root = 0)
        
        comm.Barrier() # for updating prior variance for Beta_xx

        # %% After iteration likelihood
        ######################################################################
        #### ----- Keeping track of likelihood after this iteration ----- ####
        ######################################################################
    
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

        # %% Adaptive Update tunings
        #####################################################
        ###### ----- Adaptive Update autotunings ----- ######
        #####################################################

        if iter % adapt_size == 0:
                
            gamma1 = 1 / ((iter/adapt_size + offset) ** c_1)
            gamma2 = c_0 * gamma1

            # R_t
            sigma_m_sq_Rt_list = comm.gather(sigma_m_sq_Rt, root = 0)
            num_accepted_Rt_list = comm.gather(num_accepted_Rt, root = 0)
            if rank == 0:
                for i in range(size):
                    r_hat = num_accepted_Rt_list[i]/adapt_size
                    num_accepted_Rt_list[i] = 0
                    log_sigma_m_sq_hat = np.log(sigma_m_sq_Rt_list[i]) + gamma2*(r_hat - r_opt)
                    sigma_m_sq_Rt_list[i] = np.exp(log_sigma_m_sq_hat)
            sigma_m_sq_Rt = comm.scatter(sigma_m_sq_Rt_list, root = 0)
            num_accepted_Rt = comm.scatter(num_accepted_Rt_list, root = 0)

            # phi, range, and GEV
            if rank == 0:
                # phi
                # r_hat = num_accepted['phi']/adapt_size
                # num_accepted['phi'] = 0
                ## phi_block1
                r_hat                      = num_accepted['phi_block1']/adapt_size
                num_accepted['phi_block1'] = 0
                log_sigma_m_sq_hat         = np.log(sigma_m_sq['phi_block1']) + gamma2*(r_hat - r_opt)
                sigma_m_sq['phi_block1']   = np.exp(log_sigma_m_sq_hat)
                Sigma_0_hat                = np.cov(np.array([phi_knots_trace[iter-adapt_size:iter,i].ravel() for i in range(0,3)]))
                Sigma_0['phi_block1']      = Sigma_0['phi_block1'] + gamma1*(Sigma_0_hat - Sigma_0['phi_block1'])
                ## phi_block2
                r_hat                      = num_accepted['phi_block2']/adapt_size
                num_accepted['phi_block2'] = 0
                log_sigma_m_sq_hat         = np.log(sigma_m_sq['phi_block2']) + gamma2*(r_hat - r_opt)
                sigma_m_sq['phi_block2']   = np.exp(log_sigma_m_sq_hat)
                Sigma_0_hat                = np.cov(np.array([phi_knots_trace[iter-adapt_size:iter,i].ravel() for i in range(3,6)]))
                Sigma_0['phi_block2']      = Sigma_0['phi_block2'] + gamma1*(Sigma_0_hat - Sigma_0['phi_block2'])
                ## phi_block3
                r_hat                      = num_accepted['phi_block3']/adapt_size
                num_accepted['phi_block3'] = 0
                log_sigma_m_sq_hat         = np.log(sigma_m_sq['phi_block3']) + gamma2*(r_hat - r_opt)
                sigma_m_sq['phi_block3']   = np.exp(log_sigma_m_sq_hat)
                Sigma_0_hat                = np.cov(np.array([phi_knots_trace[iter-adapt_size:iter,i].ravel() for i in range(6,9)]))
                Sigma_0['phi_block3']      = Sigma_0['phi_block3'] + gamma1*(Sigma_0_hat - Sigma_0['phi_block3'])

                # range
                # r_hat = num_accepted['range']/adapt_size
                # num_accepted['range'] = 0
                ## range_block1
                r_hat                        = num_accepted['range_block1']/adapt_size
                num_accepted['range_block1'] = 0
                log_sigma_m_sq_hat           = np.log(sigma_m_sq['range_block1']) + gamma2*(r_hat - r_opt)
                sigma_m_sq['range_block1']   = np.exp(log_sigma_m_sq_hat)
                Sigma_0_hat                  = np.cov(np.array([range_knots_trace[iter-adapt_size:iter,i].ravel() for i in range(0,3)]))
                Sigma_0['range_block1']      = Sigma_0['range_block1'] + gamma1*(Sigma_0_hat - Sigma_0['range_block1'])
                ## range_block2
                r_hat                        = num_accepted['range_block2']/adapt_size
                num_accepted['range_block2'] = 0
                log_sigma_m_sq_hat           = np.log(sigma_m_sq['range_block2']) + gamma2*(r_hat - r_opt)
                sigma_m_sq['range_block2']   = np.exp(log_sigma_m_sq_hat)
                Sigma_0_hat                  = np.cov(np.array([range_knots_trace[iter-adapt_size:iter,i].ravel() for i in range(3,6)]))
                Sigma_0['range_block2']      = Sigma_0['range_block2'] + gamma1*(Sigma_0_hat - Sigma_0['range_block2'])
                ## range_block3
                r_hat                        = num_accepted['range_block3']/adapt_size
                num_accepted['range_block3'] = 0
                log_sigma_m_sq_hat           = np.log(sigma_m_sq['range_block3']) + gamma2*(r_hat - r_opt)
                sigma_m_sq['range_block3']   = np.exp(log_sigma_m_sq_hat)
                Sigma_0_hat                  = np.cov(np.array([range_knots_trace[iter-adapt_size:iter,i].ravel() for i in range(6,9)]))
                Sigma_0['range_block3']      = Sigma_0['range_block3'] + gamma1*(Sigma_0_hat - Sigma_0['range_block3'])
                
                # # GEV
                # r_hat = num_accepted['GEV']/adapt_size
                # num_accepted['GEV'] = 0
                # sample_cov = np.cov(np.array([GEV_knots_trace[iter-adapt_size:iter,0,0].ravel(), # mu location
                #                                 GEV_knots_trace[iter-adapt_size:iter,1,0].ravel()])) # sigma scale
                # Sigma_0_hat = np.zeros((3,3)) # doing the hack because we are not updating ksi
                # Sigma_0_hat[2,2] = 1
                # Sigma_0_hat[0:2,0:2] += sample_cov
                # log_sigma_m_sq_hat = np.log(sigma_m_sq['GEV']) + gamma2*(r_hat - r_opt)
                # sigma_m_sq['GEV'] = np.exp(log_sigma_m_sq_hat)
                # Sigma_0['GEV'] = Sigma_0['GEV'] + gamma1*(Sigma_0_hat - Sigma_0['GEV'])

                # GEV coefficients
                ## Beta_mu0
                # r_hat = num_accepted['Beta_mu0']/adapt_size
                # num_accepted['Beta_mu0'] = 0
                # log_sigma_m_sq_hat    = np.log(sigma_m_sq['Beta_mu0']) + gamma2*(r_hat - r_opt)
                # sigma_m_sq['Beta_mu0'] = np.exp(log_sigma_m_sq_hat)
                # Sigma_0_hat           = np.cov(Beta_mu0_trace[iter-adapt_size:iter].T)
                # Sigma_0['Beta_mu0']    = Sigma_0['Beta_mu0'] + gamma1*(Sigma_0_hat - Sigma_0['Beta_mu0'])
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
            if iter % 50 == 0:
                print(iter)
                # print(strftime('%Y-%m-%d %H:%M:%S', localtime(time.time())))
                end_time = time.time()
                print('elapsed: ', round(end_time - start_time, 1), ' seconds')
            if iter % 100 == 0 or iter == n_iters-1: # Save and pring data every 1000 iterations

                np.save('loglik_trace', loglik_trace)
                np.save('loglik_detail_trace', loglik_detail_trace)
                np.save('R_trace_log', R_trace_log)
                np.save('phi_knots_trace', phi_knots_trace)
                np.save('range_knots_trace', range_knots_trace)
                # np.save('GEV_knots_trace', GEV_knots_trace)
                np.save('Beta_mu0_trace', Beta_mu0_trace)
                np.save('Beta_mu1_trace', Beta_mu1_trace)
                np.save('Beta_logsigma_trace', Beta_logsigma_trace)
                np.save('Beta_ksi_trace', Beta_ksi_trace)
                np.save('sigma_Beta_mu0_trace', sigma_Beta_mu0_trace)
                np.save('sigma_Beta_mu1_trace', sigma_Beta_mu1_trace)
                np.save('sigma_Beta_logsigma_trace', sigma_Beta_logsigma_trace)
                np.save('sigma_Beta_ksi_trace', sigma_Beta_ksi_trace)

                # Print traceplot thinned by 10
                xs       = np.arange(iter)
                xs_thin  = xs[0::10] # index 1, 11, 21, ...
                xs_thin2 = np.arange(len(xs_thin)) # index 1, 2, 3, ...

                loglik_trace_thin              = loglik_trace[0:iter:10,:]
                loglik_detail_trace_thin       = loglik_detail_trace[0:iter:10,:]
                R_trace_log_thin               = R_trace_log[0:iter:10,:,:]
                phi_knots_trace_thin           = phi_knots_trace[0:iter:10,:]
                range_knots_trace_thin         = range_knots_trace[0:iter:10,:]
                # GEV_knots_trace_thin           = GEV_knots_trace[0:iter:10,:,:]
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
                plt.savefig('loglik.pdf')
                plt.close()

                # ---- log-likelihood in details ----
                plt.subplots()
                for i in range(5):
                    plt.plot(xs_thin2, loglik_detail_trace_thin[:,i],label = i)
                    plt.annotate('piece ' + str(i), xy=(xs_thin2[-1], loglik_detail_trace_thin[:,i][-1]))
                plt.title('traceplot for detail log likelihood')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('log likelihood')
                plt.legend()
                plt.savefig('loglik_detail.pdf')
                plt.close()

                # ---- R_t ----
                plt.subplots()
                for i in [0,4,8]: # knots 0, 4, 8
                    for t in np.arange(Nt)[np.arange(Nt) % 15 == 0]:
                        plt.plot(xs_thin2, R_trace_log_thin[:,i,t], label = 'knot '+str(i) + ' time ' + str(t))
                        plt.annotate('knot ' + str(i) + ' time ' + str(t), xy=(xs_thin2[-1], R_trace_log_thin[:,i,t][-1]))
                plt.title('traceplot for some log(R_t)')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('log(R_t)s')
                plt.legend()
                plt.savefig('R_t.pdf')
                plt.close()

                # ---- phi ----
                plt.subplots()
                for i in range(k):
                    plt.plot(xs_thin2, phi_knots_trace_thin[:,i], label='knot ' + str(i))
                    plt.annotate('knot ' + str(i), xy=(xs_thin2[-1], phi_knots_trace_thin[:,i][-1]))
                plt.title('traceplot for phi')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('phi')
                plt.legend()
                plt.savefig('phi.pdf')
                plt.close()

                # ---- range ----
                plt.subplots()
                for i in range(k):
                    plt.plot(xs_thin2, range_knots_trace_thin[:,i], label='knot ' + str(i))
                    plt.annotate('knot ' + str(i), xy=(xs_thin2[-1], range_knots_trace_thin[:,i][-1]))
                plt.title('traceplot for range')
                plt.xlabel('iter thinned by 10')
                plt.ylabel('range')
                plt.legend()
                plt.savefig('range.pdf')
                plt.close()

                # ---- GEV ----

                # ## location coefficients
                # plt.subplots()
                # for j in range(Beta_mu0_m):
                #     plt.plot(xs_thin2, Beta_mu0_trace_thin[:,j], label = 'Beta_' + str(j))
                #     plt.annotate('Beta_' + str(j), xy=(xs_thin2[-1], Beta_mu0_trace_thin[:,j][-1]))
                # plt.title('traceplot for Beta_mu0 s')
                # plt.xlabel('iter thinned by 10')
                # plt.ylabel('Beta_mu0')
                # plt.legend()
                # plt.savefig('Beta_mu0.pdf')
                # plt.close()

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
                    plt.savefig(str(key)+'.pdf')
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
                    plt.savefig(str(key) + '.pdf')
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
                plt.savefig('Beta_logsigma.pdf')
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
                plt.savefig('sigma_Beta_xx.pdf')
                plt.close()

                # ## location Beta_xx prior variance
                # plt.subplots()
                # plt.plot(xs_thin2, sigma_Beta_mu0_trace_thin)
                # plt.title('sigma in Beta_mu0 ~ N(0, sigma^2)')
                # plt.xlabel('iter thinned by 10')
                # plt.ylabel('sigma')
                # plt.savefig('sigma_Beta_mu0.pdf')
                # plt.close()

                # ## location Beta_xx prior variance
                # plt.subplots()
                # plt.plot(xs_thin2, sigma_Beta_mu1_trace_thin)
                # plt.title('sigma in Beta_mu1 ~ N(0, sigma^2)')
                # plt.xlabel('iter thinned by 10')
                # plt.ylabel('sigma')
                # plt.savefig('sigma_Beta_mu1.pdf')
                # plt.close()

                # ## scale Beta_xx prior variance
                # plt.subplots()
                # plt.plot(xs_thin2, sigma_Beta_logsigma_trace_thin)
                # plt.title('sigma in Beta_logsigma ~ N(0, sigma^2)')
                # plt.xlabel('iter thinned by 10')
                # plt.ylabel('sigma')
                # plt.savefig('sigma_Beta_logsigma.pdf')
                # plt.close()

                # ## shape coefficients prior variance
                # plt.subplots()
                # plt.plot(xs_thin2, sigma_Beta_ksi_trace_thin)
                # plt.title('sigma in Beta_ksi ~ N(0, sigma^2)')
                # plt.xlabel('iter thinned by 10')
                # plt.ylabel('sigma')
                # plt.savefig('sigma_Beta_ksi.pdf')
                # plt.close()

        comm.Barrier() # block for drawing


    # %% 11. End of MCMC Saving Traceplot
    # 11. End of MCMC Saving Traceplot ------------------------------------------------------------------------------
    if rank == 0:
        end_time = time.time()
        print('total time: ', round(end_time - start_time, 1), ' seconds')
        print('true R: ', R_at_knots)
        np.save('loglik_trace', loglik_trace)
        np.save('loglik_detail_trace', loglik_detail_trace)
        np.save('R_trace_log', R_trace_log)
        np.save('phi_knots_trace', phi_knots_trace)
        np.save('range_knots_trace', range_knots_trace)
        # np.save('GEV_knots_trace', GEV_knots_trace)
        np.save('Beta_mu0_trace', Beta_mu0_trace)
        np.save('Beta_mu1_trace', Beta_mu1_trace)
        np.save('Beta_logsigma_trace', Beta_logsigma_trace)
        np.save('Beta_ksi_trace', Beta_ksi_trace)
        np.save('sigma_Beta_mu0_trace', sigma_Beta_mu0_trace)
        np.save('sigma_Beta_mu1_trace', sigma_Beta_mu1_trace)
        np.save('sigma_Beta_logsigma_trace', sigma_Beta_logsigma_trace)
        np.save('sigma_Beta_ksi_trace', sigma_Beta_ksi_trace)
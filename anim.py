import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# fps = 30
# nSeconds = 5
# snapshots = [ np.random.rand(5,5) for _ in range( nSeconds * fps ) ]

# # First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure( figsize=(8,8) )

# a = snapshots[0]
# im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)

# def animate_func(i):
#     if i % fps == 0:
#         print( '.', end ='' )

#     im.set_array(snapshots[i])
#     return [im]

# anim = animation.FuncAnimation(
#                                fig, 
#                                animate_func, 
#                                frames = nSeconds * fps,
#                                interval = 1000 / fps, # in ms
#                                )
# plt.show()
# anim.save('test_anim.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])

"""
This is a MCMC sampler that constantly gets updated
Scratch work and modifications are done in this file

Dec 6 2023, adding covariate for theta GEVs to the model

Notes on (installation of) Rpy2
Work around of conda+rpy2: conda install rpy2 also installs an r-base
use the conda installed R, don't use the default R on misspiggy to avoid issue (i.e. change the default R path to the end the $PATH)
Alternatively, split the MCMC into three pieces: Python generate data --> R generate X design --> Python run MCMC

Jan 23 2024, theta(s) = Beta_0 + Beta_1 * Elev(s) + splines(s) @ Beta_splines
More specifically,
    mu(s,t) = mu_0(s) + mu_1(s) * t 
    logsigma(s,t) = logsigma(s)
    ksi(s,t) = ksi(s)
where 
    t           = - Nt/2, -Nt/2 + 1, ..., 0, 1, ..., Nt/2 - 1
    mu_0(s)     = Beta_mu0_0 + Beta_mu0_1 * Elev(s) + splines(s) @ Beta_mu0_splines
    mu_1(s)     = Beta_mu1_0 + Beta_mu1_1 * Elev(s) + splines(s) @ Beta_mu1_splines
    logsigma(s) = Beta_logsigma_0 + Beta_logsigma_1 * x + Beta_logsigma_2 * y       ............ still the very simple linear surface
    ksi(s)      = Beta_ksi_0 + Beta_ksi_1 * x + Beta_ksi_2 * y                      ............ still the very simple linear surface
so we have
    Beta_mu0    = (Beta_mu0_0, Beta_mu0_1, Beta_mu0_splines)
    C_mu0(s)    = (1, Elev(s), splines(s))

Note on heatmap:
plotgrid_xy is meshgrid(order='xy') fills horizontally (x changes first, then y changes), so no need tranpose in imshow
gs_xy is meshgrid(order='ij') fills vertically (y changes first, then x changes), so NEED tranpose in imshow

Note on mgcv
It appears the basis matrix produced by smoothCon is slightly (~ 3 to 4 decimal places) different between machines
"""
# Require:
#   - utilities.py
# Example Usage:
# mpirun -n 2 -output-filename folder_name python3 MCMC.py > pyout.txt &
# mpirun -n 2 python3 MCMC.py > output.txt 2>&1 &
if __name__ == "__main__":
    # %% for reading seed from bash
    import sys
    data_seed = int(sys.argv[1]) if len(sys.argv) == 2 else 2345

    # %% Imports
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

    try: # data_seed is defined when python MCMC.py
        data_seed
    except: # when running on local machine interactively
        data_seed = 2345
    finally:
        if rank == 0: print('data_seed: ', data_seed)

    #####################################################################################################################
    # Generating Dataset ################################################################################################
    #####################################################################################################################
    # %% 0. Parameter Settings and Generate Knots 
    # 0. Parameter Settings and Generate Knots --------------------------------------

    # ----------------------------------------------------------------------------------------------------------------
    # Numbers - Ns, Nt, n_iters
    
    np.random.seed(data_seed)
    Nt = 16 # number of time replicates
    Ns = 100 # number of sites/stations
    n_iters = 5000

    # ----------------------------------------------------------------------------------------------------------------
    # Sites - random uniformly (x,y) generate site locations
    
    sites_xy = np.random.random((Ns, 2)) * 10
    sites_x = sites_xy[:,0]
    sites_y = sites_xy[:,1]

    # ----------------------------------------------------------------------------------------------------------------
    # Elevation Function - simple elevation function 1/5(|x-5| + |y-5|)
    def elevation_func(x,y):
        return(np.abs(x-5)/5 + np.abs(y-5)/5)

    # ----------------------------------------------------------------------------------------------------------------
    # Knots - uniform grid of 9 knots, should do this programatically...

    k = 9 # number of knots
    x_pos = np.linspace(0,10,5,True)[1:-1]
    y_pos = np.linspace(0,10,5,True)[1:-1]
    X_pos, Y_pos = np.meshgrid(x_pos,y_pos)
    knots_xy = np.vstack([X_pos.ravel(), Y_pos.ravel()]).T
    knots_x = knots_xy[:,0]
    knots_y = knots_xy[:,1]

    # ----------------------------------------------------------------------------------------------------------------
    # Basis Parameters - for the Gaussian and Wendland Basis

    bandwidth = 4 # range for the gaussian kernel
    radius = 4 # radius of infuence for basis, 3.5 might make some points closer to the edge of circle, might lead to numerical issues
    radius_from_knots = np.repeat(radius, k) # ?influence radius from a knot?
    assert k == len(knots_xy)

    # ----------------------------------------------------------------------------------------------------------------
    # Data Model Parameters - X_star = R^phi * g(Z)

    ## Stable S_t --> R_t
    gamma = 0.5 # this is the gamma that goes in rlevy, gamma_at_knots
    delta = 0.0 # this is the delta in levy, stays 0

    ## g(Z)
    nu = 0.5 # exponential kernel for matern with nu = 1/2
    sigsq = 1.0 # sill for Z
    range_at_knots = np.sqrt(0.3*knots_x + 0.4*knots_y)/2 # range for spatial Matern Z

    ### scenario 1
    # phi_at_knots = 0.65-np.sqrt((knots_x-3)**2/4 + (knots_y-3)**2/3)/10
    ### scenario 2
    phi_at_knots = 0.65-np.sqrt((knots_x-5.1)**2/5 + (knots_y-5.3)**2/4)/11.6
    ### scenario 3
    # phi_at_knots = 0.37 + 5*(scipy.stats.multivariate_normal.pdf(knots_xy, mean = np.array([2.5,3]), cov = 2*np.matrix([[1,0.2],[0.2,1]])) + 
    #                          scipy.stats.multivariate_normal.pdf(knots_xy, mean = np.array([7,7.5]), cov = 2*np.matrix([[1,-0.2],[-0.2,1]])))

    # ----------------------------------------------------------------------------------------------------------------
    # Extreme Value Parameters - GEV(mu, sigma, ksi)
    # Constant parameters
    mu = 0.0 # GEV location
    sigma = 1.0 # GEV scale
    ksi = 0.2 # GEV shape

    # ----------------------------------------------------------------------------------------------------------------
    # Extreme Value Parameters - GEV(mu, sigma, ksi)
    # Linear surfaces -- Specify X and Beta here

    ## Location mu0 ##
    # # simple case of mu0(s) = Beta_mu0_0 + Beta_mu0_1*x(s) + Beta_mu0_2*y(s)
    # Beta_mu0_dict = {
    #     'Beta_mu0_0' : 0.0, # intercept for mu0
    #     'Beta_mu0_1' : 0.1, # slope of beta 1 for mu0
    #     'Beta_mu0_2' : 0.1  # slope of beta 2 for mu0
    # }
    # Beta_mu0_m = len(Beta_mu0_dict)
    # C_mu0          = np.full(shape=(Beta_mu0_m, Ns, Nt), fill_value = np.nan) # mu0 design matrix
    # C_mu0[0, :, :] = 1.0 # column of 1 for the intercept
    # C_mu0[1, :, :] = np.tile(sites_x, reps = (Nt, 1)).T # column of x for beta 1
    # C_mu0[2, :, :] = np.tile(sites_y, reps = (Nt, 1)).T # column of y for beta 2


    ## Scale sigma - logscale ##
    # simple case of logsigma(s) = Beta_logsigma_0 + Beta_logsigma_1 * x(s) + Beta_logsigma_2 * y(s)
    Beta_logsigma_dict = {
        'Beta_logsigma_0' : np.log(sigma), # intercept for logsimga
        'Beta_logsigma_1' : 0.01, # slope of beta 1 for logsigma
        'Beta_logsigma_2' : 0.03, # slope of beta 2 for logsigma
    }
    Beta_logsigma_m = len(Beta_logsigma_dict)
    C_logsigma        = np.full(shape = (Beta_logsigma_m, Ns, Nt), fill_value = np.nan) # log(sigma) design matrix
    C_logsigma[0,:,:] = 1.0
    C_logsigma[1,:,:] = np.tile(sites_x, reps = (Nt, 1)).T
    C_logsigma[2,:,:] = np.tile(sites_y, reps = (Nt, 1)).T
    ## coefficients
    Beta_logsigma = np.array(list(Beta_logsigma_dict.values()))
    ## actual surface for sigma(s)
    sigma_matrix = np.exp((C_logsigma.T @ Beta_logsigma).T)

    ## Shape ksi ##
    # simple case of ksi(s) = Beta_ksi_0 + Beta_ksi_1 * x(s) + Beta_ksi_2 * y(s)
    Beta_ksi_dict = {
        'Beta_ksi_0' : ksi, # intercept for ksi
        'Beta_ksi_1' : 0.02, # slope of beta 1 for ksi
        'Beta_ksi_2' : -0.02, # slope of beta 2 for ksi
    }
    Beta_ksi_m = len(Beta_ksi_dict)
    C_ksi        = np.full(shape = (Beta_ksi_m, Ns, Nt), fill_value = np.nan) # ksi design matrix
    C_ksi[0,:,:] = 1.0
    C_ksi[1,:,:] = np.tile(sites_x, reps = (Nt, 1)).T
    C_ksi[2,:,:] = np.tile(sites_y, reps = (Nt, 1)).T
    ## coefficients
    Beta_ksi = np.array(list(Beta_ksi_dict.values()))
    ## actual surface for ksi(s)
    ksi_matrix = (C_ksi.T @ Beta_ksi).T

    # ----------------------------------------------------------------------------------------------------------------
    # Extreme Value Parameters - GEV(mu, sigma, ksi)
    # mu_0(s)     = Beta_mu0_0 + Beta_mu0_1 * Elev(s) + splines(s) @ Beta_mu0_splines

    # "knots" and prediction sites for splines -----------------------------------------------------------------------
    gs_x        = np.linspace(0, 10, 41)
    gs_y        = np.linspace(0, 10, 41)
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

    # Location mu_0(s) ----------------------------------------------------------------------------------------------
    ## coefficients
    Beta_mu0_0              = 0
    Beta_mu0_1              = 0.1
    Beta_mu0_splines_m      = 12
    Beta_mu0_splines        = np.array([0.1]*Beta_mu0_splines_m)
    Beta_mu0                = np.concatenate(([Beta_mu0_0], [Beta_mu0_1], Beta_mu0_splines))
    Beta_mu0_m              = len(Beta_mu0)
    Beta_mu0_block_idx_size = 7
    ## covariates
    C_mu0_splines = np.array(r('''
                                basis      <- smoothCon(s(x, y, k = {Beta_mu0_splines_m}, fx = TRUE), data = gs_xy_df)[[1]]
                                basis_site <- PredictMat(basis, data = sites_xy_df)
                                basis_site
                            '''.format(Beta_mu0_splines_m = Beta_mu0_splines_m))) # shaped(Ns, Beta_mu0_splines_m)
    C_mu0_1t      = np.column_stack((np.ones(Ns),
                                    elevation_func(sites_xy[:,0], sites_xy[:,1]),
                                    C_mu0_splines))
    C_mu0         = np.tile(C_mu0_1t.T[:,:,None], reps = (1, 1, Nt))
    ## mu0(s,t)
    mu0_matrix = (C_mu0.T @ Beta_mu0).T      

    # Location mu_1(s) ----------------------------------------------------------------------------------------------
    ## coefficients
    Beta_mu1_0              = 0
    Beta_mu1_1              = 0.01
    Beta_mu1_splines_m      = 12
    Beta_mu1_splines        = np.array([0.01] * Beta_mu1_splines_m)
    Beta_mu1                = np.concatenate(([Beta_mu1_0], [Beta_mu1_1], Beta_mu1_splines))
    Beta_mu1_m              = len(Beta_mu1)
    Beta_mu1_block_idx_size = 7
    ## covariates
    C_mu1_splines = np.array(r('''
                                basis      <- smoothCon(s(x, y, k = {Beta_mu1_splines_m}, fx = TRUE), data = gs_xy_df)[[1]]
                                basis_site <- PredictMat(basis, data = sites_xy_df)
                                basis_site
                            '''.format(Beta_mu1_splines_m = Beta_mu1_splines_m))) # shaped(Ns, Beta_mu1_splines_m)
    C_mu1_1t      = np.column_stack((np.ones(Ns),
                                    elevation_func(sites_xy[:,0], sites_xy[:,1]),
                                    C_mu1_splines))
    C_mu1         = np.tile(C_mu1_1t.T[:,:,None], reps = (1, 1, Nt))
    ## mu1(s,t)
    mu1_matrix = (C_mu1.T @ Beta_mu1).T

    # Location mu(s,t) -----------------------------------------------------------------------------------------------
    Time = np.linspace(-Nt/2, Nt/2-1, Nt)
    mu_matrix = mu0_matrix + mu1_matrix * Time

    # logsigma
    pass

    # ksi
    pass

    # Create Coefficient Blocks
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

    # ----------------------------------------------------------------------------------------------------------------
    # Beta Coefficient Prior Parameter - sigma_Beta_xx ~ Halt-t(4)

    ## just initial values, right? Theses are not "truth"
    sigma_Beta_mu0      = 1.0
    sigma_Beta_mu1      = 1.0
    sigma_Beta_logsigma = 1.0
    sigma_Beta_ksi      = 1.0

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

    # trial run posterior variance matrix for range rho
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

    # # trial run posterior variance matrix for constant GEV
    # GEV_post_cov = np.array([[2.88511464e-04, 1.13560517e-04, 0],
    #                         [1.13560517e-04, 6.40933053e-05,  0],
    #                         [0         , 0         , 1e-4]])
    # # GEV_post_cov = 1e-4 * np.identity(3)

    # # posterior/proposal variance matrix for linear surface mu
    # Beta_mu0_post_cov = np.array([
    #     [ 7.48055192e-03, -6.21600956e-04, -6.51251126e-04],
    #     [-6.21600956e-04,  2.19025810e-04, -4.88013752e-05],
    #     [-6.51251126e-04, -4.88013752e-05,  2.97374368e-04]])
    # # Beta_mu0_post_cov = 1e-4 * np.identity(Beta_mu0_m)
    # assert Beta_mu0_m == Beta_mu0_post_cov.shape[0]

    # posterior/proposal variance matrix for linear surface logsigma
    Beta_logsigma_post_cov = np.array([
        [ 1.09295029e-03, -5.57350333e-05, -1.26948891e-04],
        [-5.57350333e-05,  5.67940538e-05, -3.05545811e-05],
        [-1.26948891e-04, -3.05545811e-05,  7.49504590e-05]])
    # Beta_logsigma_post_cov = 1e-4 * np.identity(Beta_logsigma_m)
    assert Beta_logsigma_m == Beta_logsigma_post_cov.shape[0]

    # posterior/proposal variance matrix for linear surface ksi
    Beta_ksi_post_cov = np.array([
        [ 1.68899920e-03, -1.35994062e-04, -1.24227290e-04],
        [-1.35994062e-04,  2.85659453e-05, -2.05585256e-06],
        [-1.24227290e-04, -2.05585256e-06,  2.60318359e-05]])
    # Beta_ksi_post_cov = 1e-4 * np.identity(Beta_ksi_m)
    assert Beta_ksi_m == Beta_ksi_post_cov.shape[0]

    # trial run posterior variance matrix for Beta_mu0
    Beta_mu0_all_post_cov = 1e-4 * np.identity(Beta_mu0_m)
    assert Beta_mu0_all_post_cov.shape[0] == Beta_mu0_m
    Beta_mu0_block_post_cov_dict = {}
    for key in Beta_mu0_block_idx_dict.keys():
        start_idx                         = Beta_mu0_block_idx_dict[key][0]
        end_idx                           = Beta_mu0_block_idx_dict[key][-1]+1
        Beta_mu0_block_post_cov_dict[key] = Beta_mu0_all_post_cov[start_idx:end_idx, start_idx:end_idx]

    # trial run posterior variance matrix for Beta_mu1
    Beta_mu1_all_post_cov                 = 1e-4 * np.identity(Beta_mu1_m)
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
            # 'sigma_Beta_mu0'      : (2.4**2),
            'sigma_Beta_mu0'      : 0.03749589, # from trial run
            # 'sigma_Beta_logsigma' : (2.4**2),
            'sigma_Beta_logsigma' : 0.24878523, # from trial run
            # 'sigma_Beta_ksi'      : (2.4**2)
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
            # 'GEV'                 : 0,
            # 'Beta_mu0'            : 0,
            'Beta_logsigma'       : 0,
            'Beta_ksi'            : 0,
            'sigma_Beta_mu0'      : 0,
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

    # %% 1. Plot Spatial Domain 
    # 1. Plot Space -------------------------------------------------------------------------------------
    
    if rank == 0: # Plot the space
        plotgrid_x = np.linspace(0.1,10,25)
        plotgrid_y = np.linspace(0.1,10,25)
        plotgrid_X, plotgrid_Y = np.meshgrid(plotgrid_x, plotgrid_y)
        plotgrid_xy = np.vstack([plotgrid_X.ravel(), plotgrid_Y.ravel()]).T
        radius_from_knots = np.repeat(radius, k)
        fig, ax = plt.subplots()
        ax.plot(sites_x, sites_y, 'b.', alpha = 0.4)
        ax.plot(knots_x, knots_y, 'r+')
        space_rectangle = plt.Rectangle(xy = (0,0), width = 10, height = 10,
                                        fill = False, color = 'black')
        for i in range(k):
            circle_i = plt.Circle((knots_xy[i,0],knots_xy[i,1]), radius_from_knots[0], 
                            color='r', fill=True, fc='grey', ec = 'red', alpha = 0.2)
            ax.add_patch(circle_i)
        ax.add_patch(space_rectangle)
        plt.xlim([-2,12])
        plt.ylim([-2,12])
        # plt.show()
        # plt.savefig('point_space.pdf')
        plt.close()
    
    if rank == 0: # Plot the elevation
        plotgrid_elevations = elevation_func(plotgrid_xy[:,0], plotgrid_xy[:,1])
        graph, ax = plt.subplots()
        heatmap = ax.imshow(plotgrid_elevations.reshape(25,25), cmap='hot',interpolation='nearest',extent=[0,10,10,0])
        ax.invert_yaxis()
        graph.colorbar(heatmap)
        plt.title('elevation heatplot')
        # plt.show()
        # plt.savefig('elevation.pdf')
        plt.close()        

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


    # %% 3. Generate K then Z, and W = g(Z)
    # 3. Generate covariance matrix, Z, and W ------------------------------------------------------------------------------

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
        heatmap = ax.imshow(range_vec_for_plot.reshape(25,25), cmap ='hot', interpolation='nearest')
        ax.invert_yaxis()
        graph.colorbar(heatmap)
        # plt.show()
        # plt.savefig('heatmap range surface.pdf')
        plt.close()

        # # 3d range surface plot
        # range_vec_for_plot = gaussian_weight_matrix_for_plot @ range_at_knots
        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot(projection='3d')
        # ax2.plot_trisurf(plotgrid_xy[:,0], plotgrid_xy[:,1], range_vec_for_plot, linewidth=0.2, antialiased=True)
        # ax2.set_xlabel('X')
        # ax2.set_ylabel('Y')
        # ax2.set_zlabel('phi(s)')
        # ax2.scatter(knots_x, knots_y, range_at_knots, c='red', marker='o', s=100)
        # plt.show()
        # plt.savefig('3d range surface.pdf')
        # plt.close()


    # %% 4. Generate R^phi Scaling Factor
    # 4. Generate R^phi Scaling Factor -------------------------------------------------------------------------------------

    ## phi_vec
    phi_vec = gaussian_weight_matrix @ phi_at_knots
    # phi_vec = one_weight_matrix @ phi_at_knots

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

    if rank == 0: # plotting the phi surface
        # heatplot of phi surface
        phi_vec_for_plot = gaussian_weight_matrix_for_plot @ phi_at_knots
        graph, ax = plt.subplots()
        heatmap = ax.imshow(phi_vec_for_plot.reshape(25,25), cmap ='hot', interpolation='nearest', extent = [0, 10, 10, 0])
        ax.invert_yaxis()
        graph.colorbar(heatmap)
        # plt.show()
        # plt.savefig('heatmap phi surface.pdf')
        plt.close()

        # # 3d phi surface
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
        # plt.savefig('3d phi surface.pdf')
        # plt.close()

    # %% 5. Generate and Plot GEV Surfaces
    # 5. Generate and Plot GEV Surfaces -------------------------------------------------------------------------------------

    # Location #
    # ## coefficients
    # Beta_mu0 = np.array(list(Beta_mu0_dict.values())) # extract the coefficient values
    # assert Beta_mu0_m == len(Beta_mu0) # number of betas for mu0
    # ## actual surface for mu0(s)
    # mu0_matrix = (C_mu0.T @ Beta_mu0).T
    # if rank == 0: # plotting the location surface
    #     ## heatplot of mu0 surface
    #     C_mu0_plot = np.full(shape = (Beta_mu0_m, len(plotgrid_xy), Nt), fill_value = np.nan) # 
    #     C_mu0_plot[0, :, :] = 1.0
    #     C_mu0_plot[1, :, :] = np.tile(plotgrid_xy[:,0], reps=(Nt,1)).T
    #     C_mu0_plot[2, :, :] = np.tile(plotgrid_xy[:,1], reps=(Nt,1)).T
    #     mu_surface_plot = (C_mu0_plot.T @ Beta_mu0).T
    #     graph, ax = plt.subplots()
    #     heatmap = ax.imshow(mu_surface_plot[:,0].reshape(25,25), cmap='hot',interpolation='nearest',extent=[0,10,10,0])
    #     ax.invert_yaxis()
    #     graph.colorbar(heatmap)
    #     # plt.show()
    #     plt.savefig('heatmap mu0 surface.pdf')
    #     plt.close()

    #     ## 3d plot of mu0 surface
    #     # fig = plt.figure()
    #     # ax = fig.add_subplot(projection='3d')
    #     # ax.plot_surface(plotgrid_X, plotgrid_Y, np.matrix(mu_surface_plot[:,0]).reshape(25,25))
    #     # ax.set_xlabel('X')
    #     # ax.set_ylabel('Y')
    #     # ax.set_zlabel('mu0(s)')

    # Scale #
    if rank == 0: # heatplot of sigma(s) surface
        C_logsigma_plot = np.full(shape = (Beta_logsigma_m, len(plotgrid_xy), Nt), fill_value = np.nan)
        C_logsigma_plot[0, :, :] = 1.0
        C_logsigma_plot[1, :, :] = np.tile(plotgrid_xy[:,0], reps=(Nt,1)).T
        C_logsigma_plot[2, :, :] = np.tile(plotgrid_xy[:,1], reps=(Nt,1)).T
        sigma_surface_plot = np.exp((C_logsigma_plot.T @ Beta_logsigma).T)
        graph, ax = plt.subplots()
        heatmap = ax.imshow(sigma_surface_plot[:,0].reshape(25,25), cmap='hot',interpolation='nearest',extent=[0,10,10,0])
        ax.invert_yaxis()
        graph.colorbar(heatmap)
        plt.title('sigma surface heatplot')
        # plt.show()
        # plt.savefig('heatmap sigma surface.pdf')
        plt.close()

    # Shape #
    if rank == 0: # heatplot of ksi(s) surface
        C_ksi_plot = np.full(shape = (Beta_ksi_m, len(plotgrid_xy), Nt), fill_value = np.nan) # 
        C_ksi_plot[0, :, :] = 1.0
        C_ksi_plot[1, :, :] = np.tile(plotgrid_xy[:,0], reps=(Nt,1)).T
        C_ksi_plot[2, :, :] = np.tile(plotgrid_xy[:,1], reps=(Nt,1)).T
        ksi_surface_plot = (C_ksi_plot.T @ Beta_ksi).T
        graph, ax = plt.subplots()
        heatmap = ax.imshow(ksi_surface_plot[:,0].reshape(25,25), cmap='hot',interpolation='nearest',extent=[0,10,10,0])
        ax.invert_yaxis()
        graph.colorbar(heatmap)
        # plt.show()
        # plt.savefig('heatmap ksi surface.pdf')
        plt.close()

    # # Use mgcv tp spline as covariate for the marginal parameter surfaces
    # if rank == 0: # do R stuff (parameter surface, covariate, and coefficient) under worker 0

    #     # "knots" and prediction sites for splines
    #     gs_x        = np.linspace(0, 10, 41)
    #     gs_y        = np.linspace(0, 10, 41)
    #     gs_xy       = np.full(shape = (len(gs_x) * len(gs_y),2), fill_value = np.nan)
    #     current_row = 0
    #     for i in range(len(gs_x)):
    #         for j in range(len(gs_y)):
    #             gs_xy[current_row,0] = gs_x[i]
    #             gs_xy[current_row,1] = gs_y[j]
    #             current_row          += 1

    #     gs_x_ro     = numpy2rpy(gs_x)        # Convert to R object
    #     gs_y_ro     = numpy2rpy(gs_y)        # Convert to R object
    #     gs_xy_ro    = numpy2rpy(gs_xy)       # Convert to R object
    #     sites_xy_ro = numpy2rpy(sites_xy)    # Convert to R object

    #     r.assign("gs_x_ro", gs_x_ro)         # Note: this is a matrix in R, not df
    #     r.assign("gs_y_ro", gs_y_ro)         # Note: this is a matrix in R, not df
    #     r.assign("gs_xy_ro", gs_xy_ro)       # Note: this is a matrix in R, not df
    #     r.assign('sites_xy_ro', sites_xy_ro) # Note: this is a matrix in R, not df

    #     r("save(gs_x_ro, file='gs_x_ro.gzip', compress=TRUE)")
    #     r("save(gs_y_ro, file='gs_y_ro.gzip', compress=TRUE)")
    #     r("save(gs_xy_ro, file='gs_xy_ro.gzip', compress=TRUE)")
    #     r("save(sites_xy_ro, file='sites_xy_ro.gzip', compress=TRUE)")

    #     # "knot splines" and "site splines"
    #     mgcv = importr('mgcv')
    #     C_knot_spline_ro = r('''
    #                         gs_xy_df <- as.data.frame(gs_xy_ro)
    #                         colnames(gs_xy_df) <- c('x','y')
    #                         basis <- smoothCon(s(x, y, k = {Beta_mu0_m}, fx = TRUE), data = gs_xy_df)[[1]]
    #                         basis$X
    #                     '''.format(Beta_mu0_m = Beta_mu0_m))
    #     C_site_spline_ro = r('''
    #                         sites_xy_df           <- as.data.frame(sites_xy_ro)
    #                         colnames(sites_xy_df) <- c('x','y')
    #                         C_site_spline         <- PredictMat(basis, data = sites_xy_df)
    #                     ''')
    #     # '''         gs_xy_df <- as.data.frame(gs_xy_ro)
    #     #     colnames(gs_xy_df) <- c('x','y')
    #     #     basis <- smoothCon(s(x, y, k = {Beta_mu}, fx = TRUE), data = gs_xy_df)[[1]]
    #     #   '''.format(Beta_mu = 100)

    #     # LOCATION mu #
    #     ## mu values at sites
    #     mu0_1t            = mu_surf_generator((sites_x, sites_y))
    #     mu0_matrix        = np.tile(mu0_1t, reps = (Nt, 1)).T

    #     ## Covariates
    #     C_mu0_1t          = np.array(r('C_site_spline'))                 # shape (Ns, Nc) Nc: number of covariate/splines
    #     C_mu0             = np.tile(C_mu0_1t.T[:,:,None], reps=(1,1,Nt)) # Tranposed each C_mu0_1t, now Shaped (Nc, Ns, Nt)
    #                                                                      # for ease of matmul in np (last 2 indexes)
    #                                                                      # laster do (C.T @ Beta).T
                                                                         
    #     ## Regression Spline to get initial beta values
    #     mu0_1t_ro = numpy2rpy(mu0_1t)
    #     r.assign("mu0_1t_ro", mu0_1t_ro)
    #     r("save(mu0_1t_ro, file='mu0_1t_ro.gzip', compress=TRUE)")
    #     Beta_mu0 = np.array(r('''
    #                           Beta_mu0 <- coef(lm(c(mu0_1t_ro) ~ C_site_spline-1))
    #                           Beta_mu0'''))

    #     ## plotting the actual surface
    #     mu_surf_grid      = mu_surf_generator((gs_x, gs_y), mesh_type='structured')
    #     mu_surf_grid_ro   = numpy2rpy(mu_surf_grid)
    #     r.assign("mu_surf_grid_ro", mu_surf_grid_ro)
    #     r("save(mu_surf_grid_ro, file='mu_surf_grid_ro.gzip', compress=TRUE)")
    #     # # note that for gaussian surface, the first index runs along the x axis, and the second index runs along the y axis
    #     # mu_surf_grid.shape
    #     # mu_surf_grid
    #     # mu_surf_generator.plot()
    #     # # This is in "contradiction" to the order of filling in heatmap of imshow, which require the first index runs along the vertical y axis, and the second index runs along the horizontal x axis,
    #     # # so we should transpose the mu_surf_grid in order to propoerly use heatmap
    #     # # Also, the imshow (heatmap) function literally took the matrix as it is and "heat map" its values, so the first row gets printed on top, and the last row gets printed bottom --- this might not be what we want, as we'd like y axis to go from down to top
    #     # # hence we also need to invert the yaxis to achieve this (also don't forget to adjust the bottom top axis label in the extent parameter)
    #     graph, ax = plt.subplots()
    #     heatmap = ax.imshow(mu_surf_grid.T, cmap='hot', interpolation='nearest', extent=[0,10,10,0])  # extent (left, right, bottom, top)
    #     ax.invert_yaxis()
    #     graph.colorbar(heatmap)
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.title('mu(s) surface heatplot')
    #     plt.show()
    #     plt.savefig('heatmap mu surface.pdf')
    #     plt.close()
    # else: # broadcast from worker 0 to other workers
    #     mu0_matrix = None
    #     C_mu0      = None
    # mu0_matrix     = comm.bcast(mu0_matrix, root = 0)
    # C_mu0          = comm.bcast(C_mu0, root = 0)

    # sigma_surf_grid = sigma_surf_generator((gs_x, gs_y), mesh_type='structured')
    # sigma_surf_generator.plot()
    # graph, ax = plt.subplots()
    # heatmap = ax.imshow(np.exp(sigma_surf_grid.T), cmap='hot', interpolation='nearest', extent=[0,10,10,0])
    # ax.invert_yaxis()
    # graph.colorbar(heatmap)
        
    # %% 5.2 Plotting the GEV Surfaces
    # 5.2 Plotting the GEV Surfaces -------------------------------------------------------------------------------------
    if rank == 0:
        # mu0(s) plot
        C_mu0_plot        = np.full(shape = (Beta_mu0_m, len(gs_xy), Nt), fill_value = np.nan)
        C_mu0_plot[0,:,:] = 1.0
        C_mu0_plot[1,:,:] = np.tile(elevation_func(gs_xy[:,0], gs_xy[:,1])[:,None], reps=(1,Nt))
        # C_mu0_plot[1,:,:] = 0.0
        C_mu0_plot[2:Beta_mu0_m,:,:] = np.tile(np.array(r('''
                                                    basis <- smoothCon(s(x, y, k = {Beta_mu0_splines_m}, fx = TRUE), data = gs_xy_df)[[1]]
                                                    basis$X
                                                '''.format(Beta_mu0_splines_m = Beta_mu0_splines_m))).T[:,:,None],
                                                reps=(1,1,Nt))
        mu0_surface_plot = (C_mu0_plot.T @ Beta_mu0).T
        graph, ax = plt.subplots()
        heatmap = ax.imshow(mu0_surface_plot[:,0].reshape(len(gs_x),len(gs_y)).T, cmap='hot',interpolation='nearest',extent=[0,10,10,0])
        ax.invert_yaxis()
        graph.colorbar(heatmap)
        ax.set_title('mu0 surface plot')
        # plt.show()
        # print(mu0_surface_plot[:,0])
        # plt.savefig('heatmap mu0 surface.pdf')
        # plt.close()

        # mu(s,t) plot
        import matplotlib.animation as animation
        plt.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'
        C_mu1_plot = C_mu0_plot.copy()
        mu_surface_plot = (C_mu0_plot.T @ Beta_mu0).T + (C_mu1_plot.T @ Beta_mu1).T * Time
        graph, ax = plt.subplots()
        heatmap = ax.imshow(mu_surface_plot[:,0].reshape(len(gs_x),len(gs_y)).T, 
                            cmap='hot',interpolation='nearest',extent=[0,10,10,0],
                            vmin = np.min(mu_surface_plot), vmax = np.max(mu_surface_plot))
        ax.invert_yaxis()
        graph.colorbar(heatmap)
        title = ax.set_title('mu(s,t) surface plot')
        def animate(i):
            heatmap.set_array(mu_surface_plot[:,i].reshape(len(gs_x),len(gs_y)).T)
            title.set_text('Time {0}'.format(i))
            return [graph]
        anim = animation.FuncAnimation(graph,animate,frames = Nt, interval = 200)
        anim.save('muSurfaceOverTime.mp4', fps=1)
        # plt.show()
        plt.close()


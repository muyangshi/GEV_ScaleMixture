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
    from time import strftime, localtime
    from utilities import *
    import gstools as gs
    import rpy2.robjects as robjects
    from rpy2.robjects import r 
    from rpy2.robjects.numpy2ri import numpy2rpy
    from rpy2.robjects.packages import importr
    import pickle

    try: # data_seed is defined when python MCMC.py
        data_seed
    except: # when running on local machine interactively
        data_seed = 2345 # default seed
    finally:
        print('data_seed: ', data_seed)
    np.random.seed(data_seed)

    print('Pareto: ', norm_pareto)

    # %% Simulation Setup

    # Spatial Domain Setup --------------------------------------------------------------------------------------------

    # Numbers - Ns, Nt --------------------------------------------------------
    
    np.random.seed(data_seed)
    Nt = 50 # number of time replicates
    Ns = 10 # number of sites/stations
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
    # minX, maxX = np.floor(np.min(sites_x)), np.ceil(np.max(sites_x))
    # minY, maxY = np.floor(np.min(sites_y)), np.ceil(np.max(sites_y))

    minX, maxX = (0.0, 10.0)
    minY, maxY = (0.0, 10.0)

    # Elevation Function ------------------------------------------------------

    elev_surf_generator = gs.SRF(gs.Gaussian(dim=2, var = 1, len_scale = 2), seed=data_seed)
    elevations = elev_surf_generator((sites_x, sites_y))

    # Knots locations w/ isometric grid ---------------------------------------

    # N_outer_grid = 9
    # h_dist_between_knots     = (maxX - minX) / (int(2*np.sqrt(N_outer_grid))-1)
    # v_dist_between_knots     = (maxY - minY) / (int(2*np.sqrt(N_outer_grid))-1)
    # x_pos                    = np.linspace(minX + h_dist_between_knots/2, maxX + h_dist_between_knots/2, 
    #                                        num = int(2*np.sqrt(N_outer_grid)))
    # y_pos                    = np.linspace(minY + v_dist_between_knots/2, maxY + v_dist_between_knots/2, 
    #                                        num = int(2*np.sqrt(N_outer_grid)))
    # x_outer_pos              = x_pos[0::2]
    # x_inner_pos              = x_pos[1::2]
    # y_outer_pos              = y_pos[0::2]
    # y_inner_pos              = y_pos[1::2]
    # X_outer_pos, Y_outer_pos = np.meshgrid(x_outer_pos, y_outer_pos)
    # X_inner_pos, Y_inner_pos = np.meshgrid(x_inner_pos, y_inner_pos)
    # knots_outer_xy           = np.vstack([X_outer_pos.ravel(), Y_outer_pos.ravel()]).T
    # knots_inner_xy           = np.vstack([X_inner_pos.ravel(), Y_inner_pos.ravel()]).T
    # knots_xy                 = np.vstack((knots_outer_xy, knots_inner_xy))
    # knots_id_in_domain       = [row for row in range(len(knots_xy)) if (minX < knots_xy[row,0] < maxX and minY < knots_xy[row,1] < maxY)]
    # knots_xy                 = knots_xy[knots_id_in_domain]
    # knots_x                  = knots_xy[:,0]
    # knots_y                  = knots_xy[:,1]
    # k                        = len(knots_id_in_domain)

    # uniform grid
    x_pos        = np.linspace(0,10,5,True)[1:-1]
    y_pos        = np.linspace(0,10,5,True)[1:-1]
    X_pos, Y_pos = np.meshgrid(x_pos,y_pos)
    knots_xy     = np.vstack([X_pos.ravel(), Y_pos.ravel()]).T
    knots_x      = knots_xy[:,0]
    knots_y      = knots_xy[:,1]
    k            = len(knots_xy)
    # plotgrid_x = np.linspace(0.1,10,25)
    # plotgrid_y = np.linspace(0.1,10,25)
    # plotgrid_X, plotgrid_Y = np.meshgrid(plotgrid_x, plotgrid_y)
    # plotgrid_xy = np.vstack([plotgrid_X.ravel(), plotgrid_Y.ravel()]).T

    # Copula/Data Model Setup - X_star = R^phi * g(Z) -----------------------------------------------------------------

    # Splines -----------------------------------------------------------------

    assert k == len(knots_xy)

    radius = 4
    radius_from_knots = np.repeat(radius, k) # Wendland kernel influence radius from a knot
    
    effective_range = radius # Gaussian kernel effective range: exp(-3) = 0.05
    bandwidth = effective_range**2/6 # range for the gaussian kernel
    
    # bandwidth = radius
    
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

    # %% Plotting the Simulation Scenarios

    plotgrid_res_x       = 50
    plotgrid_res_y       = 50
    plotgrid_res_xy      = plotgrid_res_x * plotgrid_res_y
    plotgrid_x           = np.linspace(minX,maxX,plotgrid_res_x)
    plotgrid_y           = np.linspace(minY,maxY,plotgrid_res_y)
    plotgrid_X, plotgrid_Y = np.meshgrid(plotgrid_x, plotgrid_y)
    plotgrid_xy          = np.vstack([plotgrid_X.ravel(), plotgrid_Y.ravel()]).T

    gaussian_weight_matrix_for_plot = np.full(shape = (plotgrid_res_xy, k), fill_value = np.nan)
    for site_id in np.arange(plotgrid_res_xy):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)), 
                                        XB = knots_xy)
        # influence coming from each of the knots
        weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
        gaussian_weight_matrix_for_plot[site_id, :] = weight_from_knots

    # %% phi surface
    # heatplot of phi surface

    for sim_case in [1,2,3]:
        if sim_case == 1: phi_at_knots = 0.65-np.sqrt((knots_x-3)**2/4 + (knots_y-3)**2/3)/10
        if sim_case == 2: phi_at_knots = 0.65-np.sqrt((knots_x-5.1)**2/5 + (knots_y-5.3)**2/4)/11.6
        if sim_case == 3: phi_at_knots = 0.37 + 5*(scipy.stats.multivariate_normal.pdf(knots_xy, mean = np.array([2.5,3]), cov = 2*np.matrix([[1,0.2],[0.2,1]])) + 
                                                   scipy.stats.multivariate_normal.pdf(knots_xy, mean = np.array([7,7.5]), cov = 2*np.matrix([[1,-0.2],[-0.2,1]])))

        phi_vec_for_plot = gaussian_weight_matrix_for_plot @ phi_at_knots
        fig, ax = plt.subplots()
        fig.set_size_inches(8,6)
        ax.set_aspect('equal', 'box')

        vmin, vmax = (0.3, 0.7)

        heatmap = ax.imshow(phi_vec_for_plot.reshape(plotgrid_X.shape), 
                            extent=[minX, maxX, minY, maxY], origin='lower',
                            vmin = vmin, vmax = vmax,
                            cmap='RdBu_r', interpolation='nearest')
        cbar = fig.colorbar(heatmap, ax=ax)

        cbar.ax.tick_params(labelsize=20)  # Set the fontsize here
        ax.set_xticks(np.linspace(0, 10,num=5))
        ax.set_yticks(np.linspace(0, 10,num=5))

        # Add contour line at Z = 0.5
        contour = ax.contour(plotgrid_X, plotgrid_Y, phi_vec_for_plot.reshape(plotgrid_X.shape),
                            levels=[0.5], colors='black', linewidths=1, linestyles='dashed')
        # ax.clabel(contour, inline=True, fontsize=12, fmt='0.5', 
        #           manual = [(3,3)])  # Label the contour line

        # Plot knots and circles
        for i in range(k):
            circle_i = plt.Circle((knots_xy[i, 0], knots_xy[i, 1]), radius_from_knots[i],
                                color='r', fill=False, fc='None', ec='lightgrey', alpha=0.5)
            ax.add_patch(circle_i)

        # Scatter plot for sites and knots
        ax.scatter(knots_x, knots_y, marker='+', c='white', label='knot', s=300)
        for index, (x, y) in enumerate(knots_xy):
            ax.text(x+0.1, y+0.2, f'{index+1}', fontsize=12, ha='left')

        # Set axis limits and labels
        plt.xlim([0, 10])
        plt.ylim([0, 10])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.title(f'$\phi(s)$ Scenario {sim_case}', fontsize=20)

        plt.savefig(f'Surface:phi_simcase_{sim_case}.pdf', bbox_inches='tight')
        plt.show()
        plt.close()

    # %% 3 phi on 1 plot

    # Create the figure and a 1x3 grid for subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # Define vmin and vmax for colorbar normalization
    vmin, vmax = 0.3, 0.7

    # Loop over sim_cases to generate the heatmaps
    for sim_case, ax in zip([1, 2, 3], axes):
        if sim_case == 1:
            phi_at_knots = 0.65 - np.sqrt((knots_x - 3)**2 / 4 + (knots_y - 3)**2 / 3) / 10
        elif sim_case == 2:
            phi_at_knots = 0.65 - np.sqrt((knots_x - 5.1)**2 / 5 + (knots_y - 5.3)**2 / 4) / 11.6
        elif sim_case == 3:
            phi_at_knots = 0.37 + 5 * (scipy.stats.multivariate_normal.pdf(knots_xy, mean=np.array([2.5, 3]), 
                            cov=2*np.matrix([[1, 0.2], [0.2, 1]])) + 
                            scipy.stats.multivariate_normal.pdf(knots_xy, mean=np.array([7, 7.5]), 
                            cov=2*np.matrix([[1, -0.2], [-0.2, 1]])))
        
        phi_vec_for_plot = gaussian_weight_matrix_for_plot @ phi_at_knots
        
        # Plot heatmap
        heatmap = ax.imshow(phi_vec_for_plot.reshape(plotgrid_X.shape), 
                            extent=[minX, maxX, minY, maxY], origin='lower',
                            vmin=vmin, vmax=vmax,
                            cmap='RdBu_r', interpolation='nearest')
        
        ax.set_aspect('equal', 'box')
        ax.set_xticks(np.linspace(0, 10, num=5))
        ax.set_yticks(np.linspace(0, 10, num=5))
        
        # Add contour line at Z = 0.5
        ax.contour(plotgrid_X, plotgrid_Y, phi_vec_for_plot.reshape(plotgrid_X.shape),
                levels=[0.5], colors='black', linewidths=1, linestyles='dashed')
        
        # Plot knots and circles
        for i in range(k):
            circle_i = plt.Circle((knots_xy[i, 0], knots_xy[i, 1]), radius_from_knots[i],
                                color='r', fill=False, ec='lightgrey', alpha=0.5)
            ax.add_patch(circle_i)
        
        # Scatter plot for sites and knots
        ax.scatter(knots_x, knots_y, marker='+', c='white', label='knot', s=300)
        for index, (x, y) in enumerate(knots_xy):
            ax.text(x+0.1, y+0.2, f'{index+1}', fontsize=12, ha='left')

        # Set axis limits and labels
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])
        ax.set_xticks(np.linspace(0, 10, 5))
        ax.set_yticks(np.linspace(0, 10, 5))
        ax.set_xlabel('x', fontsize=14)

        if sim_case == 1:
            ax.set_ylabel('y', fontsize=14)
        else:
            ax.set_ylabel('')  # Remove y-label for second and third plots

        ax.set_title(f'$\phi(s)$ Scenario {sim_case}', fontsize=16)

    # Create a common colorbar
    cbar = fig.colorbar(heatmap, ax=axes, orientation='vertical', fraction=0.05, pad=0.01)
    cbar.ax.tick_params(labelsize=14)

    # Move the colorbar label to the top
    cbar.ax.set_xlabel('$\phi$', fontsize=20, labelpad=20)
    cbar.ax.xaxis.set_label_position('top')  # Move label to the top

    # Show the final plot
    plt.savefig(f'Surface:phi_all_scenario.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    # %% Plot range surface
    # heatplot of range surface

    range_at_knots = np.sqrt(0.3*knots_x + 0.4*knots_y)/2 # range for spatial Matern Z

    range_vec_for_plot = gaussian_weight_matrix_for_plot @ range_at_knots
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    ax.set_aspect('equal', 'box')

    # vmin, vmax = (0.3, 0.7)

    heatmap = ax.imshow(range_vec_for_plot.reshape(plotgrid_X.shape), 
                        extent=[minX, maxX, minY, maxY], origin='lower',
                        cmap='OrRd', interpolation='nearest')
    cbar = fig.colorbar(heatmap, ax=ax)

    cbar.ax.tick_params(labelsize=20)  # Set the fontsize here
    ax.set_xticks(np.linspace(0, 10,num=5))
    ax.set_yticks(np.linspace(0, 10,num=5))

    # Add contour line at Z = 0.5
    contour = ax.contour(plotgrid_X, plotgrid_Y, range_vec_for_plot.reshape(plotgrid_X.shape),
                        levels=[0.5], colors='black', linewidths=1, linestyles='dashed')
    # ax.clabel(contour, inline=True, fontsize=12, fmt='0.5', 
    #           manual = [(3,3)])  # Label the contour line

    # Plot knots and circles
    for i in range(k):
        circle_i = plt.Circle((knots_xy[i, 0], knots_xy[i, 1]), radius_from_knots[i],
                            color='r', fill=False, fc='None', ec='lightgrey', alpha=0.5)
        ax.add_patch(circle_i)

    # Scatter plot for sites and knots
    ax.scatter(knots_x, knots_y, marker='+', c='white', label='knot', s=300)
    for index, (x, y) in enumerate(knots_xy):
        ax.text(x+0.1, y+0.2, f'{index+1}', fontsize=12, ha='left')

    # Set axis limits and labels
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.title(r'$\rho(s)$ all scenarios', fontsize=20)

    plt.savefig('Surface:rho_simulation.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


    # %% all four plots together

    # Create the figure and a 1x4 grid for subplots
    fig, axes = plt.subplots(1, 4, figsize=(25, 6), constrained_layout=True)

    # Define vmin and vmax for the shared colorbar
    vmin, vmax = 0.3, 0.7

    # First 3 heatmaps with shared colorbar
    for sim_case, ax in zip([1, 2, 3], axes[:3]):
        if sim_case == 1:
            phi_at_knots = 0.65 - np.sqrt((knots_x - 3)**2 / 4 + (knots_y - 3)**2 / 3) / 10
        elif sim_case == 2:
            phi_at_knots = 0.65 - np.sqrt((knots_x - 5.1)**2 / 5 + (knots_y - 5.3)**2 / 4) / 11.6
        elif sim_case == 3:
            phi_at_knots = 0.37 + 5 * (scipy.stats.multivariate_normal.pdf(knots_xy, mean=np.array([2.5, 3]), 
                            cov=2*np.matrix([[1, 0.2], [0.2, 1]])) + 
                            scipy.stats.multivariate_normal.pdf(knots_xy, mean=np.array([7, 7.5]), 
                            cov=2*np.matrix([[1, -0.2], [-0.2, 1]])))
        
        phi_vec_for_plot = gaussian_weight_matrix_for_plot @ phi_at_knots
        
        # Plot heatmap
        heatmap = ax.imshow(phi_vec_for_plot.reshape(plotgrid_X.shape), 
                            extent=[minX, maxX, minY, maxY], origin='lower',
                            vmin=vmin, vmax=vmax,
                            cmap='RdBu_r', interpolation='nearest')
        
        ax.set_aspect('equal', 'box')
        ax.set_xticks(np.linspace(0, 10, num=5))
        ax.set_yticks(np.linspace(0, 10, num=5))
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Add contour line at Z = 0.5
        ax.contour(plotgrid_X, plotgrid_Y, phi_vec_for_plot.reshape(plotgrid_X.shape),
                levels=[0.5], colors='black', linewidths=1, linestyles='dashed')
        
        # Plot knots and circles
        for i in range(k):
            circle_i = plt.Circle((knots_xy[i, 0], knots_xy[i, 1]), radius_from_knots[i],
                                color='r', fill=False, ec='lightgrey', alpha=0.5)
            ax.add_patch(circle_i)
        
        # Scatter plot for sites and knots
        ax.scatter(knots_x, knots_y, marker='+', c='white', label='knot', s=300)
        for index, (x, y) in enumerate(knots_xy):
            ax.text(x+0.1, y+0.2, f'{index+1}', fontsize=16, ha='left')

        # Set axis limits and labels
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])
        ax.set_xticks(np.linspace(0, 10, 5))
        
        if sim_case == 1:
            ax.set_ylabel('y', fontsize=20)
        else:
            ax.set_ylabel('')  # Remove y-label for second and third plots

        ax.set_xlabel('x', fontsize=20)
        ax.set_title(f'$\phi(s)$ Scenario {sim_case}', fontsize=20)

    # Create a shared colorbar for the first three heatmaps
    cbar = fig.colorbar(heatmap, ax=axes[:3], orientation='vertical', fraction=0.04, pad=0.01)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_xlabel(r'$\phi$', fontsize=16, labelpad=10)
    cbar.ax.xaxis.set_label_position('top')  # Move label to the top

    # Fourth heatmap with its own colorbar
    ax = axes[3]  # Fourth subplot
    ax.tick_params(axis='both', which='major', labelsize=16)

    range_at_knots = np.sqrt(0.3*knots_x + 0.4*knots_y) / 2  # range for spatial Matern Z
    range_vec_for_plot = gaussian_weight_matrix_for_plot @ range_at_knots

    # Plot heatmap for the fourth case
    heatmap = ax.imshow(range_vec_for_plot.reshape(plotgrid_X.shape), 
                        extent=[minX, maxX, minY, maxY], origin='lower',
                        cmap='OrRd', interpolation='nearest')

    ax.set_aspect('equal', 'box')
    ax.set_xticks(np.linspace(0, 10, num=5))
    ax.set_yticks(np.linspace(0, 10, num=5))

    # Plot knots and circles
    for i in range(k):
        circle_i = plt.Circle((knots_xy[i, 0], knots_xy[i, 1]), radius_from_knots[i],
                            color='r', fill=False, ec='lightgrey', alpha=0.5)
        ax.add_patch(circle_i)

    # Scatter plot for sites and knots
    ax.scatter(knots_x, knots_y, marker='+', c='white', label='knot', s=300)
    for index, (x, y) in enumerate(knots_xy):
        ax.text(x+0.1, y+0.2, f'{index+1}', fontsize=16, ha='left')

    # Set axis limits and labels for the fourth plot
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_xticks(np.linspace(0, 10, 5))
    ax.set_yticks(np.linspace(0, 10, 5))
    ax.set_xlabel('x', fontsize=20)
    # ax.set_ylabel('y', fontsize=14)
    ax.set_title(r'$\rho(s)$ all scenarios', fontsize=20)

    # Create a separate colorbar for the fourth heatmap
    cbar2 = fig.colorbar(heatmap, ax=ax, orientation='vertical', fraction=0.06, pad=0.01)
    cbar2.ax.tick_params(labelsize=14)
    cbar2.ax.set_xlabel(r'$\rho$', fontsize=16, labelpad=10)
    cbar2.ax.xaxis.set_label_position('top')  # Move label to the top

    plt.savefig('Surface:all_simulation_surfaces.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

# %%

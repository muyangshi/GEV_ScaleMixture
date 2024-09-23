if __name__ == "__main__":
    # %% for reading seed from bash
    import sys
    data_seed = int(sys.argv[1]) if len(sys.argv) == 2 else 2345

    # %% imports
    # imports
    import os
    os.environ["OMP_NUM_THREADS"]        = "64" # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"]   = "64" # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"]        = "64" # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "64" # export VECLIB_MAXIMUM_THREADS=1
    os.environ["NUMEXPR_NUM_THREADS"]    = "64" # export NUMEXPR_NUM_THREADS=1
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import scipy
    import time
    from time import strftime, localtime
    from utilities import *
    import multiprocessing

    try: # data_seed is defined when python MCMC.py
        data_seed
    except: # when running on local machine interactively
        data_seed = 2345 # default seed
    finally:
        print('data_seed: ', data_seed)
    np.random.seed(data_seed)

    print('Pareto: ', norm_pareto)
    sim_case = 3

    # %% Simulation Setup

    # Spatial Domain Setup --------------------------------------------------------------------------------------------

    # Numbers - Ns, Nt --------------------------------------------------------
    
    np.random.seed(data_seed)
    Nt = 300000000 # number of time replicates
    Ns = 6 # number of sites/stations

    # Sites - random uniformly (x,y) generate site locations ------------------
    
    # sites_xy = np.random.random((Ns, 2)) * 10
    sites_xy = np.array([[0.5, 2.5],
                         [2.5, 0.5],
                         [7.5, 0.5],
                         [9.5, 2.5],
                         [8.5, 8.0],
                         [1.75, 8.5]])
    sites_x = sites_xy[:,0]
    sites_y = sites_xy[:,1]


    # # define the lower and upper limits for x and y

    minX, maxX = (0.0, 10.0)
    minY, maxY = (0.0, 10.0)


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

    # Model Parameter Setup (Truth) -----------------------------------------------------------------------------------

    # Data Model Parameters - X_star = R^phi * g(Z) ---------------------------

    rho_at_knots = np.sqrt(0.3*knots_x + 0.4*knots_y)/2 # range for spatial Matern Z

    if sim_case == 1: phi_at_knots = 0.65-np.sqrt((knots_x-3)**2/4 + (knots_y-3)**2/3)/10
    if sim_case == 2: phi_at_knots = 0.65-np.sqrt((knots_x-5.1)**2/5 + (knots_y-5.3)**2/4)/11.6
    if sim_case == 3: phi_at_knots = 0.37 + 5*(scipy.stats.multivariate_normal.pdf(knots_xy, mean = np.array([2.5,3]), cov = 2*np.matrix([[1,0.2],[0.2,1]])) + 
                                                scipy.stats.multivariate_normal.pdf(knots_xy, mean = np.array([7,7.5]), cov = 2*np.matrix([[1,-0.2],[-0.2,1]])))


    # %% Generate Simulation Data

    # Transformed Gaussian Process - W = g(Z), Z ~ MVN(0, K) ------------------

    range_vec = gaussian_weight_matrix @ rho_at_knots
    K         = ns_cov(range_vec = range_vec, sigsq_vec = sigsq_vec,
                        coords = sites_xy, kappa = nu, cov_model = "matern")
    Z         = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(Ns,)),cov=K,size=Nt).T
    W         = norm_to_Pareto(Z) 

    # Random Scaling Factor - R^phi -------------------------------------------

    phi_vec    = gaussian_weight_matrix @ phi_at_knots   # shape(Ns,)
    S_at_knots = np.array([scipy.stats.levy.rvs(loc = 0, scale = 0.5, size = k) for _ in range(Nt)]) # shape(Nt, k)
    R_vec      = (wendland_weight_matrix @ S_at_knots.T) # shape(Ns, Nt)
    X_star     = (R_vec.T ** phi_vec).T * W              # shape(Ns, Nt)

    np.save('X_eta_chi', X_star)

    # R_at_knots = np.full(shape = (k, Nt), fill_value = np.nan)
    # for t in np.arange(Nt):
    #     R_at_knots[:,t] = rlevy(n = k, m = delta, s = gamma) # generate R at time t, spatially varying k knots
    #     # should need to vectorize rlevy so in future s = gamma_at_knots (k,) vector
    #     # R_at_knots[:,t] = scipy.stats.levy.rvs(delta, gamma, k)
    #     # R_at_knots[:,t] = np.repeat(rlevy(n = 1, m = delta, s = gamma), k) # generate R at time t, spatially constant k knots
    # R_at_sites = wendland_weight_matrix @ R_at_knots
    # R_phi      = np.full(shape = (Ns, Nt), fill_value = np.nan)
    # for t in np.arange(Nt):
    #     R_phi[:,t] = np.power(R_at_sites[:,t], phi_vec)
    # X_star       = R_phi * W


    # %% Checks on Data Generation ------------------------------------------------------------------------------------

    # # Check stable variables S ------------------------------------------------

    # # levy.cdf(R_at_knots, loc = 0, scale = gamma) should look uniform
    
    # for i in range(k):
    #     scipy.stats.probplot(scipy.stats.levy.cdf(S_at_knots[i,:], scale=gamma), dist='uniform', fit=False, plot=plt)
    #     plt.axline((0,0), slope = 1, color = 'black')
    #     plt.savefig(f'QQPlot_levy_knot_{i}.png')
    #     plt.show()
    #     plt.close()

    # # Check Pareto distribution -----------------------------------------------

    # # shifted pareto.cdf(W[site_i,:] + 1, b = 1, loc = 0, scale = 1) shoud look uniform
    
    # if norm_pareto == 'shifted':
    #     for site_i in range(Ns):
    #         if site_i % 10 == 0: # don't print all sites
    #             scipy.stats.probplot(scipy.stats.pareto.cdf(W[site_i,:]+1, b = 1, loc = 0, scale = 1), dist='uniform', fit=False, plot=plt)
    #             plt.axline((0,0), slope = 1, color = 'black')
    #             plt.savefig(f'QQPlot_Pareto_site_{site_i}.png')
    #             plt.show()
    #             plt.close()

    # # standard pareto.cdf(W[site_i,:], b = 1, loc = 0, scale = 1) shoud look uniform
    # if norm_pareto == 'standard':
    #     for site_i in range(Ns):
    #         if site_i % 10 == 0:
    #             scipy.stats.probplot(scipy.stats.pareto.cdf(W[site_i,:], b = 1, loc = 0, scale = 1), dist='uniform', fit=False, plot=plt)
    #             plt.axline((0,0), slope = 1, color = 'black')
    #             plt.savefig(f'QQPlot_Pareto_site_{site_i}.png')
    #             plt.show()
    #             plt.close()

    # # Check model X_star ------------------------------------------------------

    # # pRW(X_star) should look uniform (at each site with Nt time replicates)
    # for site_i in range(Ns):
    #     if site_i % 4 == 0:
    #         unif = pRW(X_star[site_i,:], phi_vec[site_i], gamma_vec[site_i])
    #         scipy.stats.probplot(unif, dist="uniform", fit = False, plot=plt)
    #         plt.axline((0,0), slope=1, color='black')
    #         plt.savefig(f'QQPlot_Xstar_site_{site_i}.png')
    #         plt.show()
    #         plt.close()
            
    # # pRW(X_star) at each time t should deviates from uniform b/c spatial correlation
    # for t in range(Nt):
    #     if t % 5 == 0:
    #         unif = pRW(X_star[:,t], phi_vec, gamma_vec)
    #         scipy.stats.probplot(unif, dist="uniform", fit = False, plot=plt)
    #         plt.axline((0,0), slope=1, color='black')
    #         plt.savefig(f'QQPlot_Xstar_time_{t}.png')
    #         plt.show()
    #         plt.close()

    # %% Plotting the Simulation Scenarios

    plotgrid_res_x       = 500
    plotgrid_res_y       = 500
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

    
    # %% Plot phi surface and sites

    phi_vec_for_plot = gaussian_weight_matrix_for_plot @ phi_at_knots
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    ax.set_aspect('equal', 'box')

    vmin, vmax = (0.3, 0.7)

    cmap = matplotlib.cm.RdBu_r
    bounds = np.linspace(0.3, 0.7,num=9)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    heatmap = ax.imshow(phi_vec_for_plot.reshape(plotgrid_X.shape), 
                        origin='lower', extent=[minX, maxX, minY, maxY], 
                        norm = norm, cmap=cmap, interpolation='nearest')

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
                            color='r', fill=False, fc='None', ec='lightgrey', alpha=0.9)
        ax.add_patch(circle_i)

    # Scatter plot for sites and knots
    ax.scatter(knots_x, knots_y, marker='+', c='black', label='knot', s=300)
    # for index, (x, y) in enumerate(knots_xy):
        # ax.text(x+0.1, y+0.2, f'{index+1}', fontsize=12, ha='left')
    
    # Scatter the engineered sites
    markers = ['o', 's', '^', 'D', 'P', '*']
    for i, (x, y) in enumerate(zip(sites_x, sites_y)):
        ax.scatter(x, y, marker=markers[i], 
                #    c='#E6FF00', 
                   c='#c9a800',edgecolor='black',
                   s=100, 
                #    label=f'Point {i+1}',
                   label=f'$\phi(s_{i+1})$ = {round(phi_vec[i],2)}')
        ax.text(x+0.1, y+0.2, f'{i+1}', fontsize=12, ha='left', c = '#E6FF00')
    ax.legend(fontsize=15, loc='center right', bbox_to_anchor=(1.75, 0.77))  # Adjusted to place the legend on the left side
    # ax.scatter(sites_x, sites_y, marker='o', c='black', label='site', s=200)

    # Set axis limits and labels
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.title(f'$\phi(s)$', fontsize=20)

    plt.savefig(f'Simulation_eta_chi.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()

# %%
Nu = 100
us = np.linspace(0.9, 0.9999999, Nu)

# q = np.array([qRW(u, phi_vec, gamma_vec) for u in us]) # shape(Nu, Ns)

def qRW_par(args): 
    u, phi_vec, gamma_vec = args
    return qRW(u, phi_vec, gamma_vec)
args_list = []
for u in us:
    args_list.append((u, phi_vec, gamma_vec))
with multiprocessing.Pool(processes = 50) as pool:
    q_results = pool.map(qRW_par, args_list)
q = np.array(q_results) # shape(Nu, Ns)

# %% Estimate between s(1,2) for Thm2.3 a i (AD)

# \chi_{12]}

i = 0
j = 1

s1, s2 = X_star[i,:], X_star[j,:]  # Unpacking X_star
q_s1, q_s2 = q[:,i], q[:,j]      # Unpacking q

# Broadcasting comparisons for s1 and s2 against q_s1 and q_s2 for all `i` at once
co_extreme_mask = (s1[:, np.newaxis] >= q_s1) & (s2[:, np.newaxis] >= q_s2)

# Count co-extreme events across all `i` at once
count_co_extreme = np.sum(co_extreme_mask, axis=0)

# Probability of co-extreme and uni-extreme events
prob_co_extreme = count_co_extreme / len(s1)
prob_uni_extreme = np.mean(s2[:, np.newaxis] >= q_s2, axis=0)

# Chi calculation -------------------------------------------------------------
# chis = np.where(prob_uni_extreme != 0, prob_co_extreme / prob_uni_extreme, 0)

chis = prob_co_extreme / (1-us)

# Create a figure and axis object
fig, ax = plt.subplots()
fig.set_size_inches(8,6)
# ax.set_aspect('equal','box')

ax.plot(us, chis, label=r'Empirical $\chi$',
        linewidth = 3, color='tab:blue')

ax.set_xlim((0.9,1.0))
ax.set_ylim((0,0.5))
ax.set_xticks(np.linspace(0.9,1.0, 6))
ax.tick_params(axis='both', labelsize=20)

ax.set_xlabel(r'$u$', fontsize=20)
ax.set_ylabel(r'$\chi$', fontsize=20)
ax.set_title(fr'$\chi_{{{i+1}{j+1}}}$: $\phi(s_{i+1})$ = {round(phi_vec[0],2)}, $\phi(s_{j+1})$ = {round(phi_vec[1],2)}', fontsize=30)

# Add grid lines
ax.grid(True, linestyle = '--')

# Show the plot
plt.savefig(f'chi_{i+1}{j+1}.pdf', bbox_inches='tight')
# plt.show()
plt.close()

# Eta calculation -------------------------------------------------------------
etas = np.where(prob_co_extreme != 0, np.log(1-us) / np.log(prob_co_extreme), np.nan)

# Create a figure and axis object
fig, ax = plt.subplots()
fig.set_size_inches(8,6)

ax.plot(us, etas, label=r'Empirical $\eta$',
        linewidth = 3, color='tab:blue')

ax.set_xlim((0.9,1.0))
ax.set_ylim((0.2,1))
ax.set_xticks(np.linspace(0.9,1.0, 6))
ax.tick_params(axis='both', labelsize=20)

ax.set_xlabel(r'$u$', fontsize=20)
ax.set_ylabel(r'$\eta$', fontsize=20)
ax.set_title(fr'$\eta_{{{i+1}{j+1}}}$: $\phi(s_{i+1})$ = {round(phi_vec[0],2)}, $\phi(s_{j+1})$ = {round(phi_vec[1],2)}', fontsize=30)

# Add grid lines
ax.grid(True, linestyle = '--')

# Show the plot
plt.savefig(f'eta_{i+1}{j+1}.pdf', bbox_inches='tight')
# plt.show()
plt.close()

# %% Estimate between s(3,4) for Thm2.3 a ii (AI)

# \chi_{34}

i = 2
j = 3

s1, s2 = X_star[i,:], X_star[j,:]  # Unpacking X_star
q_s1, q_s2 = q[:,i], q[:,j]      # Unpacking q

# Broadcasting comparisons for s1 and s2 against q_s1 and q_s2 for all `i` at once
co_extreme_mask = (s1[:, np.newaxis] >= q_s1) & (s2[:, np.newaxis] >= q_s2)

# Count co-extreme events across all `i` at once
count_co_extreme = np.sum(co_extreme_mask, axis=0)

# Probability of co-extreme and uni-extreme events
prob_co_extreme = count_co_extreme / len(s1)
prob_uni_extreme = np.mean(s2[:, np.newaxis] >= q_s2, axis=0)

# Chi calculation -------------------------------------------------------------
# chis = np.where(prob_uni_extreme != 0, prob_co_extreme / prob_uni_extreme, 0)

chis = prob_co_extreme / (1-us)

# Create a figure and axis object
fig, ax = plt.subplots()
fig.set_size_inches(8,6)
# ax.set_aspect('equal','box')

ax.plot(us, chis, label=r'Empirical $\chi$',
        linewidth = 3, color='tab:blue')

ax.set_xlim((0.9,1.0))
ax.set_ylim((0,0.5))
ax.set_xticks(np.linspace(0.9,1.0, 6))
ax.tick_params(axis='both', labelsize=20)

ax.set_xlabel(r'$u$', fontsize=20)
ax.set_ylabel(r'$\chi$', fontsize=20)
ax.set_title(fr'$\chi_{{{i+1}{j+1}}}$: $\phi(s_{i+1})$ = {round(phi_vec[0],2)}, $\phi(s_{j+1})$ = {round(phi_vec[1],2)}', fontsize=30)

# Add grid lines
ax.grid(True, linestyle = '--')

# Show the plot
plt.savefig(f'chi_{i+1}{j+1}.pdf', bbox_inches='tight')
# plt.show()
plt.close()

# Eta calculation -------------------------------------------------------------
etas = np.where(prob_co_extreme != 0, np.log(1-us) / np.log(prob_co_extreme), np.nan)

# Create a figure and axis object
fig, ax = plt.subplots()
fig.set_size_inches(8,6)

ax.plot(us, etas, label=r'Empirical $\eta$',
        linewidth = 3, color='tab:blue')

ax.set_xlim((0.9,1.0))
ax.set_ylim((0.2,1))
ax.set_xticks(np.linspace(0.9,1.0, 6))
ax.tick_params(axis='both', labelsize=20)

ax.set_xlabel(r'$u$', fontsize=20)
ax.set_ylabel(r'$\eta$', fontsize=20)
ax.set_title(fr'$\eta_{{{i+1}{j+1}}}$: $\phi(s_{i+1})$ = {round(phi_vec[0],2)}, $\phi(s_{j+1})$ = {round(phi_vec[1],2)}', fontsize=30)

# Add grid lines
ax.grid(True, linestyle = '--')

# Show the plot
plt.savefig(f'eta_{i+1}{j+1}.pdf', bbox_inches='tight')
# plt.show()
plt.close()

# %% Estimate between s(4,5) for Thm2.3 a iii (AI)

# \chi_{45}

i = 3
j = 4

s1, s2 = X_star[i,:], X_star[j,:]  # Unpacking X_star
q_s1, q_s2 = q[:,i], q[:,j]      # Unpacking q

# Broadcasting comparisons for s1 and s2 against q_s1 and q_s2 for all `i` at once
co_extreme_mask = (s1[:, np.newaxis] >= q_s1) & (s2[:, np.newaxis] >= q_s2)

# Count co-extreme events across all `i` at once
count_co_extreme = np.sum(co_extreme_mask, axis=0)

# Probability of co-extreme and uni-extreme events
prob_co_extreme = count_co_extreme / len(s1)
prob_uni_extreme = np.mean(s2[:, np.newaxis] >= q_s2, axis=0)

# Chi calculation -------------------------------------------------------------
# chis = np.where(prob_uni_extreme != 0, prob_co_extreme / prob_uni_extreme, 0)

chis = prob_co_extreme / (1-us)

# Create a figure and axis object
fig, ax = plt.subplots()
fig.set_size_inches(8,6)
# ax.set_aspect('equal','box')

ax.plot(us, chis, label=r'Empirical $\chi$',
        linewidth = 3, color='tab:blue')

ax.set_xlim((0.9,1.0))
ax.set_ylim((0,0.5))
ax.set_xticks(np.linspace(0.9,1.0, 6))
ax.tick_params(axis='both', labelsize=20)

ax.set_xlabel(r'$u$', fontsize=20)
ax.set_ylabel(r'$\chi$', fontsize=20)
ax.set_title(fr'$\chi_{{{i+1}{j+1}}}$: $\phi(s_{i+1})$ = {round(phi_vec[0],2)}, $\phi(s_{j+1})$ = {round(phi_vec[1],2)}', fontsize=30)

# Add grid lines
ax.grid(True, linestyle = '--')

# Show the plot
plt.savefig(f'chi_{i+1}{j+1}.pdf', bbox_inches='tight')
# plt.show()
plt.close()

# Eta calculation -------------------------------------------------------------
etas = np.where(prob_co_extreme != 0, np.log(1-us) / np.log(prob_co_extreme), np.nan)

# Create a figure and axis object
fig, ax = plt.subplots()
fig.set_size_inches(8,6)

ax.plot(us, etas, label=r'Empirical $\eta$',
        linewidth = 3, color='tab:blue')

ax.set_xlim((0.9,1.0))
ax.set_ylim((0.2,1))
ax.set_xticks(np.linspace(0.9,1.0, 6))
ax.tick_params(axis='both', labelsize=20)

ax.set_xlabel(r'$u$', fontsize=20)
ax.set_ylabel(r'$\eta$', fontsize=20)
ax.set_title(fr'$\eta_{{{i+1}{j+1}}}$: $\phi(s_{i+1})$ = {round(phi_vec[0],2)}, $\phi(s_{j+1})$ = {round(phi_vec[1],2)}', fontsize=30)

# Add grid lines
ax.grid(True, linestyle = '--')

# Show the plot
plt.savefig(f'eta_{i+1}{j+1}.pdf', bbox_inches='tight')
# plt.show()
plt.close()

# %% Estimate between s(1,5) for Thm2.3 b i (AI)

# \chi_{15}

i = 0
j = 4

s1, s2 = X_star[i,:], X_star[j,:]  # Unpacking X_star
q_s1, q_s2 = q[:,i], q[:,j]      # Unpacking q

# Broadcasting comparisons for s1 and s2 against q_s1 and q_s2 for all `i` at once
co_extreme_mask = (s1[:, np.newaxis] >= q_s1) & (s2[:, np.newaxis] >= q_s2)

# Count co-extreme events across all `i` at once
count_co_extreme = np.sum(co_extreme_mask, axis=0)

# Probability of co-extreme and uni-extreme events
prob_co_extreme = count_co_extreme / len(s1)
prob_uni_extreme = np.mean(s2[:, np.newaxis] >= q_s2, axis=0)

# Chi calculation -------------------------------------------------------------
# chis = np.where(prob_uni_extreme != 0, prob_co_extreme / prob_uni_extreme, 0)

chis = prob_co_extreme / (1-us)

# Create a figure and axis object
fig, ax = plt.subplots()
fig.set_size_inches(8,6)
# ax.set_aspect('equal','box')

ax.plot(us, chis, label=r'Empirical $\chi$',
        linewidth = 3, color='tab:blue')

ax.set_xlim((0.9,1.0))
ax.set_ylim((0,0.5))
ax.set_xticks(np.linspace(0.9,1.0, 6))
ax.tick_params(axis='both', labelsize=20)

ax.set_xlabel(r'$u$', fontsize=20)
ax.set_ylabel(r'$\chi$', fontsize=20)
ax.set_title(fr'$\chi_{{{i+1}{j+1}}}$: $\phi(s_{i+1})$ = {round(phi_vec[0],2)}, $\phi(s_{j+1})$ = {round(phi_vec[1],2)}', fontsize=30)

# Add grid lines
ax.grid(True, linestyle = '--')

# Show the plot
plt.savefig(f'chi_{i+1}{j+1}.pdf', bbox_inches='tight')
# plt.show()
plt.close()

# Eta calculation -------------------------------------------------------------
etas = np.where(prob_co_extreme != 0, np.log(1-us) / np.log(prob_co_extreme), np.nan)

# Create a figure and axis object
fig, ax = plt.subplots()
fig.set_size_inches(8,6)

ax.plot(us, etas, label=r'Empirical $\eta$',
        linewidth = 3, color='tab:blue')

ax.set_xlim((0.9,1.0))
ax.set_ylim((0.2,1))
ax.set_xticks(np.linspace(0.9,1.0, 6))
ax.tick_params(axis='both', labelsize=20)

ax.set_xlabel(r'$u$', fontsize=20)
ax.set_ylabel(r'$\eta$', fontsize=20)
ax.set_title(fr'$\eta_{{{i+1}{j+1}}}$: $\phi(s_{i+1})$ = {round(phi_vec[0],2)}, $\phi(s_{j+1})$ = {round(phi_vec[1],2)}', fontsize=30)

# Add grid lines
ax.grid(True, linestyle = '--')

# Show the plot
plt.savefig(f'eta_{i+1}{j+1}.pdf', bbox_inches='tight')
# plt.show()
plt.close()

# %% Estimate between s(3,6) for Thm2.3 b ii (AI)

# \chi_{36}

i = 2
j = 5

s1, s2 = X_star[i,:], X_star[j,:]  # Unpacking X_star
q_s1, q_s2 = q[:,i], q[:,j]      # Unpacking q

# Broadcasting comparisons for s1 and s2 against q_s1 and q_s2 for all `i` at once
co_extreme_mask = (s1[:, np.newaxis] >= q_s1) & (s2[:, np.newaxis] >= q_s2)

# Count co-extreme events across all `i` at once
count_co_extreme = np.sum(co_extreme_mask, axis=0)

# Probability of co-extreme and uni-extreme events
prob_co_extreme = count_co_extreme / len(s1)
prob_uni_extreme = np.mean(s2[:, np.newaxis] >= q_s2, axis=0)

# Chi calculation -------------------------------------------------------------
# chis = np.where(prob_uni_extreme != 0, prob_co_extreme / prob_uni_extreme, 0)

chis = prob_co_extreme / (1-us)

# Create a figure and axis object
fig, ax = plt.subplots()
fig.set_size_inches(8,6)
# ax.set_aspect('equal','box')

ax.plot(us, chis, label=r'Empirical $\chi$',
        linewidth = 3, color='tab:blue')

ax.set_xlim((0.9,1.0))
ax.set_ylim((0,0.5))
ax.set_xticks(np.linspace(0.9,1.0, 6))
ax.tick_params(axis='both', labelsize=20)

ax.set_xlabel(r'$u$', fontsize=20)
ax.set_ylabel(r'$\chi$', fontsize=20)
ax.set_title(fr'$\chi_{{{i+1}{j+1}}}$: $\phi(s_{i+1})$ = {round(phi_vec[0],2)}, $\phi(s_{j+1})$ = {round(phi_vec[1],2)}', fontsize=30)

# Add grid lines
ax.grid(True, linestyle = '--')

# Show the plot
plt.savefig(f'chi_{i+1}{j+1}.pdf', bbox_inches='tight')
# plt.show()
plt.close()

# Eta calculation -------------------------------------------------------------
etas = np.where(prob_co_extreme != 0, np.log(1-us) / np.log(prob_co_extreme), np.nan)

# Create a figure and axis object
fig, ax = plt.subplots()
fig.set_size_inches(8,6)

ax.plot(us, etas, label=r'Empirical $\eta$',
        linewidth = 3, color='tab:blue')

ax.set_xlim((0.9,1.0))
ax.set_ylim((0.2,1))
ax.set_xticks(np.linspace(0.9,1.0, 6))
ax.tick_params(axis='both', labelsize=20)

ax.set_xlabel(r'$u$', fontsize=20)
ax.set_ylabel(r'$\eta$', fontsize=20)
ax.set_title(fr'$\eta_{{{i+1}{j+1}}}$: $\phi(s_{i+1})$ = {round(phi_vec[0],2)}, $\phi(s_{j+1})$ = {round(phi_vec[1],2)}', fontsize=30)

# Add grid lines
ax.grid(True, linestyle = '--')

# Show the plot
plt.savefig(f'eta_{i+1}{j+1}.pdf', bbox_inches='tight')
# plt.show()
plt.close()

# %% Estimate between s(1,4) for Thm2.3 b iii (AI)

# \chi_{14}

i = 0
j = 3

s1, s2 = X_star[i,:], X_star[j,:]  # Unpacking X_star
q_s1, q_s2 = q[:,i], q[:,j]      # Unpacking q

# Broadcasting comparisons for s1 and s2 against q_s1 and q_s2 for all `i` at once
co_extreme_mask = (s1[:, np.newaxis] >= q_s1) & (s2[:, np.newaxis] >= q_s2)

# Count co-extreme events across all `i` at once
count_co_extreme = np.sum(co_extreme_mask, axis=0)

# Probability of co-extreme and uni-extreme events
prob_co_extreme = count_co_extreme / len(s1)
prob_uni_extreme = np.mean(s2[:, np.newaxis] >= q_s2, axis=0)

# Chi calculation -------------------------------------------------------------
# chis = np.where(prob_uni_extreme != 0, prob_co_extreme / prob_uni_extreme, 0)

chis = prob_co_extreme / (1-us)

# Create a figure and axis object
fig, ax = plt.subplots()
fig.set_size_inches(8,6)
# ax.set_aspect('equal','box')

ax.plot(us, chis, label=r'Empirical $\chi$',
        linewidth = 3, color='tab:blue')

ax.set_xlim((0.9,1.0))
ax.set_ylim((0,0.5))
ax.set_xticks(np.linspace(0.9,1.0, 6))
ax.tick_params(axis='both', labelsize=20)

ax.set_xlabel(r'$u$', fontsize=20)
ax.set_ylabel(r'$\chi$', fontsize=20)
ax.set_title(fr'$\chi_{{{i+1}{j+1}}}$: $\phi(s_{i+1})$ = {round(phi_vec[0],2)}, $\phi(s_{j+1})$ = {round(phi_vec[1],2)}', fontsize=30)

# Add grid lines
ax.grid(True, linestyle = '--')

# Show the plot
plt.savefig(f'chi_{i+1}{j+1}.pdf', bbox_inches='tight')
# plt.show()
plt.close()

# Eta calculation -------------------------------------------------------------
etas = np.where(prob_co_extreme != 0, np.log(1-us) / np.log(prob_co_extreme), np.nan)

# Create a figure and axis object
fig, ax = plt.subplots()
fig.set_size_inches(8,6)

ax.plot(us, etas, label=r'Empirical $\eta$',
        linewidth = 3, color='tab:blue')

ax.set_xlim((0.9,1.0))
ax.set_ylim((0.2,1))
ax.set_xticks(np.linspace(0.9,1.0, 6))
ax.tick_params(axis='both', labelsize=20)

ax.set_xlabel(r'$u$', fontsize=20)
ax.set_ylabel(r'$\eta$', fontsize=20)
ax.set_title(fr'$\eta_{{{i+1}{j+1}}}$: $\phi(s_{i+1})$ = {round(phi_vec[0],2)}, $\phi(s_{j+1})$ = {round(phi_vec[1],2)}', fontsize=30)

# Add grid lines
ax.grid(True, linestyle = '--')

# Show the plot
plt.savefig(f'eta_{i+1}{j+1}.pdf', bbox_inches='tight')
# plt.show()
plt.close()

# %%

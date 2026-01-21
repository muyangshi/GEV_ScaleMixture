# %%
"""
coverage_analysis.py

Purpose
-------
Aggregate replicated simulation outputs and compute empirical coverage rates for
model parameters under three simulation scenarios. This script produces
binomial coverage plots reported in the simulation-study (Section 3.3 and Appendix C) of the paper.

Description
------------
For each simulation replicate and each parameter of interest, the script:
1) Loads posterior traces saved by the MCMC sampler (e.g., `phi_knots_trace.npy`),
2) Drops burn-in iterations,
3) Forms central posterior credible intervals for a grid of credible levels,
4) Computes empirical coverage indicators by checking whether the true parameter
   value lies within each interval,
5) Computes *exact* binomial proportion confidence intervals for the empirical
   coverage rate across replicates (via `scipy.stats.binomtest`),
6) Produces coverage plots with error bars.

Parameters analyzed
-------------------
- phi at knots (phi_knots_trace.npy)
- range/rho at knots (range_knots_trace.npy)
- marginal GEV parameters at knots (GEV_knots_trace.npy): mu (location) and
  tau/sigma (scale)

Inputs and assumptions
----------------------
- This script expects a directory structure with replicated simulation outputs:
  `./<data folder>/<simulation>/<scenario>/simulation_<id>/`
  where each replicate directory contains the trace `.npy` files listed above.
- The true parameter values used for coverage are defined in the “True Parameter
  Values” section below and must match the data-generation settings for the
  chosen `simulation_case`.

Outputs
-------
The script saves the following PDF figures in the current working directory:
- Empirical_Coverage_all_Phi_<scenario>.pdf
- Empirical_Coverage_all_Range_<scenario>.pdf
- Empirical_Coverage_MuSigma_<scenario>.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

# %%
# Coverage Setup ------------------------------------------------------------------------------------------------------
burnins = 6000 # length of burnin iterations
simulation_case = 'scenario3' # scenario1, scenario2, scenario3
sim_id_from = 1
sim_id_to = 50
sim_ids = np.arange(start = sim_id_from, stop = sim_id_to + 1)

# Remove bad simulations
if simulation_case == 'scenario2':
    bad_sim_ids = np.array([2]) # simulation id 2 did not finish in scenario 2
else:
    bad_sim_ids = np.array([]) # the other simulations seems fine (no good reason to remove simulation id3)
for bad_sim_id in bad_sim_ids:
    sim_ids = np.delete(sim_ids, np.argwhere(sim_ids == bad_sim_id))
nsim = len(sim_ids)
folders = ['./data_alpine/simulation coverage analysis/'+simulation_case+'/simulation_' + str(sim_id) + '/' for sim_id in sim_ids] # directory of simulations

# True Parameter Values -----------------------------------------------------------------------------------------------
# Time Replicates
Nt = 64
# Knots
x_pos = np.linspace(0,10,5,True)[1:-1]
y_pos = np.linspace(0,10,5,True)[1:-1]
X_pos, Y_pos = np.meshgrid(x_pos,y_pos)
knots_xy = np.vstack([X_pos.ravel(), Y_pos.ravel()]).T
knots_x = knots_xy[:,0]
knots_y = knots_xy[:,1]
k = len(knots_x)
# GEV
mu = 0.0 # GEV location
tau = 1.0 # GEV scale
ksi = 0.2 # GEV shape
# range
range_at_knots = np.sqrt(0.3*knots_x + 0.4*knots_y)/2 # scenario 2
# phi
match simulation_case:
    case 'scenario1':
        phi_at_knots = 0.65-np.sqrt((knots_x-3)**2/4 + (knots_y-3)**2/3)/10 # scenario 1
    case 'scenario2':
        phi_at_knots = 0.65-np.sqrt((knots_x-5.1)**2/5 + (knots_y-5.3)**2/4)/11.6 # scenario 2
    case 'scenario3':
        phi_at_knots = 10*(0.5*scipy.stats.multivariate_normal.pdf(knots_xy, 
                                                               mean = np.array([2.5,3]), 
                                                               cov = 2*np.matrix([[1,0.2],[0.2,1]])) + 
                        0.5*scipy.stats.multivariate_normal.pdf(knots_xy, 
                                                                mean = np.array([7,7.5]), 
                                                                cov = 2*np.matrix([[1,-0.2],[-0.2,1]]))) + \
                        0.37# scenario 3


# %%

# #################################################################### #
# Using ACTUAL Binomial Confidence Intervals instead of Normal Approx  #
# #################################################################### #

# %%
# phi - binomial confidence interval
alphas = np.flip(np.linspace(0.025, 0.4, 16))
creds  = 1 - alphas
q_low = alphas/2
q_high = 1 - q_low

# summary by credible levels
# PE_matrix_phi = np.full(shape = (k, nsim), fill_value = np.nan)
lower_bound_matrix_phi_alpha = np.full(shape = (k, nsim, len(alphas)), fill_value = np.nan)
upper_bound_matrix_phi_alpha = np.full(shape = (k, nsim, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    for i in range(nsim):
        folder = folders[i]
        phi_knots_trace = np.load(folder + 'phi_knots_trace.npy')
        # drop burnins
        phi_knots_trace = phi_knots_trace[burnins:]
        # phi
        # PE_matrix_phi[:,i] = np.mean(phi_knots_trace, axis = 0)
        lower_bound_matrix_phi_alpha[:,i, level_i] = np.quantile(phi_knots_trace, q = q_low[level_i], axis = 0)
        upper_bound_matrix_phi_alpha[:,i, level_i] = np.quantile(phi_knots_trace, q = q_high[level_i], axis = 0)

# coverage flag
phi_covers = np.full(shape = (k, nsim, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    for knot_id in range(k):
        phi_covers[knot_id, :, level_i] = \
            np.logical_and(lower_bound_matrix_phi_alpha[knot_id,:,level_i] < phi_at_knots[knot_id], 
                            upper_bound_matrix_phi_alpha[knot_id,:, level_i] > phi_at_knots[knot_id])

# PE: average coverage
avg_phi_covers = np.mean(phi_covers, axis = 1)

# SE: binomtest confidence interval
# l, h = scipy.stats.binomtest(k = int(sum(phi_covers[0,:,0])), # number of success (covers)
#                       n = nsim, # total number of repetition
#                       p = 1-alphas[0]).proportion_ci(confidence_level=0.95)
phi_binom_CIs = np.full(shape = (k, 2, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    for knot_k in range(k):
        l, h = scipy.stats.binomtest(k = int(sum(phi_covers[knot_k,:,level_i])), # number of success (covers)
                                     n = nsim, # total number of repetition
                                     p = 1-alphas[level_i]).proportion_ci(confidence_level=0.95)
        phi_binom_CIs[knot_k, :, level_i] = [l,h]

# errs = np.array([[avg_phi_covers[0,0] - np.array(l)], # first row contains the lower errors
#                  [np.array(h) - avg_phi_covers[0,0]]]) # second row contains the upper errors
phi_binom_errs = np.full(shape = (k, 2, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    for knot_k in range(k):
        phi_binom_errs[knot_k,:,level_i] = [avg_phi_covers[knot_k,level_i] - phi_binom_CIs[knot_k, 0, level_i],
                                            phi_binom_CIs[knot_k, 1, level_i] - avg_phi_covers[knot_k, level_i]]

# plot 1 big plot of k pieces
fig, axes = plt.subplots(3, 3, figsize=(20, 20), constrained_layout=True)
for knot_k in range(k):
    row = 2 - (knot_k // 3)   # 2 is the bottom row index; flip the row index
    col = knot_k % 3
    ax = axes[row, col]
    # ax = axes[knot_k//3, knot_k%3]
    ax.set_xlim([0.55, 1])
    ax.set_ylim([0.55, 1])
    ax.set_xticks([0.6,0.8,1.0])
    ax.set_yticks([0.6,0.8,1.0])
    ax.set_xticks([0.6,0.7,0.8,0.9,1.0], minor=True)
    ax.set_yticks([0.6,0.7,0.8,0.9,1.0], minor=True)      
    
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color = 'black')
    ax.errorbar(x = 1 - alphas,
                y = avg_phi_covers[knot_k,:],
                yerr=phi_binom_errs[knot_k,:,:],
                fmt = 'bo',
                ecolor = 'b')
    
    # Add light gridlines
    # ax.grid(True, which='major', color='gray', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Hide y-ticks and labels for subplots not in the first column
    if knot_k % 3 != 0:
        ax.set_yticklabels([])
        ax.set_ylabel('')  # Remove the y-axis label
    
    # if knot_k // 3 != 2:
    if row != 2:
        ax.set_xticklabels([])
        ax.set_xlabel('')  # Remove the y-axis label  

    ax.set_title(r'k='+str(knot_k+1), fontsize = 60)
    ax.tick_params(axis='both', which = 'major', labelsize = 60)
    # ax.tick_params(axis='both', which = 'minor', labelsize = 60)
fig.suptitle(r'Empirical Coverage for $\phi_k$', fontsize = 75)
fig.supxlabel('Credible Level', fontsize = 75)
# fig.supylabel('Empirical Coverage Rate', fontsize = 60)
# fig.text(0.5, 0.05, 'Credible Level', ha='center', va='center', fontsize = 60)
# fig.text(0.05, 0.5, 'Empirical Coverage Rate', fontsize = 60, ha='center', va='center', rotation='vertical')
fig.savefig(f"Empirical_Coverage_all_Phi_{simulation_case}.pdf")
plt.show()
plt.close()

# # plotting k subplots
# for knot_k in range(k):
#     fig, ax = plt.subplots()
#     fig.set_size_inches(8,8)
#     plt.xlim([0.5,1])
#     plt.ylim([0.5,1])
#     plt.axline((0,0), slope = 1, color = 'black')
#     plt.errorbar(x = 1 - alphas,
#                 y = avg_phi_covers[knot_k,:],
#                 yerr=phi_binom_errs[knot_k,:,:],
#                 fmt = 'bo',
#                 ecolor = 'b')
#     plt.title(r'k='+str(knot_k), fontsize = 24)
#     plt.xticks(fontsize = 20)
#     plt.yticks(fontsize = 20)
#     fig.savefig('Empirical_Coverage_Phi_Knot'+str(knot_k)+'.pdf')
#     # plt.show()
#     plt.close()


# %%
# range - binomial confidence interval
alphas = np.flip(np.linspace(0.025, 0.4, 16))
creds  = 1 - alphas
q_low = alphas/2
q_high = 1 - q_low

# summary by credible levels
# PE_matrix_range = np.full(shape = (k, nsim), fill_value = np.nan)
lower_bound_matrix_range_alpha = np.full(shape = (k, nsim, len(alphas)), fill_value = np.nan)
upper_bound_matrix_range_alpha = np.full(shape = (k, nsim, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    for i in range(nsim):
        folder = folders[i]
        range_knots_trace = np.load(folder + 'range_knots_trace.npy')
        # drop burnins
        range_knots_trace = range_knots_trace[burnins:]
        # range
        # PE_matrix_range[:,i] = np.mean(range_knots_trace, axis = 0)
        lower_bound_matrix_range_alpha[:,i, level_i] = np.quantile(range_knots_trace, q = q_low[level_i], axis = 0)
        upper_bound_matrix_range_alpha[:,i, level_i] = np.quantile(range_knots_trace, q = q_high[level_i], axis = 0)

# coverage flag
range_covers = np.full(shape = (k, nsim, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    for knot_id in range(k):
        range_covers[knot_id, :, level_i] = \
            np.logical_and(lower_bound_matrix_range_alpha[knot_id,:,level_i] < range_at_knots[knot_id], 
                            upper_bound_matrix_range_alpha[knot_id,:, level_i] > range_at_knots[knot_id])

# PE: average coverage
avg_range_covers = np.mean(range_covers, axis = 1)

# SE: binomtest confidence interval
# l, h = scipy.stats.binomtest(k = int(sum(range_covers[0,:,0])), # number of success (covers)
#                       n = nsim, # total number of repetition
#                       p = 1-alphas[0]).proportion_ci(confidence_level=0.95)
range_binom_CIs = np.full(shape = (k, 2, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    for knot_k in range(k):
        l, h = scipy.stats.binomtest(k = int(sum(range_covers[knot_k,:,level_i])), # number of success (covers)
                                     n = nsim, # total number of repetition
                                     p = 1-alphas[level_i]).proportion_ci(confidence_level=0.95)
        range_binom_CIs[knot_k, :, level_i] = [l,h]

# errs = np.array([[avg_range_covers[0,0] - np.array(l)], # first row contains the lower errors
#                  [np.array(h) - avg_range_covers[0,0]]]) # second row contains the upper errors
range_binom_errs = np.full(shape = (k, 2, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    for knot_k in range(k):
        range_binom_errs[knot_k,:,level_i] = [avg_range_covers[knot_k,level_i] - range_binom_CIs[knot_k, 0, level_i],
                                            range_binom_CIs[knot_k, 1, level_i] - avg_range_covers[knot_k, level_i]]

# plot 1 big plot of k pieces
fig, axes = plt.subplots(3, 3, figsize=(20, 20), constrained_layout=True)
for knot_k in range(k):
    row = 2 - (knot_k // 3)   # 2 is the bottom row index; flip the row index
    col = knot_k % 3
    ax = axes[row, col]
    # ax = axes[knot_k//3, knot_k%3]
    ax.set_xlim([0.55, 1])
    ax.set_ylim([0.55, 1])
    ax.set_xticks([0.6,0.8,1.0])
    ax.set_yticks([0.6,0.8,1.0])
    ax.set_xticks([0.6,0.7,0.8,0.9,1.0], minor=True)
    ax.set_yticks([0.6,0.7,0.8,0.9,1.0], minor=True)      

    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color = 'black')
    ax.errorbar(x = 1 - alphas,
                y = avg_range_covers[knot_k,:],
                yerr=range_binom_errs[knot_k,:,:],
                fmt = 'bo',
                ecolor = 'b')
    # Add light gridlines
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Hide y-ticks and labels for subplots not in the first column
    if knot_k % 3 != 0:
        ax.set_yticklabels([])
        ax.set_ylabel('')  # Remove the y-axis label
    
    # if knot_k // 3 != 2:
    if row != 2:
        ax.set_xticklabels([])
        ax.set_xlabel('')  # Remove the y-axis label  

    ax.set_title(r'k='+str(knot_k+1), fontsize = 60)
    ax.tick_params(axis='both', which = 'major', labelsize = 60)
    ax.tick_params(axis='both', which = 'minor', labelsize = 60)
fig.suptitle(r'Empirical Coverage for $\rho_k$', fontsize = 75)
fig.supxlabel('Credible Level', fontsize = 75)
# fig.supylabel('Empirical Coverage Rate', fontsize = 60)
# fig.text(0.5, 0.05, 'Credible Level', ha='center', va='center', fontsize = 32)
# fig.text(0.05, 0.5, 'Empirical Coverage Rate', fontsize = 32, ha='center', va='center', rotation='vertical')
fig.savefig(f"Empirical_Coverage_all_Range_{simulation_case}.pdf")
plt.show()
plt.close()

# # plotting k subplots
# for knot_k in range(k):
#     fig, ax = plt.subplots()
#     fig.set_size_inches(8,8)
#     plt.xlim([0.5,1])
#     plt.ylim([0.5,1])
#     plt.axline((0,0), slope = 1, color = 'black')
#     plt.errorbar(x = 1 - alphas,
#                 y = avg_range_covers[knot_k,:],
#                 yerr=range_binom_errs[knot_k,:,:],
#                 fmt = 'bo',
#                 ecolor = 'b')
#     plt.title(r'k='+str(knot_k), fontsize = 24)
#     plt.xticks(fontsize = 20)
#     plt.yticks(fontsize = 20)
#     fig.savefig('Empirical_Coverage_Range_Knot'+str(knot_k)+'.pdf')
#     # plt.show()
#     plt.close()



# %%
# location mu
alphas = np.flip(np.linspace(0.025, 0.4, 16))
creds  = 1 - alphas
q_low = alphas/2
q_high = 1 - q_low

# summary by credible levels
# PE_matrix_loc = np.full(shape = (1, nsim), fill_value = np.nan)
lower_bound_matrix_loc_alpha = np.full(shape = (1, nsim, len(alphas)), fill_value = np.nan)
upper_bound_matrix_loc_alpha = np.full(shape = (1, nsim, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    for i in range(nsim):
        folder = folders[i]
        GEV_knots_trace = np.load(folder + 'GEV_knots_trace.npy')
        loc_knots_trace = GEV_knots_trace[:,0,0]
        # drop burnins
        loc_knots_trace = loc_knots_trace[burnins:]
        # location mu
        # PE_matrix_loc[:,i] = np.mean(loc_knots_trace)
        lower_bound_matrix_loc_alpha[:, i, level_i] = np.quantile(loc_knots_trace, q = q_low[level_i], axis = 0)
        upper_bound_matrix_loc_alpha[:, i, level_i] = np.quantile(loc_knots_trace, q = q_high[level_i], axis = 0)

# coverage flag
loc_covers = np.full(shape = (1, nsim, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    loc_covers[0, :, level_i] = np.logical_and(lower_bound_matrix_loc_alpha[0,:,level_i] < mu,
                                               upper_bound_matrix_loc_alpha[0,:,level_i] > mu)

# PE: average coverage
avg_loc_covers = np.mean(loc_covers, axis = 1)

# SE: binomtest confidence interval
# l, h = scipy.stats.binomtest(k = int(sum(loc_covers[0,:,0])), # number of success (covers)
#                       n = nsim, # total number of repetition
#                       p = 1-alphas[0]).proportion_ci(confidence_level=0.95)
loc_binom_CIs = np.full(shape = (1, 2, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    l, h = scipy.stats.binomtest(k = int(sum(loc_covers[0,:,level_i])), # number of success (covers)
                                    n = nsim, # total number of repetition
                                    p = 1-alphas[level_i]).proportion_ci(confidence_level=0.95)
    loc_binom_CIs[0, :, level_i] = [l,h]

# errs = np.array([[avg_loc_covers[0,0] - np.array(l)], # first row contains the lower errors
#                  [np.array(h) - avg_loc_covers[0,0]]]) # second row contains the upper errors
loc_binom_errs = np.full(shape = (1, 2, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    loc_binom_errs[0,:,level_i] = [avg_loc_covers[0,level_i] - loc_binom_CIs[0, 0, level_i],
                                        loc_binom_CIs[0, 1, level_i] - avg_loc_covers[0, level_i]]

# fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
# ax.set_xlim([0.55, 1])
# ax.set_ylim([0.55, 1])
# ax.set_xticks([0.6,0.8,1.0])
# ax.set_yticks([0.6,0.8,1.0])
# ax.set_xticks([0.6,0.7,0.8,0.9,1.0], minor=True)
# ax.set_yticks([0.6,0.7,0.8,0.9,1.0], minor=True)
# ax.plot([0, 1], [0, 1], transform=ax.transAxes, color = 'black')
# ax.errorbar(x = 1 - alphas,
#                 y = avg_loc_covers[0,:],
#                 yerr=loc_binom_errs[0,:,:],
#                 fmt = 'bo',
#                 ecolor = 'b')
# ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
# ax.set_title(r'Empirical Coverage for $\mu$', fontsize = 60)
# ax.tick_params(axis='both', which = 'major', labelsize = 60)
# ax.tick_params(axis='both', which = 'minor', labelsize = 60)
# fig.supxlabel('Credible Level', fontsize = 60)
# fig.supylabel('Empirical Coverage Rate', fontsize = 60)
# fig.savefig(f'Empirical_Coverage_Mu_{simulation_case}.pdf')
# plt.show()
# plt.close()

# %%
# scale sigma
alphas = np.flip(np.linspace(0.025, 0.4, 16))
creds  = 1 - alphas
q_low = alphas/2
q_high = 1 - q_low

# summary by credible levels
# PE_matrix_scale = np.full(shape = (1, nsim), fill_value = np.nan)
lower_bound_matrix_scale_alpha = np.full(shape = (1, nsim, len(alphas)), fill_value = np.nan)
upper_bound_matrix_scale_alpha = np.full(shape = (1, nsim, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    for i in range(nsim):
        folder = folders[i]
        GEV_knots_trace = np.load(folder + 'GEV_knots_trace.npy')
        scale_knots_trace = GEV_knots_trace[:,1,0]
        # drop burnins
        scale_knots_trace = scale_knots_trace[burnins:]
        # scale sigma
        # PE_matrix_scale[:,i] = np.mean(scale_knots_trace)
        lower_bound_matrix_scale_alpha[:, i, level_i] = np.quantile(scale_knots_trace, q = q_low[level_i], axis = 0)
        upper_bound_matrix_scale_alpha[:, i, level_i] = np.quantile(scale_knots_trace, q = q_high[level_i], axis = 0)

# coverage flag
scale_covers = np.full(shape = (1, nsim, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    scale_covers[0, :, level_i] = np.logical_and(lower_bound_matrix_scale_alpha[0,:,level_i] < tau,
                                               upper_bound_matrix_scale_alpha[0,:,level_i] > tau)

# PE: average coverage
avg_scale_covers = np.mean(scale_covers, axis = 1)

# SE: binomtest confidence interval
# l, h = scipy.stats.binomtest(k = int(sum(scale_covers[0,:,0])), # number of success (covers)
#                       n = nsim, # total number of repetition
#                       p = 1-alphas[0]).proportion_ci(confidence_level=0.95)
scale_binom_CIs = np.full(shape = (1, 2, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    l, h = scipy.stats.binomtest(k = int(sum(scale_covers[0,:,level_i])), # number of success (covers)
                                    n = nsim, # total number of repetition
                                    p = 1-alphas[level_i]).proportion_ci(confidence_level=0.95)
    scale_binom_CIs[0, :, level_i] = [l,h]

# errs = np.array([[avg_scale_covers[0,0] - np.array(l)], # first row contains the lower errors
#                  [np.array(h) - avg_scale_covers[0,0]]]) # second row contains the upper errors
scale_binom_errs = np.full(shape = (1, 2, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    scale_binom_errs[0,:,level_i] = [avg_scale_covers[0,level_i] - scale_binom_CIs[0, 0, level_i],
                                        scale_binom_CIs[0, 1, level_i] - avg_scale_covers[0, level_i]]

# fig, ax = plt.subplots()
# fig.set_size_inches(8,8)
# ax.set_xlim([0.5, 1])
# ax.set_ylim([0.5, 1])
# ax.set_xticks([0.5,0.6,0.7,0.8,0.9,1.0])
# ax.set_yticks([0.5,0.6,0.7,0.8,0.9,1.0]) 
# ax.plot([0, 1], [0, 1], transform=ax.transAxes, color = 'black')
# ax.errorbar(x = 1 - alphas,
#                 y = avg_scale_covers[0,:],
#                 yerr=scale_binom_errs[0,:,:],
#                 fmt = 'bo',
#                 ecolor = 'b')
# ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
# ax.set_title(r'Empirical Coverage for $\sigma$', fontsize = 30)
# ax.tick_params(axis='both', which = 'major', labelsize = 20)
# ax.tick_params(axis='both', which = 'minor', labelsize = 20)
# fig.text(0.5, 0.04, 'Credible Level', ha='center', va='center', fontsize = 28)
# fig.text(0.04, 0.5, 'Empirical Coverage', fontsize = 28, ha='center', va='center', rotation='vertical')
# fig.savefig(f'Empirical_Coverage_Sigma_{simulation_case}.pdf')
# plt.show()
# plt.close()


# %%

fig, axes = plt.subplots(2, 1, figsize=(10, 20), constrained_layout=True)
ax = axes[0]
ax.set_xlim([0.55, 1])
ax.set_ylim([0.55, 1])
ax.set_xticks([0.6,0.8,1.0])
ax.set_yticks([0.6,0.8,1.0])
ax.set_xticks([0.6,0.7,0.8,0.9,1.0], minor=True)
ax.set_yticks([0.6,0.7,0.8,0.9,1.0], minor=True)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color = 'black')
ax.errorbar(x = 1 - alphas,
                y = avg_loc_covers[0,:],
                yerr=loc_binom_errs[0,:,:],
                fmt = 'bo',
                ecolor = 'b')
ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_title(r'$\mu$', fontsize = 75)
ax.tick_params(axis='both', which = 'major', labelsize = 60)
ax.tick_params(axis='both', which = 'minor', labelsize = 60)

ax = axes[1]
ax.set_xlim([0.55, 1])
ax.set_ylim([0.55, 1])
ax.set_xticks([0.6,0.8,1.0])
ax.set_yticks([0.6,0.8,1.0])
ax.set_xticks([0.6,0.7,0.8,0.9,1.0], minor=True)
ax.set_yticks([0.6,0.7,0.8,0.9,1.0], minor=True)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color = 'black')
ax.errorbar(x = 1 - alphas,
                y = avg_scale_covers[0,:],
                yerr=scale_binom_errs[0,:,:],
                fmt = 'bo',
                ecolor = 'b')
ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_title(r'$\sigma$', fontsize = 75)
ax.tick_params(axis='both', which = 'major', labelsize = 60)
ax.tick_params(axis='both', which = 'minor', labelsize = 60)

fig.suptitle(r'Empirical Coverage', fontsize = 75)
fig.supxlabel('Credible Level', fontsize = 75, horizontalalignment='center')
fig.supylabel('Empirical Coverage Rate', fontsize = 75)

fig.savefig(f'Empirical_Coverage_MuSigma_{simulation_case}.pdf')
plt.show()
plt.close()
# %%

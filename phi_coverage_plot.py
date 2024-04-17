# %%
# Read Notes!!
"""
For STAT600 Project
Make coverage plots of the phi parameter
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

# %%
# Coverage Setup ------------------------------------------------------------------------------------------------------
burnins = 10000 # length of burnin iterations

sim_id_from = 1 # inclusive
sim_id_to = 8  # inclusive
sim_ids = np.arange(start = sim_id_from, stop = sim_id_to + 1)

nsim = len(sim_ids)
folders = ['./data/standard_fixGEV/scenario2/simulation_' + str(sim_id) + '/' for sim_id in sim_ids]

# True Parameter Values -----------------------------------------------------------------------------------------------
# Time Replicates
minX = 0
maxX = 10
minY = 0
maxY = 10

# isometric knot grid
N_outer_grid = 9
x_pos                    = np.linspace(minX + 1, maxX + 1, num = int(2*np.sqrt(N_outer_grid)))
y_pos                    = np.linspace(minY + 1, maxY + 1, num = int(2*np.sqrt(N_outer_grid)))
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

# range
range_at_knots = np.sqrt(0.3*knots_x + 0.4*knots_y)/2 # scenario 2
# phi
phi_at_knots = 0.65-np.sqrt((knots_x-5.1)**2/5 + (knots_y-5.3)**2/4)/11.6 # scenario 2


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

# plotting k subplots
for knot_k in range(k):
    fig, ax = plt.subplots()
    fig.set_size_inches(8,8)
    plt.xlim([0.5,1])
    plt.ylim([0.5,1])
    plt.axline((0,0), slope = 1, color = 'black')
    plt.errorbar(x = 1 - alphas,
                y = avg_phi_covers[knot_k,:],
                yerr=phi_binom_errs[knot_k,:,:],
                fmt = 'bo',
                ecolor = 'b')
    plt.title(r'k='+str(knot_k), fontsize = 24)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    fig.savefig('Empirical_Coverage_Phi_Knot'+str(knot_k)+'.pdf')
    # plt.show()
    plt.close()

# plot 1 big plot of k pieces
fig, axes = plt.subplots(4,4)
fig.set_size_inches(24,24)
for knot_k in range(k):
    ax = axes[knot_k//4, knot_k%4]
    ax.set_xlim([0.5, 1])
    ax.set_ylim([0.5, 1])
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color = 'black')

    ax.errorbar(x = 1 - alphas,
                y = avg_phi_covers[knot_k,:],
                yerr=phi_binom_errs[knot_k,:,:],
                fmt = 'bo',
                ecolor = 'b')
    ax.set_title(r'knot '+str(knot_k+1), fontsize = 24)
    ax.tick_params(axis='both', which = 'major', labelsize = 20)
    ax.tick_params(axis='both', which = 'minor', labelsize = 20)
for i in range(13, 16):
    fig.delaxes(axes[i//4][i%4])
fig.suptitle(r'Empirical Coverage for $\phi_k$', fontsize = 36)
fig.text(0.5, 0.05, 'Credible Level', ha='center', va='center', fontsize = 24)
fig.text(0.05, 0.5, 'Empirical Coverage', fontsize = 24, ha='center', va='center', rotation='vertical')
fig.savefig("Empirical_Coverage_all_Phi.pdf")
plt.close()


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

# plotting k subplots
for knot_k in range(k):
    fig, ax = plt.subplots()
    fig.set_size_inches(8,8)
    plt.xlim([0.5,1])
    plt.ylim([0.5,1])
    plt.axline((0,0), slope = 1, color = 'black')
    plt.errorbar(x = 1 - alphas,
                y = avg_range_covers[knot_k,:],
                yerr=range_binom_errs[knot_k,:,:],
                fmt = 'bo',
                ecolor = 'b')
    plt.title(r'k='+str(knot_k), fontsize = 24)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    fig.savefig('Empirical_Coverage_Range_Knot'+str(knot_k)+'.pdf')
    # plt.show()
    plt.close()

# plot 1 big plot of k pieces
fig, axes = plt.subplots(4,4)
fig.set_size_inches(24,24)
for knot_k in range(k):
    ax = axes[knot_k//4, knot_k%4]
    ax.set_xlim([0.5, 1])
    ax.set_ylim([0.5, 1])
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color = 'black')

    ax.errorbar(x = 1 - alphas,
                y = avg_range_covers[knot_k,:],
                yerr=range_binom_errs[knot_k,:,:],
                fmt = 'bo',
                ecolor = 'b')
    ax.set_title(r'k='+str(knot_k), fontsize = 24)
    ax.tick_params(axis='both', which = 'major', labelsize = 20)
    ax.tick_params(axis='both', which = 'minor', labelsize = 20)
for i in range(13, 16):
    fig.delaxes(axes[i//4][i%4])
fig.suptitle(r'Empirical Coverage for $\rho_k$', fontsize = 36)
fig.text(0.5, 0.05, 'Credible Level', ha='center', va='center', fontsize = 24)
fig.text(0.05, 0.5, 'Empirical Coverage', fontsize = 24, ha='center', va='center', rotation='vertical')
fig.savefig("Empirical_Coverage_all_Range.pdf")
plt.close()


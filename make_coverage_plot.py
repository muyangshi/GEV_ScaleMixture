# %%
# Read Notes!!
"""
Make coverage plots for the parameters:
phi, rho (range), mu, sigma
from the 3 scenarios of 50 repeated simulations

- fix the plotting for scenario 2 (remove the unfinished simulation)
- use binomial confidence interval instead of normal approximation
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

# %%
# Coverage Setup ------------------------------------------------------------------------------------------------------
burnins = 6000 # length of burnin iterations
simulation_case = 'scenario3'
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
folders = ['./data/'+simulation_case+'/simulation_' + str(sim_id) + '/' for sim_id in sim_ids]

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
########################
# individual coverage  #
########################
# # load data and calculate statistics
# PE_matrix_phi = np.full(shape = (k, nsim), fill_value = np.nan)
# lower_bound_matrix_phi = np.full(shape = (k, nsim), fill_value = np.nan)
# upper_bound_matrix_phi = np.full(shape = (k, nsim), fill_value = np.nan)
# PE_matrix_range = np.full(shape = (k, nsim), fill_value = np.nan)
# lower_bound_matrix_range = np.full(shape = (k, nsim), fill_value = np.nan)
# upper_bound_matrix_range = np.full(shape = (k, nsim), fill_value = np.nan)
# PE_matrix_Rt_log = np.full(shape = (k, Nt, nsim), fill_value = np.nan)
# lower_bound_matrix_Rt_log = np.full(shape = (k, Nt, nsim), fill_value = np.nan)
# upper_bound_matrix_Rt_log = np.full(shape = (k, Nt, nsim), fill_value = np.nan)
# PE_matrix_loc = np.full(shape = (k, nsim), fill_value = np.nan)
# lower_bound_matrix_loc = np.full(shape = (k, nsim), fill_value = np.nan)
# upper_bound_matrix_loc = np.full(shape = (k, nsim), fill_value = np.nan)
# PE_matrix_scale = np.full(shape = (k, nsim), fill_value = np.nan)
# lower_bound_matrix_scale = np.full(shape = (k, nsim), fill_value = np.nan)
# upper_bound_matrix_scale = np.full(shape = (k, nsim), fill_value = np.nan)
# # PE_matrix_shape = np.full(shape = (k, nsim), fill_value = np.nan)
# # lower_bound_matrix_shape = np.full(shape = (k, nsim), fill_value = np.nan)
# # upper_bound_matrix_shape = np.full(shape = (k, nsim), fill_value = np.nan)

# for i in range(nsim):
#     folder = folders[i]

#     # load
#     phi_knots_trace = np.load(folder + 'phi_knots_trace.npy')
#     R_trace_log = np.load(folder + 'R_trace_log.npy')
#     range_knots_trace = np.load(folder + 'range_knots_trace.npy')
#     GEV_knots_trace = np.load(folder + 'GEV_knots_trace.npy')

#     # drop burnins
#     phi_knots_trace = phi_knots_trace[burnins:]
#     R_trace_log = R_trace_log[burnins:]
#     range_knots_trace = range_knots_trace[burnins:]
#     GEV_knots_trace = GEV_knots_trace[burnins:]

#     # phi
#     PE_matrix_phi[:,i] = np.mean(phi_knots_trace, axis = 0)
#     lower_bound_matrix_phi[:,i] = np.quantile(phi_knots_trace, q = 0.025, axis = 0)
#     upper_bound_matrix_phi[:,i] = np.quantile(phi_knots_trace, q = 0.975, axis = 0)
#     # range
#     PE_matrix_range[:,i] = np.mean(range_knots_trace, axis = 0)
#     lower_bound_matrix_range[:,i] = np.quantile(range_knots_trace, q = 0.025, axis = 0)
#     upper_bound_matrix_range[:,i] = np.quantile(range_knots_trace, q = 0.975, axis = 0)
#     # Rt
#     PE_matrix_Rt_log[:,:,i] = np.mean(R_trace_log, axis = 0)
#     lower_bound_matrix_Rt_log[:,:,i] = np.quantile(R_trace_log, q = 0.025, axis = 0)
#     upper_bound_matrix_Rt_log[:,:,i] = np.quantile(R_trace_log, q = 0.975, axis = 0)
#     # loc
#     PE_matrix_loc[:,i] = np.mean(GEV_knots_trace[:,0,:], axis = 0)
#     lower_bound_matrix_loc[:,i] = np.quantile(GEV_knots_trace[:,0,:], q = 0.025, axis = 0)
#     upper_bound_matrix_loc[:,i] = np.quantile(GEV_knots_trace[:,0,:], q = 0.975, axis = 0)
#     # scale
#     PE_matrix_scale[:,i] = np.mean(GEV_knots_trace[:,1,:], axis = 0)
#     lower_bound_matrix_scale[:,i] = np.quantile(GEV_knots_trace[:,1,:], q = 0.025, axis = 0)
#     upper_bound_matrix_scale[:,i] = np.quantile(GEV_knots_trace[:,1,:], q = 0.975, axis = 0)
#     # # shape
#     # PE_matrix_shape[:,i] = np.mean(GEV_knots_trace[:,2,:], axis = 0)
#     # lower_bound_matrix_shape[:,i] = np.quantile(GEV_knots_trace[:,2,:], q = 0.025, axis = 0)
#     # upper_bound_matrix_shape[:,i] = np.quantile(GEV_knots_trace[:,2,:], q = 0.975, axis = 0)

# # %%
# # make plots for phi
# for knot_id in range(k):
#     fig, ax = plt.subplots()
#     ax.hlines(y = phi_at_knots[knot_id], xmin = sim_id_from, xmax = sim_id_to,
#             color = 'black')
#     coloring = ['red' if type1 == True else 'green' 
#                 for type1 in np.logical_or(lower_bound_matrix_phi[knot_id,:] > phi_at_knots[knot_id], 
#                                             upper_bound_matrix_phi[knot_id,:] < phi_at_knots[knot_id])]
#     plt.errorbar(x = sim_ids, y = PE_matrix_phi[knot_id,:], 
#                 yerr = np.vstack((PE_matrix_phi[knot_id,:] - lower_bound_matrix_phi[knot_id,:], 
#                                   upper_bound_matrix_phi[knot_id,:] - PE_matrix_phi[knot_id,:])), 
#                 fmt = 'o',
#                 ecolor = coloring) # errorbar yerr is for size
#     plt.title('knot: ' + str(knot_id) + ' phi = ' + str(round(phi_at_knots[knot_id],3)))
#     plt.xlabel('simulation number')
#     plt.ylabel('phi')
#     plt.show()
#     fig.savefig('phi_knot_' + str(knot_id) + '.pdf')
#     plt.close()

# # %%
# # make plots for range
# for knot_id in range(k):
#     fig, ax = plt.subplots()
#     ax.hlines(y = range_at_knots[knot_id], xmin = sim_id_from, xmax = sim_id_to,
#             color = 'black')
#     coloring = ['red' if type1 == True else 'green' 
#             for type1 in np.logical_or(lower_bound_matrix_range[knot_id,:] > range_at_knots[knot_id], 
#                                         upper_bound_matrix_range[knot_id,:] < range_at_knots[knot_id])]
#     plt.errorbar(x = sim_ids, y = PE_matrix_range[knot_id,:], 
#                 yerr = np.vstack((PE_matrix_range[knot_id,:] - lower_bound_matrix_range[knot_id,:], 
#                                   upper_bound_matrix_range[knot_id,:] - PE_matrix_range[knot_id,:])), 
#                 fmt = 'o',
#                 ecolor = coloring)
#     plt.title('knot: ' + str(knot_id) + ' range = ' + str(round(range_at_knots[knot_id],3)))
#     plt.xlabel('simulation number')
#     plt.ylabel('range')
#     plt.show()
#     fig.savefig('range_knot_' + str(knot_id) + '.pdf')
#     plt.close()

# # %%
# # make plots for loc
# # for knot_id in range(k):
# knot_id = 0
# fig, ax = plt.subplots()
# ax.hlines(y = mu, xmin = sim_id_from, xmax = sim_id_to,
#         color = 'black')
# coloring = ['red' if type1 == True else 'green' 
#             for type1 in np.logical_or(lower_bound_matrix_loc[knot_id,:] > mu, 
#                                         upper_bound_matrix_loc[knot_id,:] < mu)]
# plt.errorbar(x = sim_ids, 
#             y = PE_matrix_loc[knot_id,:], 
#             yerr = np.vstack((PE_matrix_loc[knot_id,:] - lower_bound_matrix_loc[knot_id,:], 
#                                 upper_bound_matrix_loc[knot_id,:] - PE_matrix_loc[knot_id,:])), 
#             fmt = 'o',
#             ecolor = coloring) # errorbar yerr is for size
# plt.title('knot: ' + str(knot_id) + ' loc = ' + str(mu))
# plt.xlabel('simulation number')
# plt.ylabel('loc')
# plt.show()
# fig.savefig('loc_knot_' + str(knot_id) + '.pdf')
# plt.close()

# # %%
# # make plots for scale
# # for knot_id in range(k):
# knot_id = 0
# fig, ax = plt.subplots()
# ax.hlines(y = tau, xmin = sim_id_from, xmax = sim_id_to,
#         color = 'black')
# coloring = ['red' if type1 == True else 'green' 
#             for type1 in np.logical_or(lower_bound_matrix_scale[knot_id,:] > tau, 
#                                         upper_bound_matrix_scale[knot_id,:] < tau)]
# plt.errorbar(x = sim_ids, 
#             y = PE_matrix_scale[knot_id,:], 
#             yerr = np.vstack((PE_matrix_scale[knot_id,:] - lower_bound_matrix_scale[knot_id,:], 
#                                 upper_bound_matrix_scale[knot_id,:] - PE_matrix_scale[knot_id,:])), 
#             fmt = 'o',
#             ecolor = coloring) # errorbar yerr is for size
# plt.title('knot: ' + str(knot_id) + ' scale = ' + str(tau))
# plt.xlabel('simulation number')
# plt.ylabel('scale')
# plt.show()
# fig.savefig('scale_knot_' + str(knot_id) + '.pdf')
# plt.close()

# # # %%
# # # make plots for log(R_t)
# # #################################################################### #
# # Cannot make these plots because R_t is different for each simulation #
# # #################################################################### #
# # t = 0
# # for knot_id in range(k):
# #     fig, ax = plt.subplots()
# #     ax.hlines(y = np.log(R_at_knots[knot_id,t]), xmin = sim_id_from, xmax = nsim,
# #             color = 'black')
# #     coloring = ['red' if type1 == True else 'green' 
# #             for type1 in np.logical_or(lower_bound_matrix_Rt_log[knot_id,t,:] > np.log(R_at_knots)[knot_id,t], 
# #                                         upper_bound_matrix_Rt_log[knot_id,t,:] < np.log(R_at_knots)[knot_id,t])]
# #     plt.errorbar(x = 1 + np.arange(nsim), 
# #                 y = PE_matrix_Rt_log[knot_id,t,:], 
# #                 yerr = np.vstack((PE_matrix_Rt_log[knot_id,t,:] - lower_bound_matrix_Rt_log[knot_id,t,:], 
# #                                   upper_bound_matrix_Rt_log[knot_id,t,:] - PE_matrix_Rt_log[knot_id,t,:])), 
# #                 fmt = 'o',
# #                 ecolor = coloring)
# #     plt.title('knot: ' + str(knot_id) + ' t = ' + str(t) +' log(Rt) = ' + str(round(np.log(R_at_knots[knot_id,t]),3)))
# #     plt.xlabel('simulation number')
# #     plt.ylabel('Rt')
# #     plt.show()
# #     fig.savefig('R_knot_' + str(knot_id) + '_t_' + str(t) + '.pdf')
# #     plt.close()


# ############################
# # overall avg coverage phi #
# ############################
# # %%
# # overall coverage for phi
# alphas = np.flip(np.linspace(0.05, 0.4, 15))
# q_low = alphas/2
# q_high = 1 - q_low

# # summary by credible levels
# PE_matrix_phi = np.full(shape = (k, nsim), fill_value = np.nan)
# lower_bound_matrix_phi_alpha = np.full(shape = (k, nsim, len(alphas)), fill_value = np.nan)
# upper_bound_matrix_phi_alpha = np.full(shape = (k, nsim, len(alphas)), fill_value = np.nan)
# for level_i in range(len(alphas)):
#     for i in range(nsim):
#         folder = folders[i]
#         phi_knots_trace = np.load(folder + 'phi_knots_trace.npy')
#         # drop burnins
#         phi_knots_trace = phi_knots_trace[burnins:]
#         # phi
#         PE_matrix_phi[:,i] = np.mean(phi_knots_trace, axis = 0)
#         lower_bound_matrix_phi_alpha[:,i, level_i] = np.quantile(phi_knots_trace, q = q_low[level_i], axis = 0)
#         upper_bound_matrix_phi_alpha[:,i, level_i] = np.quantile(phi_knots_trace, q = q_high[level_i], axis = 0)

# # coverage flag
# phi_covers = np.full(shape = (k, nsim, len(alphas)), fill_value = np.nan)
# for level_i in range(len(alphas)):
#     for knot_id in range(k):
#         phi_covers[knot_id, :, level_i] = \
#             np.logical_and(lower_bound_matrix_phi_alpha[knot_id,:,level_i] < phi_at_knots[knot_id], 
#                             upper_bound_matrix_phi_alpha[knot_id,:, level_i] > phi_at_knots[knot_id])

# # average coverage
# avg_phi_covers = np.mean(phi_covers, axis = 1)
# se_phi_covers = scipy.stats.sem(phi_covers, axis = 1)

# # plotting
# for knot_id in range(k):
#     fig, ax = plt.subplots()
#     plt.xlim([0.5, 1])
#     plt.ylim([0.5, 1])
#     ax.plot([0,1],[0,1], transform=ax.transAxes, color = 'black')
#     plt.errorbar(x = 1 - alphas, 
#                 y = avg_phi_covers[knot_id,:], 
#                 yerr = 1.96*se_phi_covers[knot_id,:],
#                 fmt = 'o')
#     plt.title('phi knot ' + str(knot_id))
#     plt.ylabel('empirical coverage w/ 1.96*SE')
#     plt.xlabel('1-alpha')
#     plt.show()
#     fig.savefig('phi_knot_' + str(knot_id) + '_avg' + '.pdf')
#     plt.close()


# ##############################
# # overall avg coverage range #
# ##############################
# # %%
# # overall coverage for range

# alphas = np.flip(np.linspace(0.05, 0.4, 15))
# q_low = alphas/2
# q_high = 1 - q_low

# # summary by credible levels
# PE_matrix_range = np.full(shape = (k, nsim), fill_value = np.nan)
# lower_bound_matrix_range_alpha = np.full(shape = (k, nsim, len(alphas)), fill_value = np.nan)
# upper_bound_matrix_range_alpha = np.full(shape = (k, nsim, len(alphas)), fill_value = np.nan)
# for level_i in range(len(alphas)):
#     for i in range(nsim):
#         folder = folders[i]
#         range_knots_trace = np.load(folder + 'range_knots_trace.npy')
#         # drop burnins
#         range_knots_trace = range_knots_trace[burnins:]
#         # range
#         PE_matrix_range[:,i] = np.mean(range_knots_trace, axis = 0)
#         lower_bound_matrix_range_alpha[:,i, level_i] = np.quantile(range_knots_trace, q = q_low[level_i], axis = 0)
#         upper_bound_matrix_range_alpha[:,i, level_i] = np.quantile(range_knots_trace, q = q_high[level_i], axis = 0)

# # coverage flag
# range_covers = np.full(shape = (k, nsim, len(alphas)), fill_value = np.nan)
# for level_i in range(len(alphas)):
#     for knot_id in range(k):
#         range_covers[knot_id, :, level_i] = \
#             np.logical_and(lower_bound_matrix_range_alpha[knot_id,:,level_i] < range_at_knots[knot_id], 
#                             upper_bound_matrix_range_alpha[knot_id,:, level_i] > range_at_knots[knot_id])

# # average coverage
# avg_range_covers = np.mean(range_covers, axis = 1)
# # std_range_covers = np.std(range_covers, axis = 1)
# se_range_covers = scipy.stats.sem(range_covers, axis = 1)

# # plotting
# for knot_id in range(k):
#     fig, ax = plt.subplots()
#     plt.xlim([0.5, 1])
#     plt.ylim([0.5, 1])
#     ax.plot([0,1],[0,1], transform=ax.transAxes, color = 'black')
#     plt.errorbar(x = 1 - alphas, 
#                 y = avg_range_covers[knot_id,:], 
#                 yerr = 1.96*se_range_covers[knot_id,:],
#                 fmt = 'o')
#     plt.title('range knot ' + str(knot_id))
#     plt.ylabel('empirical coverage w/ 1.96*SE')
#     plt.xlabel('1-alpha')
#     plt.show()
#     fig.savefig('range_knot_' + str(knot_id) + '_avg' + '.pdf')
#     plt.close()

# ############################
# # overall avg coverage loc #
# ############################
# # %%
# alphas = np.flip(np.linspace(0.05, 0.4, 15))
# q_low = alphas/2
# q_high = 1 - q_low

# # summary by credible levels
# PE_matrix_loc = np.full(shape = (1, nsim), fill_value = np.nan)
# lower_bound_matrix_loc_alpha = np.full(shape = (1, nsim, len(alphas)), fill_value = np.nan)
# upper_bound_matrix_loc_alpha = np.full(shape = (1, nsim, len(alphas)), fill_value = np.nan)
# for level_i in range(len(alphas)):
#     for i in range(nsim):
#         folder = folders[i]
#         GEV_knots_trace = np.load(folder + 'GEV_knots_trace.npy')
#         loc_knots_trace = GEV_knots_trace[:,0,0]
#         scale_knots_trace = GEV_knots_trace[:,1,0]
#         # drop burnins
#         loc_knots_trace = loc_knots_trace[burnins:]
#         scale_knots_trace = scale_knots_trace[burnins:]
#         # location mu
#         PE_matrix_loc[:,i] = np.mean(loc_knots_trace)
#         lower_bound_matrix_loc_alpha[:, i, level_i] = np.quantile(loc_knots_trace, q = q_low[level_i], axis = 0)
#         upper_bound_matrix_loc_alpha[:, i, level_i] = np.quantile(loc_knots_trace, q = q_high[level_i], axis = 0)

# # coverage flag
# loc_covers = np.full(shape = (1, nsim, len(alphas)), fill_value = np.nan)
# for level_i in range(len(alphas)):
#     loc_covers[0, :, level_i] = np.logical_and(lower_bound_matrix_loc_alpha[0,:,level_i] < mu,
#                                                upper_bound_matrix_loc_alpha[0,:,level_i] > mu)

# # average coverage
# avg_loc_covers = np.mean(loc_covers, axis = 1)
# se_loc_covers = scipy.stats.sem(loc_covers, axis = 1)

# # plotting
# fig, ax = plt.subplots()
# plt.xlim([0,1])
# plt.ylim([0,1])
# ax.plot([0,1],[0,1], transform = ax.transAxes, color = 'black')
# plt.errorbar(x = 1 - alphas,
#              y = avg_loc_covers[0,:],
#              yerr = 1.96*se_loc_covers[0,:],
#              fmt = 'o')
# plt.title('loc')
# plt.ylabel('empirical coverage w/ 1.96*SE')
# plt.xlabel('1-alpha')
# plt.show()
# fig.savefig('loc_avg.pdf')
# plt.close()

# ############################
# # overall avg coverage scale #
# ############################
# # %%
# alphas = np.flip(np.linspace(0.05, 0.4, 15))
# q_low = alphas/2
# q_high = 1 - q_low

# # summary by credible levels
# PE_matrix_scale = np.full(shape = (1, nsim), fill_value = np.nan)
# lower_bound_matrix_scale_alpha = np.full(shape = (1, nsim, len(alphas)), fill_value = np.nan)
# upper_bound_matrix_scale_alpha = np.full(shape = (1, nsim, len(alphas)), fill_value = np.nan)
# for level_i in range(len(alphas)):
#     for i in range(nsim):
#         folder = folders[i]
#         GEV_knots_trace = np.load(folder + 'GEV_knots_trace.npy')
#         loc_knots_trace = GEV_knots_trace[:,0,0]
#         scale_knots_trace = GEV_knots_trace[:,1,0]
#         # drop burnins
#         loc_knots_trace = loc_knots_trace[burnins:]
#         scale_knots_trace = scale_knots_trace[burnins:]
#         # scale sigma
#         PE_matrix_scale[:,i] = np.mean(scale_knots_trace)
#         lower_bound_matrix_scale_alpha[:, i, level_i] = np.quantile(scale_knots_trace, q = q_low[level_i], axis = 0)
#         upper_bound_matrix_scale_alpha[:, i, level_i] = np.quantile(scale_knots_trace, q = q_high[level_i], axis = 0)

# # coverage flag
# scale_covers = np.full(shape = (1, nsim, len(alphas)), fill_value = np.nan)
# for level_i in range(len(alphas)):
#     scale_covers[0, :, level_i] = np.logical_and(lower_bound_matrix_scale_alpha[0,:,level_i] < tau,
#                                                upper_bound_matrix_scale_alpha[0,:,level_i] > tau)

# # average coverage
# avg_scale_covers = np.mean(scale_covers, axis = 1)
# se_scale_covers = scipy.stats.sem(scale_covers, axis = 1)

# # plotting
# fig, ax = plt.subplots()
# plt.xlim([0.5,1])
# plt.ylim([0.5,1])
# ax.plot([0,1],[0,1], transform = ax.transAxes, color = 'black')
# plt.errorbar(x = 1 - alphas,
#              y = avg_scale_covers[0,:],
#              yerr = 1.96*se_scale_covers[0,:],
#              fmt = 'o')
# plt.title('scale')
# plt.ylabel('empirical coverage w/ 1.96*SE')
# plt.xlabel('1-alpha')
# plt.show()
# fig.savefig('scale_avg.pdf')
# plt.close()




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
fig, axes = plt.subplots(3,3)
fig.set_size_inches(24,24)
for knot_k in range(k):
    ax = axes[knot_k//3, knot_k%3]
    ax.set_xlim([0.5, 1])
    ax.set_ylim([0.5, 1])
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color = 'black')

    ax.errorbar(x = 1 - alphas,
                y = avg_phi_covers[knot_k,:],
                yerr=phi_binom_errs[knot_k,:,:],
                fmt = 'bo',
                ecolor = 'b')
    ax.set_title(r'k='+str(knot_k), fontsize = 24)
    ax.tick_params(axis='both', which = 'major', labelsize = 20)
    ax.tick_params(axis='both', which = 'minor', labelsize = 20)
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
fig, axes = plt.subplots(3,3)
fig.set_size_inches(24,24)
for knot_k in range(k):
    ax = axes[knot_k//3, knot_k%3]
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
fig.suptitle(r'Empirical Coverage for $\rho_k$', fontsize = 36)
fig.text(0.5, 0.05, 'Credible Level', ha='center', va='center', fontsize = 24)
fig.text(0.05, 0.5, 'Empirical Coverage', fontsize = 24, ha='center', va='center', rotation='vertical')
fig.savefig("Empirical_Coverage_all_Range.pdf")
plt.close()



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

fig, ax = plt.subplots()
fig.set_size_inches(8,8)
plt.xlim([0.5,1])
plt.ylim([0.5,1])
plt.axline((0,0), slope = 1, color = 'black')
plt.errorbar(x = 1 - alphas,
            y = avg_loc_covers[0,:],
            yerr=loc_binom_errs[0,:,:],
            fmt = 'bo',
            ecolor = 'b')
plt.title(r'Location $\mu$', fontsize = 24)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
fig.text(0.5, 0.04, 'Credible Level', ha='center', va='center', fontsize = 24)
fig.text(0.04, 0.5, 'Empirical Coverage', fontsize = 24, ha='center', va='center', rotation='vertical')
fig.savefig('Empirical_Coverage_Mu.pdf')
# plt.show()
plt.close()

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

fig, ax = plt.subplots()
fig.set_size_inches(8,8)
plt.xlim([0.5,1])
plt.ylim([0.5,1])
plt.axline((0,0), slope = 1, color = 'black')
plt.errorbar(x = 1 - alphas,
            y = avg_scale_covers[0,:],
            yerr=scale_binom_errs[0,:,:],
            fmt = 'bo',
            ecolor = 'b')
plt.title(r'Scale $\sigma$', fontsize = 24)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
fig.text(0.5, 0.04, 'Credible Level', ha='center', va='center', fontsize = 24)
fig.text(0.04, 0.5, 'Empirical Coverage', fontsize = 24, ha='center', va='center', rotation='vertical')
fig.savefig('Empirical_Coverage_Sigma.pdf')
# plt.show()
plt.close()
# %%

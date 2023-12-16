# %%
import numpy as np
k = 10
k = 9

# %%
# Results

# folder = './data/20231026_all_adaptive_scenario2_seed2345_time64_site500_alpine/'
# folder = './data/20231116_aa_sc2_sd2345_t64_st500_10knots_misspiggy/'
folder = './data/20231206_sc2_sd79_t16_s100_stdPareto/'
phi_knots_trace = np.load(folder + 'phi_knots_trace.npy')
R_trace_log = np.load(folder + 'R_trace_log.npy')
range_knots_trace = np.load(folder + 'range_knots_trace.npy')
GEV_knots_trace = np.load(folder + 'GEV_knots_trace.npy')

Beta_mu_trace = np.load('Beta_mu_trace.npy')
Beta_mu_trace = Beta_mu_trace[0:30300]

Beta_logsigma_trace = np.load('Beta_logsigma_trace.npy')
Beta_logsigma_trace = Beta_logsigma_trace[0:30300]

Beta_ksi_trace = np.load('Beta_ksi_trace.npy')
Beta_ksi_trace = Beta_ksi_trace[0:30300]


# phi_knots_trace = phi_knots_trace[500:, :]
# range_knots_trace = range_knots_trace[22000:, :]
# GEV_knots_trace = GEV_knots_trace[2000:,:,:]

##########################################################################################
# Posterior Covariance Matrix
##########################################################################################
# %%
GEV_post_cov = np.cov(np.array([GEV_knots_trace[:,0,0].ravel(), # mu location
                                GEV_knots_trace[:,1,0].ravel()])) # tau scale

phi_post_cov = np.cov(np.array([phi_knots_trace[:,i].ravel() for i in range(k)]))
range_post_cov = np.cov(np.array([range_knots_trace[:,i].ravel() for i in range(k)]))

mu = GEV_knots_trace[:,0,0].ravel()
tau = GEV_knots_trace[:,1,0].ravel()
phi = [phi_knots_trace[:,i].ravel() for i in range(k)]

np.corrcoef(np.insert(phi, 0, [mu,tau], 0))
# np.cov(np.vstack((mu, tau,phi)))
# np.corrcoef(np.vstack((mu, tau,phi)))

Beta_mu_m = Beta_mu_trace.shape[1]
Beta_mu_post_cov = np.cov(Beta_mu_trace.T)
# Beta_mu_post_cov = np.cov(np.array([Beta_mu_trace[:,j].ravel() for j in range(Beta_mu_m)]))

Beta_logsigma_post_cov = np.cov(Beta_logsigma_trace.T)

Beta_ksi_post_cov = np.cov(Beta_ksi_trace.T)
# %%

"""
Summary informaiton regarding the posterior draws
- covariance
- summary statistics
- etc.
"""
# %%
import numpy as np

# %%
# load traceplots
# folder                    = './data/20240216_t32_s125_shifted/'
folder                    = './data/20240218_testrun_full/'
phi_knots_trace           = np.load(folder + 'phi_knots_trace.npy')
R_trace_log               = np.load(folder + 'R_trace_log.npy')
range_knots_trace         = np.load(folder + 'range_knots_trace.npy')
Beta_mu0_trace            = np.load(folder + 'Beta_mu0_trace.npy')
Beta_mu1_trace            = np.load(folder + 'Beta_mu1_trace.npy')
Beta_logsigma_trace       = np.load(folder + 'Beta_logsigma_trace.npy')
Beta_ksi_trace            = np.load(folder + 'Beta_ksi_trace.npy')
sigma_Beta_mu0_trace      = np.load(folder + 'sigma_Beta_mu0_trace.npy')
sigma_Beta_mu1_trace      = np.load(folder + 'sigma_Beta_mu1_trace.npy')
sigma_Beta_logsigma_trace = np.load(folder + 'sigma_Beta_logsigma_trace.npy')
sigma_Beta_ksi_trace      = np.load(folder + 'sigma_Beta_ksi_trace.npy')

k               = R_trace_log.shape[1]
Nt              = R_trace_log.shape[2]
Beta_mu0_m      = Beta_mu0_trace.shape[1]
Beta_mu1_m      = Beta_mu1_trace.shape[1]
Beta_logsigma_m = Beta_logsigma_trace.shape[1]
Beta_ksi_m      = Beta_ksi_trace.shape[1]

# %%
# burnins
burnin = 1500

phi_knots_trace           = phi_knots_trace[burnin:]
R_trace_log               = R_trace_log[burnin:]
range_knots_trace         = range_knots_trace[burnin:]
Beta_mu0_trace            = Beta_mu0_trace[burnin:]
Beta_mu1_trace            = Beta_mu1_trace[burnin:]
Beta_logsigma_trace       = Beta_logsigma_trace[burnin:]
Beta_ksi_trace            = Beta_ksi_trace[burnin:]
sigma_Beta_mu0_trace      = sigma_Beta_mu0_trace[burnin:]
sigma_Beta_mu1_trace      = sigma_Beta_mu1_trace[burnin:]
sigma_Beta_logsigma_trace = sigma_Beta_logsigma_trace[burnin:]
sigma_Beta_ksi_trace      = sigma_Beta_ksi_trace[burnin:]


# %%
# remove unfinished cells

R_trace_log               = R_trace_log[~np.isnan(R_trace_log)].reshape((-1,k,Nt))
phi_knots_trace           = phi_knots_trace[~np.isnan(phi_knots_trace)].reshape((-1,k))
range_knots_trace         = range_knots_trace[~np.isnan(range_knots_trace)].reshape((-1,k))
Beta_mu0_trace            = Beta_mu0_trace[~np.isnan(Beta_mu0_trace)].reshape((-1,Beta_mu0_m))
Beta_mu1_trace            = Beta_mu1_trace[~np.isnan(Beta_mu1_trace)].reshape((-1,Beta_mu1_m))
Beta_logsigma_trace       = Beta_logsigma_trace[~np.isnan(Beta_logsigma_trace)].reshape((-1,Beta_logsigma_m))
Beta_ksi_trace            = Beta_ksi_trace[~np.isnan(Beta_ksi_trace)].reshape((-1,Beta_ksi_m))
sigma_Beta_mu0_trace      = sigma_Beta_mu0_trace[~np.isnan(sigma_Beta_mu0_trace)].reshape((-1,1))
sigma_Beta_mu1_trace      = sigma_Beta_mu1_trace[~np.isnan(sigma_Beta_mu1_trace)].reshape((-1,1))
sigma_Beta_logsigma_trace = sigma_Beta_logsigma_trace[~np.isnan(sigma_Beta_logsigma_trace)].reshape((-1,1))
sigma_Beta_ksi_trace      = sigma_Beta_ksi_trace[~np.isnan(sigma_Beta_ksi_trace)].reshape((-1,1))


#######################################
##### Posterior Covariance Matrix #####
#######################################
# %%
# posterior covariance matrix
phi_cov           = np.cov(phi_knots_trace.T)
R_log_cov         = np.full(shape=(k,k,R_trace_log.shape[2]), fill_value = np.nan)
for t in range(R_trace_log.shape[2]):
    R_log_cov[:,:,t] = np.cov(R_trace_log[:,:,t].T)
range_cov         = np.cov(range_knots_trace.T)
Beta_mu0_cov      = np.cov(Beta_mu0_trace.T)
Beta_mu1_cov      = np.cov(Beta_mu1_trace.T)
Beta_logsigma_cov = np.cov(Beta_logsigma_trace.T)
Beta_ksi_cov      = np.cov(Beta_ksi_trace.T)
sigma_Beta_mu0_cov = np.cov(sigma_Beta_mu0_trace.T)
sigma_Beta_mu1_cov = np.cov(sigma_Beta_mu1_trace.T)
sigma_Beta_logsigma_cov = np.cov(sigma_Beta_logsigma_trace.T)
sigma_Beta_ksi_cov = np.cov(sigma_Beta_ksi_trace.T)

#######################################
##### Posterior Median            #####
#######################################
# Potentially use these as initial values
# %%
# posterior median
phi_median                 = np.median(phi_knots_trace, axis = 0)
R_log_median               = np.full(shape=(k,R_trace_log.shape[2]), fill_value = np.nan)
for t in range(R_trace_log.shape[2]):
    R_log_median[:,t] = np.median(R_trace_log[:,:,t], axis = 0)
range_median               = np.median(range_knots_trace, axis = 0)
Beta_mu0_median            = np.median(Beta_mu0_trace, axis = 0)
Beta_mu1_median            = np.median(Beta_mu1_trace, axis = 0)
Beta_logsigma_median       = np.median(Beta_logsigma_trace, axis = 0)
Beta_ksi_median            = np.median(Beta_ksi_trace, axis = 0)
sigma_Beta_mu0_median      = np.median(sigma_Beta_mu0_trace, axis = 0)
sigma_Beta_mu1_median      = np.median(sigma_Beta_mu1_trace, axis = 0)
sigma_Beta_logsigma_median = np.median(sigma_Beta_logsigma_trace, axis = 0)
sigma_Beta_ksi_median      = np.median(sigma_Beta_ksi_trace, axis = 0)


#######################################
##### Posterior Last Iteration    #####
#######################################
# %%
# last iteration values
phi_knots_last           = phi_knots_trace[-1]
R_last_log               = R_trace_log[-1]
range_knots_last         = range_knots_trace[-1]
Beta_mu0_last            = Beta_mu0_trace[-1]
Beta_mu1_last            = Beta_mu1_trace[-1]
Beta_logsigma_last       = Beta_logsigma_trace[-1]
Beta_ksi_last            = Beta_ksi_trace[-1]
sigma_Beta_mu0_last      = sigma_Beta_mu0_trace[-1]
sigma_Beta_mu1_last      = sigma_Beta_mu1_trace[-1]
sigma_Beta_logsigma_last = sigma_Beta_logsigma_trace[-1]
sigma_Beta_ksi_last      = sigma_Beta_ksi_trace[-1]


# %%
# GEV_post_cov = np.cov(np.array([GEV_knots_trace[:,0,0].ravel(), # mu location
#                                 GEV_knots_trace[:,1,0].ravel()])) # tau scale

# phi_post_cov = np.cov(np.array([phi_knots_trace[:,i].ravel() for i in range(k)]))
# range_post_cov = np.cov(np.array([range_knots_trace[:,i].ravel() for i in range(k)]))

# mu = GEV_knots_trace[:,0,0].ravel()
# tau = GEV_knots_trace[:,1,0].ravel()
# phi = [phi_knots_trace[:,i].ravel() for i in range(k)]

# np.corrcoef(np.insert(phi, 0, [mu,tau], 0))
# # np.cov(np.vstack((mu, tau,phi)))
# # np.corrcoef(np.vstack((mu, tau,phi)))

# %%

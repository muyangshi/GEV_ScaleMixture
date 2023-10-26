# %%
import numpy as np
k = 9

# %%
# Results

folder = './'
phi_knots_trace = np.load(folder + 'phi_knots_trace.npy')
R_trace_log = np.load(folder + 'R_trace_log.npy')
range_knots_trace = np.load(folder + 'range_knots_trace.npy')
GEV_knots_trace = np.load(folder + 'GEV_knots_trace.npy')
xs = np.arange(10000)

##########################################################################################
# Posterior Covariance Matrix
##########################################################################################

GEV_post_cov = np.cov(np.array([GEV_knots_trace[:,0,0].ravel(), # mu location
                                GEV_knots_trace[:,1,0].ravel()])) # tau scale

phi_post_cov = np.cov(np.array([phi_knots_trace[:,i].ravel() for i in range(k)]))

range_post_cov = np.cov(np.array([range_knots_trace[:,i].ravel() for i in range(k)]))

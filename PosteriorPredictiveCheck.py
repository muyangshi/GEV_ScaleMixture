"""
Making coverage plots for the posterior predictions on Y,
i.e., posterior predictive check
"""
# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt

folder = './data/20240304_gatherY/'
Y_real = np.load(folder + 'Y_sim_sc2_t32_s100_truth.npy')
Y_trace = np.load(folder + 'Y_trace.npy')
miss_matrix = np.load(folder + 'miss_matrix_bool.npy')
n_miss = np.sum(miss_matrix)
Y_miss_real = Y_real[miss_matrix]
Y_miss_trace = Y_trace[:,miss_matrix]
# %%
alphas = np.flip(np.linspace(0.025, 0.4, 16))
creds  = 1 - alphas
q_low = alphas/2
q_high = 1 - q_low

lower_bound_matrix_Y_miss = np.full(shape = (1, n_miss, len(alphas)), fill_value = np.nan)
upper_bound_matrix_Y_miss = np.full(shape = (1, n_miss, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    lower_bound_matrix_Y_miss[0,:, level_i] = np.quantile(Y_miss_trace, q = q_low[level_i], axis = 0)
    upper_bound_matrix_Y_miss[0,:, level_i] = np.quantile(Y_miss_trace, q = q_high[level_i], axis = 0)

# coverage flag
Y_miss_covers = np.full(shape = (1, n_miss, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    Y_miss_covers[0,:,level_i] = np.logical_and(lower_bound_matrix_Y_miss[0,:,level_i] < Y_miss_real,
                                               upper_bound_matrix_Y_miss[0,:,level_i] > Y_miss_real)

# PE: average coverage
avg_Y_miss_covers = np.mean(Y_miss_covers, axis = 1)

# SE: binomtest confidence interval
Y_miss_binom_CIs = np.full(shape = (1, 2, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    l, h = scipy.stats.binomtest(k = int(sum(Y_miss_covers[0,:,level_i])), # number of success (covers)
                                    n = n_miss, # total number of repetition
                                    p = 1-alphas[level_i]).proportion_ci(confidence_level=0.95)
    Y_miss_binom_CIs[0, :, level_i] = [l,h]

# errs = np.array([[avg_Y_miss_covers[0,0] - np.array(l)], # first row contains the lower errors
#                  [np.array(h) - avg_Y_miss_covers[0,0]]]) # second row contains the upper errors
Y_miss_binom_errs = np.full(shape = (1, 2, len(alphas)), fill_value = np.nan)
for level_i in range(len(alphas)):
    Y_miss_binom_errs[0,:,level_i] = [avg_Y_miss_covers[0,level_i] - Y_miss_binom_CIs[0, 0, level_i],
                                        Y_miss_binom_CIs[0, 1, level_i] - avg_Y_miss_covers[0, level_i]]

# %%
fig, ax = plt.subplots()
fig.set_size_inches(8,8)
plt.xlim([0.5,1])
plt.ylim([0.5,1])
plt.axline((0,0), slope = 1, color = 'black')
plt.errorbar(x = 1 - alphas,
            y = avg_Y_miss_covers[0,:],
            yerr=Y_miss_binom_errs[0,:,:],
            fmt = 'bo',
            ecolor = 'b')
plt.title(r'Y miss', fontsize = 24)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
fig.text(0.5, 0.04, 'Credible Level', ha='center', va='center', fontsize = 24)
fig.text(0.04, 0.5, 'Empirical Coverage', fontsize = 24, ha='center', va='center', rotation='vertical')

# %%

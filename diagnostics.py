"""
Goals for this file:
1. Make a moving window chi plot for Y
"""
# %%
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from rpy2.robjects import r 
from rpy2.robjects.numpy2ri import numpy2rpy
from rpy2.robjects.packages import importr
from math import sin, cos, sqrt, atan2, radians
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
from utilities import *
state_map = gpd.read_file('./cb_2018_us_state_20m/cb_2018_us_state_20m.shp')

# %%
def coord_to_dist(coord1: tuple, coord2: tuple):
    R = 6373.0 # Approximate radius of earth in km

    long1 = radians(coord1[0])
    lat1  = radians(coord1[1])
    long2 = radians(coord2[0])
    lat2  = radians(coord2[1])

    dlong = long2 - long1
    dlat  = lat2  - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlong / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# %%
# load data
mgcv = importr('mgcv')
r('''load('JJA_precip_maxima_nonimputed.RData')''')
GEV_estimates      = np.array(r('GEV_estimates')).T
mu0_estimates      = GEV_estimates[:,0]
mu1_estimates      = GEV_estimates[:,1]
logsigma_estimates = GEV_estimates[:,2]
ksi_estimates      = GEV_estimates[:,3]
JJA_maxima         = np.array(r('JJA_maxima_nonimputed'))
stations           = np.array(r('stations')).T
elevations         = np.array(r('elev')).T/200

Y = JJA_maxima.copy()
miss_matrix = np.isnan(Y)

Ns   = JJA_maxima.shape[0]
Nt   = JJA_maxima.shape[1] # number of time replicates
Time = np.linspace(-Nt/2, Nt/2-1, Nt)/np.std(np.linspace(-Nt/2, Nt/2-1, Nt), ddof=1)
pY   = np.full(shape = (Ns, Nt), fill_value = np.nan)
for t in range(Nt):
    pY[:,t] = pgev(Y[:,t], mu0_estimates + mu1_estimates * Time[t],
                           np.exp(logsigma_estimates),
                           ksi_estimates)

sites_xy = stations
sites_x = sites_xy[:,0]
sites_y = sites_xy[:,1]

# define the lower and upper limits for x and y
minX, maxX = np.floor(np.min(sites_x)), np.ceil(np.max(sites_x))
minY, maxY = np.floor(np.min(sites_y)), np.ceil(np.max(sites_y))

# place knots for chi plot
res_x = 9
res_y = 19
k = res_x * res_y # number of knots
# create one-dimensional arrays for x and y
x_pos = np.linspace(minX, maxX, res_x+2)[2:-2]
y_pos = np.linspace(minY, maxY, res_y+2)[2:-2]
# create the mesh based on these arrays
X_pos, Y_pos = np.meshgrid(x_pos,y_pos)
knots_xy = np.vstack([X_pos.ravel(), Y_pos.ravel()]).T
knots_x = knots_xy[:,0]
knots_y = knots_xy[:,1]   

# %%
# make a plot of the sites and knots
fig, ax = plt.subplots()
fig.set_size_inches(6,8)
ax.set_aspect('equal', 'box')
state_map.boundary.plot(ax=ax, color = 'black', linewidth = 0.5)
ax.scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.5)
ax.scatter(knots_x, knots_y, s = 15, color = 'blue', marker = '+')

rect_width = (knots_xy[0][0] - minX)*2
rect_height = (knots_xy[0][1] - minY)*2
rect_i = plt.Rectangle((knots_xy[0][0] - rect_width/2, knots_xy[0][1] - rect_height/2), 
                       width = rect_width, height = rect_height,
                       fill = False, ec = 'black', linewidth = 2)
ax.add_patch(rect_i)

plt.xlim([-105,-90])
plt.ylim([30,50])
plt.show()

# histogram of distance between sites
allsite_dist_mat = np.full(shape = (Ns, Ns), fill_value = np.nan)
for si in range(Ns):
    for sj in range(Ns):
        allsite_dist_mat[si,sj] = coord_to_dist(sites_xy[si], sites_xy[sj])
plt.hist(allsite_dist_mat[np.triu_indices(Ns, k = 1)].ravel())


# select sites within the rectangle
i = 118 # 36, 48, etc.
rect_left   = knots_xy[i][0] - rect_width/2
rect_right  = knots_xy[i][0] + rect_width/2
rect_top    = knots_xy[i][1] + rect_height/2
rect_bottom = knots_xy[i][1] - rect_height/2
sites_in_rect_mask = np.logical_and(np.logical_and(rect_left <= sites_x, sites_x <= rect_right), 
                                    np.logical_and(rect_bottom <= sites_y, sites_y <= rect_top))
sites_in_rect = sites_xy[sites_in_rect_mask]

# calculate the distance between sites inside rectangle (coords --> km)
n_sites = sites_in_rect.shape[0]
sites_dist_mat = np.full(shape = (n_sites, n_sites), fill_value = np.nan)
for si in range(n_sites):
    for sj in range(n_sites):
        sites_dist_mat[si,sj] = coord_to_dist(sites_in_rect[si], sites_in_rect[sj])
plt.hist(sites_dist_mat[np.triu_indices(n_sites, k = 1)].ravel())


# %%
# for each black box we will calculate chi
# we will plot the chi's to a heatmap

u = 0.9
h = 30

e_abs = 0.2
h_low = h * (1 - e_abs)
h_up  = h * (1 + e_abs)

# e_abs = 20
# h_low = h - e_abs
# h_up  = h + e_abs

chi_mat = np.full(shape = (len(x_pos), len(y_pos)), fill_value = np.nan)

chi_mat2 = np.full(shape = (len(y_pos), len(x_pos)), fill_value = np.nan)

for i in range(knots_xy.shape[0]):

    # select sites within the rectangle
    rect_left   = knots_xy[i][0] - rect_width/2
    rect_right  = knots_xy[i][0] + rect_width/2
    rect_top    = knots_xy[i][1] + rect_height/2
    rect_bottom = knots_xy[i][1] - rect_height/2
    sites_in_rect_mask = np.logical_and(np.logical_and(rect_left <= sites_x, sites_x <= rect_right), 
                                        np.logical_and(rect_bottom <= sites_y, sites_y <= rect_top))
    sites_in_rect = sites_xy[sites_in_rect_mask]

    # calculate the distance between sites inside rectangle (coords --> km)
    n_sites = sites_in_rect.shape[0]
    sites_dist_mat = np.full(shape = (n_sites, n_sites), fill_value = np.nan)
    for si in range(n_sites):
        for sj in range(n_sites):
            sites_dist_mat[si,sj] = coord_to_dist(sites_in_rect[si], sites_in_rect[sj])

    # select pairs: sites that are ~h km apart
    sites_h_mask = np.logical_and(np.triu(sites_dist_mat) > h_low,
                                np.triu(sites_dist_mat) < h_up)
    n_pairs = len(np.triu(sites_dist_mat)[sites_h_mask])
    site_pairs_to_check = [(np.where(sites_h_mask)[0][i], np.where(sites_h_mask)[1][i]) for i in range(n_pairs)]
    # plt.hist(sites_dist_mat[sites_h_mask])
    # plt.close()

    # # draw edges
    # if i % 10 == 0:
    #     fig, ax = plt.subplots()
    #     fig.set_size_inches(6,8)
    #     ax.set_aspect('equal', 'box')
    #     state_map.boundary.plot(ax=ax, color = 'black', linewidth = 0.5)
    #     ax.scatter(sites_in_rect[:,0], sites_in_rect[:,1], s = 5, color = 'grey', marker = 'o', alpha = 0.8)
    #     ax.scatter(knots_x, knots_y, s = 15, color = 'red', marker = '+')
    #     for site_pair in site_pairs_to_check:
    #         coord_1 = sites_in_rect[site_pair[0]]
    #         coord_2 = sites_in_rect[site_pair[1]]
    #         ax.plot((coord_1[0], coord_2[0]), (coord_1[1], coord_2[1]), 'blue', linestyle='-', linewidth = 0.5)
    #     rect_i = plt.Rectangle((knots_xy[i][0] - rect_width/2, knots_xy[i][1] - rect_height/2), 
    #                         width = rect_width, height = rect_height,
    #                         fill = False, ec = 'black', linewidth = 2)
    #     ax.add_patch(rect_i)
    #     plt.xlim([-105,-90])
    #     plt.ylim([30,50])
    #     plt.show()
    #     plt.close()

    # large pairs
    Y_in_rect     = Y[sites_in_rect_mask]
    pY_in_rect    = pY[sites_in_rect_mask]

    count_co_extreme = 0
    for site_pair in site_pairs_to_check:
        # for this pair, over time, how many co-occured extremes?
        count_co_extreme += np.sum(np.logical_and(pY_in_rect[site_pair[0]] >= u,
                                                pY_in_rect[site_pair[1]] >= u))

    prob_joint_ext = count_co_extreme / (n_pairs * Nt)
    prob_uni_ext   = np.mean(pY_in_rect >= u)
    chi            = prob_joint_ext / prob_uni_ext

    chi_mat[i % 7, i // 7] = chi
    chi_mat2[-1 - i // 7, i % 7] = chi

# %%
fig, ax = plt.subplots()
fig.set_size_inches(6,8)
ax.set_aspect('equal', 'box')
state_map.boundary.plot(ax=ax, color = 'black', linewidth = 0.5)
heatmap = ax.imshow(chi_mat2, cmap ='bwr', interpolation='nearest', 
                    extent = [min(x_pos - rect_width/8), max(x_pos + rect_width/8), 
                              min(y_pos - rect_height/8), max(y_pos+rect_height/8)])
# ax.scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.8)
ax.scatter(knots_x, knots_y, s = 15, color = 'white', marker = '+')
fig.colorbar(heatmap)
plt.xlim([-105,-90])
plt.ylim([30,50])
plt.title(rf'empirical $\chi_{{{u}}}$, h $\approx$ {h}km')
plt.show()
plt.close()
# fig, ax = plt.subplots()
# fig.set_size_inches(6,8)
# ax.set_aspect('equal', 'box')
# state_map.boundary.plot(ax=ax, color = 'black', linewidth = 0.5)
# heatmap = ax.imshow(chi_mat.T, cmap ='bwr', interpolation='nearest', extent = [min(x_pos), max(x_pos), max(y_pos), min(y_pos)])
# ax.scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.8)
# ax.scatter(knots_x, knots_y, s = 15, color = 'red', marker = '+')
# fig.colorbar(heatmap)
# plt.xlim([-105,-90])
# plt.ylim([30,50])
# plt.show()
# plt.close()

# %%

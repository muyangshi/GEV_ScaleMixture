
import os
os.chdir("/Users/LikunZhang/Desktop/Nonstat/")

# import random
import nonstat_Pareto1.model_sim as utils
import nonstat_Pareto1.generic_samplers as sampler
import nonstat_Pareto1.priors as priors
import nonstat_Pareto1.ns_cov as cov
import numpy as np
from scipy.stats import uniform
from scipy.stats import norm 

# ------------ 1. Simulation settings -------------
nu =  3/2        # Matern smoothness

n_s = 25*25      # Number of sites
n_t = 68         # Number of time points

gamma = 0.5

# -------------- 2. Location matrix and nine knots -----------------
np.random.seed(seed=1234)
Stations = np.c_[uniform.rvs(0,10,n_s),uniform.rvs(0,10,n_s)]
# loc_tmp = np.linspace(0,10,num=7)[np.array([1,3,5])]
# x_tmp,y_tmp = np.meshgrid(loc_tmp, loc_tmp)
# Knots_data = np.c_[x_tmp.flatten(), y_tmp.flatten()]
xmin=0; xmax=10; ymin=0; ymax=10; Ngrid = 9
x_vals = np.linspace(xmin + 1,  xmax + 1, num=int(2*np.sqrt(Ngrid)))
y_vals = np.linspace(ymin + 1,  ymax + 1, num=int(2*np.sqrt(Ngrid)))
part1_x, part1_y = np.meshgrid(x_vals[np.arange(0,len(x_vals)-1,2)],
              y_vals[np.arange(0,len(x_vals)-1,2)])
part2_x, part2_y = np.meshgrid(x_vals[np.arange(1,len(x_vals)-1,2)],
              y_vals[np.arange(1,len(x_vals)-1,2)]) # The seq does not include the right end
isometric_grid = np.r_[
  np.c_[part1_x.flatten(), part1_y.flatten()],
  np.c_[part2_x.flatten(), part2_y.flatten()]
]

Knots_data = isometric_grid
radius = 3.5
radius_from_knots = np.repeat(radius, Knots_data.shape[0])
# radius_from_knots[0]=4; radius_from_knots[12]=4

# import matplotlib.pyplot as plt
# circle0 = plt.Circle((Knots_data[0,0],Knots_data[0,1]), radius_from_knots[0], color='r', fill=False)
# circle1 = plt.Circle((Knots_data[1,0],Knots_data[1,1]), radius_from_knots[1], color='r', fill=False)
# circle2 = plt.Circle((Knots_data[2,0],Knots_data[2,1]), radius_from_knots[2], color='r', fill=False)
# circle3 = plt.Circle((Knots_data[3,0],Knots_data[3,1]), radius_from_knots[3], color='r', fill=False)
# circle4 = plt.Circle((Knots_data[4,0],Knots_data[4,1]), radius_from_knots[4], color='r', fill=False)
# circle5 = plt.Circle((Knots_data[5,0],Knots_data[5,1]), radius_from_knots[5], color='r', fill=False)
# circle6 = plt.Circle((Knots_data[6,0],Knots_data[6,1]), radius_from_knots[6], color='r', fill=False)
# circle7 = plt.Circle((Knots_data[7,0],Knots_data[7,1]), radius_from_knots[7], color='r', fill=False)
# circle8 = plt.Circle((Knots_data[8,0],Knots_data[8,1]), radius_from_knots[8], color='r', fill=False)
# circle9 = plt.Circle((Knots_data[9,0],Knots_data[9,1]), radius_from_knots[9], color='r', fill=False)
# circle10 = plt.Circle((Knots_data[10,0],Knots_data[10,1]), radius_from_knots[10], color='r', fill=False)
# circle11 = plt.Circle((Knots_data[11,0],Knots_data[11,1]), radius_from_knots[11], color='r', fill=False)
# circle12 = plt.Circle((Knots_data[12,0],Knots_data[12,1]), radius_from_knots[12], color='r', fill=False)

# ax = plt.gca()
# ax.cla() # clear things for fresh plot
# ax.set_xlim((0, 10))
# ax.set_ylim((0, 10))
# ax.scatter(Stations[:,0], Stations[:,1],marker='o', alpha=0.5)
# ax.scatter(Knots_data[:,0], Knots_data[:,1],marker='+', c='r')
# ax.add_patch(circle0)
# ax.add_patch(circle1)
# ax.add_patch(circle2)
# ax.add_patch(circle3)
# ax.add_patch(circle4)
# ax.add_patch(circle5)
# ax.add_patch(circle6)
# ax.add_patch(circle7)
# ax.add_patch(circle8)
# ax.add_patch(circle9)
# ax.add_patch(circle10)
# ax.add_patch(circle11)
# ax.add_patch(circle12)
# plt.xlabel('x'); plt.ylabel('y')
# figure = plt.gcf()
# figure.set_size_inches(3.4, 3.4)
# plt.savefig("/Users/LikunZhang/Desktop/PyCode/Simulation_figures/R_knots.png", dpi=94)


## ***********************
test = 1 
np.random.seed(seed=test)
## ***********************


# -------------- 3. Generate covariance matrix -----------------
from scipy.spatial import distance


# Range values at the knots
# range_at_knots = (-1 + 0.3*Knots_data[:,0] + 0.4*Knots_data[:,1])/4
range_at_knots = np.sqrt(0.3*Knots_data[:,0] + 0.4*Knots_data[:,1])/2


bw = 4 # bandwidth
range_vec = np.repeat(np.nan, n_s)
num_knots = np.repeat(np.nan, n_s)
phi_range_weights = np.empty((n_s,Knots_data.shape[0]))
for idx in np.arange(n_s):
  d_tmp = distance.cdist(Stations[idx,:].reshape((-1,2)),Knots_data)
  weights = utils.weights_fun(d_tmp,radius,bw,cutoff=False)
  phi_range_weights[idx,:] = weights
  range_vec[idx] = np.sum(weights*range_at_knots)
  num_knots[idx]=np.sum(weights>0)


# import matplotlib.pyplot as plt
# plt.scatter(Stations[:,0], Stations[:,1],c=range_vec, marker='o', alpha=0.5, cmap='jet')
# plt.colorbar()
# plt.title("Range vector");

# Calculate covariance:
Cov = cov.ns_cov(range_vec, np.repeat(1, Stations.shape[0]), Stations, kappa = nu, cov_model = "matern")

# plt.scatter(Stations[:,0], Stations[:,1],c=Cov[495,:], marker='o', alpha=0.5, cmap='jet')
# plt.colorbar()
# plt.title("Covariance cencered at location 495");

# plt.scatter(Stations[:,0], Stations[:,1],c=Cov[100,:], marker='o', alpha=0.5, cmap='jet')
# plt.colorbar()
# plt.title("Covariance cencered at location 100");


eig_Cor = np.linalg.eigh(Cov) #For symmetric matrices
V = eig_Cor[1]
d = eig_Cor[0]


# -------------- 4. Generate scaling factor -----------------
# phi values at the knots
## Scenario 1
phi_at_knots = 0.65-np.sqrt((Knots_data[:,0]-3)**2/4 + (Knots_data[:,1]-3)**2/3)/10

## Scenario 2
# from scipy.stats import multivariate_normal
# phi_at_knots = np.empty(Knots_data.shape[0])
# for wh in np.arange(Knots_data.shape[0]):
#     phi_at_knots[wh] = 13*multivariate_normal.pdf(Knots_data[wh,:], mean=np.array([5,6]), 
#                        cov = 4*np.array([[1,0.6],[0.6,1]]))+0.25

## Scenario 3
# from scipy.stats import multivariate_normal
# phi_at_knots = np.empty(Knots_data.shape[0])
# for wh in np.arange(Knots_data.shape[0]):
#     phi_at_knots[wh] = 27*(0.4*multivariate_normal.pdf(Knots_data[wh,:], mean=np.array([2,3]), cov = 2.7*np.array([[1,0.2],[0.2,1]]))+
#   0.6*multivariate_normal.pdf(Knots_data[wh,:], mean=np.array([8,8.5]), cov = 6*np.array([[1,-0.3],[-0.3,1]])))+0.17
   
# phi_vec = np.repeat(np.nan, n_s)
# for idx in np.arange(n_s):
#   d_tmp = distance.cdist(Stations[idx,:].reshape((-1,2)),Knots_data)
#   weights = utils.weights_fun(d_tmp,radius,bw, cutoff=False)
#   phi_vec[idx] = np.sum(weights*phi_at_knots)

n_phi_range_knots = len(phi_at_knots)
phi_vec = phi_range_weights @ phi_at_knots

# import matplotlib.pyplot as plt
# plt.scatter(Stations[:,0], Stations[:,1],c=phi_vec, marker='o', alpha=0.5, cmap='jet')
# plt.colorbar()
# plt.title(r"$\phi(s)$");

R_at_knots = np.empty((Knots_data.shape[0],n_t))
n_Rt_knots = R_at_knots.shape[0]
R_s = np.empty((n_s,n_t))
R_s[:] = np.nan
R_weights = np.empty((n_s,Knots_data.shape[0]))
gamma_vec = np.repeat(np.nan, n_s)
for idx in np.arange(n_t):
    S = utils.rlevy(Knots_data.shape[0],m=0,s=gamma)
    R_at_knots[:,idx] = S
    for idy in np.arange(n_s):
        d_tmp = distance.cdist(Stations[idy,:].reshape((-1,2)),Knots_data)
        weights = utils.wendland_weights_fun(d_tmp,radius_from_knots)
        R_s[idy,idx] = np.sum(weights*S)
        if idx ==1: 
            R_weights[idy,:] = weights
            gamma_vec[idy] = np.sum(np.sqrt(weights[np.nonzero(weights)]*gamma))**2 #only save once

## Same as the following for gamma_vec
def gamma_func_apply(vec):
    return np.sum(np.sqrt(vec[np.nonzero(vec)]))**2*gamma
gamma_vec = np.apply_along_axis(gamma_func_apply, 1, R_weights)
# plt.scatter(Stations[:,0], Stations[:,1],c=R_s[:,31], marker='o', alpha=0.5, cmap='jet')
# plt.colorbar()
# plt.title(r"$R(s)$");


# -------------- 5. Generate X(s) and X_s(s) -----------------
X = np.empty((n_s,n_t))
X[:] = np.nan

Z = np.empty((n_s,n_t))
Z[:] = np.nan
for idy in np.arange(n_t):
  Z_t = utils.eig2inv_times_vector(V, np.sqrt(d), norm.rvs(size=n_s))
  Z_to_W_s = 1/(1-norm.cdf(Z_t))-1
  tmp = (R_s[:,idy]**phi_vec)*Z_to_W_s
  X[:,idy] = tmp 
  Z[:,idy] = Z_t

# import matplotlib.pyplot as plt
# plt.scatter(Stations[:,0], Stations[:,1],c=Z[:,1], marker='o', alpha=0.5, cmap='jet')
# plt.colorbar()
# plt.title(r"$Z(s)$");

# plt.scatter(Stations[:,0], Stations[:,1],c=np.log(X_s[:,1]), marker='o', alpha=0.5, cmap='jet')
# plt.colorbar()
# plt.title(r"$\log\{R(s)^{\phi(s)}W(s)\}$");

# from matplotlib.backends.backend_pdf import PdfPages


# for idy in np.arange(n_t):
#     pdf_filename = "./Simulation_replicates/sim"+str(idy)+".pdf"
#     fig, axs = plt.subplots(2, 2, figsize=(8,8))
#     pp = PdfPages(pdf_filename,)
#     axs[0, 0].scatter(Stations[:,0], Stations[:,1],c=phi_vec, marker='o', alpha=0.5, cmap='jet')
#     # axs[0, 0].colorbar()
#     axs[0, 0].set_title(r"$\phi(s)$")
#     axs[0, 1].scatter(Stations[:,0], Stations[:,1],c=R_s[:,idy], marker='o', alpha=0.5, cmap='jet')
#     # axs[0, 1].colorbar()
#     axs[0, 1].set_title(r"$R(s)$")
#     axs[1, 0].scatter(Stations[:,0], Stations[:,1],c=Z[:,idy], marker='o', alpha=0.5, cmap='jet')
#     # axs[1, 0].colorbar()
#     axs[1, 0].set_title(r"$Z(s)$")
#     axs[1, 1].scatter(Stations[:,0], Stations[:,1],c=np.log(X_s[:,idy]), marker='o', alpha=0.5, cmap='jet')
#     # axs[1, 1].colorbar()
#     axs[1, 1].set_title(r"$\log\{R(s)^{\phi(s)}W(s)\}$")
#     pp.savefig(fig)
#     pp.close()
#     plt.close()




# ------------ 3. Marginal transformation -----------------
# Design_mat = np.c_[np.repeat(1,n_s), Stations[:,1]]
Design_mat = np.c_[np.repeat(1,n_s), np.repeat(0,n_s)]
n_covariates = Design_mat.shape[1]

beta_loc0 = np.array([20,0])
loc0 = Design_mat @beta_loc0

beta_loc1 = np.array([0,0])
loc1 = Design_mat @beta_loc1

Time = np.arange(n_t)
Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(Time,n_s)
Loc = Loc.reshape((n_s,n_t),order='F')


beta_scale = np.array([1,0])
scale = Design_mat @beta_scale
Scale = np.tile(scale, n_t)
Scale = Scale.reshape((n_s,n_t),order='F')

beta_shape = np.array([0.2,0])
shape = Design_mat @beta_shape
Shape = np.tile(shape, n_t)
Shape = Shape.reshape((n_s,n_t),order='F')

beta_gev_params = np.array([beta_loc0[0], beta_scale[0], beta_shape[0]])
n_beta_gev_params = beta_gev_params.shape[0]
   
Y = np.empty((n_s,n_t))
Y[:] = np.nan
unifs = np.empty((n_s,n_t))
unifs[:] = np.nan
for idx in np.arange(n_s):
    Y[idx,:] = utils.RW_2_gev(X[idx,:], phi_vec[idx], gamma_vec[idx], Loc[idx,:], Scale[idx,:], Shape[idx,:])
    unifs[idx,:] = utils.pgev(Y[idx,:], Loc[idx,:], Scale[idx,:], Shape[idx,:])




# thresh_X =  utils.qRW_me_interp(prob_below, xp, surv_p, tau_sqd, phi, gamma)
# thresh_X_above =  utils.qRW_me_interp(prob_above, xp, surv_p, tau_sqd, phi, gamma)



# ------------ 4. Save initial values -----------------
data = {'Knots':Knots_data,
        'phi_range_weights': phi_range_weights,
        'R_weights':R_weights,
        'Stations':Stations,
        'phi_vec':phi_vec,
        'phi_at_knots':phi_at_knots,
        'radius_from_knots':radius_from_knots,
        'gamma':gamma,
        'gamma_vec':gamma_vec,
        'range_vec':range_vec,
        'range_at_knots':range_at_knots,
        'nu':nu,
        'X':X,
        'R_at_knots':R_at_knots,
        'R_s':R_s,
        'Design_mat':Design_mat,
        'beta_loc0':beta_loc0,
        'beta_loc1':beta_loc1,
        'Time':Time,
        'beta_scale':beta_scale,
        'beta_shape':beta_shape,
        }
n_updates = 1001    
sigma_m   = {'phi_radius':np.sqrt(2.4**2/(n_phi_range_knots+4)),
             'radius':0.00042, #np.sqrt(2.4**2/n_Rt_knots),
             'Rt':np.sqrt(2.4**2/n_Rt_knots),
             'phi':np.sqrt(2.4**2/n_phi_range_knots),
             'range':np.sqrt(2.4**2/n_phi_range_knots),
             'gev_params':np.sqrt(2.4**2/n_beta_gev_params),
             'theta_c':2.4**2/2,
             'R_1t':2.4**2,
             'beta_loc0':2.4**2/n_covariates,
             'beta_loc1':2.4**2/n_covariates,
             'beta_scale':2.4**2/n_covariates,
             'beta_shape':2.4**2/n_covariates,
             }
prop_sigma   = {'phi_radius':np.eye(n_phi_range_knots+4),
                'radius':np.eye(n_Rt_knots)*1e-3,
                'Rt':np.eye(n_Rt_knots),
                'phi':np.eye(n_phi_range_knots)*1e-3,
                'range':np.eye(n_phi_range_knots),
                'gev_params':np.eye(n_beta_gev_params)*1e-3,
                'theta_c':np.eye(2),
                'beta_loc0':np.eye(n_covariates),
                'beta_loc1':np.eye(n_covariates),
                'beta_scale':np.eye(n_covariates),
                'beta_shape':np.eye(n_covariates)
                }

from pickle import dump
with open('./data_sim1.pkl', 'wb') as f:
     dump(Y, f)
     dump(data, f)
     dump(sigma_m, f)
     dump(prop_sigma, f)



## ---------------------------------------------------------
## ------------------------ For phi ------------------------
## ---------------------------------------------------------

import matplotlib.pyplot as plt
choose = 2
phi_at_knots[choose]
X_star = np.empty(X.shape)
from scipy.linalg import cholesky
cholesky_U = cholesky(Cov,lower=False)
phi_at_knots_proposal = np.empty(phi_at_knots.shape)

def test(phi):
    phi_at_knots_proposal[:] = phi_at_knots
    phi_at_knots_proposal[choose] = phi
    phi_vec_star = phi_range_weights @ phi_at_knots_proposal
    Star_lik = np.empty(n_t)
    for rank in np.arange(n_t):
        X_star[:,rank] = utils.gev_2_RW(Y[:,rank], phi_vec_star, gamma_vec, 
                                  Loc[:,rank], Scale[:,rank], Shape[:,rank])
        Star_lik[rank] = utils.marg_transform_data_mixture_likelihood_1t(Y[:,rank], X_star[:,rank], Loc[:,rank], Scale[:,rank],
                                                 Shape[:,rank], phi_vec_star, gamma_vec, R_s[:,rank],
                                                 cholesky_U)
    return np.sum(Star_lik)

from scipy.linalg import lapack
def test_1dim(phi, rank):
    phi_at_knots_proposal[:] = phi_at_knots
    phi_at_knots_proposal[0] = phi
    phi_vec_star = phi_range_weights @ phi_at_knots_proposal
    X_star = np.empty(n_s)
    X_star[:] = utils.gev_2_RW(Y[:,rank], phi_vec_star, gamma_vec, 
                                  Loc[:,rank], Scale[:,rank], Shape[:,rank])
    W_vec = X_star/R_s[:,rank]**phi_vec_star
    Z_vec = utils.pareto_to_Norm(W_vec)
    
    cholesky_inv = lapack.dpotrs(cholesky_U,Z_vec)
    part1 = -0.5*np.sum(Z_vec*cholesky_inv[0])-np.sum(np.log(np.diag(cholesky_U))) # multivariate density
  
    ## Jacobian determinant
    part21 = 0.5*np.sum(Z_vec**2) # 1/standard Normal densities of each Z_j
    part22 = np.sum(-phi_vec_star*np.log(R_s[:,rank])-2*np.log(W_vec+1)) # R_j^phi_j/X_j^2
    part23 = np.sum(utils.dgev(Y[:,rank], Loc=Loc[:,rank], Scale=Scale[:,rank], Shape=Shape[:,rank], log=True)-np.log(utils.dRW(X_star, phi_vec_star, gamma_vec)))

    return (part1, part21, part22, part23)

Phi = np.linspace(phi_at_knots[choose]-0.2,phi_at_knots[choose]+0.2, num=20)
Lik = np.zeros(len(Phi))
for idx, phi_tmp in enumerate(Phi):
    print(idx)
    Lik[idx] = test(phi_tmp)
plt.plot(Phi, Lik, color='black', linestyle='solid')
plt.axvline(phi_at_knots[choose], color='r', linestyle='--');


random_generator = np.random.RandomState()
Res = sampler.static_metr(Y, 0.58, utils.phi_update_mixture_me_likelihood, priors.interval_unif, 
                  np.array([0.1,0.7]),1000, 
                  random_generator,
                  np.nan, 0.005, True, 
                  R, Z, cen, cen_above, 
                  prob_below, prob_above, Loc, Scale, Shape, tau_sqd, gamma) 
plt.plot(np.arange(1000),Res['trace'][0,:],linestyle='solid')


Res = sampler.adaptive_metr(Y, 0.552, utils.phi_update_mixture_me_likelihood, priors.interval_unif, 
                          np.array([0.1,0.7]),5000,
                          random_generator,
                          np.nan, False, False,
                          .234, 10, .8, 10, 
                          R, Z, cen, cen_above, 
                          prob_below, prob_above, Loc, Scale, Shape, tau_sqd, gamma)
plt.plot(np.arange(5000)[3:],Res['trace'][0,3:],linestyle='solid')
plt.hlines(phi, 0, 5000, colors='r', linestyles='--');


## -------------------------------------------------------
## ----------------------- For tau -----------------------
## -------------------------------------------------------
# t_chosen = 24
# def test(tau_sqd):
#     return utils.tau_update_mixture_me_likelihood(Y[:,t_chosen], tau_sqd, X_s[:,t_chosen], cen[:,t_chosen], cen_above[:,t_chosen], 
#                     prob_below, prob_above, Loc[:,t_chosen], Scale[:,t_chosen], Shape[:,t_chosen], 
#                     phi, gamma, xp, surv_p, den_p)

# Tau = np.arange(8,12,step=0.1)
# Lik = np.zeros(len(Tau))
# for idx, t in enumerate(Tau):
#     Lik[idx] = test(t) 
# plt.plot(Tau, Lik, color='black', linestyle='solid')
# plt.axvline(tau_sqd, color='r', linestyle='--');

def test(tau_sqd):
    return utils.tau_update_mixture_me_likelihood(Y, tau_sqd, X_s, cen, cen_above, 
                    prob_below, prob_above, Loc, Scale, Shape, 
                    phi, gamma, xp, surv_p, den_p)

Tau = np.arange(8,12,step=0.1)
Lik = np.zeros(len(Tau))
for idx, t in enumerate(Tau):
    Lik[idx] = test(t) 
plt.plot(Tau, Lik, color='black', linestyle='solid')
plt.axvline(tau_sqd, color='r', linestyle='--');



Res = sampler.static_metr(Y, 2.0, utils.tau_update_mixture_me_likelihood, priors.invGamma_prior, 
                          np.array([0.1,0.1]),1000, 
                          random_generator,
                          np.nan, 2.1, True, 
                          X_s, cen, cen_above, 
                          prob_below, prob_above, Loc, Scale, Shape, 
                          phi, gamma, xp, surv_p, den_p)
plt.plot(np.arange(1000),Res['trace'][0,:],linestyle='solid')


Res = sampler.adaptive_metr(Y, 2, utils.tau_update_mixture_me_likelihood, priors.invGamma_prior, 
                          np.array([0.1,0.1]),5000,
                          random_generator,
                          np.nan, False, False,
                          .234, 10, .8, 10, 
                          X_s, cen, cen_above, 
                          prob_below, prob_above, Loc, Scale, Shape, 
                          phi, gamma, xp, surv_p, den_p)
plt.plot(np.arange(5000)[3:],Res['trace'][0,3:],linestyle='solid')
plt.hlines(tau_sqd, 0, 5000, colors='r', linestyles='--');



## --------------------------------------------------------------------
## ----------------------- For radii from knots -----------------------
## --------------------------------------------------------------------
from scipy.linalg import cholesky
cholesky_U = cholesky(Cov,lower=False)

# Distance from knots
Distance_from_stations_to_knots = np.empty((n_s, n_Rt_knots))
for ind in np.arange(n_s):
    Distance_from_stations_to_knots[ind,:] = distance.cdist(Stations[ind,:].reshape((-1,2)),Knots_data)
 
R_weights_star = np.empty((n_s,Knots_data.shape[0]))
R_s_star = np.empty(R_s.shape)
gamma_vec_star = np.repeat(np.nan, n_s)
X_star = np.empty(X.shape)

def tmpf(radius_knot1):
    radius_from_knots_proposal = np.repeat(radius_knot1,n_Rt_knots)
    # radius_from_knots_proposal[0] = radius_knot1
 
    # Not broadcasting but generating at each node
    for idy in np.arange(n_s):
        tmp_weights = utils.wendland_weights_fun(Distance_from_stations_to_knots[idy,:],
                                                       radius_from_knots_proposal)
        R_weights_star[idy,:] = tmp_weights
        gamma_vec_star[idy] = np.sum(np.sqrt(tmp_weights[np.nonzero(tmp_weights)]*gamma))**2 #only save once
    for rank in np.arange(n_t):
        R_s_star[:,rank] = R_weights_star @ R_at_knots[:,rank]
        X_star[:,rank] = utils.gev_2_RW(Y[:,rank], phi_vec, gamma_vec_star, 
                                  Loc[:,rank], Scale[:,rank], Shape[:,rank])

    # Evaluate likelihood at new values
    if np.all(np.logical_and(radius_from_knots_proposal>0, radius_from_knots_proposal<10)):           
        Star_lik = utils.marg_transform_data_mixture_likelihood(Y, X_star, Loc, Scale, 
                                          Shape, phi_vec, gamma_vec_star, R_s_star, 
                                          cholesky_U)
    else:
        Star_lik = -np.inf
    return Star_lik

try_size = 30
x = np.linspace(3.3,3.7, try_size)

func = np.empty(try_size)
for idx,xi in enumerate(x):
         print(idx,xi)
         func[idx] = tmpf(xi)

import matplotlib.pyplot as plt
plt.plot(x, func, linestyle='solid')




## ----------------------------------------------------------------------------------
## ----------------------- For radii from knots (Monte Carlo) -----------------------
## ----------------------------------------------------------------------------------
# Distance from knots
Distance_from_stations_to_knots = np.empty((n_s, n_Rt_knots))
for ind in np.arange(n_s):
    Distance_from_stations_to_knots[ind,:] = distance.cdist(Stations[ind,:].reshape((-1,2)),Knots_data)
 
R_weights_star = np.empty((n_s,Knots_data.shape[0]))
R_s_star = np.empty(R_s.shape)
gamma_vec_star = np.repeat(np.nan, n_s)
X_star = np.empty(X.shape)
from scipy.linalg import cholesky
cholesky_U = cholesky(Cov,lower=False)

## Get Monte Carlo Samples
from pickle import load
   
with open('nonstat_progress_0.pkl', 'rb') as f:
     Y = load(f)
     initial_values = load(f)
     sigma_m = load(f)
     prop_Sigma = load(f)
     iter = load(f)
     Rt_knots_trace = load(f)
     # radius_knots_trace = load(f)
     range_knots_trace = load(f)
     phi_knots_radius_trace = load(f)
     beta_gev_params_trace = load(f)
    
     f.close()

choose = np.floor(np.linspace(5450, 5800,  num=20))
choose = choose.astype(int)

phi_1_vals = np.mean(phi_knots_radius_trace[choose,0])+(phi_knots_radius_trace[choose,0]-np.mean(phi_knots_radius_trace[choose,0]))*200

loc_vals = np.mean(beta_gev_params_trace[choose,0])+(beta_gev_params_trace[choose,0]-np.mean(beta_gev_params_trace[choose,0]))*200
scale_vals = np.mean(beta_gev_params_trace[choose,1])+(beta_gev_params_trace[choose,1]-np.mean(beta_gev_params_trace[choose,1]))*200
shape_vals = np.mean(beta_gev_params_trace[choose,2])+(beta_gev_params_trace[choose,2]-np.mean(beta_gev_params_trace[choose,2]))*200

GEV_phi_params = np.c_[loc_vals, scale_vals, shape_vals, phi_1_vals]


def tmpf(radius_knot1, GEV_phi_params):
    radius_from_knots_proposal = np.repeat(radius_knot1,n_Rt_knots)
    # radius_from_knots_proposal[0] = radius_knot1
 
    # Not broadcasting but generating at each node
    for idy in np.arange(n_s):
        tmp_weights = utils.wendland_weights_fun(Distance_from_stations_to_knots[idy,:],
                                                       radius_from_knots_proposal)
        R_weights_star[idy,:] = tmp_weights
        gamma_vec_star[idy] = np.sum(np.sqrt(tmp_weights[np.nonzero(tmp_weights)]*gamma))**2 #only save once
    for rank in np.arange(n_t):
            R_s_star[:,rank] = R_weights_star @ R_at_knots[:,rank]
            
    numiter = GEV_phi_params.shape[0]
    Lik = np.empty(numiter)
    for id_iter in np.arange(numiter):
        print(id_iter, "out of ", numiter)
        beta_gev_params_star = GEV_phi_params[id_iter,:3]
        loc0_star = Design_mat @np.array([beta_gev_params_star[0],0])
        scale_star = Design_mat @np.array([beta_gev_params_star[1],0])
        shape_star = Design_mat @np.array([beta_gev_params_star[2],0])
        Loc_star = np.tile(loc0_star, n_t) + np.tile(loc1, n_t)*np.repeat(Time,n_s)
        Loc_star = Loc_star.reshape((n_s,n_t),order='F')
        Scale_star = np.tile(scale_star, n_t)
        Scale_star = Scale_star.reshape((n_s,n_t),order='F')
        Shape_star = np.tile(shape_star, n_t)
        Shape_star = Shape_star.reshape((n_s,n_t),order='F')
        
        
        phi_at_knots_proposal=np.empty(n_phi_range_knots)
        phi_at_knots_proposal[:] = phi_at_knots
        phi_at_knots_proposal[0] = GEV_phi_params[id_iter,3]
        phi_vec_star = phi_range_weights @ phi_at_knots_proposal
            
        for rank in np.arange(n_t):
            X_star[:,rank] = utils.gev_2_RW(Y[:,rank], phi_vec_star, gamma_vec_star,
                                  Loc_star[:,rank], Scale_star[:,rank], Shape_star[:,rank])
        
        print("--------------------------")
        # Evaluate likelihood at new values
        if np.all(np.logical_and(radius_from_knots_proposal>0, radius_from_knots_proposal<10)):
            Star_lik = utils.marg_transform_data_mixture_likelihood(Y, X_star, Loc, Scale,
                                          Shape, phi_vec, gamma_vec_star, R_s_star,
                                          cholesky_U)
        else:
            Star_lik = -np.inf
            
        Lik[id_iter] = Star_lik
        
    return (np.nanmean(Lik),np.sum(np.isnan(Lik)))

try_size = 20
x = np.linspace(3.4,3.6, try_size)

func = np.empty(try_size)
numnan = np.empty(try_size)
for idx,xi in enumerate(x):
         print(idx,xi)
         tmp = tmpf(xi, GEV_phi_params)
         func[idx] = tmp[0]
         numnan[idx] = tmp[1]

import matplotlib.pyplot as plt
plt.plot(x, func, 'y', linestyle='solid')

plt.scatter(x, func)





## -------------------------------------------------------
## ------------------------ For Rt -----------------------
## -------------------------------------------------------

# R[0] = 0.074
rank = 1
add_small = np.linspace(-1,10,num=20)
Xt = X[:,rank]
Rt_s = R_s[:,rank]
Rt_at_knots = R_at_knots[:,rank]
from scipy.linalg import cholesky
cholesky_U = cholesky(Cov,lower=False)
def test(x):
    #Propose new values
    Rt_s_star = np.empty(Rt_s.shape)
       
    #Propose Rt under every worker
    Rt_at_knots_star = Rt_at_knots + np.concatenate((np.array([x]), np.repeat(0,n_Rt_knots-1)))
    Rt_s_star[:] = R_weights @ Rt_at_knots_star 
    return utils.marg_transform_data_mixture_likelihood_1t(Y[:,rank], Xt, Loc[:,rank], Scale[:,rank], 
                                                 Shape[:,rank], phi_vec, gamma_vec, Rt_s_star, 
                                                 cholesky_U)

Lik = np.zeros(len(add_small))
for idx, r in enumerate(add_small):
    print(idx)
    Lik[idx] = test(r) 
import matplotlib.pyplot as plt
plt.plot(add_small, Lik, linestyle='solid')
# plt.axvline(R[t_chosen], color='r', linestyle='--');



Res = sampler.adaptive_metr(Y[:,t_chosen], R[t_chosen], utils.Rt_update_mixture_me_likelihood, 
                            priors.R_prior, gamma, 5000, random_generator,
                            np.nan, True, False, .234, 10, .8,  10,
                            X[:,t_chosen], Z[:,t_chosen], cen[:,t_chosen], cen_above[:,t_chosen], 
                            prob_below, prob_above, Loc[:,t_chosen], Scale[:,t_chosen], Shape[:,t_chosen], tau_sqd, phi, gamma,
                            xp, surv_p, den_p, thresh_X, thresh_X_above)
plt.plot(np.arange(5000), Res['trace'][0,:], linestyle='solid')
plt.hlines(R[t_chosen], 0, 5000, colors='r', linestyles='--');






## --------------------------------------------------------------
## ----------------------- For GEV params -----------------------
## --------------------------------------------------------------
from scipy.linalg import cholesky
cholesky_U = cholesky(Cov,lower=False)
X_star = np.empty(X.shape)
def tmpf(x,y):
    beta_gev_params_star = np.array([x,y,0.2])

    # Evaluate likelihood at new values
    # Not broadcasting but generating at each node
    loc0_star = Design_mat @np.array([beta_gev_params_star[0],0])
    scale_star = Design_mat @np.array([beta_gev_params_star[1],0])
    shape_star = Design_mat @np.array([beta_gev_params_star[2],0])
    Loc_star = np.tile(loc0_star, n_t) + np.tile(loc1, n_t)*np.repeat(Time,n_s)
    Loc_star = Loc_star.reshape((n_s,n_t),order='F')
    Scale_star = np.tile(scale_star, n_t)
    Scale_star = Scale_star.reshape((n_s,n_t),order='F')
    Shape_star = np.tile(shape_star, n_t)
    Shape_star = Shape_star.reshape((n_s,n_t),order='F')
    for idx in np.arange(n_s):
           X_star[idx,:] = utils.gev_2_RW(Y[idx,:], phi_vec[idx], gamma_vec[idx], 
                                         Loc_star[idx,:], Scale_star[idx,:], Shape_star[idx,:])
    Star_lik = utils.marg_transform_data_mixture_likelihood(Y, X_star, Loc_star, Scale_star, 
                                          Shape_star, phi_vec, gamma_vec, R_s, 
                                          cholesky_U)
    return Star_lik




try_size = 20
x = np.linspace(19.5,20.5, try_size)
y = np.linspace(0.95,1.05, try_size)

func = np.empty((try_size,try_size))
for idy,yi in enumerate(y):
    for idx,xi in enumerate(x):
         print(idx,idy)
         func[idy,idx] = tmpf(xi,yi)

import matplotlib.pyplot as plt
plt.contourf(x, y, func, 20, cmap='RdGy')
plt.colorbar();

## Use prop_Sigma
Res = sampler.adaptive_metr_ratio(Design_mat, np.array([0.2,-1]), utils.loc0_gev_update_mixture_me_likelihood, 
                            priors.unif_prior, 20, 5000, random_generator, prop_Sigma, -0.2262189, 0.2827393557113686, True,
                            False, .234, 10, .8,  10,
                            Y, X_s, cen, cen_above, prob_below, prob_above, 
                            tau_sqd, phi, gamma, loc1, Scale, Shape, Time, xp, surv_p, den_p, 
                            thresh_X, thresh_X_above)


# (2) loc1: 0.1, -0.1
def test(x):
    return utils.loc1_gev_update_mixture_me_likelihood(Design_mat, np.array([x,-0.1]), Y, X_s, cen, cen_above, prob_below, prob_above, 
                     tau_sqd, phi, gamma, loc0, Scale, Shape, Time, xp, surv_p, den_p, 
                     thresh_X, thresh_X_above)
Coef = np.arange(0.098,0.14,step=0.0005)
Lik = np.zeros(len(Coef))
for idx, coef in enumerate(Coef):
    Lik[idx] = test(coef)
plt.plot(Coef, Lik, color='black', linestyle='solid')
plt.axvline(0.1, color='r', linestyle='--');


def test(x):
    return utils.loc1_gev_update_mixture_me_likelihood(Design_mat, np.array([0.1,x]), Y, X_s, cen, cen_above, prob_below, prob_above, 
                     tau_sqd, phi, gamma, loc0, Scale, Shape, Time, xp, surv_p, den_p, 
                     thresh_X, thresh_X_above)

Coef = np.arange(-0.11,-0.08,step=0.001)
Lik = np.zeros(len(Coef))
for idx, coef in enumerate(Coef):
    Lik[idx] = test(coef)
plt.plot(Coef, Lik, color='black', linestyle='solid')
plt.axvline(-0.1, color='r', linestyle='--');



Res = sampler.adaptive_metr(Design_mat, np.array([0.1,-0.1]), utils.loc1_gev_update_mixture_me_likelihood, 
                            priors.unif_prior, 20, 5000, random_generator, 
                            np.nan, True, False, .234, 10, .8,  10,
                            Y, X_s, cen, cen_above, prob_below, prob_above, 
                            tau_sqd, phi, gamma, loc0, Scale, Shape, Time, xp, surv_p, den_p, 
                            thresh_X, thresh_X_above)



plt.plot(np.arange(5000),Res['trace'][0,:],linestyle='solid')
plt.hlines(0.1, 0, 5000, colors='r', linestyles='--');

plt.plot(np.arange(5000),Res['trace'][1,:],linestyle='solid')
plt.hlines(-0.1, 0, 5000, colors='r', linestyles='--');
plt.plot(*Res['trace'])



def tmpf(x,y):
    return utils.loc1_gev_update_mixture_me_likelihood(Design_mat, np.array([x,y]), 
                            Y, X_s, cen, cen_above, prob_below, prob_above, 
                            tau_sqd, phi, gamma, loc0, Scale, Shape, Time, xp, surv_p, den_p, 
                            thresh_X, thresh_X_above)
try_size = 50
x = np.linspace(0.098, 0.103, try_size)
y = np.linspace(-0.103, -0.096, try_size)

func  = np.empty((try_size,try_size))
for idy,yi in enumerate(y):
    for idx,xi in enumerate(x):
         func[idy,idx] = tmpf(xi,yi)

plt.contourf(x, y, func, 20, cmap='RdGy')
plt.colorbar();



# (3) scale: 0.1,1
def test(x):
    return utils.scale_gev_update_mixture_me_likelihood(Design_mat, np.array([x,beta_scale[1]]), Y, X_s, cen, cen_above, prob_below, prob_above, 
                     tau_sqd, phi, gamma, Loc, Shape, Time, xp, surv_p, den_p, 
                     thresh_X, thresh_X_above)
Coef = np.arange(beta_scale[0]-0.01,beta_scale[0]+0.01,step=0.0005)
Lik = np.zeros(len(Coef))
for idx, coef in enumerate(Coef):
    Lik[idx] = test(coef)
plt.plot(Coef, Lik, color='black', linestyle='solid')
plt.axvline(beta_scale[0], color='r', linestyle='--');


def test(x):
    return utils.scale_gev_update_mixture_me_likelihood(Design_mat, np.array([beta_scale[0],x]), Y, X_s, cen, cen_above, prob_below, prob_above, 
                     tau_sqd, phi, gamma, Loc, Shape, Time, xp, surv_p, den_p, 
                     thresh_X, thresh_X_above)

Coef = np.arange(beta_scale[1]-0.01,beta_scale[1]+0.01,step=0.0005)
Lik = np.zeros(len(Coef))
for idx, coef in enumerate(Coef):
    Lik[idx] = test(coef)
plt.plot(Coef, Lik, color='black', linestyle='solid')
plt.axvline(beta_scale[1], color='r', linestyle='--');



Res = sampler.adaptive_metr(Design_mat, np.array([0.1,1]), utils.scale_gev_update_mixture_me_likelihood, 
                            priors.unif_prior, 20, 5000, random_generator, 
                            np.nan, True, False, .234, 10, .8,  10, 
                            Y, X_s, cen, cen_above, prob_below, prob_above, 
                            tau_sqd, phi, gamma, Loc, Shape, Time, xp, surv_p, den_p, 
                            thresh_X, thresh_X_above)


plt.plot(np.arange(5000),Res['trace'][0,:], linestyle='solid')
plt.hlines(0.1, 0, 5000, colors='r', linestyles='--');

plt.plot(np.arange(5000),Res['trace'][1,:], linestyle='solid')
plt.hlines(1, 0, 5000, colors='r', linestyles='--');
plt.plot(*Res['trace'])


def tmpf(x,y):
    return utils.scale_gev_update_mixture_me_likelihood(Design_mat, np.array([x,y]), Y, X_s, cen, cen_above, prob_below, prob_above, 
                            tau_sqd, phi, gamma, Loc, Shape, Time, xp, surv_p, den_p, 
                            thresh_X, thresh_X_above)
try_size = 50
x = np.linspace(0.097, 0.104, try_size)
y = np.linspace(0.996, 1.005, try_size)

func = np.empty((try_size,try_size))
for idy,yi in enumerate(y):
    for idx,xi in enumerate(x):
         func[idy,idx] = tmpf(xi,yi)

plt.contourf(x, y, func, 20, cmap='RdGy')
plt.colorbar();



# (4) shape: -0.02,0.2
def test(x):
    return utils.shape_gev_update_mixture_me_likelihood(Design_mat, np.array([x,0.2]), Y, X_s, cen, cen_above, prob_below, prob_above,
                     tau_sqd, phi, gamma, Loc, Scale, Time, xp, surv_p, den_p, 
                     thresh_X, thresh_X_above)
Coef = np.arange(-0.03,0.,step=0.0005)
Lik = np.zeros(len(Coef))
for idx, coef in enumerate(Coef):
    Lik[idx] = test(coef)
plt.plot(Coef, Lik, color='black', linestyle='solid')
plt.axvline(-0.02, color='r', linestyle='--');


def test(x):
    return utils.shape_gev_update_mixture_me_likelihood(Design_mat, np.array([-0.02,x]), Y, X_s, cen, cen_above, prob_below, prob_above,
                     tau_sqd, phi, gamma, Loc, Scale, Time, xp, surv_p, den_p, 
                     thresh_X, thresh_X_above)

Coef = np.arange(0.18,0.22,step=0.001)
Lik = np.zeros(len(Coef))
for idx, coef in enumerate(Coef):
    Lik[idx] = test(coef)
plt.plot(Coef, Lik, color='black', linestyle='solid')
plt.axvline(0.2, color='r', linestyle='--');



Res = sampler.adaptive_metr(Design_mat, np.array([-0.02,0.2]), utils.shape_gev_update_mixture_me_likelihood, 
                            priors.unif_prior, 20, 5000, random_generator, 
                            np.nan, True, False, .234, 10, .8,  10,
                            Y, X_s, cen, cen_above, prob_below, prob_above,
                            tau_sqd, phi, gamma, Loc, Scale, Time, xp, surv_p, den_p, 
                            thresh_X, thresh_X_above)


plt.plot(np.arange(5000),Res['trace'][0,:], linestyle='solid')
plt.hlines(-0.02, 0, 5000, colors='r', linestyles='--');

plt.plot(np.arange(5000),Res['trace'][1,:], linestyle='solid')
plt.hlines(0.2, 0, 5000, colors='r', linestyles='--');
plt.plot(*Res['trace'])



def tmpf(x,y):
    return utils.shape_gev_update_mixture_me_likelihood(Design_mat, np.array([x,y]), Y, X_s, cen, cen_above, prob_below, prob_above,
                            tau_sqd, phi, gamma, Loc, Scale, Time, xp, surv_p, den_p, 
                            thresh_X, thresh_X_above)
try_size = 50
x = np.linspace(-0.0215, -0.0185, try_size)
y = np.linspace(0.1985, 0.2020, try_size)

func = np.empty((try_size,try_size))
for idy,yi in enumerate(y):
    for idx,xi in enumerate(x):
         func[idy,idx] = tmpf(xi,yi)

plt.contourf(x, y, func, 20, cmap='RdGy')
plt.colorbar();









## -------------------------------------------------------
## --------------------- For theta_c ---------------------
## -------------------------------------------------------

# 1, 1.5
def test(x):
    return utils.theta_c_update_mixture_me_likelihood(Z, np.array([x,1.5]), S)

Range = np.arange(0.5,1.3,step=0.01)
Lik = np.zeros(len(Range))
for idx, r in enumerate(Range):
    Lik[idx] = test(r) 
plt.plot(Range, Lik, color='black', linestyle='solid')
plt.axvline(range, color='r', linestyle='--');


def test(x):
    return utils.theta_c_update_mixture_me_likelihood(Z, np.array([1,x]), S)

Nu = np.arange(0.9,1.8,step=0.01)
Lik = np.zeros(len(Nu))
for idx, r in enumerate(Nu):
    Lik[idx] = test(r) 
plt.plot(Nu, Lik, color='black', linestyle='solid')
plt.axvline(nu, color='r', linestyle='--');


Res = sampler.adaptive_metr(Z, np.array([1,1.5]), utils.theta_c_update_mixture_me_likelihood, 
                            priors.unif_prior, 20, 5000, random_generator,
                            np.nan, True, False, .234, 10, .8,  10,
                            S)

plt.plot(np.arange(5000),Res['trace'][0,:], linestyle='solid')
plt.hlines(1, 0, 5000, colors='r', linestyles='--');

plt.plot(np.arange(5000),Res['trace'][1,:], linestyle='solid')
plt.hlines(1.5, 0, 5000, colors='r', linestyles='--');
plt.plot(*Res['trace'])



def tmpf(x,y):
    return utils.theta_c_update_mixture_me_likelihood(Z, np.array([x,y]), S)

try_size = 50
x = np.linspace(0.85, 1.15, try_size)
y = np.linspace(1.37, 1.7, try_size)

func  = np.empty((try_size,try_size))
for idy,yi in enumerate(y):
    for idx,xi in enumerate(x):
         func[idy,idx] = tmpf(xi,yi)

plt.contourf(x, y, func, 20, cmap='RdGy')
plt.colorbar();





## -------------------------------------------------------
## ------------------------ For Z -----------------------
## -------------------------------------------------------

t_chosen = 1; idx = 20
prop_Z = np.empty(n_s)
prop_Z[:] = Z[:,t_chosen]
def test(x):
    prop_Z[idx] =x
    prop_X_s_idx = (R[t_chosen]**phi)*utils.norm_to_Pareto(x)
    return utils.marg_transform_data_mixture_me_likelihood_uni(Y[idx, t_chosen], X[idx, t_chosen], prop_X_s_idx, 
                       cen[idx, t_chosen], cen_above[idx, t_chosen], prob_below, prob_above, 
                       Loc[idx, t_chosen], Scale[idx, t_chosen], Shape[idx, t_chosen], tau_sqd, phi, gamma, 
                       xp, surv_p, den_p, thresh_X, thresh_X_above) + utils.Z_likelihood_conditional(prop_Z, V, d)
Zt = np.arange(Z[idx, t_chosen]-3,Z[idx, t_chosen]+1,step=0.01)
Lik = np.zeros(len(Zt))
for idy, x in enumerate(Zt):
    Lik[idy] = test(x) 
plt.plot(Zt, Lik, color='black', linestyle='solid')
plt.axvline(Z[idx, t_chosen], color='r', linestyle='--');


n_updates = 5000
Z_trace = np.empty((3,n_updates))

Z_new = np.empty(n_s)
Z_new[:] = Z[:, t_chosen]
K=10; k=3
r_opt = .234; c_0 = 10; c_1 = .8
accept = np.zeros(n_s)
Sigma_m = np.repeat(np.sqrt(tau_sqd),n_s)
for idx in np.arange(n_updates):
    tmp = utils.Z_update_onetime(Y[:,t_chosen], X[:,t_chosen], R[t_chosen],  Z_new, cen[:,t_chosen], cen_above[:,t_chosen], prob_below, prob_above,
                                   tau_sqd, phi, gamma, Loc[:,t_chosen], Scale[:,t_chosen], Shape[:,t_chosen], xp, surv_p, den_p,
                                   thresh_X, thresh_X_above, V, d, Sigma_m, random_generator)
    Z_trace[:,idx] = np.array([Z_new[4],Z_new[6],Z_new[7]])
    accept = accept + tmp
    
    if (idx % K) == 0:
        print('Finished ' + str(idx) + ' out of ' + str(n_updates) + ' iterations ')
        gamma2 = 1 / ((idx/K) + k)**(c_1)
        gamma1 = c_0*gamma2
        R_hat = accept/K
        Sigma_m = np.exp(np.log(Sigma_m) + gamma1*(R_hat - r_opt))
        accept[:] = 0


plt.plot(np.arange(n_updates),Z_trace[0,:],linestyle='solid')
plt.hlines(Z[4, t_chosen], 0, n_updates, colors='r', linestyles='--');

plt.plot(np.arange(n_updates),Z_trace[1,:],linestyle='solid')
plt.hlines(Z[6, t_chosen], 0, n_updates, colors='r', linestyles='--');

plt.plot(np.arange(n_updates),Z_trace[2,:],linestyle='solid')
plt.hlines(Z[7, t_chosen], 0, n_updates, colors='r', linestyles='--');



## -------------------------------------------------------
## --------------------- For Sampler ---------------------
## -------------------------------------------------------
from pickle import load
with open('nonstat_progress_0.pkl', 'rb') as f:
     Y_tmp=load(f)
     cen_tmp=load(f)
     cen_above_tmp=load(f)
     initial_values_tmp=load(f)
     sigma_m=load(f)
     prop_sigma=load(f)
     iter_tmp=load(f)
     phi_trace_tmp=load(f)
     tau_sqd_trace=load(f)
     theta_c_trace_tmp=load(f)
     beta_loc0_trace_tmp=load(f)
     beta_loc1_trace_tmp=load(f)
     beta_scale_trace_tmp=load(f)
     beta_shape_trace_tmp=load(f)
                   
     X_s_1t_trace=load(f)
     R_1t_trace=load(f)
     Y_onetime=load(f)
     X_onetime=load(f)
     X_s_onetime=load(f)
     R_onetime=load(f)





loc0=initial_values_tmp['Design_mat']@initial_values_tmp['beta_loc0']
loc1=initial_values_tmp['Design_mat']@initial_values_tmp['beta_loc1']
scale=initial_values_tmp['Design_mat']@initial_values_tmp['beta_scale']
shape=initial_values_tmp['Design_mat']@initial_values_tmp['beta_shape']

n_t = Y_tmp.shape[1]
n_s = Y_tmp.shape[0]
Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(initial_values_tmp['Time'],n_s)
Loc = Loc.reshape((n_s,n_t),order='F')
Scale = np.tile(scale, n_t)
Scale = Scale.reshape((n_s,n_t),order='F')
Shape = np.tile(shape, n_t)
Shape = Shape.reshape((n_s,n_t),order='F')



## ----------------------------------------------------------------------------------------------
## ----------------------- Theoretical distribution function verification -----------------------
## ----------------------------------------------------------------------------------------------
# Levy distribution
samples = utils.rlevy(100000)
x_vals = np.linspace(0.01,20,1000)
d_vals = utils.dlevy(x_vals)
censor = 2-2*norm.cdf(np.sqrt(1/(100)))
d_vals = d_vals/censor
import seaborn as sns
import matplotlib.pyplot as plt
plt.clf()
sns.distplot(samples[samples<100], hist=True, kde=True, bins=1000)
plt.plot(x_vals,d_vals)
plt.xlim(0,20)

# R = sum(weights*S)
gamma=0.8
weights = R_weights[1,:]
n_knots = weights.shape[0]
R_vec = np.empty(100000)
for idx in np.arange(100000):
    S = utils.rlevy(n_knots,m=0,s=gamma)
    R_vec[idx] = np.sum(weights*S)

gamma_new=np.sum(np.sqrt(weights[np.nonzero(weights)]))**2*gamma
x_vals = np.linspace(0.01,40,1000)
d_vals = utils.dlevy(x_vals, m=0, s=gamma_new)
censor = 2-2*norm.cdf(np.sqrt(gamma_new/(200)))
d_vals = d_vals/censor
import seaborn as sns
plt.clf()
sns.distplot(R_vec[R_vec<200], hist=True, kde=True, bins=2000)
plt.plot(x_vals,d_vals)
plt.xlim(0,40)

# R^phi
phi=0.35
R_phi = R_vec**phi
for idx in np.arange(x_vals.shape[0]):
    d_vals[idx] = utils.dR_power_phi(x_vals[idx], phi, m=0, s=gamma_new, log=False)
censor = 2-2*norm.cdf(np.sqrt(gamma_new/(200**(1/phi))))
d_vals = d_vals/censor
import seaborn as sns
plt.clf()
sns.distplot(R_phi[R_phi<200], hist=True, kde=True, bins=2000)
plt.plot(x_vals,d_vals)
plt.xlim(0,40)

# R^phi*W
X = R_phi*(1/(1-norm.cdf(norm.rvs(size=100000))))
x_vals = np.linspace(0.01,40,1000)
d_vals = utils.dRW(x_vals, phi, gamma_new)
censor = utils.pRW(200, phi, gamma_new)
d_vals = d_vals/censor
import seaborn as sns
plt.clf()
sns.distplot(X[X<200], hist=True, kde=True, bins=2000)
plt.plot(x_vals,d_vals)
plt.xlim(0,40)

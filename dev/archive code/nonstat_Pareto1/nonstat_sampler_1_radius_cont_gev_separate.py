#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 08:39:47 2022

@author: LikunZhang
"""




###################################################################################
## Main sampler cont'd

 
      

if __name__ == "__main__":
   import nonstat_Pareto1.model_sim as utils
   # import nonstat_Pareto1.generic_samplers as sampler
   # import nonstat_Pareto1.priors as priors
   import nonstat_Pareto1.ns_cov as cov
   import os
   import numpy as np
   # import time
   import matplotlib.pyplot as plt
   from matplotlib.backends.backend_pdf import PdfPages
   from pickle import load
   from pickle import dump
   # from scipy.linalg import lapack
   from scipy.linalg import cholesky
   from scipy.spatial import distance
   
   # Check whether the 'mpi4py' is installed
   test_mpi = os.system("python -c 'from mpi4py import *' &> /dev/null")
   if test_mpi != 0:
      import sys
      sys.exit("mpi4py import is failing, aborting...")
   
   # get rank and size
   from mpi4py import MPI
  
   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
   size = comm.Get_size()
   thinning = 10; echo_interval = 50; n_updates = 60001
  
   import warnings
   warnings.filterwarnings("ignore", category=RuntimeWarning) 
     
   # Filename for storing the intermediate results
   filename='./nonstat_progress_'+str(rank)+'.pkl'
   
   n_updates_thinned = np.int(np.ceil(n_updates/thinning))
   # Load data input
   if rank==0:
       with open(filename, 'rb') as f:
           Y = load(f)
           initial_values = load(f)
           sigma_m = load(f)
           prop_Sigma = load(f)
           iter_current = load(f)
           Rt_knots_trace = load(f)
           # radius_knots_trace = load(f)
           range_knots_trace = load(f)
           phi_knots_radius_trace = load(f)
           beta_gev_params_trace = load(f)
           f.close()
           
           if(Rt_knots_trace.shape[0]<n_updates_thinned):
               add_length = n_updates_thinned - Rt_knots_trace.shape[0]
               Rt_knots_trace = np.pad(Rt_knots_trace, ((0, add_length),(0,0)), 'constant', constant_values=np.nan)
               # radius_knots_trace = np.pad(radius_knots_trace, ((0, add_length),(0,0)), 'constant', constant_values=np.nan)
               range_knots_trace = np.pad(range_knots_trace, ((0, add_length),(0,0)),'constant', constant_values=np.nan)
               phi_knots_radius_trace = np.pad(phi_knots_radius_trace, ((0, add_length),(0,0)),'constant', constant_values=np.nan)
               beta_gev_params_trace = np.pad(beta_gev_params_trace, ((0, add_length),(0,0)),'constant', constant_values=np.nan)
   else:
       with open(filename, 'rb') as f:
           Y = load(f)
           initial_values = load(f)
           sigma_m = load(f)
           prop_Sigma = load(f)
           iter_current = load(f)
           Rt_knots_trace = load(f)
           f.close()
           
           if(Rt_knots_trace.shape[0]<n_updates_thinned):
               add_length = n_updates_thinned - Rt_knots_trace.shape[0]
               Rt_knots_trace = np.pad(Rt_knots_trace, ((0, add_length),(0,0)),'constant', constant_values=np.nan)

   
   # Generate multiple independent random streams
   random_generator = np.random.RandomState()
  
   # Constants to control adaptation of the Metropolis sampler
   c_0 = 1
   c_1 = 0.8
   offset = 3  # the iteration offset
   r_opt_1d = .41
   r_opt_2d = .35
   eps = 1e-6 # a small number
  
   # Hyper parameters for the prior of the mixing distribution parameters and 
   hyper_params_phi = np.array([0.5,0.7])
   hyper_params_tau_sqd = np.array([0.1,0.1])
   hyper_params_theta_c = np.array([0, 20])
   hyper_params_theta_gev = 25
   # hyper_params_range = np.array([0.5,1.5]) # in case where roughness is not updated
    
   # Load initial values
   Knots =  initial_values['Knots']
   phi_range_weights = initial_values['phi_range_weights']
   R_weights =  initial_values['R_weights']
   Stations =  initial_values['Stations']
   phi_vec = initial_values['phi_vec']
   phi_at_knots = initial_values['phi_at_knots']
   phi_at_knots_and_radius = initial_values['phi_at_knots_and_radius']
   radius_from_knots = initial_values['radius_from_knots']
   gamma = initial_values['gamma']
   gamma_vec = initial_values['gamma_vec']
   range_vec = initial_values['range_vec']
   range_at_knots = initial_values['range_at_knots']
   nu = initial_values['nu']
   X = initial_values['X']; Xt = X[:,rank]
   Rt_at_knots = initial_values['Rt_at_knots']
   R_s = initial_values['R_s']; Rt_s = R_s[:,rank]
   Design_mat = initial_values['Design_mat']
   beta_gev_params = initial_values['beta_gev_params']
   Time = initial_values['Time']
   
   
   n_beta_gev_params = beta_gev_params.shape[0]
   
   
   # Bookkeeping
   n_s = Y.shape[0]
   n_t = Y.shape[1]
   n_phi_range_knots = len(phi_at_knots)
   n_Rt_knots = len(Rt_at_knots)
   if n_t != size:
      import sys
      sys.exit("Make sure the number of cpus (N) = number of time replicates (n_t), i.e.\n     srun -N python nonstat_sampler.py")
   # n_covariates = len(beta_loc0)
   
   
   wh_to_plot_Xs = n_s*np.array([0.25,0.5,0.75])
   wh_to_plot_Xs = wh_to_plot_Xs.astype(int)
   
   # Distance from knots
   Distance_from_stations_to_knots = np.empty((n_s, n_Rt_knots))
   for ind in np.arange(n_s):
       Distance_from_stations_to_knots[ind,:] = distance.cdist(Stations[ind,:].reshape((-1,2)),Knots)
   
   
   # Eigendecomposition of the correlation matrix
   one_vec = np.ones(n_s)
   Cor = cov.ns_cov(range_vec, one_vec, Stations, kappa = nu, cov_model = "matern")
   # eig_Cor = np.linalg.eigh(Cor) #For symmetric matrices
   # V = eig_Cor[1]
   # d = eig_Cor[0]
   cholesky_U = cholesky(Cor,lower=False)

   # Marginal GEV parameters: per location x time
   beta_gev_params = comm.bcast(beta_gev_params, root=0)
   loc0 = Design_mat @np.array([beta_gev_params[0],0])
   loc0 = loc0.astype('float64')
   loc1 = Design_mat @np.array([0,0])
   loc1 = loc1.astype('float64')
   Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(Time,n_s)
   Loc = Loc.reshape((n_s,n_t),order='F')

   scale = Design_mat @np.array([beta_gev_params[1],0])
   scale = scale.astype('float64')
   Scale = np.tile(scale, n_t)
   Scale = Scale.reshape((n_s,n_t),order='F')
   
   shape = Design_mat @np.array([beta_gev_params[2],0])
   shape = shape.astype('float64')
   Shape = np.tile(shape, n_t)
   Shape = Shape.reshape((n_s,n_t),order='F')
   
   Current_Rt_prior = np.sum(utils.dlevy(Rt_at_knots, m=0, s=gamma, log=True))
   Current_lik = utils.marg_transform_data_mixture_likelihood_1t(Y[:,rank], Xt, Loc[:,rank], Scale[:,rank], 
                                             Shape[:,rank], phi_vec, gamma_vec, Rt_s, 
                                             cholesky_U)
   Current_Lik_recv = comm.gather(Current_lik,root=0)
   
   
   # Initialize trace objects   
   R_knots_within_thinning = np.empty((n_Rt_knots,thinning)); R_knots_within_thinning[:] = np.nan
   if rank == 0:
       # radius_knots_within_thinning = np.empty((n_Rt_knots,thinning)); radius_knots_within_thinning[:] = np.nan
       range_knots_within_thinning = np.empty((n_phi_range_knots,thinning)); range_knots_within_thinning[:] = np.nan
       phi_knots_radius_within_thinning = np.empty((n_phi_range_knots+1,thinning)); phi_knots_radius_within_thinning[:] = np.nan
       beta_gev_params_within_thinning = np.empty((n_beta_gev_params,thinning)); beta_gev_params_within_thinning[:] = np.nan
   
   
   R_accept = 0
   radius_accept = 0
   range_accept = 0 
   phi_radius_accept = 0
   beta_gev_accept = 0
   
   phi_at_knots_and_radius = phi_at_knots_and_radius[:(n_phi_range_knots+1)]
   if(len(prop_Sigma['gev_params'])==0 or prop_Sigma['phi_radius'].shape[0]!=(n_phi_range_knots+1)):
       prop_Sigma['gev_params'] = prop_Sigma['phi_radius'][(n_phi_range_knots+1):, (n_phi_range_knots+1):]
       prop_Sigma['phi_radius'] = prop_Sigma['phi_radius'][:(n_phi_range_knots+1), :(n_phi_range_knots+1)]
       sigma_m['gev_params'] = sigma_m['phi_radius']
   
   # -----------------------------------------------------------------------------------
   # -----------------------------------------------------------------------------------
   # --------------------------- Start Metropolis Updates ------------------------------
   # -----------------------------------------------------------------------------------
   # -----------------------------------------------------------------------------------
   for iter in np.arange(iter_current+1,n_updates):
       index_within = (iter-1)%thinning
       # start_time = time.time()
       
       # --------- Update Rt -----------
       #Propose new values
       Rt_s_star = np.empty(Rt_s.shape)
       
       #Propose Rt under every worker
       tmp_upper = cholesky(prop_Sigma['Rt'],lower=False)
       tmp_params_star = sigma_m['Rt']*random_generator.standard_normal(n_Rt_knots)
       Rt_at_knots_star = Rt_at_knots + np.matmul(tmp_upper.T , tmp_params_star)
       Rt_s_star[:] = R_weights @ Rt_at_knots_star 
       
       # Evaluate likelihood at new values
       # Not broadcasting but evaluating at each node 
       if np.any(Rt_at_knots_star<0):
           Star_Rt_prior = -np.inf
           Star_lik = -np.inf
       else:    
           Star_Rt_prior = np.sum(utils.dlevy(Rt_at_knots_star, m=0, s=gamma, log=True))
           Star_lik = utils.marg_transform_data_mixture_likelihood_1t(Y[:,rank], Xt, Loc[:,rank], Scale[:,rank], 
                                                 Shape[:,rank], phi_vec, gamma_vec, Rt_s_star, 
                                                 cholesky_U)
       
       # Determine update or not
       # Not gathering but evaluating at each node 
       r = np.exp(Star_Rt_prior + Star_lik - Current_Rt_prior - Current_lik)
       if ~np.isfinite(r):
           r = 0
       if random_generator.uniform(0,1,1)<r:
           Rt_at_knots[:] = Rt_at_knots_star
           Rt_s[:] = Rt_s_star 
           Current_lik = Star_lik
           Current_Rt_prior = Star_Rt_prior
           R_accept = R_accept + 1  
       
       # Gather anyways
       R_knots_within_thinning[:, index_within] = Rt_at_knots
       Current_Lik_recv = comm.gather(Current_lik,root=0)
       # R_s_recv = comm.gather(Rt_s,root=0)
       # if rank ==0: 
       #     R_s[:] = np.vstack(R_s_recv).T
       
      
       accept = 0
       # --------- Update range_vec -----------
       #Propose new values
       range_vec_star = np.empty(n_s)
       # V_star = np.empty(V.shape)
       # d_star = np.empty(d.shape)
       cholesky_U_star = np.empty(cholesky_U.shape)
       
       if rank==0:
           tmp_upper = cholesky(prop_Sigma['range'],lower=False)
           tmp_params_star = sigma_m['range']*random_generator.standard_normal(n_phi_range_knots)
           range_at_knots_proposal = range_at_knots + np.matmul(tmp_upper.T , tmp_params_star)
           range_vec_star[:] = phi_range_weights @ range_at_knots_proposal
       range_vec_star = comm.bcast(range_vec_star,root=0)
       
       # Evaluate likelihood at new values
       if np.all(range_vec_star>0):
           # Not broadcasting but generating at each node
           Cor_star = cov.ns_cov(range_vec_star, one_vec, Stations, kappa = nu, cov_model = "matern")
           # eig_Cor = np.linalg.eigh(Cor_star) #For symmetric matrices
           # V_star[:] = eig_Cor[1]
           # d_star[:] = eig_Cor[0]   
           cholesky_U_star[:] = cholesky(Cor_star,lower=False)
           Star_lik = utils.marg_transform_data_mixture_likelihood_1t(Y[:,rank], Xt, Loc[:,rank], Scale[:,rank], 
                                                 Shape[:,rank], phi_vec, gamma_vec, Rt_s, 
                                                 cholesky_U_star)
       else:
           Star_lik = -np.inf
       
       Star_Lik_recv = comm.gather(Star_lik,root=0)
       
       # Determine update or not
       if rank==0:
           log_num = np.sum(Star_Lik_recv)
           log_denom = np.sum(Current_Lik_recv)
           r = np.exp(log_num - log_denom)
           if ~np.isfinite(r):
               r = 0
           if random_generator.uniform(0,1,1)<r:
               range_at_knots[:] = range_at_knots_proposal
               range_vec[:] = range_vec_star 
               Current_Lik_recv[:] = Star_Lik_recv
               accept = 1
               range_accept = range_accept + 1
           range_knots_within_thinning[:, index_within] = range_at_knots
       
       # Broadcast according to accept
       accept = comm.bcast(accept,root=0)
       if accept==1:
           # V[:] = V_star
           # d[:] = d_star
           cholesky_U[:] = cholesky_U_star
           Current_lik = Star_lik
           
           
       accept = 0
       # --------- Update radius + phi_at_knots -----------
       #Propose new values
       phi_vec_star = np.empty(n_s)
       radius_proposal = np.empty(1)
       R_weights_star = np.empty((n_s,Knots.shape[0]))
       gamma_vec_star = np.repeat(np.nan, n_s)
       Xt_star = np.empty(n_s)
       if rank==0:
           tmp_upper = cholesky(prop_Sigma['phi_radius'],lower=False)
           tmp_params_star = sigma_m['phi_radius']*random_generator.standard_normal(n_phi_range_knots+1)
           phi_at_knots_and_radius_proposal = phi_at_knots_and_radius + np.matmul(tmp_upper.T , tmp_params_star)
           phi_vec_star[:] = phi_range_weights @ phi_at_knots_and_radius_proposal[:n_phi_range_knots]
           radius_proposal = phi_at_knots_and_radius_proposal[-1]
       phi_vec_star = comm.bcast(phi_vec_star,root=0)
       radius_proposal = comm.bcast(radius_proposal,root=0)

       # Not broadcasting but generating at each node
       for idy in np.arange(n_s):
           tmp_weights = utils.wendland_weights_fun(Distance_from_stations_to_knots[idy,:],
                                                           np.repeat(radius_proposal, n_Rt_knots))
           R_weights_star[idy,:] = tmp_weights
           gamma_vec_star[idy] = np.sum(np.sqrt(tmp_weights[np.nonzero(tmp_weights)]*gamma))**2 #only save once


       # Evaluate likelihood at new values
       if np.any(phi_vec_star>=1) or np.any(phi_vec_star<=0) or radius_proposal<0 or radius_proposal>10: #U(0,1) priors
           Star_lik = -np.inf
       else:
           Rt_s_star[:] = R_weights_star @ Rt_at_knots
           Xt_star[:] = utils.gev_2_RW(Y[:,rank], phi_vec_star, gamma_vec_star,
                                      Loc[:,rank], Scale[:,rank], Shape[:,rank])
           Star_lik = utils.marg_transform_data_mixture_likelihood_1t(Y[:,rank], Xt_star, Loc[:,rank], Scale[:,rank],
                                              Shape[:,rank], phi_vec_star, gamma_vec_star, Rt_s_star,
                                              cholesky_U)
           Star_lik_d = utils.marg_transform_data_mixture_likelihood_1t_detail(Y[:,rank], Xt_star, Loc[:,rank], Scale[:,rank],
                                              Shape[:,rank], phi_vec_star, gamma_vec_star, Rt_s_star,
                                              cholesky_U)
           Current_lik_d = utils.marg_transform_data_mixture_likelihood_1t_detail(Y[:,rank], Xt, Loc[:,rank], Scale[:,rank],
                                              Shape[:,rank], phi_vec, gamma_vec, Rt_s,
                                              cholesky_U)
       Star_Lik_recv = comm.gather(Star_lik,root=0)

       # Determine update or not
       if rank==0:
           log_num = np.sum(Star_Lik_recv)
           log_denom = np.sum(Current_Lik_recv)
           r = np.exp(log_num - log_denom)
           if ~np.isfinite(r):
               r = 0
           if random_generator.uniform(0,1,1)<r:
               phi_at_knots_and_radius[:] = phi_at_knots_and_radius_proposal
               phi_vec[:] = phi_vec_star
               Current_Lik_recv[:] = Star_Lik_recv
               phi_radius_accept = phi_radius_accept + 1
               accept = 1
           phi_knots_radius_within_thinning[:, index_within] = phi_at_knots_and_radius

       # Broadcast anyways
       accept = comm.bcast(accept,root=0)
       if accept==1:
           phi_vec[:] = phi_vec_star
           Xt[:] = Xt_star
           R_weights[:] = R_weights_star
           gamma_vec[:] = gamma_vec_star
           Rt_s[:] = Rt_s_star
           Current_lik = Star_lik


       accept = 0
       # --------- Update GEV params -----------
       #Propose new values
       beta_gev_params_star = np.empty(beta_gev_params.shape)
       if rank==0:
           tmp_upper = cholesky(prop_Sigma['gev_params'],lower=False)
           tmp_params_star = sigma_m['gev_params']*random_generator.standard_normal(n_beta_gev_params)
           beta_gev_params_star[:] = beta_gev_params + np.matmul(tmp_upper.T , tmp_params_star)

       beta_gev_params_star = comm.bcast(beta_gev_params_star,root=0)

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

       Xt_star[:] = utils.gev_2_RW(Y[:,rank], phi_vec, gamma_vec, 
                                      Loc_star[:,rank], Scale_star[:,rank], Shape_star[:,rank])

       Star_lik = utils.marg_transform_data_mixture_likelihood_1t(Y[:,rank], Xt_star, Loc_star[:,rank], Scale_star[:,rank], 
                                              Shape_star[:,rank], phi_vec, gamma_vec, Rt_s, 
                                              cholesky_U)
       Star_Lik_recv = comm.gather(Star_lik,root=0)

       # Determine update or not
       if rank==0:
           log_num = np.sum(Star_Lik_recv)
           log_denom = np.sum(Current_Lik_recv)
           r = np.exp(log_num - log_denom)
           if ~np.isfinite(r):
               r = 0
           if random_generator.uniform(0,1,1)<r:
               beta_gev_params[:] = beta_gev_params_star
               Current_Lik_recv[:] = Star_Lik_recv
               accept = 1  
               beta_gev_accept = beta_gev_accept + 1
           beta_gev_params_within_thinning[:, index_within] = beta_gev_params

       # Broadcast anyways
       accept = comm.bcast(accept,root=0)
       if accept==1:
           loc0[:] = loc0_star
           scale[:] = scale_star
           shape[:] = shape_star
           Loc[:] = Loc_star
           Scale[:] = Scale_star
           Shape[:] = Shape_star
           Xt[:] = Xt_star
           Current_lik = Star_lik
           beta_gev_params[:] = beta_gev_params_star
       # time_spent = time.time()-start_time
       # print(rank, time_spent)  
     
       
      
   
       # ----------------------------------------------------------------------------------------
       # --------------------------- Summarize every 'thinning' steps ---------------------------
       # ----------------------------------------------------------------------------------------
       if (iter % thinning) == 0:
           index = np.int(iter/thinning)
           
           # Fill in trace objects
           Rt_knots_trace[index,:] = Rt_at_knots
           if rank == 0:
               # radius_knots_trace[index,:] = radius_from_knots
               range_knots_trace[index,:] = range_at_knots
               phi_knots_radius_trace[index,:] = phi_at_knots_and_radius[:(n_phi_range_knots+1)]
               beta_gev_params_trace[index,:] = beta_gev_params
           
           # Adapt via Shaby and Wells (2010)
           gamma1 = 1 / (index + offset)**(c_1)
           gamma2 = c_0*gamma1
           
           sigma_m['Rt'] = np.exp(np.log(sigma_m['Rt']) + gamma2*(R_accept/thinning - r_opt_2d))
           R_accept = 0
           prop_Sigma['Rt'] = prop_Sigma['Rt'] + gamma1*(np.cov(R_knots_within_thinning) - prop_Sigma['Rt'])
           check_chol_cont = True
           while check_chol_cont:
               try:
                   # Initialize prop_C
                   np.linalg.cholesky(prop_Sigma['Rt'])
                   check_chol_cont = False
               except  np.linalg.LinAlgError:
                   prop_Sigma['Rt'] = prop_Sigma['Rt'] + eps*np.eye(n_Rt_knots)
                   print("Oops. Proposal covariance matrix is now:\n")
                   print(prop_Sigma['Rt'])
                   
                   
                   
           if rank == 0:
               sigma_m['range'] = np.exp(np.log(sigma_m['range']) + gamma2*(range_accept/thinning - r_opt_2d))
               range_accept = 0
               prop_Sigma['range'] = prop_Sigma['range'] + gamma1*(np.cov(range_knots_within_thinning) - prop_Sigma['range'])
               check_chol_cont = True
               while check_chol_cont:
                   try:
                       # Initialize prop_C
                       np.linalg.cholesky(prop_Sigma['range'])
                       check_chol_cont = False
                   except  np.linalg.LinAlgError:
                       prop_Sigma['range'] = prop_Sigma['range'] + eps*np.eye(n_phi_range_knots)
                       print("Oops. Proposal covariance matrix is now:\n")
                       print(prop_Sigma['range'])
               
               sigma_m['phi_radius'] = np.exp(np.log(sigma_m['phi_radius']) + gamma2*(phi_radius_accept/thinning - r_opt_2d))
               phi_radius_accept = 0
               prop_Sigma['phi_radius'] = prop_Sigma['phi_radius'] + gamma1*(np.cov(phi_knots_radius_within_thinning) - prop_Sigma['phi_radius'])
               check_chol_cont = True
               while check_chol_cont:
                   try:
                       # Initialize prop_C
                       np.linalg.cholesky(prop_Sigma['phi_radius'])
                       check_chol_cont = False
                   except  np.linalg.LinAlgError:
                       prop_Sigma['phi_radius'] = prop_Sigma['phi_radius'] + eps*np.eye(n_phi_range_knots+1)
                       print("Oops. Proposal covariance matrix is now:\n")
                       print(prop_Sigma['phi_radius'])
                       
               sigma_m['gev_params'] = np.exp(np.log(sigma_m['gev_params']) + gamma2*(beta_gev_accept/thinning - r_opt_2d))
               beta_gev_accept = 0
               prop_Sigma['gev_params'] = prop_Sigma['gev_params'] + gamma1*(np.cov(beta_gev_params_within_thinning) - prop_Sigma['gev_params'])
               check_chol_cont = True
               while check_chol_cont:
                   try:
                       # Initialize prop_C
                       np.linalg.cholesky(prop_Sigma['gev_params'])
                       check_chol_cont = False
                   except  np.linalg.LinAlgError:
                       prop_Sigma['gev_params'] = prop_Sigma['gev_params'] + eps*np.eye(n_beta_gev_params)
                       print("Oops. Proposal covariance matrix is now:\n")
                       print(prop_Sigma['gev_params'])
               
                    
               
        
        
        
       # ----------------------------------------------------------------------------------------
       # -------------------------- Echo & save every 'thinning' steps --------------------------
       # ----------------------------------------------------------------------------------------
       if (iter / thinning) % echo_interval == 0:    
           # Temporarily not saving because they can recovered from the saved results
           # R_s_recv = comm.gather(Rt_s,root=0)
           # if rank ==0: 
           #     R_s[:] = np.vstack(R_s_recv).T
           
           # X_recv = comm.gather(Xt,root=0)
           # if rank ==0: 
           #     X[:] = np.vstack(X_recv).T     
             
           if rank == 0:
               print('Done with '+str(index)+" updates while thinned by "+str(thinning)+" steps,\n")
               
               # Save the intermediate results to filename
               initial_values = {'Knots':Knots,
                                 'phi_range_weights': phi_range_weights,
                                 'R_weights':R_weights,
                                 'Stations':Stations,
                                 'phi_vec':phi_vec,
                                 'phi_at_knots':phi_at_knots,
                                 'phi_at_knots_and_radius': phi_at_knots_and_radius,
                                 'radius_from_knots':radius_from_knots,
                                 'gamma':gamma,
                                 'gamma_vec':gamma_vec,
                                 'range_vec':range_vec,
                                 'range_at_knots':range_at_knots,
                                 'nu':nu,
                                 'X':X,
                                 'Rt_at_knots':Rt_at_knots,
                                 'R_s':R_s,
                                 'Design_mat':Design_mat,
                                 'beta_gev_params':beta_gev_params,
                                 'Time':Time
                                }
               with open(filename, 'wb') as f:
                   dump(Y, f)
                   dump(initial_values, f)
                   dump(sigma_m, f)
                   dump(prop_Sigma, f)
                   dump(iter, f)
                   dump(Rt_knots_trace, f)
                   # dump(radius_knots_trace, f)
                   dump(range_knots_trace, f)
                   dump(phi_knots_radius_trace, f)
                   dump(beta_gev_params_trace, f)
                   f.close()
                   
               # Echo trace plots
               pdf_pages = PdfPages('./progress.pdf')
               grid_size = (4,2)
               #-page-1
               fig = plt.figure(figsize = (8.75, 11.75))
               plt.subplot2grid(grid_size, (0,0)) 
               plt.plot(Rt_knots_trace[:,0], color='gray', linestyle='solid')
               plt.ylabel(r'$R_{knots,rank 0}$[0]')
               plt.subplot2grid(grid_size, (0,1)) 
               plt.plot(Rt_knots_trace[:,1], color='gray', linestyle='solid')
               plt.ylabel(r'$R_{knots,rank 0}$[1]')
               plt.subplot2grid(grid_size, (1,0)) 
               plt.plot(Rt_knots_trace[:,2], color='gray', linestyle='solid')
               plt.ylabel(r'$R_{knots,rank 0}$[2]')
               plt.subplot2grid(grid_size, (1,1))
               plt.plot(Rt_knots_trace[:,3], color='gray', linestyle='solid')
               plt.ylabel(r'$R_{knots,rank 0}$[3]')
               plt.subplot2grid(grid_size, (2,0)) 
               plt.plot(Rt_knots_trace[:,4], color='gray', linestyle='solid')
               plt.ylabel(r'$R_{knots,rank 0}$[4]')
               plt.subplot2grid(grid_size, (2,1)) 
               plt.plot(Rt_knots_trace[:,5], color='gray', linestyle='solid')
               plt.ylabel(r'$R_{knots,rank 0}$[5]')
               plt.subplot2grid(grid_size, (3,0)) 
               plt.plot(Rt_knots_trace[:,6], color='gray', linestyle='solid')
               plt.ylabel(r'$R_{knots,rank 0}$[6]')
               plt.subplot2grid(grid_size, (3,1)) 
               plt.plot(Rt_knots_trace[:,7], color='gray', linestyle='solid')
               plt.ylabel(r'$R_{knots,rank 0}$[7]')
               plt.tight_layout()
               pdf_pages.savefig(fig)
               plt.close()
                   
               #-page-2
               fig = plt.figure(figsize = (8.75, 11.75))
               plt.subplot2grid(grid_size, (0,0)) 
               plt.plot(range_knots_trace[:,0], color='gray', linestyle='solid')
               plt.ylabel(r'$range_{knots}$[0]')
               plt.subplot2grid(grid_size, (0,1))
               plt.plot(range_knots_trace[:,1], color='gray', linestyle='solid')
               plt.ylabel(r'$range_{knots}$[1]')
               plt.subplot2grid(grid_size, (1,0))  
               plt.plot(range_knots_trace[:,2], color='gray', linestyle='solid')
               plt.ylabel(r'$range_{knots}$[2]')
               plt.subplot2grid(grid_size, (1,1)) 
               plt.plot(range_knots_trace[:,3], color='gray', linestyle='solid')
               plt.ylabel(r'$range_{knots}$[3]')
               plt.subplot2grid(grid_size, (2,0))  
               plt.plot(range_knots_trace[:,4], color='gray', linestyle='solid')
               plt.ylabel(r'$range_{knots}$[4]')
               plt.subplot2grid(grid_size, (2,1))  
               plt.plot(range_knots_trace[:,5], color='gray', linestyle='solid')
               plt.ylabel(r'$range_{knots}$[5]')
               plt.subplot2grid(grid_size, (3,0))  
               plt.plot(range_knots_trace[:,6], color='gray', linestyle='solid')
               plt.ylabel(r'$range_{knots}$[6]')
               plt.subplot2grid(grid_size, (3,1))  
               plt.plot(range_knots_trace[:,7], color='gray', linestyle='solid')
               plt.ylabel(r'$range_{knots}$[7]')
               plt.tight_layout()
               pdf_pages.savefig(fig)
               plt.close()
               

               #-page-3
               fig = plt.figure(figsize = (8.75, 11.75))
               plt.subplot2grid(grid_size, (0,0)) 
               plt.plot(phi_knots_radius_trace[:,0], color='gray', linestyle='solid')
               plt.ylabel(r'$phi_{knots}$[0]')
               plt.subplot2grid(grid_size, (0,1))
               plt.plot(phi_knots_radius_trace[:,1], color='gray', linestyle='solid')
               plt.ylabel(r'$phi_{knots}$[1]')
               plt.subplot2grid(grid_size, (1,0))  
               plt.plot(phi_knots_radius_trace[:,2], color='gray', linestyle='solid')
               plt.ylabel(r'$phi_{knots}$[2]')
               plt.subplot2grid(grid_size, (1,1)) 
               plt.plot(phi_knots_radius_trace[:,3], color='gray', linestyle='solid')
               plt.ylabel(r'$phi_{knots}$[3]')
               plt.subplot2grid(grid_size, (2,0))  
               plt.plot(phi_knots_radius_trace[:,4], color='gray', linestyle='solid')
               plt.ylabel(r'$phi_{knots}$[4]')
               plt.subplot2grid(grid_size, (2,1))  
               plt.plot(phi_knots_radius_trace[:,5], color='gray', linestyle='solid')
               plt.ylabel(r'$phi_{knots}$[5]')
               plt.subplot2grid(grid_size, (3,0))  
               plt.plot(phi_knots_radius_trace[:,6], color='gray', linestyle='solid')
               plt.ylabel(r'$phi_{knots}$[6]')
               plt.subplot2grid(grid_size, (3,1))  
               plt.plot(phi_knots_radius_trace[:,7], color='gray', linestyle='solid')
               plt.ylabel(r'$phi_{knots}$[7]')
               plt.tight_layout()
               pdf_pages.savefig(fig)
               plt.close()
               
               #-page-4
               fig = plt.figure(figsize = (8.75, 11.75))
               plt.subplot2grid(grid_size, (0,0)) 
               plt.plot(phi_knots_radius_trace[:,8], color='gray', linestyle='solid')
               plt.ylabel(r'$phi_{knots}$[8]')
               plt.subplot2grid(grid_size, (0,1))
               plt.plot(phi_knots_radius_trace[:,-1], color='gray', linestyle='solid')
               plt.ylabel(r'$r$')
               plt.subplot2grid(grid_size, (1,0)) # loc0
               plt.plot(beta_gev_params_trace[:,0], color='gray', linestyle='solid')
               plt.ylabel(r'$\mu$')
               plt.subplot2grid(grid_size, (1,1)) # loc1
               plt.plot(beta_gev_params_trace[:,1], color='gray', linestyle='solid')
               plt.ylabel(r'$\sigma$')
               plt.subplot2grid(grid_size, (2,0)) # scale
               plt.plot(beta_gev_params_trace[:,2], color='gray', linestyle='solid')
               plt.ylabel(r'$\xi$')
               plt.tight_layout()
               pdf_pages.savefig(fig)
               plt.close()
               pdf_pages.close()
           else:
               initial_values = {'Knots':Knots,
                                 'phi_range_weights': phi_range_weights,
                                 'R_weights':R_weights,
                                 'Stations':Stations,
                                 'phi_vec':phi_vec,
                                 'phi_at_knots':phi_at_knots, # not broadcasted
                                 'phi_at_knots_and_radius': phi_at_knots_and_radius,
                                 'radius_from_knots':radius_from_knots, # not broadcasted
                                 'gamma':gamma,
                                 'gamma_vec':gamma_vec,
                                 'range_vec':range_vec, # not broadcasted
                                 'range_at_knots':range_at_knots, # not broadcaste
                                 'nu':nu,
                                 'X':X,
                                 'Rt_at_knots':Rt_at_knots,
                                 'R_s':R_s,
                                 'Design_mat':Design_mat,
                                 'beta_gev_params':beta_gev_params, # not broadcasted
                                 'Time':Time
                                }
               with open(filename, 'wb') as f:
                   dump(Y, f)
                   dump(initial_values, f)
                   dump(sigma_m, f)
                   dump(prop_Sigma, f)
                   dump(iter, f)
                   dump(Rt_knots_trace, f)
                   f.close()

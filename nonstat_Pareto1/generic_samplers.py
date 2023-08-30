from __future__ import print_function
# from scipy.stats import norm
# from scipy.stats import uniform
import numpy as np

## --------------------------------------------------------------------- ##
#  A generic Metropolis sampler.  You have to supply the log likelihood   #
#  function, which need not really be a likelihood function at all.       #
#
#  Translated from Shaby's R code.
#
# z ............................... 'data' term in the likelihood functions
# starting_theta ........................................... initial values
# likelihood_fn ..................................... likelihood in-between
# prior_fn ....................................... prior function for theta
# hyper_params ........................................ for prior functions
# n_updates .................................. number of Metropolis updates
# prop_Sigma ................................... proposal covariance matrix
# sigma_m ..................................... scaling factor for proposal
# verbose ..................................... print out the error message
#                                                                         #

def static_metr(z, starting_theta, likelihood_fn,
                        prior_fn, hyper_params, n_updates, 
                        random_generator, 
                        prop_Sigma = np.nan, sigma_m=np.nan, 
                        verbose=False, *argv):

    if type(starting_theta).__module__!='numpy' or isinstance(starting_theta, np.float64):
      starting_theta = np.array(starting_theta)
    p = starting_theta.size
    invalid = False
    
    # If the supplied proposal covariance matrix is either not given or invalid,
    # just use the identity.
    if np.any(np.isnan(prop_Sigma)) or prop_Sigma.size != p**2:
        prop_Sigma = np.eye(p)
        prop_C = np.eye(p)
        invalid = True
    else:
        try:
            # Initialize prop_C
            prop_C = np.linalg.cholesky(prop_Sigma)
        except  np.linalg.LinAlgError:
            prop_Sigma = np.eye(p)
            prop_C = np.eye(p)
            invalid = True
    if verbose and invalid:
        print("Invalid or missing proposal covariance matrix.  Using identity.\n")
    
    # Initialize sigma_m to the rule of thumb if it's not supplied
    if np.isnan(sigma_m):
        sigma_m = (2.4/p)**2
 
    # Set up and initialize trace objects
    trace = np.zeros((p, n_updates))
    jump_trace = np.zeros(n_updates)

    trace[:, 0] = starting_theta
    
    # Initialize Metropolis
    theta = starting_theta
    likelihood = likelihood_fn(z, theta, *argv)
    prior = prior_fn(theta, hyper_params)
  
  
    #########################################################
    # Begin main loop
    for i in np.arange(1,n_updates):
        theta_star = theta + sigma_m * random_generator.standard_normal(p) @ prop_C
        # print(str(theta_star)+', '+str(hyper_params))
        prior_star = prior_fn(theta_star, hyper_params)
        if prior_star != -np.inf:
                likelihood_star = likelihood_fn(z, theta_star, *argv)
      
                if np.isnan(likelihood_star): likelihood_star = -np.inf
                with np.errstate(over='raise'):
                    try:
                        metr_ratio = np.exp(prior_star + likelihood_star -
                                  prior - likelihood)  # this gets caught and handled as an exception
                    except FloatingPointError:
                        print('- '+likelihood_fn.__name__+': theta_star='+str(theta_star)+", likelihood_star="+str(likelihood_star)+', theta='+str(theta)+", likelihood="+str(likelihood))
                        metr_ratio=1
                
                if np.isnan(metr_ratio):  metr_ratio = 0
                
                if metr_ratio > random_generator.uniform(0,1,1): 
                        theta = theta_star
                        prior = prior_star
                        likelihood = likelihood_star
                        jump_trace[i] = 1

        # Update the trace objects
        trace[:, i] = theta

        # Echo every 100 iterations
        if (i % 100) == 0:
            if verbose: print("Finished " + str(i) + " out of "+ str(n_updates) + " iterations.\n")
      # End main loop
      #########################################################

    # Collect trace objects to return
    res = {'acc_prob':jump_trace[1:n_updates].mean(),
           'trace':trace}

    return res


          
#                                                                         #
## --------------------------------------------------------------------- ##



## --------------------------------------------------------------------- ##
#  A generic Metropolis sampler.  You have to supply the log likelihood   #
#  function, which need not really be a likelihood function at all.       #
#
#  Translated from Shaby's R code.
#
#  Uppercase K is the size of the blocks of iterations used for
#  adapting the proposal.
#  Lowercase k is the offset to get rid of wild swings in adaptation
#  process that otherwise happen the early
#  iterations.
#

# z ............................... 'data' term in the likelihood functions
# starting_theta ........................................... initial values
# likelihood_fn ..................................... likelihood in-between
# prior_fn ....................................... prior function for theta
# hyper_params ........................................ for prior functions
# n_updates .................................. number of Metropolis updates
# prop_Sigma ................................... proposal covariance matrix

# adapt_cov ......................... whether to update proposal covariance
# return_prop_Sigma_trace........ save proposal covariance from each update
# r_opt ........................................... optimal acceptance rate
# c_0, c_1 .. two coefficients for updating sigma_m and proposal covariance
# K .............................................. adapt every K iterations
#                                                                         #


def adaptive_metr(z, starting_theta, likelihood_fn,
                          prior_fn, hyper_params, n_updates,
                          random_generator,
                          prop_Sigma = np.nan, adapt_cov = False,
                          return_prop_Sigma_trace = False,
                          r_opt = .234, c_0 = 10, c_1 = .8,
                          K = 10, *argv):


    eps = .001
    k = 3  # the iteration offset

    if type(starting_theta).__module__!='numpy' or isinstance(starting_theta, np.float64):
       starting_theta = np.array(starting_theta)
    p = starting_theta.size
    invalid = False
    
    # If the supplied proposal covariance matrix is either not given or invalid,
    # just use the identity.
    if np.any(np.isnan(prop_Sigma)) or prop_Sigma.size != p**2:
        prop_Sigma = np.eye(p)
        prop_C = np.eye(p)
        invalid = True
    else:
        try:
            # Initialize prop_C
            prop_C = np.linalg.cholesky(prop_Sigma)
        except  np.linalg.LinAlgError:
            prop_Sigma = np.eye(p)
            prop_C = np.eye(p)
            invalid = True
    if invalid:
        print("Invalid or missing proposal covariance matrix.  Using identity.\n")
    
    # Initialize sigma_m to the rule of thumb
    sigma_m = 2.4**2/p
    r_hat = 0
  
    # Set up and initialize trace objects
    trace = np.zeros((p, n_updates))
    sigma_m_trace = np.zeros(n_updates)
    r_trace = np.zeros(n_updates)
    jump_trace = np.zeros(n_updates)

    trace[:, 0] = starting_theta
    sigma_m_trace[0] = sigma_m
    
    if return_prop_Sigma_trace:
        prop_Sigma_trace = np.zeros((n_updates, p, p))
        prop_Sigma_trace[0,:,:] = prop_Sigma
  
    
    
    # Initialize Metropolis
    theta = starting_theta
    likelihood = likelihood_fn(z, theta, *argv)
    prior = prior_fn(theta, hyper_params)
  
  
    #########################################################
    # Begin main loop
    for i in np.arange(1,n_updates):
      theta_star= theta + sigma_m * random_generator.standard_normal(p) @ prop_C
      prior_star = prior_fn(theta_star, hyper_params)
      if prior_star != -np.inf:
          likelihood_star = likelihood_fn(z, theta_star, *argv)
      
          if np.isnan(likelihood_star): likelihood_star = -np.inf
      
          metr_ratio = np.exp(prior_star + likelihood_star -
                                  prior - likelihood)
          if np.isnan(metr_ratio):  metr_ratio = 0
                
          if metr_ratio > random_generator.uniform(0,1,1): 
              theta = theta_star
              prior = prior_star
              likelihood = likelihood_star
              jump_trace[i] = 1


      ########################################################
      # Adapt via my method                                  #
      if (i % K) == 0:
        gamma2 = 1 / ((i/K) + k)**(c_1)
        gamma1 = c_0*gamma2
      
        r_hat = jump_trace[(i - K + 1) : i].mean()

        sigma_m = np.exp(np.log(sigma_m) + gamma1*(r_hat - r_opt))

        if adapt_cov:
           prop_Sigma = prop_Sigma + gamma2*(np.cov(trace[:,(i - K + 1) : i]) - prop_Sigma)
           
           check_chol_cont = True
           while check_chol_cont:
               try:
                   # Initialize prop_C
                   prop_C = np.linalg.cholesky(prop_Sigma)
                   check_chol_cont = False
               except  np.linalg.LinAlgError:
                   prop_Sigma = prop_Sigma + eps*np.eye(p)
                   print("Oops. Proposal covariance matrix is now:\n")
                   print(prop_Sigma)
        
      #                                                      #
      ########################################################


    
      # Update the trace objects
      trace[:, i] = theta
      sigma_m_trace[i] = sigma_m
      r_trace[i] = r_hat
      if return_prop_Sigma_trace:
        prop_Sigma_trace[i,:,:] = prop_Sigma
      

      # Echo every 100 iterations
      if (i % 100) == 0:
        print("Finished "+str(i)+ " out of " + str(n_updates), " iterations.\n")
    
    # End main loop
    #########################################################

    # Collect trace objects to return
    res = {'trace':trace,
           'sigma_m_trace':sigma_m_trace,
           'r_trace':r_trace,
           'acc_prob':jump_trace.mean()}
    if return_prop_Sigma_trace:
      res['prop_Sigma_trace'] =prop_Sigma_trace

    return res
##                                                                        #
## --------------------------------------------------------------------- ##






## --------------------------------------------------------------------- ##
#  A generic Metropolis sampler.  You have to supply the log likelihood   #
#  function, which need not really be a likelihood function at all.       #
#
#  Translated from Shaby's R code.
#
#  Uppercase K is the size of the blocks of iterations used for
#  adapting the proposal.
#  Lowercase k is the offset to get rid of wild swings in adaptation
#  process that otherwise happen the early
#  iterations.
#

# z ............................... 'data' term in the likelihood functions
# starting_theta ........................................... initial values
# likelihood_fn ..................................... likelihood in-between
# prior_fn ....................................... prior function for theta
# hyper_params ........................................ for prior functions
# n_updates .................................. number of Metropolis updates
# prop_Sigma ................................... proposal covariance matrix

# adapt_cov ......................... whether to update proposal covariance
# return_prop_Sigma_trace........ save proposal covariance from each update
# r_opt ........................................... optimal acceptance rate
# c_0, c_1 .. two coefficients for updating sigma_m and proposal covariance
# K .............................................. adapt every K iterations
#                                                                         #


def adaptive_metr_ratio(z, starting_theta, likelihood_fn,
                          prior_fn, hyper_params, n_updates, 
                          random_generator,
                          prop_Sigma = np.nan, init_corr = np.nan, sd_ratio = np.nan,
                          adapt_cov = False,
                          return_prop_Sigma_trace = False,
                          r_opt = .234, c_0 = 10, c_1 = .8,
                          K = 10, *argv):


    eps = .001
    k = 3  # the iteration offset

    if type(starting_theta).__module__!='numpy' or isinstance(starting_theta, np.float64):
       starting_theta = np.array(starting_theta)
    p = starting_theta.size
    invalid = False
    
    # If the supplied proposal covariance matrix is either not given or invalid,
    # just use the identity.
    if np.any(np.isnan(prop_Sigma)) or prop_Sigma.size != p**2:
        prop_Sigma = np.eye(p)
        prop_C = np.eye(p)
        invalid = True
    else:
        try:
            # Initialize prop_C
            prop_C = np.linalg.cholesky(prop_Sigma)
        except  np.linalg.LinAlgError:
            prop_Sigma = np.eye(p)
            prop_C = np.eye(p)
            invalid = True
    if invalid:
        print("Invalid or missing proposal covariance matrix.  Using identity.\n")
    
    # Initialize sigma_m to the rule of thumb
    sigma_m = 2.4**2/p
    r_hat = 0
  
    # Set up and initialize trace objects
    trace = np.zeros((p, n_updates))
    sigma_m_trace = np.zeros(n_updates)
    r_trace = np.zeros(n_updates)
    jump_trace = np.zeros(n_updates)

    trace[:, 0] = starting_theta
    sigma_m_trace[0] = sigma_m
    
    if return_prop_Sigma_trace:
        prop_Sigma_trace = np.zeros((n_updates, p, p))
        prop_Sigma_trace[0,:,:] = prop_Sigma
  
    
    
    # Initialize Metropolis
    theta = starting_theta
    likelihood = likelihood_fn(z, theta, *argv)
    prior = prior_fn(theta, hyper_params)
  
  
    #########################################################
    # Begin main loop
    for i in np.arange(1,n_updates):
      theta_star= theta + sigma_m * random_generator.standard_normal(p) @ prop_C
      prior_star = prior_fn(theta_star, hyper_params)
      if prior_star != -np.inf:
          likelihood_star = likelihood_fn(z, theta_star, *argv)
      
          if np.isnan(likelihood_star): likelihood_star = -np.inf
      
          metr_ratio = np.exp(prior_star + likelihood_star -
                                  prior - likelihood)
          if np.isnan(metr_ratio):  metr_ratio = 0
                
          if metr_ratio > random_generator.uniform(0,1,1): 
              theta = theta_star
              prior = prior_star
              likelihood = likelihood_star
              jump_trace[i] = 1


      ########################################################
      # Adapt via my method                                  #
      if (i % K) == 0:
        gamma2 = 1 / ((i/K) + k)**(c_1)
        gamma1 = c_0*gamma2
      
        r_hat = jump_trace[(i - K + 1) : i].mean()

        sigma_m = np.exp(np.log(sigma_m) + gamma1*(r_hat - r_opt))
        
        if adapt_cov:
           if r_hat>0:
               sd_ratio_hat = np.std(trace[0,(i - K + 1) : i]) / np.std(trace[1,(i - K + 1) : i])
           else:
               sd_ratio_hat = 1
           
           if sd_ratio_hat == 0 or np.isnan(sd_ratio_hat):
               print("Haha")
               sd_ratio_hat = 1
           sd_ratio = np.exp(np.log(sd_ratio) + gamma1*(np.log(sd_ratio_hat) - np.log(sd_ratio)))
           prop_Sigma = np.array([[1, init_corr/sd_ratio], [init_corr/sd_ratio, 1/(sd_ratio*sd_ratio)]])
           
           check_chol_cont = True
           while check_chol_cont:
               try:
                   # Initialize prop_C
                   prop_C = np.linalg.cholesky(prop_Sigma)
                   check_chol_cont = False
               except  np.linalg.LinAlgError:
                   prop_Sigma = prop_Sigma + eps*np.eye(p)
                   print("Oops. Proposal covariance matrix is now:\n")
                   print(prop_Sigma)
        
      #                                                      #
      ########################################################


    
      # Update the trace objects
      trace[:, i] = theta
      sigma_m_trace[i] = sigma_m
      r_trace[i] = r_hat
      if return_prop_Sigma_trace:
        prop_Sigma_trace[i,:,:] = prop_Sigma
      

      # Echo every 100 iterations
      if (i % 100) == 0:
        print("Finished "+str(i)+ " out of " + str(n_updates), " iterations.\n")
    
    # End main loop
    #########################################################

    # Collect trace objects to return
    res = {'trace':trace,
           'sigma_m_trace':sigma_m_trace,
           'r_trace':r_trace,
           'acc_prob':jump_trace.mean()}
    if return_prop_Sigma_trace:
      res['prop_Sigma_trace'] =prop_Sigma_trace

    return res
##                                                                        #
## --------------------------------------------------------------------- ##

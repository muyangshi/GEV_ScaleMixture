###############################################################################
# run_all.sh
#
# Purpose
# -------
# Top-level wrapper script that reproduces the analysis workflow for the paper.
# This script sequentially runs the data-analysis pipeline on the real dataset
# and documents the expected outputs (figures and tables) at each step.
#
# Scope
# -----
# - Data analysis section: fully reproducible end-to-end (given data access and
#   required dependencies).
# - Simulation study section: provides the post-processing steps; 
#   full reproduction requires HPC resources due to computational cost.
#
# See also
# --------
# - README.md (Pipeline section): 
#   - detailed mapping from scripts to figure numbers.
#   - instructions for running the simulation study.
###############################################################################


# Preparation -----------------------------------------------------------------
# Ensure all dependencies listed in the README Preparation section are installed.
# Compile the shared library for numerical integration.
# Adjust include and library paths if necessary.
# This command generates the shared library object `RW_inte_cpp.so`.
g++ -I$GSL_INCLUDE -I$BOOST_INC -L$GSL_LIBRARY -L$BOOST_LIBRARY -std=c++11 -Wall -pedantic RW_inte_cpp.cpp -shared -fPIC -o RW_inte_cpp.so -lgsl -lgslcblas


###############################################################################
######## Data Analysis ########################################################
###############################################################################


# MCMC Sampling on Real Data --------------------------------------------------
# Run the MCMC sampler with 75 parallel processes, by default, on the real dataset.
# Produces traceplot items as described in the README.md.
# Reproduces Figure 6(b): Model configuration illustration (Plot_stations{modelname}.pdf).
mpirun -n 75 python3 MCMC.py


# Exploratory Analysis and Results Figures ------------------------------------
# Reproduce various figures for exploratory data analysis and results.
# Figure 6(a): Scatter plot of stations (stations train and test combined.pdf).
# Figure 8: Horizontal boxplot of log-likelihoods (ll_boxplot_all_horizontal.pdf).
# Figure 9: QQ plots for test data at each site (QQPlot_R_Test_MCMC_Site_<site>.pdf).
# Figure 10: Interpolated posterior mean surfaces (Surface_phi.pdf,
#            Surface_rho.pdf, Surface_mu0_pred.pdf, Surface_mu1_pred.pdf,
#            Surface_logsigma_pred.pdf, and Surface_xi_pred.pdf.)
python3 results_and_diagnostics.py

# Reproduce Figure 7: unique dates of yearly maxima (Fig_unique_peak_dates_combined.pdf).
python3 plot-timing.py

# Reproduce Figure 1, Figure 11 and Appendix E Figures 16 - 17 related to chi surfaces:
# Surface_model_chi_h=<h>.pdf, Surface_data_chi_fittedGEV_h=<h>.pdf,
# Surface_model_chi_LBUB_h=<h>_u=<u>.pdf, Surface_data_chi_LBUB_h=<h>_u=<u>.pdf.
# Surface_chi_diff_h={h}.pdf
python3 plot-chi.py
python3 plot-chi-diff.py

# Reproduce Figure 14 (Appendix D): smooth marginal parameter surfaces (Fig_elev_logsigma_xi_5panel.pdf).
python3 plot-GEV.py

# Reproduce Figure 15 (Appendix D): CO2/greenhouse gas time trends (Fig_co2_time_trends_3panel.pdf).
python3 plot-co2.py


###############################################################################
######## Simulation Study #####################################################
###############################################################################

# Empirical Analysis of Eta and Chi -------------------------------------------
# Perform numerical study of eta and chi parameters.
# Reproduces Figure 2: parameter surface plot. (Simulation_eta_chi.pdf)
# Reproduces Figure 3: empirical estimates. (chi_{i}{j}.pdf and eta_{i}{j}.pdf, 8 subfigures).
python3 empirical_eta_chi_mev.py


# Simulation Study ------------------------------------------------------------
# Reproduce Figure 4: Dependence model parameter surfaces for simulation study (Surface_all_simulation_surfaces.pdf).
python3 plot-simulation-surface.py

# Full simulation-study reproduction requires many replicated MCMC fits and 
# is intended to be executed on an HPC cluster. See README.md for setup.
# Once all simulation replicates have finished, running the post-processing step below 
# will aggregate results and generate the coverage figures.
# Reproduces: Figure 5 and Appendix C Figures 12–13 (binomial coverage plots),
# including:
#   - Empirical_Coverage_all_Phi_{simulation_case}.pdf
#   - Empirical_Coverage_all_Range_{simulation_case}.pdf
#   - Empirical_Coverage_MuSigma_{simulation_case}.pdf
python3 coverage_analysis.py

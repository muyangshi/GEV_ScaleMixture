# Preparation ---------------------------------------------
# Make sure the dependencies listed in the README file are installed
# Compile the shared library for numerical integration; Customize the include and library paths if necessary
# This step should produce a shared library object `RW_inte_cpp.so`
g++ -I$GSL_INCLUDE -I$BOOST_INC -L$GSL_LIBRARY -L$BOOST_LIBRARY -std=c++11 -Wall -pedantic RW_inte_cpp.cpp -shared -fPIC -o RW_inte_cpp.so -lgsl -lgslcblas


# Empirical Analysis of Eta and Chi -----------------------
# This performs the numerical study of eta and chi
# Figure 2: Simulation_eta_chi.pdf
# Figure 3: chi_{i}{j}.pdf and eta_{i}{j}.pdf
python3 empirical_eta_chi_mev.py # This will produce Figure 2 and Figure 3 (the 8 subfigures of it)


# MCMC sampler --------------------------------------------
# This runs the sampler on the real data by default.
mpirun -n 75 python3 MCMC.py
# produce traceplot and stuffs detailed in the README.md
# Figure 6(b) Mode configuration illustration Plot_stations<modelname>.pdf

# Exploratory Analysis and Results figures ------------------------------
# produce figure XYZ
python3 results_and_diagnostics.py
# Figure 6(a) Station trained and test combined.pdf
# Figure 8: ll_boxplot_all_horizontal.pdf
# Figure 9: QQPlot_R_Test_MCMC_Site_<site>.pdf
# Figure 10: the interpoltaed surfaces of posteriro mean. Surface_phi.pdf, Surface_rho.pdf, Surface_mu0_pred.pdf, urface_mu1_pred.pdf, Surface_logsigma_pred.pdf, Surface_xi_pred.pdf

python3 revision-timing-plot.py # produce Figure 7: Fig_unique_peak_dates_combined.pdf

python3 revision-chi-plot.py
# Produce Figure 11: Surface_model_chi_h=<h>.pdf; Surface_data_chi_fittedGEV_h=<h>.pdf
# Produce Appendix E Figure: Surface_model_chi_LBUB_h=<h>_u=<u>.pdf, Surface_data_chi_LBUB_h=<h>_u=<u>.pdf

python3 revision-GEV-plot.py # Produce Figure 14 (Appendix): Fig_elev_logsigma_xi_5panel.pdf
python3 revision-co2-plot.py # Produce Figure 15 (Appendix): Fig_co2_time_trends_3panel.pdf

python3 revision-plot-chi-diff.py # Produce Figure 16 Appendix E: Surface_chi_diff_h={h}.pdf

# simulation study ----------------------------------------
python3 plot_simulation_surface.py # this will reproduce Figure 4 dependece model parameter surfaces for simulation study Surface_all_simulation_surfaces.pdf
# Read README on how to set it up on a cluster HPC
To create N replicated simulations, first create N folders to house each simulation;
Then within each, 
    python3 simulate_data <seed> <simulation scenario> # to create the dataaset
    python3 MCMC.py # swap line 114 load read data to load simulated_data.RData
After all is run,
    python3 coverage analysis.py # will reproduce Figure 5 and FIgure 12 and 13 (in the Appendix) the binomial coverage plots
# FIgure 5 "Empirical_Coverage_all_Phi_{simulation_case}.pdf", "Empirical_Coverage_all_Range_{simulation_case}.pdf", 'Empirical_Coverage_MuSigma_{simulation_case}.pdf'
# Figure5,  12 and 13 are respective with different simulation_cases
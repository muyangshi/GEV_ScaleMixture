# Documentation

This is to document the files in the `/GEV_ScaleMixture` folder, along with any preparation that needs to be down to run the sampler. All files should be placed under the same directory.

```
GEV_ScaleMixture/
  │
  │   # data files
  ├── cb_2018_us_state_20m/
  ├── JJA_precip_maxima_nonimputed.RData
  ├── blockMax_JJA_centralUS_test.RData
  │
  │   # helper functions
  ├── RW_inte_cpp.cpp
  ├── RW_inte_cpp.so
  ├── RW_inte.py
  ├── utilities.py
  │
  │   # main sampler
  ├── MCMC.py
  │
  │   # other utility scripts
  ├── empirical_eta_chi_mev.py
  ├── simulate_data.py
  ├── coverage_analysis.py
  ├── results_and_diagnostics.py
  ├── proposal_cov.py
  ├── plot-chi.py
  ├── plot-chi-diff.py
  ├── plot-co2.py
  ├── plot-GEV.py
  ├── plot-simulation-surface.py
  └── plot-timing.py
```


## Preparation

This section includes some dependencies packages/modules to be downloaded and imported, instructions on a function library to be built, and specification on the dataset that feeds into the sampler. 

### Imports/Dependencies

**Parallel Computation**:

- openmpi
- openmpi-mpicxx

**Cpp**:

- GSL
- Boost

**Python**: (packages are managed through `conda` and/or `pip`)

- numpy
- scipy=1.11
- mpi4py=3.1.4
- gstools
- mpmath
- gmpy2
- numba
- rpy2
- matplotlib
    - Statemap `cb_2018_us_state_20m/` downloaded from [US census Bureau](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html)
- geopandas

**R**:

- mgcv
- mev


### Utilities
Files:

- `RW_inte_cpp.cpp`
- `RW_inte_cpp.so`
- `RW_inte.py`
- `utilities.py`

Numerical Integration functions are written in the `RW_inte_cpp.cpp` script. This script is compiled to a shared library object `RW_inte_cpp.so` (as a function library), for example using the terminal command: 

```
g++ -I$GSL_INCLUDE -I$BOOST_INC -L$GSL_LIBRARY -L$BOOST_LIBRARY -std=c++11 -Wall -pedantic RW_inte_cpp.cpp -shared -fPIC -o RW_inte_cpp.so -lgsl -lgslcblas
```

where `GSL_INCLUDE`, `BOOST_INCLUDE`, `GSL_LIBRARY`, and `BOOST_LIBRARY` are the corresponding `include/` and `lib/` folders for those packages.

Then these compiled functions are "packaged" into `RW_inte.py`, which is then imported in the set of all utilities/helper function in `utilities.py`.

### Dataset

#### Using a real dataset:

File:

- `JJA_precip_maxima_nonimputed.RData`
- `blockMax_JJA_centralUS_test.Rdata`

This is an example real dataset of central US summer time block-maxima precipitation. This `.RData` file contains the following items:

- `JJA_maxima_nonimputed`: an ($N_s$, $N_t$) matrix of type `double` of the non-imputed (contains NA) summer time block-maxima precipitation at the $N_s$ stations across $N_t$ times. Each column $t$ represents the $N_s$ observations at time $t$.
- `GEV_estimates`: an ($N_s$ $\times$ 4) `dataframe` of the marginal parameter estimates (by fitting a simple GEV) at each station. Each of the four columns represents the estimates for $\mu_0$, $\mu_1$, $\log(\sigma)$, and $\xi$ for each station. 
- `stations`: an ($N_s$ $\times$ 2) `dataframe` of the coordinates of each station. The first column is longitude and the second column represents latitude.
- `elev`: an ($N_s$,) vector of type `double` of the elevation of the geographic location of each station.

where $N_s$ denotes the number of stations and $N_t$ denotes the number of years recorded in the dataset.

#### Generating a Simulated Dataset:

If we are not performing an application on a real dataset, we could simulate a dataset to test the model.

File:

- `simulate_data.py`

This is a python script to generate a simulated dataset. 
 Within this file, specifies $N_s$, $N_t$, $\phi$ and/or $\rho$ surfaces as well as the marginal parameters ($\Beta's$ for the $\mu_0, \mu_1$ as well as $\sigma$ and $\xi$) surfaces to fine tune the truth parameter setup. Some additional comments in the script might be helpful. 

To run this file:

```
python3 simulate_data.py <random_seed>
```
where `<random_seed>` is a seed used to randomly generate the data;
 e.g. `python3 simulate_data.py 2345` can be used to generate a simulated dataset using the random seed of 2345.


Outputs:

Running this script will generate the following dataset and/or plots:
 (assuming we have simulated a dataset of $N_t=24$ time replicates at $N_s = 300$ sites using simulate scenario $sc = 2$)
- `simulated_data.RData`: The generated dataset, which consists of the 4 items matching those described in the real dataset above.
- Additional formats/pieces of the same simulated dataset (for easier checking): 
    - `Y_full_sim_sc2_t24_s300.npy`: simulated observation $Y$ with all observations (no missing)
    - `miss_matrix_bool.npy`: simulated missing indicator matrix
    - `Y_miss_sim_sc2_t24_s300.npy`: simulated observation $Y$ after applying the `miss_matrix` (some observation become `NA`s as mimicing missing at random in data)
- `.png` Plots (for checking generated dataset):
    - QQPlots: QQplots will be generated for the Levy variables at the knots, the Pareto $W=g(Z)$ at the sites, and the process $X^*$ across the sites $s$ or time $t$. All qqplots are transformed to the uniform margin. Other than the QQPlot for $X^*$ across all sites at a specific time $t$ should deviates from 1:1 line (because of spatial correlation), the others should all approximately be linearly following diagonal line.
    - Histogram: MLE fitted GEV models on the dataset at each site $s$ across all time $t$ is pooled together into a histogram; the values of the $\mu, \sigma, \xi$ should roughly reflect that in the parameter surface setting. (Not precise but should/could serve as a quick check)


## Sampler

This section includes some information on running the sampler `MCMC.py` script, as well as some notes on the output (plots, traceplots, intermediate model states) that running this script will produce.

**Required Files** (in addition to the files described in the previous section):

- `proposal_cov.py`
- `MCMC.py`

**Optional Files** (saved model states from prior runs that `MCMC.py` will generate periodically):

- `iter.pkl`
- `sigma_m_sq_Rt_list.pkl`
- `sigma_m_sq.pkl`
- `Sigma_0.pkl`


`MCMC.py` is the main sampler file, and it uses a random-walk Metropolis algorithm using Log-Adaptive Proposal (LAP) as an adaptive tuning strategy (Shaby and Wells, 2010). This script takes in a dataset file (e.g. `JJA_precip_maxima_nonimputed.RData`) placed under the same directory. Additional dependencies are specified in the previous section. 
 
Sometimes (especially when running on clusters) we can't afford to have the sampler be continuously running until it finishes, and so we have to "chop" it into pieces and "daisychaining" the subsequent runs. This script will automatically create and save the traceplots for the variables/parameters, as well as saving the model states when the script is stopped (e.g. run into the time limit). When invoking this script, it will check if there are saved model states (the optional files) saved in the directory and will pick up from there.

This `MCMC.py` script is split into the following sections (more detailed comments are made within the script):

- Load the dataset
- Setup the spline smoothings
- Estimate the initial starting points for parameters
  - Plot the initially estimated parameter surfaces
- Specify the block-update structure for MCMC updating the parameters
- Metropolis-Hasting MCMC Loop:
  - Update $R_t$
  - Update $\phi$
  - Update $\rho$
  - Update $Y$ (imputations)
  - Update $\beta(\mu_0)$
  - Update $\beta(\mu_1)$
  - Update $\beta(\log(\sigma))$
  - Update $\beta(\xi)$
  - Update the adaptive metropolis strategy (periodically, once every certain \# of iterations)

The `proposal_cov.py` is the initial proposal covariance matrix $\Sigma_0$ for this LAP tuning strategy. 
Without specific knowledge on the covariance of the proposals, one can set the variables in the `proposal_cov.py` script to `None`, as this would make the sampler default to initialize with identity $I$ proposals.
This is only used when starting the chains fresh, as the later continuance/"daisychain" will load the proposal scalar variance and covariance from the `.pkl` files saved from previous runs.

To **run this sampler script**:

```
mpirun -n <Nt> python3 MCMC.py
```

where `<Nt>` is the number of time replicates in the dataset and hence (by the parallelization of the code) the number of cores used to invoke this parallelized job using mpi.

**Outputs**:

Running the sampler will generate the following results files

- Plots

    - Geo/spatial informations on the dataset:
        - `Plot_US.pdf`, `Plot_stations.pdf`: scatterplots of the stations (longitude, latitude) with overlaying state boundary
        - `Plot_station_elevation.pdf`: scatterplots of the stations with color coding their elevations
    - Initial Parameter Estimates:
        - `Plot_initial_heatmap_phi_surface.pdf`, `Plot_initial_heatmap_rho_surface.pdf`: heatmaps of the $\phi$ and $\rho$ surfaces coming from initial parameter estimation.
        - `Plot_initial_mu0_estimates.pdf`, `Plot_initial_mu1_estimates.pdf`, `Plot_initial_logsigma_estimates.pdf`, and `Plot_initial_ksi_estimates.pdf`: Comparison of the intial GEV fitted parameters at the sites versus the spline smoothed marginal parameters at the sites (color represents value)
    - Plots of the Traceplots:
      - Overall log-likelihoods: `Traceplot_loglik.pdf`, `Traceplot_loglik_detail.pdf`
      - Copula parameters: `Traceplot_Rt_<t>.pdf` (`<t>` in 1, ..., $N_t$), `Traceplot_phi.pdf`, and `Traceplot_range.pdf`
      - Marginal model coefficients and regularization terms: `Traceplot_<Beta_mu0_block_idx>.pdf` (for $\Beta$'s for $\mu_0$ in that block update), `Traceplot_<Beta_mu1_block_idx>.pdf`, `Traceplot_Beta_logsigma.pdf`, `Traceplot_Beta_ksi.pdf`, and `Traceplot_sigma_Beta_xx.pdf`.

- Traceplot `.npy` Matrix

    - The traceplot items are periodically saved (currently after every 50 iterations), including
        - log-likelihood trace: `loglik_trace.npy`, `loglik_detail_trace.npy`
        - copula parameter trace: `R_trace_log.npy`, `phi_knots_trace.npy`, `range_knots_trace.npy`, 
        - marginal model parameter trace: `Beta_mu0_trace.npy`, `Beta_mu1_trace.npy`, `Beta_logsigma_trace.npy`, `Beta_ksi_trace.npy` and their regularization hyper parameter trace `sigma_Beta_mu0_trace.npy`, `sigma_Beta_mu1_trace.npy`, `sigma_Beta_logsigma_trace.npy`
        - records on imputation (conditional gaussian draws) on `Y_trace.npy`

- Periodically saved model/chain states in `.pkl` pickles (to be picked up at each consecutive run)

    - `iter.pkl`: the number of iteration this chain has reached; a consecutive run will "restart" the chain at the last saved `iter.pkl` iteration)
    - Adapted proposal scalar variance and/or covariance matrix
        - the proposal scalar variance for the stable variables $R_t$'s: `sigma_m_sq_Rt_list.pkl`
        - the proposal scalar variance and covariance matrix for any other parameters: `sigma_m_sq.pkl`, `Sigma_0.pkl`


## Pipeline

This section documents the end-to-end computational pipeline used in the paper.
The workflow is divided into two components:

1. **Data Analysis on Real Dataset** — fully reproducible end-to-end (given the required environment and data files).
2. **Simulation Study** — computationally intensive; scripts and the full workflow are provided, but complete reproduction requires HPC resources.

The top-level wrapper script `run_all.sh` executes the steps below in order.

---

### Preparation

Before running any analysis, ensure all dependencies listed in the **Preparation** section (above) are installed. In particular, the numerical integration routines must be compiled into a shared library.

```bash
g++ -I$GSL_INCLUDE -I$BOOST_INC -L$GSL_LIBRARY -L$BOOST_LIBRARY \
    -std=c++11 -Wall -pedantic RW_inte_cpp.cpp \
    -shared -fPIC -o RW_inte_cpp.so -lgsl -lgslcblas
```

This command generates the shared object `RW_inte_cpp.so`, which is imported by the Python utilities used throughout the project.

---

## Data Analysis

This section reproduces all real-data results and figures (main text and appendices) produced from the application dataset.

### MCMC Sampling on Real Data

Run the MCMC sampler with 75 parallel MPI processes:

```bash
mpirun -n 75 python3 MCMC.py
```

This step:
- runs the Bayesian MCMC sampler on the real dataset,
- produces traceplot items saved to disk,
- saves posterior samples and model states for downstream analysis.

**Reproduces:**
- **Figure 6(b)** — Model configuration illustration (`Plot_stations{modelname}.pdf`)

---

### Exploratory Analysis and Results Figures

Post-processing and visualization of the MCMC output are handled by:

```bash
python3 results_and_diagnostics.py
```

**Reproduces:**
- **Figure 6(a)** — Station locations (training and test combined)
- **Figure 8** — Horizontal boxplot of log-likelihoods (`ll_boxplot_all_horizontal.pdf`)
- **Figure 9** — QQ plots for test data at each site (`QQPlot_R_Test_MCMC_Site_<site>.pdf`)
- **Figure 10** — Interpolated posterior mean surfaces:
  - `Surface_phi.pdf`
  - `Surface_rho.pdf`
  - `Surface_mu0_pred.pdf`
  - `Surface_mu1_pred.pdf`
  - `Surface_logsigma_pred.pdf`
  - `Surface_xi_pred.pdf`

Timing information of the Data can be generated with:

```bash
python3 plot-timing.py
```

**Reproduces:**
- **Figure 7** — Unique dates of yearly maxima (`Fig_unique_peak_dates_combined.pdf`)

The empirical moving window estimation of $\chi$ is generated with:

```bash
python3 plot-chi.py
python3 plot-chi-diff.py
```

**Reproduces:**
- **Figure 1** and **Figure 11** — Empirical and model-based chi surfaces
- **Appendix E figures** — chi lower/upper bound surfaces and difference surfaces, including:
  - `Surface_model_chi_h=<h>.pdf`
  - `Surface_data_chi_fittedGEV_h=<h>.pdf`
  - `Surface_model_chi_LBUB_h=<h>_u=<u>.pdf`
  - `Surface_data_chi_LBUB_h=<h>_u=<u>.pdf`
  - `Surface_chi_diff_h=<h>.pdf`

To Generate the marginal parameter spline smoothing figure:

```bash
python3 plot-GEV.py
```

**Reproduces:**
- **Figure 14 (Appendix D)** — Smooth marginal parameter surfaces (`Fig_elev_logsigma_xi_5panel.pdf`)

To Generate the CO₂ / greenhouse gas figure:

```bash
python3 plot-co2.py
```

**Reproduces:**
- **Figure 15 (Appendix D)** — CO₂ / greenhouse gas time trends (`Fig_co2_time_trends_3panel.pdf`)

---

## Simulation Study

The simulation study would requires running many replications, which is computationally heavy, so **full reproduction is intended for execution on an HPC cluster**.

This repository provides:
- scripts for generating simulation-study figures,
- the workflow to run replicated simulations,
- post-processing code to aggregate results and compute coverage.

---

### Empirical Analysis of $\eta$ and $\chi$

This script performs the numerical study in **Illustration 1**:

```bash
python3 empirical_eta_chi_mev.py
```

**Reproduces:**
- **Figure 2** — Parameter surface plot (`Simulation_eta_chi.pdf`)
- **Figure 3** — Empirical chi/eta estimates (`chi_{i}{j}.pdf`, `eta_{i}{j}.pdf`, 8 subfigures)

---

### Simulation Parameter Surfaces

```bash
python3 plot-simulation-surface.py
```

**Reproduces:**
- **Figure 4** — Dependence model parameter surfaces (`Surface_all_simulation_surfaces.pdf`)


### Replicated Simulation Runs and Coverage Analysis (HPC)

Full simulation-study reproduction requires:
1. Creating multiple simulation-replication directories.
2. Running data generation and MCMC sampling independently within each directory.
3. Collecting completed outputs and running the coverage analysis below.

Once all simulation replicates have finished, run:

```bash
python3 coverage_analysis.py
```

**Reproduces:**
- **Figure 5** — Empirical coverage for dependence and marginal parameters
- **Appendix C Figures 12–13** — Additional binomial coverage plots, including:
  - `Empirical_Coverage_all_Phi_{simulation_case}.pdf`
  - `Empirical_Coverage_all_Range_{simulation_case}.pdf`
  - `Empirical_Coverage_MuSigma_{simulation_case}.pdf`

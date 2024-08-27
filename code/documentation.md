# Documentation

This is to document the files in the `/code` folder, along with any preparation that needs to be down to run the sampler. All files should be placed under the same directory.

```
code/
  │
  ├── cb_2018_us_state_20m/
  ├── RW_inte_cpp.cpp
  ├── RW_inte_cpp.so
  ├── RW_inte.py
  ├── utilities.py
  │
  ├── JJA_precip_maxima_nonimputed.RData
  ├── simulate_data.py
  ├── MCMC.py
  │
  └── posterior_and_diagnostics.py
```


---
## Preparation

This section includes some dependencies packages/modules to be downloaded and imported, instructions on a function library to be built, and specification on the dataset that feeds into the `MCMC.py` sampler. 

### Imports/Dependencies

**Parallel Computation**:

- openmpi
- openmpi-mpicxx

**Cpp**:

- GSL
- Boost

**Python**: (packages are managed through `Conda`)

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
 Within this file, specifies $N_s$, $N_t$, $\phi$ and/or $\rho$ surfaces as well as the marginal parameters ($\Beta's$ for the $\mu_0, \mu1$ as well as $\sigma$ and $\xi$) surfaces to fine tune the truth parameter setup. Some additional comments in the script might be helpful. 

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


---
## Sampler

File:

- `MCMC.py`

This is the main sampler file. What does it do. How to run it. What does it produce.

To run this file:

```
mpirun -n 75 python3 MCMC.py
```





---
## Posterior Summary

File:

- `posterior_and_diagnostics.py`

This is the posterior summary script, that summarizes the chains resulting from running the `MCMC.py` sampler. 

What does it do.

What does it produce.

To fun this file:
```
python3 posterior_and_diagnostics.py
```


# Documentation

This is to document the files in the `/code` folder, along with any preparation that needs to be down to run the sampler.




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

**Format**: 

- What items in this RData

- What shape is each datachart

#### Generating a Simulated Dataset:

File:

- `simulation.py`

This is a python script to generate a simulated dataset. Some details in the file on how to change the generation specifications. What files does it produce?

To run this file:

```
python3 bla bla simulation.py
```




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


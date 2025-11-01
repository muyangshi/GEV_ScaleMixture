#!/bin/bash
#SBATCH --account=csu70_alpine1
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --job-name=simulation_36
#SBATCH --constraint=ib
#SBATCH --partition=amilan
#SBATCH --output=simulation_36.out
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
module purge
module load anaconda
module load gcc
module load openmpi
SLURM_EXPORT_ENV=ALL
module load gsl
module load boost
conda activate alpine_MCMC
$CXX -std=c++11 -Wall -pedantic -I$CURC_GSL_INC -I$CURC_BOOST_INC -L$CURC_GSL_LIB -L$CURC_BOOST_LIB p_inte.cpp -shared -fPIC -o p_inte.so -lgsl -lgslcblas
$CXX -std=c++11 -Wall -pedantic -I$CURC_GSL_INC -I$CURC_BOOST_INC -L$CURC_GSL_LIB -L$CURC_BOOST_LIB RW_inte_cpp.cpp -shared -fPIC -o RW_inte_cpp.so -lgsl -lgslcblas
mpirun -n 64 python3 MCMC_scenario2.py 265
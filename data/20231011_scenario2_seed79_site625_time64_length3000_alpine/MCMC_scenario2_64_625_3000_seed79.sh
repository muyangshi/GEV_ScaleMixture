#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 64
#SBATCH --job-name MCMC_scenario2_64_625_3000_seed79
#SBATCH --constraint ib
#SBATCH --partition amilan
#SBATCH --time 24:00:00
#SBATCH --output MCMC_scenario2_64_625_3000_seed79.out

module purge

module load gcc
module load openmpi
SLURM_EXPORT_ENV=ALL
module load gsl
module load boost
module load anaconda

conda activate testenv

$CXX -std=c++11 -Wall -pedantic -I$CURC_GSL_INC -I$CURC_BOOST_INC -L$CURC_GSL_LIB -L$CURC_BOOST_LIB p_inte.cpp -shared -fPIC -o p_inte.so -lgsl -lgslcblas

mpirun -n 64 python3 MCMC_scenario2_64_625_3000_seed79.py
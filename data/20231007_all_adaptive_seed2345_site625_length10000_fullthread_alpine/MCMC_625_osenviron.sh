#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 64
#SBATCH --job-name MCMC_625_osenviron
#SBATCH --constraint ib
#SBATCH --partition amilan
#SBATCH --time 24:00:00
#SBATCH --output MCMC_625_osenviron.out

module purge

module load gcc
module load openmpi
SLURM_EXPORT_ENV=ALL
module load gsl
module load boost
module load anaconda

conda activate testenv

$CXX -std=c++11 -Wall -pedantic -I$CURC_GSL_INC -I$CURC_BOOST_INC -L$CURC_GSL_LIB -L$CURC_BOOST_LIB p_inte.cpp -shared -fPIC -o p_inte.so -lgsl -lgslcblas

mpirun -n 32 python3 MCMC_625_osenviron.py
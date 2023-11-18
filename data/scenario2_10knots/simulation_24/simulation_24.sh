#!/bin/bash
#SBATCH --account=csu70_alpine1
#SBATCH --nodes 1
#SBATCH --ntasks 64
#SBATCH --job-name simulation_24
#SBATCH --constraint ib
#SBATCH --partition amilan
#SBATCH --time 24:00:00
#SBATCH --output simulation_24.out
module purge
module load anaconda
module load gcc
module load openmpi
SLURM_EXPORT_ENV=ALL
module load gsl
module load boost
conda activate alpine_MCMC
$CXX -std=c++11 -Wall -pedantic -I$CURC_GSL_INC -I$CURC_BOOST_INC -L$CURC_GSL_LIB -L$CURC_BOOST_LIB p_inte.cpp -shared -fPIC -o p_inte.so -lgsl -lgslcblas
mpirun -n 64 python3 MCMC_10knots.py 181
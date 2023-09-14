# Create folders and bash scripts necessary to execute the coverage analysis
# This file should be placed under the root directory of this project
import os
import shutil

directories = []
bash_files = []

n_sim = 10

# %%
# Create folder and files for EACH simulation repetition
# ------------------------------------------------------

# Create a simulation_# folder under /data/scenario2
# Create a simulation_#.sh bash script under /data/scenario2/simulation_#/
# Create the necessary p_inte.cpp, MCMC.py, model_sim.py files to /data/scenario2/simulation_#

for i in range(n_sim):
    sim_id = str(i + 1)
    lines = ['#!/bin/bash',
            '#SBATCH --nodes 1',
            '#SBATCH --ntasks 32',
            '#SBATCH --job-name simulation_' + sim_id,
            '#SBATCH --constraint ib',
            '#SBATCH --partition amilan',
            '#SBATCH --time 0:60:00',
            '#SBATCH --output simulation_' + sim_id + '.out',

            'module purge',

            'module load gcc',
            'module load openmpi',
            'SLURM_EXPORT_ENV=ALL',
            'module load gsl',
            'module load boost',
            'module load anaconda',

            'conda activate testenv',

            '$CXX -std=c++11 -Wall -pedantic -I$CURC_GSL_INC -I$CURC_BOOST_INC -L$CURC_GSL_LIB -L$CURC_BOOST_LIB p_inte.cpp -shared -fPIC -o p_inte.so -lgsl -lgslcblas',

            'mpirun -n 32 python3 MCMC_test.py']
    
    # Create subfolder ./data/scenario2/simulation_#
    os.makedirs('./data/scenario2/simulation_' + sim_id, exist_ok=True)

    # Create the bash script
    with open('./data/scenario2/simulation_' + sim_id + '/' + 'simulation_' + sim_id + '.sh','w') as f:
        f.write('\n'.join(lines))

    # Paste over the necessary files
    shutil.copy('./p_inte.cpp', './data/scenario2/simulation_' + sim_id)
    shutil.copy('./MCMC_test.py', './data/scenario2/simulation_' + sim_id)
    shutil.copy('./model_sim.py', './data/scenario2/simulation_' + sim_id)

    sim_id = str(i + 1)
    directory = './data/scenario2/simulation_' + sim_id
    directories.append(directory)

    bash_file = 'simulation_' + sim_id + '.sh'
    bash_files.append(bash_file)


# %%
# Create master bash script that submit jobs
# ------------------------------------------------------

with open('master_scenario2.sh', 'w') as f:
    for i in range(n_sim):
        f.write('cd' + ' ' + '/projects/$USER/GEV_simulation/' + directories[i] + '\n')
        f.write('bash' + ' ' + bash_files[i] + '\n')
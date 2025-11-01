#%%
# Imports and Set Parameters
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print('my rank is: ', rank)

from RW_inte import *
print(rank, pRW_transformed_cpp(20, 0.5, 0.5))

if rank == 0:
    Nt = size
    k = 8
    # R_log_cov = np.array([t*np.eye(k) for t in range(Nt)])
    R_log_cov_to_scat = [[0.5*i] * k for i in range(Nt)] # shape(Nt, k) scatter along axis 0
else:
    R_log_cov_to_scat = None
scattered_R_log_cov = comm.scatter(R_log_cov_to_scat, root = 0)
print('rank:',rank,scattered_R_log_cov,type(scattered_R_log_cov[0]))
# %%

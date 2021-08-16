#!/bin/bash -e
#PJM -L  "node=1"                          # Number of assign node 8 (1 dimention format)
#PJM -L  "freq=2200"                         
#PJM -L "rscgrp=small"         # Specify resource group
#PJM -L  "elapse=00:05:00"                 # Elapsed time limit 1 hour
#PJM --mpi "max-proc-per-node=48"          # Maximum number of MPI processes created per node
#PJM -s                                    # Statistical information output

export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh
# bash $deepmd_root/script/x86_64/build_deepmd.sh

export PLE_MPI_STD_EMPTYFILE=off
# export PRINT_TIME=1

export TF_INTRA_OP_PARALLELISM_THREADS=1
export HAVE_PREPROCESSED=1
export TF_CPP_MIN_LOG_LEVEL=3

export OMP_NUM_THREADS=1
export DEEPMD_NUM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=1
mpiexec -n 16 lmp_mpi -echo screen -in ../lmp/in.water_compress_48


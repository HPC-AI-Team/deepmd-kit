#!/bin/bash
#PJM -L  "node=1"                          # Number of assign node 8 (1 dimention format)
#PJM -L  "freq=2200"                         
#PJM -L "rscgrp=small"         # Specify resource group
#PJM -L  "elapse=01:00:00"                 # Elapsed time limit 1 hour
#PJM --mpi "shape=1"
#PJM --mpi "max-proc-per-node=48"          # Maximum number of MPI processes created per node
#PJM -s                                    # Statistical information output

set -ex

deepmd_root=$HOME/gzq/deepmd-kit
source $deepmd_root/script/fugaku/env.sh
bash $deepmd_root/script/fugaku/build_deepmd.sh

export PLE_MPI_STD_EMPTYFILE=off
# export PRINT_TIME=1
export OMP_NUM_THREADS=1
export TF_INTRA_OP_PARALLELISM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=1

likwid-pin -c 0 lmp -echo screen -in ./in.water_compress_1

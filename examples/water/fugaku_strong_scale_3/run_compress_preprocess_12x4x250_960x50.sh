#!/bin/bash -e
#PJM -L  "node=250"                          # Number of assign node 8 (1 dimention format)
#PJM -L  "freq=2200"                         
#PJM -L "rscgrp=small"         # Specify resource group
#PJM -L  "elapse=00:05:00"                 # Elapsed time limit 1 hour
#PJM --mpi "max-proc-per-node=12"          # Maximum number of MPI processes created per node
#PJM -s                                    # Statistical information output


if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi
source $deepmd_root/script/fugaku/env.sh
# bash $deepmd_root/script/fugaku/build_deepmd.sh

export OMPI_MCA_plm_ple_memory_allocation_policy=bind_local
export PLE_MPI_STD_EMPTYFILE=off
export OMP_NUM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=-1
export TF_INTRA_OP_PARALLELISM_THREADS=1
export HAVE_PREPROCESSED=1
export TF_CPP_MIN_LOG_LEVEL=3


export DEEPMD_NUM_THREADS=4
mpiexec lmp_mpi -echo screen -in ../lmp/in.water_compress_preprocess_960x50

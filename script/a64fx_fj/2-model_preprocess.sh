#!/bin/bash
#PJM -L  "node=1"                           # Number of assign node 8 (1 dimention format)
#PJM -L  "freq=2200"                         
#PJM -L  "rscgrp=int"                     # Specify resource group
#PJM -L  "elapse=00:20:00"                  # Elapsed time limit 1 hour
#PJM --mpi "max-proc-per-node=1"            # Maximum number of MPI processes created per node
export PLE_MPI_STD_EMPTYFILE=off

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi

source $deepmd_root/script/a64fx_fj/env.sh

set -ex

export OMP_NUM_THREADS=1
export TF_INTRA_OP_PARALLELISM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=1

export LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1

# modify for your path ----------------------------------------------------------

# Compressed model path (input path)
origin_model=
# preprocessed model path (output path)
preprocessed_model=

# -------------------------------------------------------------------------------

if [ -z $origin_model ]
then
    echo "origin model path is not set !!!"
    exit -1
fi 

if [ -z $preprocessed_model ]
then
    echo "preprocessed model path is not set !!!"
    exit -1
fi 

dp preprocess -i $origin_model -o $preprocessed_model

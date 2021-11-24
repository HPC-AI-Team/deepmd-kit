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
    exit -1
fi

source $deepmd_root/script/a64fx_fj/env.sh

set -ex

export OMP_NUM_THREADS=1
export TF_INTRA_OP_PARALLELISM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=1

export LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1


# modify for your path ----------------------------------------------------------

# train config file path. 
training_config=$deepmd_root/examples/water/se_e2_a/input_100.json
# The path of the model to be compressed
origin_model=$deepmd_root/examples/water/fugaku/model_optimized.pb
# Compressed model path (output path)
compressed_model=$deepmd_root/examples/water/fugaku/model_optimized_compressed.pb

# -------------------------------------------------------------------------------

if [ -z $origin_model ]
then
    echo "origin model path is not set !!!"
    exit -1
fi 

if [ -z $training_config ]
then
    echo "training config path is not set !!!"
    exit -1
fi 

if [ -z $compressed_model ]
then
    echo "compressed model path is not set !!!"
    exit -1
fi 

dp compress -t $training_config -i $origin_model -o $compressed_model
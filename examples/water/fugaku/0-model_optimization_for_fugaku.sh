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


export OMP_NUM_THREADS=1
export TF_INTRA_OP_PARALLELISM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=1

export LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1

# modify for your path ----------------------------------------------------------

# trained model path you have prepared (non compressed)
raw_model=$deepmd_root/examples/water/model/double/graph.pb
# train config file. modify the trainning step (numb_steps or stop_betch) to 100.
# (we just need a model pattern and model weights will be transfered from your trained model) 
training_config=$deepmd_root/examples/water/se_e2_a/input_100.json
# optimized model path (output path)
optimized_model=$deepmd_root/examples/water/fugaku/model_optimized.pb

# -------------------------------------------------------------------------------

if [ -z $raw_model ]
then
    echo "raw model path is not set !!!"
    exit -1
fi 

if [ -z $training_config ]
then
    echo "training config path is not set !!!"
    exit -1
fi 

if [ -z $optimized_model ]
then
    echo "optimized model path is not set !!!"
    exit -1
fi 

set -ex

dp train $training_config
dp freeze -o $optimized_model

dp transfer -O $raw_model -r $optimized_model -o$optimized_model
./link.sh double test

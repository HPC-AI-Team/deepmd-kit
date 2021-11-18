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

raw_model=$deepmd_root/examples/water/model/double/original/graph-original-baseline.pb
training_config=$deepmd_root/examples/water/se_e2_a/input_1.json
optimized_model=$deepmd_root/examples/water/fugaku/model_optimized.pb

dp train $training_config
dp freeze -o $optimized_model

dp transfer -O $raw_model -r $optimized_model -o$optimized_model

cp $deepmd_root/examples/water/fugaku/model_optimized.pb $deepmd_root/examples/water/model/double/original/graph-original-test.pb
./link.sh double test


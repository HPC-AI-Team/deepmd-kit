#!/bin/bash
#PJM -L  "node=1"                           # Number of assign node 8 (1 dimention format)
#PJM -L  "freq=2200"                         
#PJM -L  "rscgrp=int"                     # Specify resource group
#PJM -L  "elapse=00:20:00"                  # Elapsed time limit 1 hour
#PJM --mpi "max-proc-per-node=1"            # Maximum number of MPI processes created per node
export PLE_MPI_STD_EMPTYFILE=off

set -ex

source $deepmd_root/script/fugaku/env.sh
# bash $deepmd_root/script/fugaku/build_deepmd.sh

export LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1

export TF_CPP_MIN_LOG_LEVEL=3
export TF_PROFILE=1
export DEEPMD_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=1
export TF_INTRA_OP_PARALLELISM_THREADS=1

rm -f profiler.json_*

dp test -m ../model/graph-original.pb -s ../data/data_3 -n 1

python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/tools/profiler_visualization_topk.py profiler 

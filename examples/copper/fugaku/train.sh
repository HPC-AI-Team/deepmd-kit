#!/bin/bash
#PJM -L  "node=1"                           # Number of assign node 8 (1 dimention format)
#PJM -L  "freq=2200"                         
#PJM -L  "rscgrp=int"                     # Specify resource group
#PJM -L  "elapse=00:20:00"                  # Elapsed time limit 1 hour
#PJM --mpi "max-proc-per-node=1"            # Maximum number of MPI processes created per node
export PLE_MPI_STD_EMPTYFILE=off


set -ex



source $deepmd_root/script/fugaku/env.sh
bash $deepmd_root/script/fugaku/build_deepmd.sh

export OMP_NUM_THREADS=1
export TF_INTRA_OP_PARALLELISM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=1

export LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1

# dp train ../se_e2_a/input_float_1000.json
# dp freeze -o ../model/float/original/graph-original-baseline.pb
# dp test -m ../model/float/original/graph-original-baseline.pb -s ../data/data_3 -n 1

# rm -rf model.ckpt.*
# rm -rf checkpoint


dp train ../se_e2_a/input_double_1000.json
dp freeze -o ../model/double/original/graph-original-gemm_tanh.pb
dp test -m ../model/double/original/graph-original-gemm_tanh.pb -s ../data/data_3 -n 1

rm -rf model.ckpt.*
rm -rf checkpoint
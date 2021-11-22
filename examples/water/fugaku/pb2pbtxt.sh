#!/bin/bash -e
#PJM -L  "node=1"                           # Number of assign node 8 (1 dimention format)
#PJM -L  "freq=2200"                         
#PJM -L  "rscgrp=int"                     # Specify resource group
#PJM -L  "elapse=00:10:00"                  # Elapsed time limit 1 hour
#PJM --mpi "max-proc-per-node=1"            # Maximum number of MPI processes created per node
export PLE_MPI_STD_EMPTYFILE=off

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
    exit -1
fi
source $deepmd_root/script/a64fx_fj/env.sh

export LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1

input_model=$deepmd_root/examples/water/model/double/compress/graph-compress-test.pb
output_model=$deepmd_root/examples/water/fugaku/graph-compress-test.pbtxt

dp pb2pbtxt -i $input_model -o $output_model
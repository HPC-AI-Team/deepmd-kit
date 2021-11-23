#!/bin/bash
#PJM -L  "node=1"                           # Number of assign node 8 (1 dimention format)
#PJM -L  "freq=2200"                         
#PJM -L  "rscgrp=int"                     # Specify resource group
#PJM -L  "elapse=00:20:00"                  # Elapsed time limit 1 hour
#PJM --mpi "max-proc-per-node=1"            # Maximum number of MPI processes created per node
export PLE_MPI_STD_EMPTYFILE=off

source $deepmd_root/script/fugaku/env.sh

python_script=$deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/tools/pb2pbtxt.py

export LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1

python $python_script ../model/float/compress/graph-compress-baseline.pb ../model/float/compress/graph-compress-baseline.pbtxt
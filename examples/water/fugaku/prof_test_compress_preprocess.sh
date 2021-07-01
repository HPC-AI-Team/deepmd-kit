#!/bin/bash
#PJM -L "node=1"               # Number of node
#PJM -L "freq=2200"                         
#PJM -L "rscgrp=small"     # Specify resource group
#PJM -L "elapse=60:00"         # Job run time limit value
#PJM -S     

set -ex


deepmd_root=$HOME/gzq/deepmd-kit
source $deepmd_root/script/fugaku/env.sh
bash $deepmd_root/script/fugaku/build_deepmd.sh

export TF_INTRA_OP_PARALLELISM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=1

export TF_PROFILE=1

rm -f profiler.json_*

export HAVE_PREPROCESSED=1

likwid-pin -c 0 dp test -m ../model/graph-compress-preprocess.pb -s ../data/data_3 -n 1

python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/tools/profiler_visualization_topk.py profiler 
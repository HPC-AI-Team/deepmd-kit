#!/bin/bash
#PJM -L "node=1"               # Number of node
#PJM -L  "freq=2200"                         
#PJM -L "rscgrp=small"     # Specify resource group
#PJM -L "elapse=60:00"         # Job run time limit value
#PJM -S     

set -ex

deepmd_root=$HOME/gzq/deepmd-kit
source $deepmd_root/script/fugaku/env.sh
bash $deepmd_root/script/fugaku/build_deepmd.sh

export HAVE_PREPROCESSED=1
export TF_CPP_MIN_LOG_LEVEL=3

likwid-perfctr -C 0 -g FLOPS_DP lmp -echo screen -in ./in.water_compress_preprocess_1
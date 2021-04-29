#!/bin/bash
#PJM -L "node=1"               # Number of node
#PJM -L  "freq=2200"                         
#PJM -L "rscgrp=small"     # Specify resource group
#PJM -L "elapse=60:00"         # Job run time limit value
#PJM -S     

set -ex

source $deepmd_root/script/fugaku/env.sh
bash $deepmd_root/script/fugaku/build_deepmd.sh

export TF_PROFILE=1

likwid-pin -c 0 dp test -m $deepmd_root/water/model/graph.pb -s $deepmd_root/water/data -n 1

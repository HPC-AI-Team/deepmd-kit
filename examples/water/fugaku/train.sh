#!/bin/bash
#PJM -L "node=1"               # Number of node
#PJM -L  "freq=2200"                         
#PJM -L "rscgrp=small"     # Specify resource group
#PJM -L "elapse=04:00:00"         # Job run time limit value
#PJM -S     

set -ex

deepmd_root=$HOME/gzq/deepmd-kit
source $deepmd_root/script/fugaku/env.sh
bash $deepmd_root/script/fugaku/build_deepmd.sh

export OMP_NUM_THREADS=48
export TF_INTRA_OP_PARALLELISM_THREADS=0
export TF_INTER_OP_PARALLELISM_THREADS=0

dp train ../se_e2_a_tebd/input_1000.json
dp train ../se_e2_a/input_1000.json
dp freeze -o ../model/graph.pb
dp test -m ../model/graph.pb -s ../data/data_3 -n 1

rm -rf model.ckpt.*
rm -rf checkpoint

dp compress ../se_e2_a/input_1000.json -i ../model/graph.pb -o ../model/graph-compress_1.pb

rm -rf model.ckpt.*
rm -rf checkpoint
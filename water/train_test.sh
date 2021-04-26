#!/bin/bash
#PJM -L "node=1"               # Number of node
#PJM -L  "freq=2200"                         
#PJM -L "rscgrp=small"     # Specify resource group
#PJM -L "elapse=01:00:00"         # Job run time limit value
#PJM -S     

set -ex

if [ $deepmd_root == "" ]
then
    echo "not found envoriment variable : deepmd_root"
fi

if [ $lammps_root == "" ]
then
    echo "not found envoriment variable : lammps_root"
fi

source $HOME/gzq/fj_env.sh

bash $deepmd_root/script/build_deepmd.sh

dp train train/water_se_a.json
dp freeze -o model/graph.pb
dp test -m model/graph.pb -s ./data -n 1
rm -rf model.ckpt.*
rm -rf checkpoint

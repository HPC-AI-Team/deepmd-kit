#!/bin/bash
#PJM -L "node=1"               # Number of node
#PJM -L  "freq=2200"                         
#PJM -L "rscgrp=small"     # Specify resource group
#PJM -L "elapse=60:00"         # Job run time limit value
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

source $deepmd_root/script/fugaku/env.sh

bash $deepmd_root/script/fugaku/build_deepmd.sh

likwid-perfctr -C 0 -g FLOPS_DP $lammps_root/src/lmp_serial -echo screen -in ./in.water_1
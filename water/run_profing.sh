#!/bin/bash
#PJM -L "node=1"               # Number of node
#PJM -L "rscgrp=eap-small"     # Specify resource group
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

bash $deepmd_root/script/build_deepmd.sh

export TF_INTRA_OP_PARALLELISM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3

likwid-perfctr -C 0 -g FLOPS_DP likwid-pin -c 0 $lammps_root/src/lmp_serial -echo screen -in ./lmp/in.water_1

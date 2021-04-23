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


$lammps_root/src/lmp_serial -echo screen -in ./lmp/in.water_5

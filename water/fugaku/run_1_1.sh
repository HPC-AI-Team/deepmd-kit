#!/bin/bash

#!/bin/sh -x
#PJM -L  "node=1"                          # Number of assign node 8 (1 dimention format)
#PJM -L  "freq=2200"                         
#PJM -L "rscgrp=small"         # Specify resource group
#PJM -L  "elapse=01:00:00"                 # Elapsed time limit 1 hour
#PJM --mpi "shape=1"
#PJM --mpi "max-proc-per-node=48"          # Maximum number of MPI processes created per node
#PJM -s                                    # Statistical information output

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
export PLE_MPI_STD_EMPTYFILE=off
# export PRINT_TIME=1

$lammps_root/src/lmp_serial -echo screen -in ./in.water_1
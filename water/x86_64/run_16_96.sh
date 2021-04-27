#!/bin/bash

set -ex

export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh

bash $deepmd_root/script/x86_64/build_deepmd.sh

export TF_INTRA_OP_PARALLELISM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=1
# export PRINT_TIME=1

mpirun -n 16 $lammps_root/src/lmp_mpi -echo screen -in ./in.water_96
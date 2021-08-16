#!/bin/bash

set -e

export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh
# bash $deepmd_root/script/x86_64/build_deepmd.sh

export DEEPMD_NUM_THREADS=2
export OMP_NUM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=1
export TF_INTRA_OP_PARALLELISM_THREADS=1
# export PRINT_TIME=1

mpiexec -n 1 lmp_mpi -echo screen -in ../lmp/in.water_4
# mpiexec -n 1 lmp_mpi -echo screen -in ../lmp/in.water_1
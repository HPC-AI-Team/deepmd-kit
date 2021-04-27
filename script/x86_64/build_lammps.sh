#!/bin/bash 

set -ex

export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh

cd $deepmd_root/source/build

make lammps

cd $lammps_root/src

cp -r $deepmd_root/source/build/USER-DEEPMD .

make clean-all
make no-user-deepmd

make yes-user-deepmd
make yes-kspace
make serial -j16
make mpi -j16
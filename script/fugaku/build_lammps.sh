#!/bin/bash 

set -ex

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
    exit -1
fi

source $deepmd_root/script/fugaku/env.sh

cd $deepmd_root/source/build

make lammps

cp -r $deepmd_root/script/fugaku/fugaku_lammps_patch/* $lammps_root/src

cd $lammps_root/src

cp -r $deepmd_root/source/build/USER-DEEPMD .

make clean-all
make no-user-deepmd

make yes-user-deepmd
make yes-kspace
make serial -j16
make mpi -j16


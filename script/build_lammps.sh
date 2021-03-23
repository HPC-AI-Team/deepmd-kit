#!/bin/bash 

set -ex

if [ $deepmd_root == "" ]
then
    echo "not found envoriment variable : deepmd_root"
fi

if [ $lammps_root == "" ]
then
    echo "not found envoriment variable : lammps_root"
fi

cd $deepmd_root/source/build

make lammps

cd $lammps_root/src

make clean-all
make no-user-deepmd
rm -rf ./USER-DEEPMD

cp -r $deepmd_root/source/build/USER-DEEPMD .
make yes-user-deepmd
make yes-kspace
make serial -j16
make mpi -j16
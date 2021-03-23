#!/bin/bash 

set -ex

if [ $deepmd_root == "" ]
then
    echo "not found envoriment variable : deepmd_root"
fi


cd $deepmd_root

mkdir -p source/build
cd source/build

cmake -DXDRFILE_ROOT=$xdrfile_root -DTENSORFLOW_ROOT=$tensorflow_root -DCMAKE_INSTALL_PREFIX=$deepmd_root ..
make -j16
make install 
make lammps





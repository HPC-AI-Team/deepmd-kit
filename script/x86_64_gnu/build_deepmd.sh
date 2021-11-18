#!/bin/bash

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi

source $deepmd_root/script/x86_64_gnu/env.sh

set -ex

cd $deepmd_root
# rm -rf build
mkdir -p build
cd build

cmake   -DTENSORFLOW_ROOT=$tensorflow_root      \
        -DCMAKE_INSTALL_PREFIX=$deepmd_root     \
        -DLAMMPS_SOURCE_ROOT=$lammps_root       \
        ../source

make VERBOSE=1 -j48
make install

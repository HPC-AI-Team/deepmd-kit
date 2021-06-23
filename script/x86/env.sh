#!/bin/bash 


export deepmd_root=$HOME/deepmd-kit
export tensorflow_root=$HOME/software/tensorflow-gpu-2.4



if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi

export LD_LIBRARY_PATH=$deepmd_root/dp/lib:$LD_LIBRARY_PATH
export CPATH=$deepmd_root/dp/include:$CPATH
export PATH=$deepmd_root/dp/bin:$PATH

source $tensorflow_root/env.sh


INSTALL_PREFIX=$deepmd_root/dp
FLOAT_PREC=high
DEEPMD_BUILD_DIR=$deepmd_root/source/build
LAMMPS_BUILD_DIR=$deepmd_root/source/build_lammps

export CC="gcc -Ofast -fopenmp"
export CXX="g++ -Ofast -fopenmp"

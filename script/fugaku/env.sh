#!/bin/bash 


export deepmd_root=$HOME/gzq/deepmd-kit
export tensorflow_root=$HOME/gzq/fj_software/tensorflow/TensorFlow-2.2.0
export lammps_root=$HOME/gzq/deepmd-kit/source/build_lammps/lammps-stable_29Oct2020

alias "build_deempd=$deepmd_root/script/fugaku/build_deepmd.sh"

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi

export LD_LIBRARY_PATH=$deepmd_root/dp/lib:$LD_LIBRARY_PATH
export CPATH=$deepmd_root/dp/include:$CPATH
export PATH=$deepmd_root/dp/bin:$PATH

export PATH=$tensorflow_root/bin:$PATH
export CPATH=$tensorflow_root/include:$CPATH
export LD_LIBRARY_PATH=$tensorflow_root/lib:$LD_LIBRARY_PATH

# lammps
export PATH=$lammps_root/src:$PATH

INSTALL_PREFIX=$deepmd_root/dp
FLOAT_PREC=high
DEEPMD_BUILD_DIR=$deepmd_root/source/build
LAMMPS_BUILD_DIR=$deepmd_root/source/build_lammps

export DP_VARIANT=cpu
export DP_FLOAT_PREC=high

export CC="fcc -Nclang -Ofast -fopenmp -mcpu=a64fx -march=armv8.3-a+sve -D_GLIBCXX_USE_CXX11_ABI=0"
export CXX="FCC -Nclang -Ofast -fopenmp -mcpu=a64fx -march=armv8.3-a+sve -D_GLIBCXX_USE_CXX11_ABI=0"

# export CC="fcc -Nclang -D_GLIBCXX_USE_CXX11_ABI=0"
# export CXX="FCC -Nclang -D_GLIBCXX_USE_CXX11_ABI=0"
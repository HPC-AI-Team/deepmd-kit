#!/bin/bash -e

export deepmd_root=$HOME/deepmd-kit


export tensorflow_root=$HOME/software/tensorflow-gpu-2.4
export lammps_root=$deepmd_root/source/build_lammps/lammps-stable_29Oct2020/

spack load cmake
spack load openblas

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi

# deeppmd path
export LD_LIBRARY_PATH=$deepmd_root/dp/lib:$LD_LIBRARY_PATH
export CPATH=$deepmd_root/dp/include:$CPATH
export PATH=$deepmd_root/dp/bin:$PATH

# lammps path
export PATH=$lammps_root/src:$PATH

# tensorflow path
source $tensorflow_root/env.sh

INSTALL_PREFIX=$deepmd_root/dp
FLOAT_PREC=high
DEEPMD_BUILD_DIR=$deepmd_root/source/build
LAMMPS_BUILD_DIR=$deepmd_root/source/build_lammps

export DP_VARIANT=cuda
export DP_FLOAT_PREC=high

export CUDA_VISIBLE_DEVICES=1

# # gcc
export CC="gcc -Ofast -fopenmp -lopenblas"
export CXX="g++ -Ofast -fopenmp -lopenblas"

# icc
# source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh -arch intel64 -platform linux
# export LD_PRELOAD=/opt/intel/compilers_and_libraries_2020.2.254/linux/mkl/lib/intel64_lin/libmkl_rt.so
# export CC="icc -Ofast -fopenmp -mkl=parallel -liomp5 -lpthread -lm -ldl"
# export CXX="icpc -Ofast -fopenmp -mkl=parallel -liomp5 -lpthread -lm -ldl" 

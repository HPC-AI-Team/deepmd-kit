#!/bin/bash -e

export deepmd_root=$HOME/deepmd-kit


export tensorflow_root=$HOME/software/tensorflow-gpu-2.4
export lammps_root=$deepmd_root/source/build_lammps/lammps-stable_29Oct2020/

source $HOME/software/cmake/env.sh

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

export DP_VARIANT=cpu
export DP_FLOAT_PREC=high

export CUDA_VISIBLE_DEVICES=-1

export LD_PRELOAD=/data/home/guozhuoqiang/software/spack-0.16.2/opt/spack/linux-centos7-haswell/gcc-4.8.5/gcc-7.5.0-apfqefcs5zty75lid2nxwyh5f4uagvtp/lib64/libgomp.so.1
# # gcc
export CC="gcc -Ofast -fopenmp -lopenblas"
export CXX="g++ -Ofast -fopenmp -lopenblas"

# icc
# source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh -arch intel64 -platform linux
# export LD_PRELOAD=/opt/intel/compilers_and_libraries_2020.2.254/linux/mkl/lib/intel64_lin/libmkl_rt.so
# export CC="icc -Ofast -fopenmp -mkl=parallel -liomp5 -lpthread -lm -ldl"
# export CXX="icpc -Ofast -fopenmp -mkl=parallel -liomp5 -lpthread -lm -ldl" 

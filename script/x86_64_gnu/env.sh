#!/bin/bash 

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi

# export tensorflow_root=/vol0004/hp200266/u01036/gzq/fj_software/tensorflow/TensorFlow-2.2.0
export tensorflow_root=$deepmd_root/../dependents/tensorflow-gpu-2.4

lammps_version=stable_29Sep2021
export lammps_root=$deepmd_root/../dependents/lammps-$lammps_version

export LD_LIBRARY_PATH=$deepmd_root/lib:$LD_LIBRARY_PATH
export CPATH=$deepmd_root/include:$CPATH
export PATH=$deepmd_root/bin:$PATH

source $tensorflow_root/env_no_cuda.sh

export DP_VARIANT=cpu

# spack load gcc@7.5.0
spack load cmake
# spack load openmpi
# spack load openblas

# export CC="gcc -Ofast -fopenmp -lopenblas"
# export CXX="g++ -Ofast -fopenmp -lopenblas"

source /opt/intel/parallel_studio_xe_2020/psxevars.sh
# export LD_PRELOAD=/opt/intel/compilers_and_libraries_2020.2.254/linux/mkl/lib/intel64_lin/libmkl_rt.so
export CC="icc -Ofast -fopenmp -mkl=parallel -liomp5 -lpthread -lm -ldl"
export CXX="icpc -Ofast -fopenmp -mkl=parallel -liomp5 -lpthread -lm -ldl" 
#!/bin/bash 


export deepmd_root=/home/hp200266/u01036/gzq/deepmd-kit
export tensorflow_root=/vol0004/hp200266/u01036/gzq/fj_software/tensorflow/TensorFlow-2.2.0
export lammps_root=$deepmd_root/package/lammps-stable_29Oct2020


if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi

export LD_LIBRARY_PATH=$deepmd_root/dp/lib:$LD_LIBRARY_PATH
export CPATH=$deepmd_root/dp/include:$CPATH
export PATH=$deepmd_root/dp/bin:$PATH

export numpy_root=$tensorflow_root/lib/python3.8/site-packages/numpy/core
export PATH=$tensorflow_root/bin:$PATH
export CPATH=$tensorflow_root/include/python3.8:$CPATH
export CPATH=$tensorflow_root/include:$CPATH
export CPATH=$numpy_root/include:$CPATH
export LD_LIBRARY_PATH=$tensorflow_root/lib:$LD_LIBRARY_PATH

# lammps
export PATH=$lammps_root/src:$PATH

INSTALL_PREFIX=$deepmd_root/dp
FLOAT_PREC=high
DEEPMD_BUILD_DIR=$deepmd_root/source/build_lammps
LAMMPS_BUILD_DIR=$deepmd_root/source/build_lammps

export CC="fcc -Nclang -Ofast -fopenmp -ffj-no-largepage -mcpu=a64fx -march=armv8.3-a+sve -D_GLIBCXX_USE_CXX11_ABI=0"
export CXX="FCC -Nclang -Ofast -fopenmp -ffj-no-largepage -mcpu=a64fx -march=armv8.3-a+sve -D_GLIBCXX_USE_CXX11_ABI=0"

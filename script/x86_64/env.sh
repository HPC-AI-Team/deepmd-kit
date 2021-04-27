#!/bin/bash 

set -ex

if [ $deepmd_root == "" ]
then
    echo "not found envoriment variable : deepmd_root"
fi

if [ $tensorflow_root == "" ]
then
    echo "not found envoriment variable : tensorflow_root"
fi

export LD_LIBRARY_PATH=$deepmd_root/lib:$LD_LIBRARY_PATH
export CPATH=$deepmd_root/include:$CPATH
export PATH=$deepmd_root/bin:$PATH

# tensorflow
tf_version=2.2.0
export tensorflow_root=$deepmd_root/package/TensorFlow-$tf_version

export numpy_root=$tensorflow_root/lib/python3.8/site-packages/numpy/core
export PATH=$tensorflow_root/bin:$PATH
export CPATH=$tensorflow_root/include/python3.8:$CPATH
export CPATH=$tensorflow_root/include:$CPATH
export CPATH=$numpy_root/include:$CPATH
export LD_LIBRARY_PATH=$tensorflow_root/lib:$LD_LIBRARY_PATH

# lammps
export lammps_root=$deepmd_root/package/lammps-stable_29Oct2020
export PATH=$lammps_root/src:$PATH


export CC="fcc -Nclang -Kfast -Knolargepage"
export CXX="FCC -Nclang -Kfast -Knolargepage"
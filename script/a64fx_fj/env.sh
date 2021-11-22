#!/bin/bash 

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi

export tensorflow_root=$deepmd_root/../dependents/TensorFlow-2.2.0

lammps_version=stable_29Sep2021
export lammps_root=$deepmd_root/../dependents/lammps-$lammps_version

# deepmd path 
export LD_LIBRARY_PATH=$deepmd_root/lib:$LD_LIBRARY_PATH
export CPATH=$deepmd_root/include:$CPATH
export PATH=$deepmd_root/bin:$PATH

# tensorflow path
export PATH=$tensorflow_root/bin:$PATH
export CPATH=$tensorflow_root/include:$tensorflow_root/lib/python3.8/site-packages/tensorflow/include:$CPATH
export LD_LIBRARY_PATH=$tensorflow_root/lib:$tensorflow_root/lib64:$LD_LIBRARY_PATH

export DP_VARIANT=cpu

export CC="mpifcc -Nclang -Ofast -fopenmp -lfjlapacksve -mcpu=a64fx -march=armv8.3-a+sve -D_GLIBCXX_USE_CXX11_ABI=0"
export CXX="mpiFCC -Nclang -Ofast -fopenmp -lfjlapacksve -mcpu=a64fx -march=armv8.3-a+sve -D_GLIBCXX_USE_CXX11_ABI=0"

# export LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1

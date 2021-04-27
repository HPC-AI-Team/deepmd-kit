#!/bin/bash 

set -ex

export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh

echo tensorflow_root : $tensorflow_root
echo deepmd_root : $deepmd_root

cd $deepmd_root

mkdir -p source/build
cd source/build

cmake -DTENSORFLOW_ROOT=$tensorflow_root -DCMAKE_INSTALL_PREFIX=$deepmd_root -DUSE_CUDA_TOOLKIT=false ..
make -j16
make install 

cd $deepmd_root

python ./setup.py install


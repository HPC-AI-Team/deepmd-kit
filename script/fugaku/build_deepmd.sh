#!/bin/bash 

set -ex

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
    exit -1
fi

source $deepmd_root/script/fugaku/env.sh

cd $deepmd_root

mkdir -p source/build
cd source/build

cmake -DTENSORFLOW_ROOT=$tensorflow_root -DCMAKE_INSTALL_PREFIX=$deepmd_root -DUSE_CUDA_TOOLKIT=false ..
make -j16
make install 

cd $deepmd_root

python ./setup.py install


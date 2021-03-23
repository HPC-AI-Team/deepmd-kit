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

cd $deepmd_root

mkdir -p source/build
cd source/build

echo tensorflow_root : $tensorflow_root
echo deepmd_root : $deepmd_root

cmake -DTENSORFLOW_ROOT=$tensorflow_root -DCMAKE_INSTALL_PREFIX=$deepmd_root -DUSE_CUDA_TOOLKIT=false ..
make -j16
make install 

cd $deepmd_root

python ./setup.py install


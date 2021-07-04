#!/bin/bash -e

deepmd_root=$HOME/gzq/deepmd-kit
source $deepmd_root/script/fugaku/env.sh

cd $deepmd_root

# rm -rf source/build

mkdir -p $DEEPMD_BUILD_DIR
cd $DEEPMD_BUILD_DIR

cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DFLOAT_PREC=${FLOAT_PREC} -DINSTALL_TENSORFLOW=FALSE -DTENSORFLOW_ROOT=$tensorflow_root -DUSE_CUDA_TOOLKIT=FALSE ..

make -j48
make install 

cd $deepmd_root

python ./setup.py install -j48


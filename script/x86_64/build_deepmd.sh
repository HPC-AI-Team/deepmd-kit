#!/bin/bash -e

deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh

cd $deepmd_root

# rm -rf source/build

mkdir -p $DEEPMD_BUILD_DIR
cd $DEEPMD_BUILD_DIR

cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DFLOAT_PREC=${FLOAT_PREC} -DINSTALL_TENSORFLOW=FALSE -DTENSORFLOW_ROOT=$tensorflow_root -DUSE_CUDA_TOOLKIT=FALSE ..
# cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DFLOAT_PREC=${FLOAT_PREC} -DINSTALL_TENSORFLOW=FALSE -DTENSORFLOW_ROOT=$tensorflow_root -DUSE_CUDA_TOOLKIT=TRUE ..

make -j16 VERBOSE=1
make install 

cd $deepmd_root

pip install packaging
pip install scikit-build

export DP_VARIANT=cpu
# export DP_VARIANT=cuda

python ./setup.py install -j48


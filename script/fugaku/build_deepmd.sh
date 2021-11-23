#!/bin/bash
#PJM -L  "node=1"                           # Number of assign node 8 (1 dimention format)
#PJM -L  "freq=2200"                         
#PJM -L  "rscgrp=int"                     # Specify resource group
#PJM -L  "elapse=00:20:00"                  # Elapsed time limit 1 hour
#PJM --mpi "max-proc-per-node=1"            # Maximum number of MPI processes created per node
export PLE_MPI_STD_EMPTYFILE=off

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi
source $deepmd_root/script/fugaku/env.sh

cd $deepmd_root

# rm -rf source/build

mkdir -p $DEEPMD_BUILD_DIR
cd $DEEPMD_BUILD_DIR

cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DFLOAT_PREC=${FLOAT_PREC} -DINSTALL_TENSORFLOW=FALSE -DTENSORFLOW_ROOT=$tensorflow_root -DUSE_CUDA_TOOLKIT=FALSE ..

make -j48 VERBOSE=1
make install 

cd $deepmd_root

# python ./setup.py install -j48


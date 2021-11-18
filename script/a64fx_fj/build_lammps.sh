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

source $deepmd_root/script/a64fx_fj/env.sh

if [ ! -d $lammps_root ]
then
    if [ ! -e "$deepmd_root/../package/${lammps_version}.tar.gz" ]
    then
        wget https://github.com/lammps/lammps/archive/refs/tags/${lammps_version}.tar.gz
        mv ${lammps_version}.tar.gz $deepmd_root/../package
    fi
    cd $deepmd_root/../dependents
    tar -xzvf $deepmd_root/../package/${lammps_version}.tar.gz
fi

cd $lammps_root

# rm -rf build
mkdir -p build
cd build

cmake -D PKG_PLUGIN=ON -D PKG_KSPACE=ON -D LAMMPS_INSTALL_RPATH=ON -D BUILD_SHARED_LIBS=yes -D CMAKE_INSTALL_PREFIX=${deepmd_root} -D CMAKE_INSTALL_LIBDIR=lib -D CMAKE_INSTALL_FULL_LIBDIR=${deepmd_root}/lib ../cmake
make VERBOSE=1 -j48
make install

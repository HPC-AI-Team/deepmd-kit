#!/bin/bash

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi

source $deepmd_root/script/x86_64_gnu/env.sh

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

rm -rf build
mkdir -p build
cd build

cmake -D PKG_PLUGIN=ON -D PKG_KSPACE=ON -D LAMMPS_INSTALL_RPATH=ON -D BUILD_SHARED_LIBS=yes -D CMAKE_INSTALL_PREFIX=${deepmd_root} -D CMAKE_INSTALL_LIBDIR=lib -D CMAKE_INSTALL_FULL_LIBDIR=${deepmd_root}/lib ../cmake
make VERBOSE=1 -j4
make install

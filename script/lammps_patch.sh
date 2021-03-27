#!bin/bash

if [ $deepmd_root == "" ]
then
    echo "not found envoriment variable : deepmd_root"
fi

set -ex

cd $HOME/gzq

package_dir=$HOME/gzq/package/stable_29Oct2020.tar.gz

rm -rf $HOME/gzq/lammps-stable_29Oct2020

tar -xzf $HOME/gzq/package/stable_29Oct2020.tar.gz

cp $deepmd_root/script/fugaku_lammps_patch/* $HOME/gzq/lammps-stable_29Oct2020/src/


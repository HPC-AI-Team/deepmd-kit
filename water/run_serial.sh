#!/bin/bash


if [ $deepmd_root == "" ]
then
    echo "not found envoriment variable : deepmd_root"
fi

if [ $lammps_root == "" ]
then
    echo "not found envoriment variable : lammps_root"
fi

bash $deepmd_root/script/build_deepmd.sh

$lammps_root/src/lmp_serial -echo screen -in ./lmp/in.water

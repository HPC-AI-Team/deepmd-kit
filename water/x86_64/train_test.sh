#!/bin/bash
set -ex

if [ $deepmd_root == "" ]
then
    echo "not found envoriment variable : deepmd_root"
fi

if [ $lammps_root == "" ]
then
    echo "not found envoriment variable : lammps_root"
fi

source $deepmd_root/script/x86_64/env.sh

bash $deepmd_root/script/x86_64/build_deepmd.sh

dp train $deepmd_root/water/train/water_se_a.json
dp freeze -o $deepmd_root/water/model/graph.pb
dp test -m $deepmd_root/water/model/graph.pb -s $deepmd_root/water/data -n 1
rm -rf model.ckpt.*
rm -rf checkpoint

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

dp train train/water_se_a.json
dp freeze -o model/graph.pb
dp test -m model/graph.pb -s ./data -n 1
rm -rf model.ckpt.*
rm -rf checkpoint

#!/bin/bash

set -ex

export if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi
source $deepmd_root/script/x86_64/env.sh
# bash $deepmd_root/script/x86_64/build_deepmd.sh


export TF_PROFILE=1

rm -f profiler.json_*
export CUDA_VISIBLE_DEVICES=1

likwid-pin -c 0 dp test -m ../model/graph-original.pb -s ../data/data_3 -n 1

# python $deepmd_root/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/tools/profiler_visualization_topk.py profiler 
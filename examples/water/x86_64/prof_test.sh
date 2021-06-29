#!/bin/bash

set -ex


export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh
bash $deepmd_root/script/x86_64/build_deepmd.sh

export TF_INTRA_OP_PARALLELISM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=1

export TF_PROFILE=1

rm -f profiler.json_*

likwid-pin -c 0 dp test -m ../model/graph-compress.pb -s ../data/data_3 -n 1

python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/tools/profiler_visualization_topk.py profiler 
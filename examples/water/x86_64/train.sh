#!/bin/bash

set -ex

export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh
bash $deepmd_root/script/x86_64/build_deepmd.sh

dp train ../se_e2_a/input_double_1000.json
dp freeze -o ../model/double/original/graph-original-gemm_tanh_fusion.pb
dp test -m ../model/double/original/graph-original-gemm_tanh_fusion.pb -s ../data/data_3 -n 1


#!/bin/bash

set -ex

export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh
bash $deepmd_root/script/x86_64/build_deepmd.sh

# dp train ../train/input_v2_compat_train_2000.json
# dp freeze -o ../model/double/original/graph-original-gemm_tanh.pb
# dp test -m ../model/double/original/graph-original-gemm_tanh.pb -s ../data/init/cu.fcc.02x02x02/02.md/sys-0032/deepmd -n 1


dp train ../train/input_v2_compat_train_float_2000.json
dp freeze -o ../model/float/graph.pb
dp test -m ../model/float/graph.pb -s ../data/init/cu.fcc.02x02x02/02.md/sys-0032/deepmd -n 1


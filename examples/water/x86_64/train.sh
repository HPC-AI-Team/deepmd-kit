#!/bin/bash

set -ex

export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh
bash $deepmd_root/script/x86_64/build_deepmd.sh

mkdir -p ../model/float

dp train ../train/input_v2_compat_train_float_2000.json
dp freeze -o ../model/float/graph.pb
dp test -m ../model/float/graph.pb -s ../data/data_3 -n 1

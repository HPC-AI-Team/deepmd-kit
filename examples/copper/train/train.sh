#!/bin/bash

set -ex

export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh
# bash $deepmd_root/script/x86_64/build_deepmd.sh

dp train ./input_v2_compat.json
dp freeze -o ./graph.pb
# dp test -m ../model/graph.pb -s ../data/iter.000047/02.fp/data.051 -n 1

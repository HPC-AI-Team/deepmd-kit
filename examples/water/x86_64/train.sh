#!/bin/bash

set -ex

export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh
bash $deepmd_root/script/x86_64/build_deepmd.sh

dp train ../se_e2_a/input_double_100000.json
dp freeze -o ../model/double/graph.pb
dp test -m ../model/double/graph.pb -s ../data/data_3 -n 1

cp ../model/double/graph.pb ../model/double/original/graph-original-baseline.pb

dp compress ../se_e2_a/input_double.json -i ../model/double/original/graph-original-baseline.pb -o ../model/double/compress/graph-compress-baseline.pb

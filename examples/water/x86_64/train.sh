#!/bin/bash

set -ex

export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh
bash $deepmd_root/script/x86_64/build_deepmd.sh

dp train ../se_e2_a/input_1000.json
dp freeze -o ../model/graph.pb
dp test -m ../model/graph.pb -s ../data/data_3 -n 1

rm -rf model.ckpt.*
rm -rf checkpoint

dp compress ../se_e2_a/input_1000.json -i ../model/graph.pb -o ../model/graph-compress.pb

rm -rf model.ckpt.*
rm -rf checkpoint
#!/bin/bash

set -ex

export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh
bash $deepmd_root/script/x86_64/build_deepmd.sh

export CUDA_VISIBLE_DEVICES=1
dp train ../train/input_v2_compat_100000.json
dp freeze -o ../model/double/graph.pb
dp test -m ../model/double/graph.pb -s ../data.init/Al-dpgen/init/al.bcc.02x02x02/02.md/sys-0016/deepmd -n 1


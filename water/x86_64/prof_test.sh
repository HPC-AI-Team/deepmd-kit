#!/bin/bash

set -ex

export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh
bash $deepmd_root/script/x86_64/build_deepmd.sh
export TF_PROFILE=1

likwid-pin -c 0 dp test -m ../model/graph.pb -s ../data -n 1

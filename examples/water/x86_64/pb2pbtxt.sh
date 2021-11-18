#!/bin/bash

source $deepmd_root/script/x86_64/env.sh

python $deepmd_root/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/tools/pb2pbtxt.py    $deepmd_root/../model/water/double/compress/graph-compress-baseline.pb        $deepmd_root/../model/water/double/compress/graph-compress-baseline.pbtxt
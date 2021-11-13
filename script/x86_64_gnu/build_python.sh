#!/bin/bash

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi

source $deepmd_root/script/x86_64_gnu/env.sh

cd $deepmd_root

python ./setup.py install -j48


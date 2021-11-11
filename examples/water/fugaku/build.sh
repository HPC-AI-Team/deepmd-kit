#!/bin/bash

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi

pjsub --interact $deepmd_root/script/a64fx_fj/build_deepmd.sh

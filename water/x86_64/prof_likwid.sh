#!/bin/bash

set -ex

export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh

bash $deepmd_root/script/x86_64/build_deepmd.sh

likwid-perfctr -C 0 -g FLOPS_DP likwid-pin -c 0 $lammps_root/src/lmp_serial -echo screen -in ../lmp/in.water_1

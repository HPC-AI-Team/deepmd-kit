#!/bin/bash


if [ $deepmd_root == "" ]
then
    echo "not found envoriment variable : deepmd_root"
fi

if [ $lammps_root == "" ]
then
    echo "not found envoriment variable : lammps_root"
fi

bash $deepmd_root/script/build_deepmd.sh

run_serial_with_inter_intra(){
    inter=$1
    intra=$2
    echo "run serial with inter : $inter , intra : $intra"
    export TF_INTER_OP_PARALLELISM_THREADS=$inter
    export TF_INTRA_OP_PARALLELISM_THREADS=$intra
    $lammps_root/src/lmp_serial -echo screen -in ./lmp/in.water &> out/out.water_125_${inter}_${intra}
    mv ./profile/profiler.json_0 ./profile/profiler_125_${inter}_${intra}.json_0 
}

run_serial_with_inter_intra 0 0

# run_serial_with_inter_intra 1 1
# run_serial_with_inter_intra 1 2
# run_serial_with_inter_intra 1 4
# run_serial_with_inter_intra 1 8
# run_serial_with_inter_intra 1 16

# run_serial_with_inter_intra 2 1
# run_serial_with_inter_intra 2 2
# run_serial_with_inter_intra 2 4
# run_serial_with_inter_intra 2 8

# run_serial_with_inter_intra 4 1
# run_serial_with_inter_intra 4 2
# run_serial_with_inter_intra 4 4

# run_serial_with_inter_intra 8 1
# run_serial_with_inter_intra 8 2

# run_serial_with_inter_intra 16 1
#!/bin/bash 


export deepmd_root=$HOME/deepmd-kit
export lammps_root=$HOME/lammps-stable_29Oct2020                                                                                                                                                                                                                                           
export tensorflow_root=$HOME/software/tensorflow-2.2.2

if [ $deepmd_root == "" ]
then
    echo "not found envoriment variable : deepmd_root"
fi

if [ $tensorflow_root == "" ]
then
    echo "not found envoriment variable : tensorflow_root"
fi

if [ $lammps_root == "" ]
then
    echo "not found envoriment variable : tensorflow_root"
fi

export LD_LIBRARY_PATH=$deepmd_root/lib:$LD_LIBRARY_PATH
export CPATH=$deepmd_root/include:$CPATH
export PATH=$deepmd_root/bin:$PATH

# tensorflow
tf_version=2.2.2                                                                                                                                                                                                                                                                           
export tensorflow_root=$HOME/software/tensorflow-$tf_version
export CPATH=$tensorflow_root/include:$CPATH
export LD_LIBRARY_PATH=$tensorflow_root/lib:$LD_LIBRARY_PATH
source $tensorflow_root/python_venv/bin/activate

# lammps
export PATH=$lammps_root/src:$PATH
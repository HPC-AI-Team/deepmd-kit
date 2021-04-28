#!/bin/bash 

set -ex

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi

source $deepmd_root/script/fugaku/env.sh

# Copy tensorflow provided by fugaku to your own directory


cd $deepmd_root/package
if [ ! -f "tensorflow_complete.flag" ];then 
    tf_version=2.2.0
    rm -rf ./TensorFlow-$tf_version
    mkdir -p ./TensorFlow-$tf_version
    cd ./TensorFlow-$tf_version
    cp -r /home/apps/oss/TensorFlow-$tf_version/bin .
    cp -r /home/apps/oss/TensorFlow-$tf_version/build .
    cp -r /home/apps/oss/TensorFlow-$tf_version/doc .
    cp -r /home/apps/oss/TensorFlow-$tf_version/include .
    cp -r /home/apps/oss/TensorFlow-$tf_version/lib .
    cp -r /home/apps/oss/TensorFlow-$tf_version/LICENSE .
    cp -r /home/apps/oss/TensorFlow-$tf_version/share .
    cp -r ./lib/python3.8/site-packages/tensorflow/include/* ./include/
    # make flag
    cd ..
    touch "tensorflow_complete.flag"
    echo "install tensorflow complete !!!"
fi

# install pip package
pip install scikit-build setuptools_scm
# compile deepmd
bash $deepmd_root/script/fugaku/build_deepmd.sh


# download and uncompress lammps
cd $deepmd_root/package

if [ ! -f "lammps_complete.flag" ];then
    wget https://github.com/lammps/lammps/archive/refs/tags/stable_29Oct2020.tar.gz
    tar -xzf ./stable_29Oct2020.tar.gz
    touch "lammps_complete.flag"
    echo "install lammps complete !!!"
fi

# compile lammps+deepmd
bash $deepmd_root/script/fugaku/build_lammps.sh

echo "install complete !!!"


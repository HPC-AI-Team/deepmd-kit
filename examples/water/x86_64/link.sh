#!/bin/bash

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi
source $deepmd_root/script/x86_64/env.sh

if [ $# != 2 ]
then
    echo "usage : $0 precision name"
    exit -1
fi

precision=$1
name=$2

echo "precision : $precision"
echo "name : $name"

model_path=$deepmd_root/model/water

type_list=(original compress compress-preprocess)

for type in ${type_list[*]}
do 
    origin_model_path=$model_path/$precision/$type/graph-$type-$name.pb
    link_path=$model_path/graph-$type.pb
    if [ -e $origin_model ]
    then
        ln -sf $origin_model_path $link_path
	echo "$link_path -> $origin_model_path"
    else
        echo "$origin_model_path not exist !!!"
        exit -1
    fi
done

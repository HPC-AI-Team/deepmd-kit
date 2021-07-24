#!/bin/bash


# for file in $(ls run_*.sh)
# do 
#     pjsub $file
# done 


for file in $(ls run_compress_preprocess_4x12x1_*.sh)
do 
    pjsub $file
done 

watch pjstat
#!/bin/bash


# for file in $(ls run_*.sh)
# do 
#     pjsub $file
# done 


# for file in $(ls run_compress_preprocess_4x12x1_*.sh)
# do 
#     pjsub $file
# done 

pjsub run_compress_preprocess_8x6x1_1.sh
# pjsub run_compress_preprocess_8x6x1_48.sh
pjsub run_compress_preprocess_8x6x1_288.sh
pjsub run_compress_preprocess_8x6x1_576.sh
pjsub run_compress_preprocess_8x6x1_864.sh
# pjsub run_compress_preprocess_8x6x1_960.sh

watch pjstat
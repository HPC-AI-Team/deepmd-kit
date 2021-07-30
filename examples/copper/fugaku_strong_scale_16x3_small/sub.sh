#!/bin/bash


# for file in $(ls run_*.sh)
# do 
#     pjsub $file
# done 


pjsub run_compress_preprocess_16x3x5_1008x5.sh
pjsub run_compress_preprocess_16x3x11_1008x5.sh
pjsub run_compress_preprocess_16x3x25_1008x5.sh
pjsub run_compress_preprocess_16x3x57_1008x5.sh
pjsub run_compress_preprocess_16x3x114_1008x5.sh
pjsub run_compress_preprocess_16x3x228_1008x5.sh
pjsub run_compress_preprocess_16x3x456_1008x5.sh

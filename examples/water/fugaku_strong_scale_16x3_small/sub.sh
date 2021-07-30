#!/bin/bash


# for file in $(ls run_*.sh)
# do 
#     pjsub $file
# done 


pjsub run_compress_preprocess_16x3x5_864x5.sh
# pjsub run_compress_preprocess_16x3x11_864x5.sh
# pjsub run_compress_preprocess_16x3x25_864x5.sh
# pjsub run_compress_preprocess_16x3x57_864x5.sh
# pjsub run_compress_preprocess_16x3x114_864x5.sh
# pjsub run_compress_preprocess_16x3x228_864x5.sh
# pjsub run_compress_preprocess_16x3x456_864x5.sh

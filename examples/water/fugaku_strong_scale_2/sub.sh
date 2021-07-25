#!/bin/bash


# for file in $(ls run_*.sh)
# do 
#     pjsub $file
# done 

# pjsub run_compress_preprocess_8x6x50_864x50.sh
pjsub run_compress_preprocess_8x6x110_864x50.sh
pjsub run_compress_preprocess_8x6x250_864x50.sh
pjsub run_compress_preprocess_8x6x570_864x50.sh
pjsub run_compress_preprocess_8x6x1140_864x50.sh
pjsub run_compress_preprocess_8x6x2280_864x50.sh
pjsub run_compress_preprocess_8x6x4560_864x50.sh

# pjsub run_compress_preprocess_16x3x50_864x50.sh
# pjsub run_compress_preprocess_16x3x110_864x50.sh
# pjsub run_compress_preprocess_16x3x250_864x50.sh
# pjsub run_compress_preprocess_16x3x570_864x50.sh
# pjsub run_compress_preprocess_16x3x1140_864x50.sh
# pjsub run_compress_preprocess_16x3x2280_864x50.sh
# pjsub run_compress_preprocess_16x3x4560_864x50.sh

watch pjstat
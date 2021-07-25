#!/bin/bash


# for file in $(ls run_*.sh)
# do 
#     pjsub $file
# done 

# pjsub run_compress_preprocess_4x12x1_960.sh
# pjsub run_compress_preprocess_4x12x2_960.sh
# pjsub run_compress_preprocess_4x12x4_960.sh
# pjsub run_compress_preprocess_4x12x8_960.sh
# pjsub run_compress_preprocess_4x12x16_960.sh
# pjsub run_compress_preprocess_4x12x32_960.sh
# pjsub run_compress_preprocess_4x12x64_960.sh

# pjsub run_compress_preprocess_8x6x1_960.sh
# pjsub run_compress_preprocess_8x6x2_960.sh
# pjsub run_compress_preprocess_8x6x4_960.sh
# pjsub run_compress_preprocess_8x6x8_960.sh
# pjsub run_compress_preprocess_8x6x16_960.sh
# pjsub run_compress_preprocess_8x6x32_960.sh
# pjsub run_compress_preprocess_8x6x64_960.sh

# pjsub run_compress_preprocess_12x4x1_672.sh
# pjsub run_compress_preprocess_12x4x2_672.sh
# pjsub run_compress_preprocess_12x4x4_672.sh
# pjsub run_compress_preprocess_12x4x8_672.sh
# pjsub run_compress_preprocess_12x4x16_672.sh
# pjsub run_compress_preprocess_12x4x32_672.sh
# pjsub run_compress_preprocess_12x4x64_672.sh

# pjsub run_compress_preprocess_16x3x1_864.sh
# pjsub run_compress_preprocess_16x3x2_864.sh
# pjsub run_compress_preprocess_16x3x4_864.sh
# pjsub run_compress_preprocess_16x3x8_864.sh
# pjsub run_compress_preprocess_16x3x16_864.sh
# pjsub run_compress_preprocess_16x3x32_864.sh
# pjsub run_compress_preprocess_16x3x64_864.sh

pjsub run_compress_preprocess_24x2x1_624.sh
pjsub run_compress_preprocess_24x2x2_624.sh
pjsub run_compress_preprocess_24x2x4_624.sh
pjsub run_compress_preprocess_24x2x8_624.sh
pjsub run_compress_preprocess_24x2x16_624.sh
pjsub run_compress_preprocess_24x2x32_624.sh
pjsub run_compress_preprocess_24x2x64_624.sh

pjsub run_compress_preprocess_48x1x1_576.sh
pjsub run_compress_preprocess_48x1x2_576.sh
pjsub run_compress_preprocess_48x1x4_576.sh
pjsub run_compress_preprocess_48x1x8_576.sh
pjsub run_compress_preprocess_48x1x16_576.sh
pjsub run_compress_preprocess_48x1x32_576.sh
pjsub run_compress_preprocess_48x1x64_576.sh

watch pjstat
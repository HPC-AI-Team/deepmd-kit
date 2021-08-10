#!/bin/bash

rm -f *.err
rm -f *.out
rm -f *.stats

# pjsub run_48x1x1_24.sh
# pjsub run_compress_48x1x1_24.sh
pjsub run_compress_preprocess_48x1x1_24.sh

# pjsub run_compress_preprocess_4x12x1_24.sh
# pjsub run_compress_preprocess_8x6x1_24.sh
# pjsub run_compress_preprocess_12x4x1_24.sh
# pjsub run_compress_preprocess_16x3x1_24.sh
# pjsub run_compress_preprocess_24x2x1_24.sh


# watch pjstat

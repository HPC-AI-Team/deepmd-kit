#!/bin/bash

rm -f *.err
rm -f *.stats
rm -f *.out

# pjsub run_48x1x1_48.sh
# pjsub run_compress_48x1x1_48.sh
# pjsub run_compress_preprocess_48x1x1_48.sh

# pjsub run_48x1x1_144.sh
# pjsub run_48x1x1_192.sh
# pjsub run_48x1x1_240.sh
# pjsub run_48x1x1_288.sh

# pjsub run_48x1x1_96.sh
# pjsub run_compress_48x1x1_96.sh
pjsub run_compress_preprocess_48x1x1_96.sh

# pjsub run_compress_preprocess_4x12x1_96.sh
# pjsub run_compress_preprocess_8x6x1_96.sh
# pjsub run_compress_preprocess_12x4x1_96.sh
# pjsub run_compress_preprocess_16x3x1_96.sh
# pjsub run_compress_preprocess_24x2x1_96.sh

# watch pjstat

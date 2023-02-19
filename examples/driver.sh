#!/bin/bash

set -e

rm -rf output/

python stripped_down_tango_example_kinsol_gptune.py \
       --beta 0.4

python stripped_down_tango_example_kinsol_gptune.py \
       --beta 0.282386 --mAA 5 --delay 1

python stripped_down_tango_example_kinsol_gptune.py \
       --beta 0.4 --mAA 3 --delayAA 5

./plot_output.py \
    output/*Fresid.txt

# python stripped_down_tango_example_kinsol_gptune.py \
#        --beta 0.06 --p 10 --IC lin

# python stripped_down_tango_example_kinsol_gptune.py \
#        --beta 0.125355 --p 10 --IC lin

# python stripped_down_tango_example_kinsol_gptune.py \
#        --beta 0.06 --mAA 4 --p 10 --IC lin --delay 30

# python stripped_down_tango_example_kinsol_gptune.py \
#        --beta 0.06 --mAA 5 --adaptmAA --p 10 --IC lin

# ./plot_output.py \
#     output/*Fresid.txt

# ./plot_output.py \
#     output-compare-norms/*Fresid_L2_*.txt \
#     output-compare-norms/*Fresid_Max_*.txt \
#     output-compare-norms/*Fresid_RMS_*.txt \
#     output-compare-norms/*Rresid_L2_*.txt \
#     output-compare-norms/*Rresid_Max_*.txt \
#     output-compare-norms/*Rresid_RMS_*.txt \
#     --legend "L2" "MAX" "RMS" "R-L2" "R-MAX" "R-RMS"

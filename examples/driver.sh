#!/bin/bash

set -e

# ./plot_output.py \
#     output/*Fresid.txt
# exit 0

rm -rf output/

# adapt beta and m
python stripped_down_tango_example_kinsol_gptune.py \
       --p 2 --beta 0.347423 --mAA 9 --delayAA 12 \
       --adapt_mAA --adapt_mAA_factor 39.210692 \
       --adapt_beta --adapt_beta_factor 0.174528 \
       --maxIters 30 --useMaxIters

python stripped_down_tango_example_kinsol_gptune.py \
       --p 2 --beta 0.150308 --mAA 1 \
       --adapt_mAA --adapt_mAA_factor 1.0 \
       --adapt_beta --adapt_beta_factor 0.533333 \
       --maxIters 30 --useMaxIters

# adapt beta
python stripped_down_tango_example_kinsol_gptune.py \
       --p 2 --beta 0.282037 --mAA 7 --delayAA 5 \
       --adapt_beta --adapt_beta_factor 0.450109 \
       --maxIters 30 --useMaxIters

python stripped_down_tango_example_kinsol_gptune.py \
       --p 2 --beta 0.25195 --mAA 10 \
       --adapt_beta --adapt_beta_factor 0.239981 \
       --maxIters 30 --useMaxIters

# adapt m (not useful, opt max = 1)
python stripped_down_tango_example_kinsol_gptune.py \
       --p 2 --beta 0.380332 --mAA 1 --delayAA 2 \
       --adapt_mAA --adapt_mAA_factor 100.0 \
       --maxIters 30 --useMaxIters

python stripped_down_tango_example_kinsol_gptune.py \
       --p 2 --beta 0.349495 --mAA 1 \
       --adapt_mAA --adapt_mAA_factor 100.0 \
       --maxIters 30 --useMaxIters

# fixed beta
python stripped_down_tango_example_kinsol_gptune.py \
       --p 2 --beta 0.431238 --mAA 9 --delayAA 2 \
       --maxIters 30 --useMaxIters

python stripped_down_tango_example_kinsol_gptune.py \
       --p 2 --beta 0.446298 --mAA 3 \
       --maxIters 30 --useMaxIters

python stripped_down_tango_example_kinsol_gptune.py \
       --p 2 --beta 0.43317 \
       --maxIters 30 --useMaxIters

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

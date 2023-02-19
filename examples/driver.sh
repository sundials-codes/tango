#!/bin/bash

set -e

# python stripped_down_tango_example_kinsol_gptune.py \
#        --beta 0.10

# python stripped_down_tango_example_kinsol_gptune.py \
#        --beta 0.20

# python stripped_down_tango_example_kinsol_gptune.py \
#        --beta 0.30

# python stripped_down_tango_example_kinsol_gptune.py \
#        --beta 0.40

# python stripped_down_tango_example_kinsol_gptune.py \
#        --beta 0.50

# python stripped_down_tango_example_kinsol_gptune.py \
#        --beta 0.60

rm -rf output/
rm -rf gptune.db/

python stripped_down_tango_example_kinsol_gptune.py \
       --beta 0.06 --p 10 --IC lin

python stripped_down_tango_example_kinsol_gptune.py \
       --beta 0.125355 --p 10 --IC lin

python stripped_down_tango_example_kinsol_gptune.py \
       --beta 0.06 --mAA 3 --p 10 --IC lin

# python stripped_down_tango_example_kinsol_gptune.py \
#        --beta_adapt --mAA 3

python stripped_down_tango_example_kinsol_gptune.py \
       --beta 0.06 --mAA 5 --adaptmAA --p 10 --IC lin

# python stripped_down_tango_example_kinsol_gptune.py \
#        --beta 0.4 --beta_adapt --mAA 5 --adaptmAA

./plot_output.py \
    output/*Fresid.txt


# ./plot_output.py \
#     output-compare-norms/*Fresid_L2_*.txt \
#     output-compare-norms/*Fresid_Max_*.txt \
#     output-compare-norms/*Fresid_RMS_*.txt \
#     output-compare-norms/*Rresid_L2_*.txt \
#     output-compare-norms/*Rresid_Max_*.txt \
#     output-compare-norms/*Rresid_RMS_*.txt \
#     --legend "L2" "MAX" "RMS" "R-L2" "R-MAX" "R-RMS"

# python stripped_down_tango_example_kinsol_gptune.py \
#        --gptune

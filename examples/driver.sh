#!/bin/bash

# python stripped_down_tango_example_kinsol_2.py \
#        --beta 0.10

# python stripped_down_tango_example_kinsol_2.py \
#        --beta 0.20

# python stripped_down_tango_example_kinsol_2.py \
#        --beta 0.30

# python stripped_down_tango_example_kinsol_2.py \
#        --beta 0.40

# python stripped_down_tango_example_kinsol_2.py \
#        --beta 0.50

# python stripped_down_tango_example_kinsol_2.py \
#        --beta 0.60

# python stripped_down_tango_example_kinsol_2.py \
#        --beta 0.10 --kinsol

# python stripped_down_tango_example_kinsol_2.py \
#        --beta 0.20 --kinsol

# python stripped_down_tango_example_kinsol_2.py \
#        --beta 0.30 --kinsol

# python stripped_down_tango_example_kinsol_2.py \
#        --beta 0.40 --kinsol

# python stripped_down_tango_example_kinsol_2.py \
#        --beta 0.50 --kinsol

# python stripped_down_tango_example_kinsol_2.py \
#        --beta 0.60 --kinsol

rm -rf output/

python stripped_down_tango_example_kinsol_2.py \
       --beta 0.1 --beta_adapt --kinsol --mAA 3 --delayAA 1

python stripped_down_tango_example_kinsol_2.py \
       --beta 0.1 --kinsol --mAA 3 --delayAA 5

python stripped_down_tango_example_kinsol_2.py \
       --beta 0.4 --kinsol

./plot_output.py output/*

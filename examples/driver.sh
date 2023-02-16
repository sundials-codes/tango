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
       --beta 0.4 --kinsol --norm Max

python stripped_down_tango_example_kinsol_2.py \
       --kinsol --beta 0.4 --mAA 3 --delayAA 5  --norm Max

# in the normalized residual plots this on has an edge
python stripped_down_tango_example_kinsol_2.py \
       --kinsol --beta 0.4 --beta_adapt --mAA 3 --delayAA 5  --norm Max

# finally seeing a very slight edge over fixed mAA but NOT in the normalized
# residual plots - may need some adjustment factors
python stripped_down_tango_example_kinsol_2.py \
       --kinsol --beta 0.4 --mAA 5 --adaptmAA --delayAA 5  --norm Max

python stripped_down_tango_example_kinsol_2.py \
       --kinsol --beta 0.4 --beta_adapt --mAA 5 --adaptmAA  --norm Max

./plot_output.py \
    output/*Fresid_Max_history.txt

./plot_output.py \
    output/*Rresid_Max_history.txt

# ./plot_output.py \
#     output/kinsol_p_2.0_alpha_1.0_beta_0.4_adapt-beta_False_adapt-beta-factor_0.5_m_0_delay_0_adapt-m_False_Fresid_RMS_history.txt \
#     output/kinsol_p_2.0_alpha_1.0_beta_0.4_adapt-beta_False_adapt-beta-factor_0.5_m_0_delay_0_adapt-m_False_Rresid_RMS_history.txt

#!/bin/bash

set -e

# ------------
# 1e-11 tuning
# ------------

run_11()
{
    rm -rf output/

    # default
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin \
           --beta 0.06 \
           --maxIters 100 --useMaxIters

    # --- fixed beta ---

    # beta (20)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin \
           --beta 0.523842 \
           --maxIters 100 --useMaxIters

    # beta and m (18)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin \
           --beta 0.881753 --mAA 4 \
           --maxIters 100 --useMaxIters

    # beta, m, and delay (16)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin \
           --beta 0.140555 --mAA 10 --delayAA 40 \
           --maxIters 100 --useMaxIters

    # --- adapt m ---

    # beta and m (17)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin --adapt_mAA \
           --beta 0.079727 --mAA 5 --adapt_mAA_factor 95.945563 \
           --maxIters 100 --useMaxIters

    # beta, m, and delay (17)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin --adapt_mAA \
           --beta 0.092451 --mAA 10 --delayAA 40 --adapt_mAA_factor 1.021288 \
           --maxIters 100 --useMaxIters

    # --- adapt beta ---

    # beta and m (23)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin --adapt_beta \
           --beta 0.002236 --mAA 10 --adapt_beta_factor 0.933427 \
           --maxIters 100 --useMaxIters

    # beta, m, and delay (19)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin --adapt_beta \
           --beta 0.146264 --mAA 10 --delayAA 40 --adapt_beta_factor 0.999784 \
           --maxIters 100 --useMaxIters

    # --- adapt beta and m ---

    # 17
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin --adapt_mAA --adapt_beta \
           --beta 0.644886 --mAA 8 --adapt_mAA_factor 89.082557 --adapt_beta_factor 0.314233 \
           --maxIters 100 --useMaxIters

    # 17
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin --adapt_mAA --adapt_beta \
           --beta 0.882374 --mAA 1 --delayAA 27 --adapt_mAA_factor 73.873326 --adapt_beta_factor 0.773408 \
           --maxIters 100 --useMaxIters

    mv output output_p_10_tol_1.0e-11
}

# ------------
# 1e-8 tuning
# ------------

run_8()
{
    rm -rf output/

    # default
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin \
           --beta 0.06 \
           --maxIters 100 --useMaxIters

    # --- fixed beta ---

    # beta (12)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin \
           --beta 0.13877 \
           --maxIters 100 --useMaxIters

    # beta and m (12)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin \
           --beta 1.2e-05 --mAA 1 \
           --maxIters 100 --useMaxIters

    # beta, m, and delay (12) -- delay removes acceleration
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin \
           --beta 0.121349 --mAA 4 --delayAA 12 \
           --maxIters 100 --useMaxIters

    # --- adapt m ---

    # beta and m (11)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin --adapt_mAA \
           --beta 0.036541 --mAA 1 --adapt_mAA_factor 99.608998 \
           --maxIters 100 --useMaxIters

    # beta, m, and delay (11)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin --adapt_mAA \
           --beta 0.049767 --mAA 7 --delayAA 1 --adapt_mAA_factor 1.0 \
           --maxIters 100 --useMaxIters

    # --- adapt beta ---

    # beta and m (16)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin --adapt_beta \
           --beta 0.775008 --mAA 6 --adapt_beta_factor 0.16105 \
           --maxIters 100 --useMaxIters

    # beta, m, and delay (11)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin --adapt_beta \
           --beta 0.511188 --mAA 9 --delayAA 38 --adapt_beta_factor 0.397119 \
           --maxIters 100 --useMaxIters

    # --- adapt beta and m ---

    # (13)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin --adapt_mAA --adapt_beta \
           --beta 0.401667 --mAA 7 --adapt_mAA_factor 2.88898 --adapt_beta_factor 0.644927 \
           --maxIters 100 --useMaxIters

    # (12) -- delay removes acceleration
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin --adapt_mAA --adapt_beta \
           --beta 0.110625 --mAA 10 --delayAA 40 --adapt_mAA_factor 100.0 --adapt_beta_factor 0.998988 \
           --maxIters 100 --useMaxIters

    mv output output_p_10_tol_1.0e-8
}


# ------------
# 1e-6 tuning
# ------------

run_6()
{
    rm -rf output/

    # default
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin \
           --beta 0.06 \
           --maxIters 100 --useMaxIters

    # --- fixed beta ---

    # beta (9)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin \
           --beta 0.143118 \
           --maxIters 100 --useMaxIters

    # beta and m (8)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin \
           --beta 0.012759 --mAA 3 \
           --maxIters 100 --useMaxIters

    # beta, m, and delay (10) -- delay removes acceleration
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin \
           --beta 8.6e-05 --mAA 1 --delayAA 1 \
           --maxIters 100 --useMaxIters

    # --- adapt m ---

    # beta and m (9)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin --adapt_mAA \
           --beta 0.703276 --mAA 1 --adapt_mAA_factor 3.591444 \
           --maxIters 100 --useMaxIters

    # beta, m, and delay (9)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin --adapt_mAA \
           --beta 0.055041 --mAA 3 --delayAA 20 --adapt_mAA_factor 88.293587 \
           --maxIters 100 --useMaxIters

    # --- adapt beta ---

    # beta and m (14)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin --adapt_beta \
           --beta 0.000278 --mAA 10 --adapt_beta_factor 0.999941 \
           --maxIters 100 --useMaxIters

    # beta, m, and delay (9)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin --adapt_beta \
           --beta 0.141604 --mAA 9 --delayAA 22 --adapt_beta_factor 0.139932 \
           --maxIters 100 --useMaxIters

    # --- adapt beta and m ---

    # (9)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin --adapt_mAA --adapt_beta \
           --beta 0.025248 --mAA 10 --adapt_mAA_factor 93.655498 --adapt_beta_factor 0.929059 \
           --maxIters 100 --useMaxIters

    # (9) -- delay removes acceleration
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 10 --IC lin --adapt_mAA --adapt_beta \
           --beta 0.123857 --mAA 10 --delayAA 35 --adapt_mAA_factor 100.0 --adapt_beta_factor 0.999713 \
           --maxIters 100 --useMaxIters

    mv output output_p_10_tol_1.0e-6
}


plot_compare()
{
    # 1. Optimize beta
    ./plot_output.py \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.06_resid_F.txt \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.523842_resid_F.txt \
        output_p_10_tol_1.0e-8/p_10.0_beta_0.13877_resid_F.txt \
        output_p_10_tol_1.0e-6/p_10.0_beta_0.143118_resid_F.txt

    # 2. Optimize beta and m
    ./plot_output.py \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.06_resid_F.txt \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.881753_m_4_resid_F.txt \
        output_p_10_tol_1.0e-8/p_10.0_beta_1.2e-05_m_1_resid_F.txt \
        output_p_10_tol_1.0e-6/p_10.0_beta_0.012759_m_3_resid_F.txt \

    # 3. Optimize beta, m, and delay
    ./plot_output.py \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.06_resid_F.txt \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.140555_m_10_delay_40_resid_F.txt \
        output_p_10_tol_1.0e-8/p_10.0_beta_0.121349_m_4_delay_12_resid_F.txt \
        output_p_10_tol_1.0e-6/p_10.0_beta_8.6e-05_m_1_delay_1_resid_F.txt

    # 4. Optimize beta and adaptive m
    ./plot_output.py \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.06_resid_F.txt \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.079727_m_5_adapt-m_True_adapt-m-factor_95.945563_resid_F.txt \
        output_p_10_tol_1.0e-8/p_10.0_beta_0.036541_m_1_adapt-m_True_adapt-m-factor_99.608998_resid_F.txt \
        output_p_10_tol_1.0e-6/p_10.0_beta_0.703276_m_1_adapt-m_True_adapt-m-factor_3.591444_resid_F.txt

    ./plot_kinsol_log.py \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.079727_m_5_adapt-m_True_adapt-m-factor_95.945563.log \
        output_p_10_tol_1.0e-8/p_10.0_beta_0.036541_m_1_adapt-m_True_adapt-m-factor_99.608998.log \
        output_p_10_tol_1.0e-6/p_10.0_beta_0.703276_m_1_adapt-m_True_adapt-m-factor_3.591444.log

    # 5. Optimize beta, adaptive m, and delay
    ./plot_output.py \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.06_resid_F.txt \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.092451_m_10_adapt-m_True_adapt-m-factor_1.021288_delay_40_resid_F.txt \
        output_p_10_tol_1.0e-8/p_10.0_beta_0.049767_m_7_adapt-m_True_adapt-m-factor_1.0_delay_1_resid_F.txt \
        output_p_10_tol_1.0e-6/p_10.0_beta_0.055041_m_3_adapt-m_True_adapt-m-factor_88.293587_delay_20_resid_F.txt

    ./plot_kinsol_log.py \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.092451_m_10_adapt-m_True_adapt-m-factor_1.021288_delay_40.log \
        output_p_10_tol_1.0e-8/p_10.0_beta_0.049767_m_7_adapt-m_True_adapt-m-factor_1.0_delay_1.log \
        output_p_10_tol_1.0e-6/p_10.0_beta_0.055041_m_3_adapt-m_True_adapt-m-factor_88.293587_delay_20.log

    # 6. Optimize adaptive beta and m
    ./plot_output.py \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.06_resid_F.txt \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.002236_m_10_adapt-beta_True_adapt-beta-factor_0.933427_resid_F.txt \
        output_p_10_tol_1.0e-8/p_10.0_beta_0.775008_m_6_adapt-beta_True_adapt-beta-factor_0.16105_resid_F.txt \
        output_p_10_tol_1.0e-6/p_10.0_beta_0.000278_m_10_adapt-beta_True_adapt-beta-factor_0.999941_resid_F.txt

    ./plot_kinsol_log.py \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.002236_m_10_adapt-beta_True_adapt-beta-factor_0.933427.log \
        output_p_10_tol_1.0e-8/p_10.0_beta_0.775008_m_6_adapt-beta_True_adapt-beta-factor_0.16105.log \
        output_p_10_tol_1.0e-6/p_10.0_beta_0.000278_m_10_adapt-beta_True_adapt-beta-factor_0.999941.log

    # 7. Optimize adaptive beta, m, and delay
    ./plot_output.py \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.06_resid_F.txt \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.146264_m_10_adapt-beta_True_adapt-beta-factor_0.999784_delay_40_resid_F.txt \
        output_p_10_tol_1.0e-8/p_10.0_beta_0.511188_m_9_adapt-beta_True_adapt-beta-factor_0.397119_delay_38_resid_F.txt \
        output_p_10_tol_1.0e-6/p_10.0_beta_0.141604_m_9_adapt-beta_True_adapt-beta-factor_0.139932_delay_22_resid_F.txt

    ./plot_kinsol_log.py \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.146264_m_10_adapt-beta_True_adapt-beta-factor_0.999784_delay_40.log \
        output_p_10_tol_1.0e-8/p_10.0_beta_0.511188_m_9_adapt-beta_True_adapt-beta-factor_0.397119_delay_38.log \
        output_p_10_tol_1.0e-6/p_10.0_beta_0.141604_m_9_adapt-beta_True_adapt-beta-factor_0.139932_delay_22.log

    # 8. Optimize adaptive beta and adaptive m
    ./plot_output.py \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.06_resid_F.txt \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.644886_m_8_adapt-m_True_adapt-m-factor_89.082557_adapt-beta_True_adapt-beta-factor_0.314233_resid_F.txt \
        output_p_10_tol_1.0e-8/p_10.0_beta_0.401667_m_7_adapt-m_True_adapt-m-factor_2.88898_adapt-beta_True_adapt-beta-factor_0.644927_resid_F.txt \
        output_p_10_tol_1.0e-6/p_10.0_beta_0.025248_m_10_adapt-m_True_adapt-m-factor_93.655498_adapt-beta_True_adapt-beta-factor_0.929059_resid_F.txt

    ./plot_kinsol_log.py \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.644886_m_8_adapt-m_True_adapt-m-factor_89.082557_adapt-beta_True_adapt-beta-factor_0.314233.log \
        output_p_10_tol_1.0e-8/p_10.0_beta_0.401667_m_7_adapt-m_True_adapt-m-factor_2.88898_adapt-beta_True_adapt-beta-factor_0.644927.log \
        output_p_10_tol_1.0e-6/p_10.0_beta_0.025248_m_10_adapt-m_True_adapt-m-factor_93.655498_adapt-beta_True_adapt-beta-factor_0.929059.log

    # 9. Optimize adaptive beta and adaptive m
    ./plot_output.py \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.06_resid_F.txt \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.882374_m_1_adapt-m_True_adapt-m-factor_73.873326_adapt-beta_True_adapt-beta-factor_0.773408_delay_27_resid_F.txt \
        output_p_10_tol_1.0e-8/p_10.0_beta_0.110625_m_10_adapt-m_True_adapt-m-factor_100.0_adapt-beta_True_adapt-beta-factor_0.998988_delay_40_resid_F.txt \
        output_p_10_tol_1.0e-6/p_10.0_beta_0.123857_m_10_adapt-m_True_adapt-m-factor_100.0_adapt-beta_True_adapt-beta-factor_0.999713_delay_35_resid_F.txt

    ./plot_kinsol_log.py \
        output_p_10_tol_1.0e-11/p_10.0_beta_0.882374_m_1_adapt-m_True_adapt-m-factor_73.873326_adapt-beta_True_adapt-beta-factor_0.773408_delay_27.log \
        output_p_10_tol_1.0e-8/p_10.0_beta_0.110625_m_10_adapt-m_True_adapt-m-factor_100.0_adapt-beta_True_adapt-beta-factor_0.998988_delay_40.log \
        output_p_10_tol_1.0e-6/p_10.0_beta_0.123857_m_10_adapt-m_True_adapt-m-factor_100.0_adapt-beta_True_adapt-beta-factor_0.999713_delay_35.log
}

# Generate Plot Data
# run_11
# run_8
# run_6

# ./plot_output.py \
#     output_p_10_tol_1.0e-6/*F.txt
# exit 0

# Compare different configurations
# plot_compare

# "Best" methods from above
# ./plot_output.py \
#     output_p_10_tol_1.0e-11/p_10.0_beta_0.06_resid_F.txt \
#     output_p_10_tol_1.0e-6/p_10.0_beta_0.143118_resid_F.txt \
#     output_p_10_tol_1.0e-6/p_10.0_beta_0.012759_m_3_resid_F.txt \
#     output_p_10_tol_1.0e-11/p_10.0_beta_0.140555_m_10_delay_40_resid_F.txt \
#     output_p_10_tol_1.0e-8/p_10.0_beta_0.121349_m_4_delay_12_resid_F.txt \
#     output_p_10_tol_1.0e-11/p_10.0_beta_0.079727_m_5_adapt-m_True_adapt-m-factor_95.945563_resid_F.txt \
#     output_p_10_tol_1.0e-11/p_10.0_beta_0.092451_m_10_adapt-m_True_adapt-m-factor_1.021288_delay_40_resid_F.txt \
#     output_p_10_tol_1.0e-8/p_10.0_beta_0.049767_m_7_adapt-m_True_adapt-m-factor_1.0_delay_1_resid_F.txt \
#     output_p_10_tol_1.0e-6/p_10.0_beta_0.123857_m_10_adapt-m_True_adapt-m-factor_100.0_adapt-beta_True_adapt-beta-factor_0.999713_delay_35_resid_F.txt

# Down select from the "Best"
./plot_output.py \
    output_p_10_tol_1.0e-11/p_10.0_beta_0.06_resid_F.txt \
    output_p_10_tol_1.0e-6/p_10.0_beta_0.143118_resid_F.txt \
    output_p_10_tol_1.0e-6/p_10.0_beta_0.012759_m_3_resid_F.txt \
    output_p_10_tol_1.0e-11/p_10.0_beta_0.140555_m_10_delay_40_resid_F.txt \
    output_p_10_tol_1.0e-8/p_10.0_beta_0.121349_m_4_delay_12_resid_F.txt \
    output_p_10_tol_1.0e-8/p_10.0_beta_0.049767_m_7_adapt-m_True_adapt-m-factor_1.0_delay_1_resid_F.txt \
    --save

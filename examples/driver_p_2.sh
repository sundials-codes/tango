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
           --p 2 --IC pow \
           --beta 0.3 \
           --maxIters 30 --useMaxIters

    # --- fixed beta ---

    # beta (20)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow \
           --beta 0.428274 \
           --maxIters 30 --useMaxIters

    # beta and m (18)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow \
           --beta 0.127969 --mAA 8 \
           --maxIters 30 --useMaxIters

    # beta, m, and delay (16)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow \
           --beta 0.318384 --mAA 5 --delayAA 1 \
           --maxIters 30 --useMaxIters

    # --- adapt m ---

    # beta and m (17)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow --adapt_mAA \
           --beta 0.43463 --mAA 5 --adapt_mAA_factor 1.0 \
           --maxIters 30 --useMaxIters

    # beta, m, and delay (17)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow --adapt_mAA \
           --beta 0.454919 --mAA 10 --delayAA 8 --adapt_mAA_factor 1.0 \
           --maxIters 30 --useMaxIters

    # --- adapt beta ---

    # beta and m (23)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow --adapt_beta \
           --beta 0.218874 --mAA 10 --adapt_beta_factor 0.016974 \
           --maxIters 30 --useMaxIters

    # beta, m, and delay (19)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow --adapt_beta \
           --beta 0.398877 --mAA 5 --delayAA 4 --adapt_beta_factor 0.900131 \
           --maxIters 30 --useMaxIters

    # --- adapt beta and m ---

    # 17
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow --adapt_mAA --adapt_beta \
           --beta 0.470486 --mAA 1 --adapt_mAA_factor 48.563464 --adapt_beta_factor 0.599727 \
           --maxIters 30 --useMaxIters

    # 17
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow --adapt_mAA --adapt_beta \
           --beta 0.471083 --mAA 1 --delayAA 2 --adapt_mAA_factor 1.0 --adapt_beta_factor 0.604706 \
           --maxIters 30 --useMaxIters

    mv output output_p_2_tol_1.0e-11
}

# ------------
# 1e-8 tuning
# ------------

run_8()
{
    rm -rf output/

    # default
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow \
           --beta 0.3 \
           --maxIters 30 --useMaxIters

    # --- fixed beta ---

    # beta (12)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow \
           --beta 0.398905 \
           --maxIters 30 --useMaxIters

    # beta and m (12)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow \
           --beta 0.212943 --mAA 3 \
           --maxIters 30 --useMaxIters

    # beta, m, and delay (12) -- delay removes acceleration
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow \
           --beta 0.410807 --mAA 8 --delayAA 17 \
           --maxIters 30 --useMaxIters

    # --- adapt m ---

    # beta and m (11)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow --adapt_mAA \
           --beta 0.371451 --mAA 1 --adapt_mAA_factor 84.200901 \
           --maxIters 30 --useMaxIters

    # beta, m, and delay (11)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow --adapt_mAA \
           --beta 0.379135 --mAA 8 --delayAA 40 --adapt_mAA_factor 100.0 \
           --maxIters 30 --useMaxIters

    # --- adapt beta ---

    # beta and m (16)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow --adapt_beta \
           --beta 0.261884 --mAA 4 --adapt_beta_factor 0.174546 \
           --maxIters 30 --useMaxIters

    # beta, m, and delay (11)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow --adapt_beta \
           --beta 0.378612 --mAA 1 --delayAA 40 --adapt_beta_factor 0.999871 \
           --maxIters 30 --useMaxIters

    # --- adapt beta and m ---

    # (13)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow --adapt_mAA --adapt_beta \
           --beta 0.447478 --mAA 1 --adapt_mAA_factor 99.999997 --adapt_beta_factor 0.782052 \
           --maxIters 30 --useMaxIters

    # (12) -- delay removes acceleration
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow --adapt_mAA --adapt_beta \
           --beta 0.388333 --mAA 9 --delayAA 27 --adapt_mAA_factor 29.420087 --adapt_beta_factor 0.782349 \
           --maxIters 30 --useMaxIters

    mv output output_p_2_tol_1.0e-8
}


# ------------
# 1e-6 tuning
# ------------

run_6()
{
    rm -rf output/

    # default
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow \
           --beta 0.3 \
           --maxIters 30 --useMaxIters

    # --- fixed beta ---

    # beta (9)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow \
           --beta 0.385962 \
           --maxIters 30 --useMaxIters

    # beta and m (8)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow \
           --beta 0.146448 --mAA 3 \
           --maxIters 30 --useMaxIters

    # beta, m, and delay (10) -- delay removes acceleration
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow \
           --beta 0.404793 --mAA 4 --delayAA 27 \
           --maxIters 30 --useMaxIters

    # --- adapt m ---

    # beta and m (9)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow --adapt_mAA \
           --beta 0.366449 --mAA 3 --adapt_mAA_factor 90.89083 \
           --maxIters 30 --useMaxIters

    # beta, m, and delay (9)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow --adapt_mAA \
           --beta 0.364744 --mAA 1 --delayAA 1 --adapt_mAA_factor 52.277289 \
           --maxIters 30 --useMaxIters

    # --- adapt beta ---

    # beta and m (14)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow --adapt_beta \
           --beta 0.410391 --mAA 2 --adapt_beta_factor 0.930117 \
           --maxIters 30 --useMaxIters

    # beta, m, and delay (9)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow --adapt_beta \
           --beta 0.134305 --mAA 10 --delayAA 4 --adapt_beta_factor 0.242516 \
           --maxIters 30 --useMaxIters

    # --- adapt beta and m ---

    # (9)
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow --adapt_mAA --adapt_beta \
           --beta 0.382421 --mAA 1 --adapt_mAA_factor 95.910891 --adapt_beta_factor 0.999974 \
           --maxIters 30 --useMaxIters

    # (9) -- delay removes acceleration
    python stripped_down_tango_example_kinsol_gptune.py \
           --p 2 --IC pow --adapt_mAA --adapt_beta \
           --beta 0.369516 --mAA 4 --delayAA 39 --adapt_mAA_factor 75.179673 --adapt_beta_factor 0.764456 \
           --maxIters 30 --useMaxIters

    mv output output_p_2_tol_1.0e-6
}


plot_compare()
{
    # 1. Optimize beta
    ./plot_output.py \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.3_resid_F.txt \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.428274_resid_F.txt \
        output_p_2_tol_1.0e-8/p_2.0_beta_0.398905_resid_F.txt \
        output_p_2_tol_1.0e-6/p_2.0_beta_0.385962_resid_F.txt

    # 2. Optimize beta and m
    ./plot_output.py \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.3_resid_F.txt \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.127969_m_8_resid_F.txt \
        output_p_2_tol_1.0e-8/p_2.0_beta_0.212943_m_3_resid_F.txt \
        output_p_2_tol_1.0e-6/p_2.0_beta_0.146448_m_3_resid_F.txt

    # 3. Optimize beta, m, and delay
    ./plot_output.py \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.3_resid_F.txt \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.318384_m_5_delay_1_resid_F.txt \
        output_p_2_tol_1.0e-8/p_2.0_beta_0.410807_m_8_delay_17_resid_F.txt \
        output_p_2_tol_1.0e-6/p_2.0_beta_0.404793_m_4_delay_27_resid_F.txt

    # 4. Optimize beta and adaptive m
    ./plot_output.py \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.3_resid_F.txt \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.43463_m_5_adapt-m_True_adapt-m-factor_1.0_resid_F.txt \
        output_p_2_tol_1.0e-8/p_2.0_beta_0.371451_m_1_adapt-m_True_adapt-m-factor_84.200901_resid_F.txt \
        output_p_2_tol_1.0e-6/p_2.0_beta_0.366449_m_3_adapt-m_True_adapt-m-factor_90.89083_resid_F.txt

    ./plot_kinsol_log.py \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.43463_m_5_adapt-m_True_adapt-m-factor_1.0.log \
        output_p_2_tol_1.0e-8/p_2.0_beta_0.371451_m_1_adapt-m_True_adapt-m-factor_84.200901.log \
        output_p_2_tol_1.0e-6/p_2.0_beta_0.366449_m_3_adapt-m_True_adapt-m-factor_90.89083.log

    # 5. Optimize beta, adaptive m, and delay
    ./plot_output.py \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.3_resid_F.txt \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.454919_m_10_adapt-m_True_adapt-m-factor_1.0_delay_8_resid_F.txt \
        output_p_2_tol_1.0e-8/p_2.0_beta_0.379135_m_8_adapt-m_True_adapt-m-factor_100.0_delay_40_resid_F.txt \
        output_p_2_tol_1.0e-6/p_2.0_beta_0.364744_m_1_adapt-m_True_adapt-m-factor_52.277289_delay_1_resid_F.txt

    ./plot_kinsol_log.py \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.454919_m_10_adapt-m_True_adapt-m-factor_1.0_delay_8.log \
        output_p_2_tol_1.0e-8/p_2.0_beta_0.379135_m_8_adapt-m_True_adapt-m-factor_100.0_delay_40.log \
        output_p_2_tol_1.0e-6/p_2.0_beta_0.364744_m_1_adapt-m_True_adapt-m-factor_52.277289_delay_1.log

    # 6. Optimize adaptive beta and m
    ./plot_output.py \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.3_resid_F.txt \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.218874_m_10_adapt-beta_True_adapt-beta-factor_0.016974_resid_F.txt \
        output_p_2_tol_1.0e-8/p_2.0_beta_0.261884_m_4_adapt-beta_True_adapt-beta-factor_0.174546_resid_F.txt \
        output_p_2_tol_1.0e-6/p_2.0_beta_0.410391_m_2_adapt-beta_True_adapt-beta-factor_0.930117_resid_F.txt

    ./plot_kinsol_log.py \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.218874_m_10_adapt-beta_True_adapt-beta-factor_0.016974.log \
        output_p_2_tol_1.0e-8/p_2.0_beta_0.261884_m_4_adapt-beta_True_adapt-beta-factor_0.174546.log \
        output_p_2_tol_1.0e-6/p_2.0_beta_0.410391_m_2_adapt-beta_True_adapt-beta-factor_0.930117.log

    # 7. Optimize adaptive beta, m, and delay
    ./plot_output.py \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.3_resid_F.txt \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.398877_m_5_adapt-beta_True_adapt-beta-factor_0.900131_delay_4_resid_F.txt \
        output_p_2_tol_1.0e-8/p_2.0_beta_0.378612_m_1_adapt-beta_True_adapt-beta-factor_0.999871_delay_40_resid_F.txt \
        output_p_2_tol_1.0e-6/p_2.0_beta_0.134305_m_10_adapt-beta_True_adapt-beta-factor_0.242516_delay_4_resid_F.txt

    ./plot_kinsol_log.py \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.398877_m_5_adapt-beta_True_adapt-beta-factor_0.900131_delay_4.log \
        output_p_2_tol_1.0e-8/p_2.0_beta_0.378612_m_1_adapt-beta_True_adapt-beta-factor_0.999871_delay_40.log \
        output_p_2_tol_1.0e-6/p_2.0_beta_0.134305_m_10_adapt-beta_True_adapt-beta-factor_0.242516_delay_4.log

    # 8. Optimize adaptive beta and adaptive m
    ./plot_output.py \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.3_resid_F.txt \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.470486_m_1_adapt-m_True_adapt-m-factor_48.563464_adapt-beta_True_adapt-beta-factor_0.599727_resid_F.txt \
        output_p_2_tol_1.0e-8/p_2.0_beta_0.447478_m_1_adapt-m_True_adapt-m-factor_99.999997_adapt-beta_True_adapt-beta-factor_0.782052_resid_F.txt \
        output_p_2_tol_1.0e-6/p_2.0_beta_0.382421_m_1_adapt-m_True_adapt-m-factor_95.910891_adapt-beta_True_adapt-beta-factor_0.999974_resid_F.txt

    ./plot_kinsol_log.py \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.470486_m_1_adapt-m_True_adapt-m-factor_48.563464_adapt-beta_True_adapt-beta-factor_0.599727.log \
        output_p_2_tol_1.0e-8/p_2.0_beta_0.447478_m_1_adapt-m_True_adapt-m-factor_99.999997_adapt-beta_True_adapt-beta-factor_0.782052.log \
        output_p_2_tol_1.0e-6/p_2.0_beta_0.382421_m_1_adapt-m_True_adapt-m-factor_95.910891_adapt-beta_True_adapt-beta-factor_0.999974.log

    # 9. Optimize adaptive beta and adaptive m
    ./plot_output.py \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.3_resid_F.txt \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.471083_m_1_adapt-m_True_adapt-m-factor_1.0_adapt-beta_True_adapt-beta-factor_0.604706_delay_2_resid_F.txt \
        output_p_2_tol_1.0e-8/p_2.0_beta_0.388333_m_9_adapt-m_True_adapt-m-factor_29.420087_adapt-beta_True_adapt-beta-factor_0.782349_delay_27_resid_F.txt \
        output_p_2_tol_1.0e-6/p_2.0_beta_0.369516_m_4_adapt-m_True_adapt-m-factor_75.179673_adapt-beta_True_adapt-beta-factor_0.764456_delay_39_resid_F.txt

    ./plot_kinsol_log.py \
        output_p_2_tol_1.0e-11/p_2.0_beta_0.471083_m_1_adapt-m_True_adapt-m-factor_1.0_adapt-beta_True_adapt-beta-factor_0.604706_delay_2.log \
        output_p_2_tol_1.0e-8/p_2.0_beta_0.388333_m_9_adapt-m_True_adapt-m-factor_29.420087_adapt-beta_True_adapt-beta-factor_0.782349_delay_27.log \
        output_p_2_tol_1.0e-6/p_2.0_beta_0.369516_m_4_adapt-m_True_adapt-m-factor_75.179673_adapt-beta_True_adapt-beta-factor_0.764456_delay_39.log
}

# Generate Plot Data
# run_11
# run_8
# run_6

# Compare different configurations
# plot_compare

# "Best" methods from above
./plot_output.py \
    output_p_2_tol_1.0e-11/p_2.0_beta_0.3_resid_F.txt \
    output_p_2_tol_1.0e-8/p_2.0_beta_0.398905_resid_F.txt \
    output_p_2_tol_1.0e-11/p_2.0_beta_0.127969_m_8_resid_F.txt \
    output_p_2_tol_1.0e-11/p_2.0_beta_0.318384_m_5_delay_1_resid_F.txt \
    output_p_2_tol_1.0e-6/p_2.0_beta_0.366449_m_3_adapt-m_True_adapt-m-factor_90.89083_resid_F.txt \
    output_p_2_tol_1.0e-6/p_2.0_beta_0.364744_m_1_adapt-m_True_adapt-m-factor_52.277289_delay_1_resid_F.txt \
    output_p_2_tol_1.0e-6/p_2.0_beta_0.134305_m_10_adapt-beta_True_adapt-beta-factor_0.242516_delay_4_resid_F.txt \
    output_p_2_tol_1.0e-11/p_2.0_beta_0.470486_m_1_adapt-m_True_adapt-m-factor_48.563464_adapt-beta_True_adapt-beta-factor_0.599727_resid_F.txt \
    output_p_2_tol_1.0e-11/p_2.0_beta_0.471083_m_1_adapt-m_True_adapt-m-factor_1.0_adapt-beta_True_adapt-beta-factor_0.604706_delay_2_resid_F.txt

# Down select from the "Best"
./plot_output.py \
    output_p_2_tol_1.0e-11/p_2.0_beta_0.3_resid_F.txt \
    output_p_2_tol_1.0e-8/p_2.0_beta_0.398905_resid_F.txt \
    output_p_2_tol_1.0e-11/p_2.0_beta_0.318384_m_5_delay_1_resid_F.txt \
    output_p_2_tol_1.0e-6/p_2.0_beta_0.366449_m_3_adapt-m_True_adapt-m-factor_90.89083_resid_F.txt \
    output_p_2_tol_1.0e-6/p_2.0_beta_0.134305_m_10_adapt-beta_True_adapt-beta-factor_0.242516_delay_4_resid_F.txt

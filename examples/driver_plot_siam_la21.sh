# # paramter options
# power=2
# noise="no-noise"

# outdir=output_kinsol_gfun-p_${noise}_p_${power}_N_500_IC_orig_maxit_200

# # reference
# ref=${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_*_beta_1.0_gamma_1.0_m_0_delay_0_damp_1.0_residual_error_history.txt

# # no delay
# ./plot_heatmap.py \
#     ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_*_delay_0_damp_*_residual_error_history.txt \
#     --rthresh 1.0e-11 --normalize_idx 0 2 --save

# # delay = 1
# ./plot_heatmap.py \
#     ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_0_delay_0_damp_*_residual_error_history.txt \
#     ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_*_delay_1_damp_*_residual_error_history.txt \
#     --rthresh 1.0e-11 --normalize_idx 0 2 --save

# # delay = 1
# ./plot_heatmap.py \
#     ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_0_delay_0_damp_*_residual_error_history.txt \
#     ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_*_delay_2_damp_*_residual_error_history.txt \
#     --rthresh 1.0e-11 --normalize_idx 0 2 --save

# # delay = 1
# ./plot_heatmap.py \
#     ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_0_delay_0_damp_*_residual_error_history.txt \
#     ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_*_delay_3_damp_*_residual_error_history.txt \
#     --rthresh 1.0e-11 --normalize_idx 0 2 --save

# # delay = 1
# ./plot_heatmap.py \
#     ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_0_delay_0_damp_*_residual_error_history.txt \
#     ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_*_delay_4_damp_*_residual_error_history.txt \
#     --rthresh 1.0e-11 --normalize_idx 0 2 --save

# # delay = 5
# ./plot_heatmap.py \
#     ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_0_delay_0_damp_*_residual_error_history.txt \
#     ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_*_delay_5_damp_*_residual_error_history.txt \
#     --rthresh 1.0e-11 --normalize_idx 0 2 --save

# # delay = 6
# ./plot_heatmap.py \
#     ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_0_delay_0_damp_*_residual_error_history.txt \
#     ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_*_delay_6_damp_*_residual_error_history.txt \
#     --rthresh 1.0e-11 --normalize_idx 0 2 --save


# paramter options
# power=2
# noise="add-noise"
# delay=5

# outdir=output_kinsol_gfun-p_${noise}_p_${power}_N_500_IC_orig_maxit_200

# ./plot_heatmap.py \
#     ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_*_delay_0_damp_*_residual_error_history.txt \
#     ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_*_delay_${delay}_damp_*_residual_error_history.txt \
#     --rthresh 5e-3 --normalize_idx 0 2 --nrm_min 7 --nrm_max 14 --save

# for i in 1 2 3 4 5 6; do
#     ./plot_heatmap.py \
#         ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_0_delay_0_damp_*_residual_error_history.txt \
#         ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_*_delay_${i}_damp_*_residual_error_history.txt \
#         --save --rthresh 1e-4 --normalize_idx 0 2
# done

#--rthresh 5e-3 --normalize_idx 0 2 --nrm_min 7 --nrm_max 14 --save

# paramter options
power=2
noise="add-noise"
# delay=5

outdir=output_kinsol_gfun-p_${noise}_p_${power}_N_500_IC_orig_maxit_200

# ./plot_history.py \
#     ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_0_delay_0_damp_*_residual_error_history.txt \
#     --save

# for j in 1 2 3 4 5; do
#     for i in 0 1 2 3 4 5 6; do
#         ./plot_history.py \
#             ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_${j}_delay_${i}_damp_*_residual_error_history.txt \
#             --save --maxiter 25
#     done
# done

./plot_history.py \
    ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_0_delay_0_damp_0.3_residual_error_history.txt \
    ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_0_delay_0_damp_0.4_residual_error_history.txt \
    ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_1_delay_3_damp_0.3_residual_error_history.txt \
    ${outdir}/kinsol_gfun-p_${noise}_p_${power}_alpha_1.0_beta_1.0_gamma_1.0_m_1_delay_3_damp_0.4_residual_error_history.txt \
    --maxiter 15 --legendtitle None --legend "b = 0.3, m = 0" "b = 0.3, m = 1" "b = 0.4, m = 0" "b = 0.4, m = 1" --linestyle dashed solid dashed solid --save

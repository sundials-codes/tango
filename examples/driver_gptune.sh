#!/bin/bash

set -e
set -o pipefail

rm -rf gptune.db/
output_dir="gptune-output"

power=2

# -------------
# No adaptivity
# -------------

test_name="p_${power}_beta"
python stripped_down_tango_example_kinsol_gptune.py \
       --p ${power} --gptune beta \
       2>&1 | tee "${test_name}.out"

mkdir -p "${output_dir}/${test_name}"
mv gptune.db "${test_name}.out" "${output_dir}/${test_name}/."

test_name="p_${power}_beta_mAA"
python stripped_down_tango_example_kinsol_gptune.py \
       --p ${power} --gptune beta mAA \
       2>&1 | tee "${test_name}.out"

mkdir -p "${output_dir}/${test_name}"
mv gptune.db "${test_name}.out" "${output_dir}/${test_name}/."

test_name="p_${power}_beta_mAA_delayAA"
python stripped_down_tango_example_kinsol_gptune.py \
       --p ${power} --gptune beta mAA delayAA \
       2>&1 | tee "${test_name}.out"

mkdir -p "${output_dir}/${test_name}"
mv gptune.db "${test_name}.out" "${output_dir}/${test_name}/."

# -------------
# Adaptive beta
# -------------

test_name="p_${power}_adapt-beta_mAA"
python stripped_down_tango_example_kinsol_gptune.py \
       --p ${power} --adapt_beta --gptune beta mAA adapt_beta_factor \
       2>&1 | tee "${test_name}.out"

mkdir -p "${output_dir}/${test_name}"
mv gptune.db "${test_name}.out" "${output_dir}/${test_name}/."

test_name="p_${power}_adapt-beta_mAA_delayAA"
python stripped_down_tango_example_kinsol_gptune.py \
       --p ${power} --adapt_beta --gptune beta mAA adapt_beta_factor delayAA \
       2>&1 | tee "${test_name}.out"

mkdir -p "${output_dir}/${test_name}"
mv gptune.db "${test_name}.out" "${output_dir}/${test_name}/."

# -------------
# Adaptive m
# -------------

test_name="p_${power}_beta_adapt-mAA"
python stripped_down_tango_example_kinsol_gptune.py \
       --p ${power} --adapt_mAA --gptune beta mAA adapt_mAA_factor \
       2>&1 | tee "${test_name}.out"

mkdir -p "${output_dir}/${test_name}"
mv gptune.db "${test_name}.out" "${output_dir}/${test_name}/."

test_name="p_${power}_beta_adapt-mAA_delayAA"
python stripped_down_tango_example_kinsol_gptune.py \
       --p ${power} --adapt_mAA --gptune beta mAA adapt_mAA_factor delayAA \
       2>&1 | tee "${test_name}.out"

mkdir -p "${output_dir}/${test_name}"
mv gptune.db "${test_name}.out" "${output_dir}/${test_name}/."

# -------------------
# Adaptive beta and m
# -------------------

test_name="p_${power}_adapt-beta_adapt-mAA"
python stripped_down_tango_example_kinsol_gptune.py \
       --p ${power} --adapt_beta --adapt_mAA --gptune beta mAA adapt_mAA_factor adapt_beta_factor \
       2>&1 | tee "${test_name}.out"

mkdir -p "${output_dir}/${test_name}"
mv gptune.db "${test_name}.out" "${output_dir}/${test_name}/."

test_name="p_${power}_adapt-beta_adapt-mAA_delayAA"
python stripped_down_tango_example_kinsol_gptune.py \
       --p ${power} --adapt_beta --adapt_mAA --gptune beta mAA adapt_mAA_factor adapt_beta_factor delayAA \
       2>&1 | tee "${test_name}.out"

mkdir -p "${output_dir}/${test_name}"
mv gptune.db "${test_name}.out" "${output_dir}/${test_name}/."

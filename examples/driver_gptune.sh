#!/bin/bash

set -e
set -o pipefail

run_tests() {

    power=$1
    IC=$2
    tol=$3

    output_dir="gptune_output_p_$1_tol_$3"

    # -------------
    # No adaptivity
    # -------------

    test_name="p_${power}_beta"
    python stripped_down_tango_example_kinsol_gptune.py \
           --p ${power} --IC ${IC} --tol ${tol} \
           --gptune beta \
           --samples 50 \
           2>&1 | tee "${test_name}.out"

    mkdir -p "${output_dir}/${test_name}"
    mv gptune.db "${test_name}.out" "${output_dir}/${test_name}/."

    test_name="p_${power}_beta_mAA"
    python stripped_down_tango_example_kinsol_gptune.py \
           --p ${power} --IC ${IC} --tol ${tol} \
           --gptune beta mAA \
           --samples 50 \
           2>&1 | tee "${test_name}.out"

    mkdir -p "${output_dir}/${test_name}"
    mv gptune.db "${test_name}.out" "${output_dir}/${test_name}/."

    test_name="p_${power}_beta_mAA_delayAA"
    python stripped_down_tango_example_kinsol_gptune.py \
           --p ${power} --IC ${IC} --tol ${tol} \
           --gptune beta mAA delayAA \
           --samples 50 \
           2>&1 | tee "${test_name}.out"

    mkdir -p "${output_dir}/${test_name}"
    mv gptune.db "${test_name}.out" "${output_dir}/${test_name}/."

    # -------------
    # Adaptive beta
    # -------------

    test_name="p_${power}_adapt-beta_mAA"
    python stripped_down_tango_example_kinsol_gptune.py \
           --p ${power} --IC ${IC} --tol ${tol} \
           --adapt_beta --gptune beta mAA adapt_beta_factor \
            --samples 50 \
           2>&1 | tee "${test_name}.out"

    mkdir -p "${output_dir}/${test_name}"
    mv gptune.db "${test_name}.out" "${output_dir}/${test_name}/."

    test_name="p_${power}_adapt-beta_mAA_delayAA"
    python stripped_down_tango_example_kinsol_gptune.py \
           --p ${power} --IC ${IC} --tol ${tol} \
           --adapt_beta --gptune beta mAA adapt_beta_factor delayAA \
           --samples 50 \
           2>&1 | tee "${test_name}.out"

    mkdir -p "${output_dir}/${test_name}"
    mv gptune.db "${test_name}.out" "${output_dir}/${test_name}/."

    # -------------
    # Adaptive m
    # -------------

    test_name="p_${power}_beta_adapt-mAA"
    python stripped_down_tango_example_kinsol_gptune.py \
           --p ${power} --IC ${IC} --tol ${tol} \
           --adapt_mAA --gptune beta mAA adapt_mAA_factor \
            --samples 50 \
           2>&1 | tee "${test_name}.out"

    mkdir -p "${output_dir}/${test_name}"
    mv gptune.db "${test_name}.out" "${output_dir}/${test_name}/."

    test_name="p_${power}_beta_adapt-mAA_delayAA"
    python stripped_down_tango_example_kinsol_gptune.py \
           --p ${power} --IC ${IC} --tol ${tol} \
           --adapt_mAA --gptune beta mAA adapt_mAA_factor delayAA \
            --samples 50 \
           2>&1 | tee "${test_name}.out"

    mkdir -p "${output_dir}/${test_name}"
    mv gptune.db "${test_name}.out" "${output_dir}/${test_name}/."

    # -------------------
    # Adaptive beta and m
    # -------------------

    test_name="p_${power}_adapt-beta_adapt-mAA"
    python stripped_down_tango_example_kinsol_gptune.py \
           --p ${power} --IC ${IC} --tol ${tol} \
           --adapt_beta --adapt_mAA \
           --gptune beta mAA adapt_mAA_factor adapt_beta_factor \
            --samples 50 \
           2>&1 | tee "${test_name}.out"

    mkdir -p "${output_dir}/${test_name}"
    mv gptune.db "${test_name}.out" "${output_dir}/${test_name}/."

    test_name="p_${power}_adapt-beta_adapt-mAA_delayAA"
    python stripped_down_tango_example_kinsol_gptune.py \
           --p ${power} --IC ${IC} --tol ${tol} \
           --adapt_beta --adapt_mAA \
           --gptune beta mAA adapt_mAA_factor adapt_beta_factor delayAA \
           --samples 50 \
           2>&1 | tee "${test_name}.out"

    mkdir -p "${output_dir}/${test_name}"
    mv gptune.db "${test_name}.out" "${output_dir}/${test_name}/."
}

rm -rf gptune.db/

run_tests 2 pow 1.0e-4
run_tests 10 lin 1.0e-4

run_tests 2 pow 1.0e-6
run_tests 10 lin 1.0e-6

run_tests 2 pow 1.0e-8
run_tests 10 lin 1.0e-8

run_tests 2 pow 1.0e-11
run_tests 10 lin 1.0e-11

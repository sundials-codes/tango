#!/usr/bin/env python3

import os
import stat
import numpy as np


def write_command(jobfile, form, noise, power, alpha, beta, aa_m, aa_delay,
                  aa_damping, test_num, total_tests):

    setup = (f"Test {test_num} of {total_tests}: ")
    setup += (f"Form = {form}, Noise = {noise}, Power = {power:d}, "
              f"Alpha = {alpha:f}, Beta = {beta:f}, m = {aa_m:d}, "
              f"aa_delay = {aa_delay:d}, aa_damping = {aa_damping:f}")

    # adjust damping for higher powers of p
    if power > 2:
        if alpha < 1.0:
            alpha = alpha / p
        if beta < 1.0:
            beta = beta / p
        if aa_damping < 1.0:
            aa_damping = aa_damping / p

    if f == "p":
        cmd = f"echo \"{setup}\"\n"
        cmd += f"./stripped_down_tango_example_pq_kinsol.py \\\n"
        if noise:
            cmd += "    --noise \\\n"
        cmd += (f"    --p {power} \\\n"
                f"    --alpha {alpha:f} \\\n"
                f"    --beta {beta:f} \\\n"
                f"    --max_iterations 200 \\\n"
                f"    --aa_m {aa_m:d} \\\n"
                f"    --aa_delay {aa_delay:d} \\\n"
                f"    --aa_damping {aa_damping:f} \\\n"
                f"    --plot_off\n\n")

    jobfile.write(cmd)

    return 1

# ---------------
# Test parameters
# ---------------

# problem setup options
forms = ["p"]   # relax state or diffusion
noise = [False]  # use noisy flux
powers = [2]     # flux power

# solver options
damping = np.linspace(0.1, 0.9, 9)  # damping values
accel = range(0,5)                  # acceleration space depth
delay = [0, 5]                      # delay length

# number of tests to create
if 0 in accel:
    tests_per_setup = 4 * len(damping) * ((len(accel) - 1) * len(delay) + 1)
else:
    tests_per_setup = 4 * len(damping) * len(accel) * len(delay)

total_tests = len(forms) * len(noise) * len(powers) * tests_per_setup

print(f"Creating {total_tests} tests, {tests_per_setup} per setup")

# ----------
# Make tests
# ----------

for f in forms:
    for n in noise:
        for p in powers:

            # create new script for PQ vs D form, +noise, and power
            fname = f"run_tango_{f}"
            if n:
                fname += '_noise'
            fname += f"_p_{p}.sh"

            c = 1

            with open(fname, "w") as fn:

                fn.write("#!/bin/bash\n\n")
                fn.write("source env.sh\n\n")

                for m in accel:
                    if m == 0:
                        # alpha damping
                        for a in damping:
                            c += write_command(fn, f, n, p, a, 1.0, m, 0,
                                               1.0, c, tests_per_setup)
                        # beta damping
                        for b in damping:
                            c += write_command(fn, f, n, p, 1.0, b, m, 0,
                                               1.0, c, tests_per_setup)
                        # alpha & beta damping
                        for ab in damping:
                            c += write_command(fn, f, n, p, ab, ab, m, 0,
                                               1.0, c, tests_per_setup)
                        # Anderson damping
                        for ad in damping:
                            c += write_command(fn, f, n, p, 1.0, 1.0, m, 0,
                                               ad, c, tests_per_setup)
                    else:
                        for d in delay:
                            # alpha damping
                            for a in damping:
                                c += write_command(fn, f, n, p, a, 1.0, m, d,
                                                   1.0, c, tests_per_setup)
                            # beta damping
                            for b in damping:
                                c += write_command(fn, f, n, p, 1.0, b, m, d,
                                                   1.0, c, tests_per_setup)
                            # alpha & beta damping
                            for ab in damping:
                                c += write_command(fn, f, n, p, ab, ab, m, d,
                                                   1.0, c, tests_per_setup)
                            # Anderson damping
                            for ad in damping:
                                c += write_command(fn, f, n, p, 1.0, 1.0, m, d,
                                                   ad, c, tests_per_setup)

            # make script executable
            st = os.stat(fname)
            os.chmod(fname, st.st_mode | stat.S_IEXEC)

print("Done")

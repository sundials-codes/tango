#!/usr/bin/env python3


def main():

    import os
    import stat
    import numpy as np

    # ---------------
    # Test parameters
    # ---------------

    # problem setup options
    forms = ["p"]     # relax state or diffusion
    noise = [False]   # use noisy flux
    powers = [10]     # flux power
    #tests = ['alpha', 'beta', 'alpha-beta', 'aa-damp']
    tests = ['alpha', 'alpha-beta']
    #tests = ['alpha-beta']
    ic = 'linear'

    # solver options
    maxiters = 2000
    damping = np.linspace(0.1, 0.9, 9)  # damping values
    accel = range(0,5)                  # acceleration space depth
    delay = [30]                        # delay length

    # number of tests to create
    if 0 in accel:
        tests_per_setup = len(tests) * len(damping) * ((len(accel) - 1) *
                                                       len(delay) + 1)
    else:
        tests_per_setup = len(tests) * len(damping) * len(accel) * len(delay)

    total_tests = len(forms) * len(noise) * len(powers) * tests_per_setup

    print(f"Creating {total_tests} tests, {tests_per_setup} per setup")

    # ----------
    # Make tests
    # ----------

    for f in forms:
        for n in noise:
            for p in powers:

                # create new script for PQ vs D form, +noise, and power
                fname = f"run_tango_gfun-{f}"
                if n:
                    fname += '_add-noise'
                else:
                    fname += '_no-noise'
                fname += f"_p_{p}.sh"

                c = 1

                with open(fname, "w") as fn:

                    fn.write("#!/bin/bash\n\n")
                    fn.write("source env.sh\n\n")

                    for m in accel:
                        if m == 0:
                            # alpha damping
                            if "alpha" in tests:
                                for a in damping:
                                    c += write_command(fn,
                                                       f, n, p, ic,
                                                       a, 1.0,
                                                       maxiters,
                                                       m, 0, 1.0,
                                                       c, tests_per_setup)
                            # beta damping
                            if "beta" in tests:
                                for b in damping:
                                    c += write_command(fn,
                                                       f, n, p, ic,
                                                       1.0, b,
                                                       maxiters,
                                                       m, 0, 1.0,
                                                       c, tests_per_setup)
                            # alpha & beta damping
                            if "alpha-beta" in tests:
                                for ab in damping:
                                    c += write_command(fn,
                                                       f, n, p, ic,
                                                       ab, ab,
                                                       maxiters,
                                                       m, 0, 1.0,
                                                       c, tests_per_setup)
                            # Anderson damping
                            if "aa-damp" in tests:
                                for ad in damping:
                                    c += write_command(fn,
                                                       f, n, p, ic,
                                                       1.0, 1.0,
                                                       maxiters,
                                                       m, 0, ad,
                                                       c, tests_per_setup)
                        else:
                            for d in delay:
                                # alpha damping
                                if "alpha" in tests:
                                    for a in damping:
                                        c += write_command(fn,
                                                           f, n, p, ic,
                                                           a, 1.0,
                                                           maxiters,
                                                           m, d, 1.0,
                                                           c, tests_per_setup)
                                # beta damping
                                if "beta" in tests:
                                    for b in damping:
                                        c += write_command(fn,
                                                           f, n, p, ic,
                                                           1.0, b,
                                                           maxiters,
                                                           m, d, 1.0,
                                                           c, tests_per_setup)
                                # alpha & beta damping
                                if "alpha-beta" in tests:
                                    for ab in damping:
                                        c += write_command(fn,
                                                           f, n, p, ic,
                                                           ab, ab,
                                                           maxiters,
                                                           m, d, 1.0,
                                                           c, tests_per_setup)
                                # Anderson damping
                                if "aa-damp" in tests:
                                    for ad in damping:
                                        c += write_command(fn,
                                                           f, n, p, ic,
                                                           1.0, 1.0,
                                                           maxiters,
                                                           m, d, ad,
                                                           c, tests_per_setup)

                    # make script executable
                    st = os.stat(fname)
                    os.chmod(fname, st.st_mode | stat.S_IEXEC)

    print("Done")


def write_command(jobfile, form, noise, power, ic, alpha, beta, maxiters,
                  aa_m, aa_delay, aa_damping, test_num, total_tests):

    # adjust damping for higher powers of p
    if power > 2:
        if alpha < 1.0:
            alpha = alpha / power
        if beta < 1.0:
            beta = beta / power
        if aa_damping < 1.0:
            aa_damping = aa_damping / power

    setup = (f"Test {test_num} of {total_tests}: ")
    setup += (f"Form = {form}, "
              f"Noise = {noise}, "
              f"Power = {power:d}, "
              f"IC = {ic}, "
              f"Alpha = {alpha:f}, "
              f"Beta = {beta:f}, "
              f"maxiters = {maxiters:d}, "
              f"m = {aa_m:d}, "
              f"aa_delay = {aa_delay:d}, "
              f"aa_damping = {aa_damping:f}")

    if form == "p":
        cmd = "echo \"====================\"\n"
        cmd += f"echo \"{setup}\"\n"
        cmd += f"./stripped_down_tango_example_pq_kinsol.py \\\n"
        if noise:
            cmd += "    --noise \\\n"
        cmd += (f"    --p {power:d} \\\n"
                f"    --initial_condition {ic} \\\n"
                f"    --alpha {alpha:f} \\\n"
                f"    --beta {beta:f} \\\n"
                f"    --max_iterations {maxiters:d} \\\n"
                f"    --aa_m {aa_m:d} \\\n"
                f"    --aa_delay {aa_delay:d} \\\n"
                f"    --aa_damping {aa_damping:f} \\\n"
                f"    --plot_off\n\n")
        cmd += "echo \"====================\"\n"

    jobfile.write(cmd)

    return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

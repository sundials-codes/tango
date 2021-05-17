#!/usr/bin/env python3

import os
import stat
import numpy as np

test_count = 0


def write_command(jobfile, f, n, p, a, b, m, d):

        global test_count
        test_count += 1
        print(f"{test_count}: ", end='')
        print(f"Form = {f}, Noise = {n}, Power = {p:d}, Alpha = {a:f}, Beta = {b:f}, m = {m:d}, delay = {d:d}")

        # adjust damping for higher powers of p
        if p > 2 and a < 1.0:
            a = a / p
        if p > 2 and b < 1.0:
            b = b / p

        if f == "pq":
            cmd = f"./stripped_down_tango_example_pq_kinsol.py \\\n"
            if n:
                cmd += "    --noise \\\n"
            cmd += (f"    --p {p} \\\n"
                    f"    --alpha {a:f} \\\n"
                    f"    --beta {b:f} \\\n"
                    f"    --max_iterations 200 \\\n"
                    f"    --aa_m {m:d} \\\n"
                    f"    --aa_delay {d:d} \\\n"
                    f"    --plot_off\n\n")

        jobfile.write(cmd)


# ---------------
# Test parameters
# ---------------

forms = ["pq"]
noise = [False]
powers = [2]
damping = np.linspace(0.1, 0.9, 9)
accel = range(0,4)
delay = [0, 5]

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

            with open(fname, "w") as jobfile:

                jobfile.write("#!/bin/bash\n\n")
                jobfile.write("source env.sh \n\n")

                for m in accel:
                    if m == 0:
                        # alpha damping
                        for a in damping:
                            write_command(jobfile, f, n, p, a, 1.0, m, 0)
                        # beta damping
                        for b in damping:
                            write_command(jobfile, f, n, p, 1.0, b, m, 0)
                        # alpha & beta damping
                        for c in damping:
                            write_command(jobfile, f, n, p, c, c, m, 0)
                    else:
                        for d in delay:
                            # alpha damping
                            for a in damping:
                                write_command(jobfile, f, n, p, a, 1.0, m, d)
                            # beta damping
                            for b in damping:
                                write_command(jobfile, f, n, p, 1.0, b, m, d)
                            # alpha & beta damping
                            for c in damping:
                                write_command(jobfile, f, n, p, c, c, m, d)

            # make script executable
            st = os.stat(fname)
            os.chmod(fname, st.st_mode | stat.S_IEXEC)

#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Script to extract data from a file given a specific key
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# main routine
# -----------------------------------------------------------------------------
def main():

    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    parser = argparse.ArgumentParser(description='Extract data from files')

    parser.add_argument('filenames', type=str, nargs="+",
                        help='Files to parse')

    # parse command line args
    args = parser.parse_args()

    # create figure and axes
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    for fn in args.filenames:

        # get data
        nni, beta, gain, lAA = read_log(fn)

        # parse file name to get run settings
        fname = os.path.splitext(os.path.basename(fn))[0].split("_")

        # get parameters
        p = float(fname[1])
        b = float(fname[3])
        adapt_b = int(fname[5] == 'True')
        adapt_b_factor = float(fname[7])
        m = int(fname[9])
        delay = int(fname[11])
        adapt_m = int(fname[13] == 'True')
        adapt_m_factor = float(fname[15])

        # create label for this data
        label = ''
        if adapt_b:
            label = f'$\\beta^*_0$={b:.2f} ({adapt_b_factor:.2f})'
        else:
            label = f'$\\beta$={b:.2f}'
        if m > 0 and not adapt_m:
            label += f', m={m}'
        if m > 0 and adapt_m:
            label += f', m=0..{m} ({adapt_m_factor:.2f})'
        if delay > 0:
            label += f', delay={delay}'

        ax1.plot(range(len(beta)), beta, label=label)
        ax2.plot(range(len(gain)), gain, label=label)
        ax3.plot(range(len(lAA)), lAA, label=label)

    ax1.set_title("Beta")
    ax2.set_title("Gain")
    ax3.set_title("lAA")

    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')

    plt.show()

def read_log(logfile):

    import shlex

    print(f"Reading {logfile}")

    nni = list()
    beta = list()
    gain = list()
    lAA = list()

    # read output file line by line
    with open(logfile,'r') as fn:

        index = 0

        for line in fn:

            # skip empty lines
            if not line.strip():
                continue

            # split line into list
            text = shlex.split(line)

            if "nni" in line:
                nni.append(int(text[2][:-1]))

            if "beta" in line:
                if "gain" in line:
                    beta.append(float(text[-1]))
                    gain.append(float(text[8]))
                else:
                    beta.append(float(text[2]))
                    gain.append(float('nan'))

            if "lAA" in line:
                lAA.append(int(text[2]))

    return nni, beta, gain, lAA


# -----------------------------------------------------------------------------
# run the main routine
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    sys.exit(main())

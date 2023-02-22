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

    parser.add_argument('--save', action='store_true',
                        help='save figure to file')

    # parse command line args
    args = parser.parse_args()

    # create figure and axes
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    for fn in args.filenames:

        # get data
        lAA, beta, gain = read_log(fn)

        # parse file name to get run settings
        fname_list = os.path.splitext(os.path.basename(fn))[0].split("_")
        fname_dict = {fname_list[i]: fname_list[i + 1] for i in range(0, len(fname_list), 2)}

        # get method name and parameters, set title
        method = "KINSOL"
        if "p" in fname_dict:
            power = float(fname_dict["p"])
        else:
            power = 0
        if "beta" in fname_dict:
            b = float(fname_dict["beta"])
        else:
            b = 1.0
        if "m" in fname_dict:
            m = int(fname_dict["m"])
        else:
            m = 0
        if "delay" in fname_dict:
            delay = int(fname_dict["delay"])
        else:
            delay = 0
        if "adapt-m" in fname_dict:
            adapt_m = int(fname_dict["adapt-m"] == 'True')
        else:
            adapt_m = 0
        if "adapt-m-factor" in fname_dict:
            adapt_m_factor = float(fname_dict["adapt-m-factor"])
        else:
            adapt_m_factor = 1.0
        if "adapt-beta" in fname_dict:
            adapt_b = int(fname_dict["adapt-beta"] == 'True')
        else:
            adapt_b = 0
        if "adapt-beta-factor" in fname_dict:
            adapt_b_factor = float(fname_dict["adapt-beta-factor"])
        else:
            adapt_b_factor = 0.5

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

        ax1.plot(range(len(lAA)), lAA, label=label, marker='.')
        ax2.plot(range(len(beta)), beta, label=label, marker='.')
        ax3.plot(range(len(gain)), gain, label=label, marker='.')

    ax1.set_title("Depth History")
    ax2.set_title("$\\beta$ History")
    ax3.set_title("Gain History")

    ax1.set_xlabel('Iteration')
    ax2.set_xlabel('Iteration')
    ax3.set_xlabel('Iteration')

    ax1.set_ylabel('$m_k$')
    ax2.set_ylabel('$\\beta_k$')
    ax3.set_ylabel('$\\theta_k$')

    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')

    if args.save:
        fig1.savefig("fig-depth.pdf", bbox_inches='tight')
        fig2.savefig("fig-beta.pdf", bbox_inches='tight')
        fig3.savefig("fig-gain.pdf", bbox_inches='tight')
    else:
        plt.show()

def read_log(logfile):

    import shlex

    print(f"Reading {logfile}")

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

            if "beta" in line:
                beta.append(float(text[-1]))
                if float(text[2]) < 0:
                    gain.append(float('nan'))
                else:
                    gain.append(float(text[2]))

            if "lAA" in line:
                lAA.append(int(text[2]))

    return lAA, beta, gain


# -----------------------------------------------------------------------------
# run the main routine
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    sys.exit(main())

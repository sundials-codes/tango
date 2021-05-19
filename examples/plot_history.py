#!/usr/bin/env python

# ****** Main ***** #
def main():

    import argparse
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Run Shestakov example')

    # output files to plot
    parser.add_argument('outfiles', type=str, nargs='+',
                        help='Output files to pliot')


    parser.add_argument('--row', type=int, default=0,
                        help='Which row to plots')

    parser.add_argument('--legend', type=str, nargs='+',
                        help='set legend entries')



    # debugging options
    parser.add_argument('--debug', action='store_true',
                        help='Enable debugging output')

    # parse command line args
    args = parser.parse_args()

    # create figure and axes
    fig, ax = plt.subplots()

    # iterate over files and add to figure
    for i, outfile in enumerate(args.outfiles):

        if args.debug:
            print(i, outfile)

        # parse file name to get run settings
        fname = os.path.basename(outfile).split("_")

        if "gfunp" in fname:
            gfun = 'p'
        if "gfunpq" in fname:
            gfun = 'pq'
        if "gfund" in fname:
            gfun = 'd'

        power = fname[3]
        alpha = fname[5]
        beta = fname[7]
        gamma = fname[9]
        aa_m = fname[11]
        aa_delay = fname[13]
        aa_damp = fname[15]

        # load data
        data = np.loadtxt(outfile)

        # number of columns
        num_iters = np.shape(data)[1]

        iters = range(num_iters)

        if args.legend:
            label = args.legend[i]
        else:
            label = None

        # plot method convergence
        ax.semilogy(iters, data[args.row,:], nonpositive='clip', label=label)

    ax.legend(loc='best')

    plt.grid()
    plt.show()


# ****** run main ****** #
if __name__ == '__main__':
    import sys
    sys.exit(main())
